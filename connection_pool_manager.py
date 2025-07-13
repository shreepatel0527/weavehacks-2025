"""
Advanced connection pooling for database, Redis, and API connections
"""
import asyncio
import aiohttp
import aioredis
import asyncpg
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import time
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import weave

class ConnectionType(Enum):
    DATABASE = "database"
    REDIS = "redis"
    HTTP = "http"
    WEBSOCKET = "websocket"

@dataclass
class ConnectionStats:
    """Statistics for connection pool"""
    created: int = 0
    active: int = 0
    idle: int = 0
    failed: int = 0
    total_requests: int = 0
    total_wait_time: float = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def average_wait_time(self) -> float:
        return self.total_wait_time / self.total_requests if self.total_requests > 0 else 0

class DatabaseConnectionPool:
    """PostgreSQL connection pool with advanced features"""
    
    def __init__(self, 
                 dsn: str,
                 min_size: int = 10,
                 max_size: int = 20,
                 max_queries: int = 50000,
                 max_inactive_time: int = 300):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.max_queries = max_queries
        self.max_inactive_time = max_inactive_time
        
        self.pool = None
        self.stats = ConnectionStats()
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger('db_pool')
        
        # Connection health tracking
        self._health_check_interval = 60  # seconds
        self._health_check_task = None
        
    async def initialize(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_time,
                command_timeout=60
            )
            
            self.stats.created = self.min_size
            self.logger.info(f"Database pool initialized with {self.min_size} connections")
            
            # Start health check
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
        except Exception as e:
            self.stats.errors.append(str(e))
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close connection pool"""
        if self._health_check_task:
            self._health_check_task.cancel()
            
        if self.pool:
            await self.pool.close()
            self.logger.info("Database pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        start_time = time.time()
        
        async with self._lock:
            self.stats.total_requests += 1
            
        try:
            async with self.pool.acquire() as connection:
                wait_time = time.time() - start_time
                self.stats.total_wait_time += wait_time
                self.stats.active += 1
                
                yield connection
                
        except Exception as e:
            self.stats.failed += 1
            self.stats.errors.append(f"Connection error: {str(e)}")
            raise
        finally:
            self.stats.active -= 1
    
    async def execute(self, query: str, *args, timeout: float = None):
        """Execute query with connection from pool"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)
    
    async def fetch(self, query: str, *args, timeout: float = None):
        """Fetch results with connection from pool"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)
    
    async def fetchrow(self, query: str, *args, timeout: float = None):
        """Fetch single row with connection from pool"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)
    
    async def _health_check_loop(self):
        """Periodic health check of connections"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                # Test connection
                async with self.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                
                # Update stats
                pool_stats = self.pool._get_stats()
                self.stats.idle = pool_stats['idle']
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                self.stats.errors.append(f"Health check: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            'created': self.stats.created,
            'active': self.stats.active,
            'idle': self.stats.idle,
            'failed': self.stats.failed,
            'total_requests': self.stats.total_requests,
            'average_wait_time_ms': self.stats.average_wait_time * 1000,
            'recent_errors': self.stats.errors[-10:]
        }

class RedisConnectionPool:
    """Redis connection pool with pub/sub support"""
    
    def __init__(self,
                 url: str,
                 min_connections: int = 5,
                 max_connections: int = 10,
                 decode_responses: bool = True):
        self.url = url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        
        self.pool = None
        self.pubsub_pool = None
        self.stats = ConnectionStats()
        self.logger = logging.getLogger('redis_pool')
        
    async def initialize(self):
        """Initialize Redis pools"""
        try:
            # Main connection pool
            self.pool = await aioredis.create_redis_pool(
                self.url,
                minsize=self.min_connections,
                maxsize=self.max_connections,
                encoding='utf-8' if self.decode_responses else None
            )
            
            # Separate pool for pub/sub
            self.pubsub_pool = await aioredis.create_redis_pool(
                self.url,
                minsize=2,
                maxsize=5,
                encoding='utf-8' if self.decode_responses else None
            )
            
            self.stats.created = self.min_connections + 2
            self.logger.info("Redis pools initialized")
            
        except Exception as e:
            self.stats.errors.append(str(e))
            self.logger.error(f"Failed to initialize Redis pool: {e}")
            raise
    
    async def close(self):
        """Close Redis pools"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            
        if self.pubsub_pool:
            self.pubsub_pool.close()
            await self.pubsub_pool.wait_closed()
            
        self.logger.info("Redis pools closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire Redis connection"""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            yield self.pool
            
            wait_time = time.time() - start_time
            self.stats.total_wait_time += wait_time
            
        except Exception as e:
            self.stats.failed += 1
            self.stats.errors.append(f"Redis error: {str(e)}")
            raise
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        async with self.acquire() as redis:
            return await redis.get(key)
    
    async def set(self, key: str, value: str, expire: int = None):
        """Set value in Redis"""
        async with self.acquire() as redis:
            await redis.set(key, value, expire=expire)
    
    async def delete(self, key: str):
        """Delete key from Redis"""
        async with self.acquire() as redis:
            await redis.delete(key)
    
    async def publish(self, channel: str, message: str):
        """Publish message to channel"""
        async with self.pubsub_pool.get() as conn:
            await conn.publish(channel, message)
    
    async def subscribe(self, *channels):
        """Subscribe to channels"""
        async with self.pubsub_pool.get() as conn:
            ch = conn.channel()
            await ch.subscribe(*channels)
            return ch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            'created': self.stats.created,
            'total_requests': self.stats.total_requests,
            'failed': self.stats.failed,
            'average_wait_time_ms': self.stats.average_wait_time * 1000,
            'pool_size': self.pool.size if self.pool else 0,
            'pool_free': self.pool.freesize if self.pool else 0
        }

class HTTPConnectionPool:
    """HTTP connection pool with advanced features"""
    
    def __init__(self,
                 connector_limit: int = 100,
                 connector_limit_per_host: int = 30,
                 timeout: aiohttp.ClientTimeout = None,
                 keepalive_timeout: int = 30):
        
        self.connector_limit = connector_limit
        self.connector_limit_per_host = connector_limit_per_host
        self.keepalive_timeout = keepalive_timeout
        
        # Default timeout
        self.timeout = timeout or aiohttp.ClientTimeout(
            total=30,
            connect=5,
            sock_connect=5,
            sock_read=10
        )
        
        self.connector = None
        self.session = None
        self.stats = ConnectionStats()
        self.logger = logging.getLogger('http_pool')
        
        # Request tracking
        self._active_requests = set()
        self._request_history = deque(maxlen=1000)
        
    async def initialize(self):
        """Initialize HTTP session with connection pool"""
        try:
            # Create connector with connection pooling
            self.connector = aiohttp.TCPConnector(
                limit=self.connector_limit,
                limit_per_host=self.connector_limit_per_host,
                ttl_dns_cache=300,
                keepalive_timeout=self.keepalive_timeout,
                enable_cleanup_closed=True
            )
            
            # Create session
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'LabAssistant/3.0',
                    'Accept': 'application/json'
                }
            )
            
            self.stats.created = 1
            self.logger.info("HTTP connection pool initialized")
            
        except Exception as e:
            self.stats.errors.append(str(e))
            self.logger.error(f"Failed to initialize HTTP pool: {e}")
            raise
    
    async def close(self):
        """Close HTTP session and connector"""
        # Wait for active requests
        if self._active_requests:
            await asyncio.gather(
                *self._active_requests,
                return_exceptions=True
            )
        
        if self.session:
            await self.session.close()
            
        if self.connector:
            await self.connector.close()
            
        self.logger.info("HTTP connection pool closed")
    
    @asynccontextmanager
    async def request(self, method: str, url: str, **kwargs):
        """Make HTTP request with connection pooling"""
        start_time = time.time()
        request_id = f"{method}:{url}:{start_time}"
        self._active_requests.add(request_id)
        self.stats.total_requests += 1
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                wait_time = time.time() - start_time
                self.stats.total_wait_time += wait_time
                
                # Track request
                self._request_history.append({
                    'method': method,
                    'url': url,
                    'status': response.status,
                    'duration': wait_time,
                    'timestamp': datetime.now()
                })
                
                yield response
                
        except Exception as e:
            self.stats.failed += 1
            self.stats.errors.append(f"HTTP error: {str(e)}")
            raise
        finally:
            self._active_requests.discard(request_id)
    
    async def get(self, url: str, **kwargs):
        """GET request"""
        async with self.request('GET', url, **kwargs) as response:
            return await response.json()
    
    async def post(self, url: str, **kwargs):
        """POST request"""
        async with self.request('POST', url, **kwargs) as response:
            return await response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        # Calculate request rate
        recent_requests = [
            r for r in self._request_history
            if (datetime.now() - r['timestamp']).total_seconds() < 60
        ]
        
        requests_per_minute = len(recent_requests)
        
        # Calculate average response time
        if recent_requests:
            avg_response_time = sum(r['duration'] for r in recent_requests) / len(recent_requests)
        else:
            avg_response_time = 0
        
        return {
            'active_requests': len(self._active_requests),
            'total_requests': self.stats.total_requests,
            'failed_requests': self.stats.failed,
            'requests_per_minute': requests_per_minute,
            'average_response_time_ms': avg_response_time * 1000,
            'connector_limit': self.connector_limit,
            'recent_errors': self.stats.errors[-5:]
        }

class ConnectionPoolManager:
    """Manage all connection pools"""
    
    def __init__(self):
        self.pools = {}
        self.logger = logging.getLogger('pool_manager')
        self._initialized = False
        
        # Initialize W&B
        weave.init('connection-pooling')
    
    async def initialize(self, config: Dict[str, Any]):
        """Initialize all connection pools"""
        if self._initialized:
            return
            
        # Database pool
        if 'database' in config:
            db_config = config['database']
            self.pools['database'] = DatabaseConnectionPool(
                dsn=db_config['dsn'],
                min_size=db_config.get('min_size', 10),
                max_size=db_config.get('max_size', 20)
            )
            await self.pools['database'].initialize()
        
        # Redis pool
        if 'redis' in config:
            redis_config = config['redis']
            self.pools['redis'] = RedisConnectionPool(
                url=redis_config['url'],
                min_connections=redis_config.get('min_connections', 5),
                max_connections=redis_config.get('max_connections', 10)
            )
            await self.pools['redis'].initialize()
        
        # HTTP pool
        if 'http' in config:
            http_config = config['http']
            self.pools['http'] = HTTPConnectionPool(
                connector_limit=http_config.get('connector_limit', 100),
                connector_limit_per_host=http_config.get('connector_limit_per_host', 30)
            )
            await self.pools['http'].initialize()
        
        self._initialized = True
        self.logger.info("All connection pools initialized")
        
        # Start monitoring
        asyncio.create_task(self._monitor_pools())
    
    async def close(self):
        """Close all connection pools"""
        for name, pool in self.pools.items():
            try:
                await pool.close()
                self.logger.info(f"Closed {name} pool")
            except Exception as e:
                self.logger.error(f"Error closing {name} pool: {e}")
        
        self._initialized = False
    
    def get_pool(self, pool_type: str):
        """Get specific connection pool"""
        return self.pools.get(pool_type)
    
    @property
    def database(self) -> DatabaseConnectionPool:
        """Get database pool"""
        return self.pools.get('database')
    
    @property
    def redis(self) -> RedisConnectionPool:
        """Get Redis pool"""
        return self.pools.get('redis')
    
    @property
    def http(self) -> HTTPConnectionPool:
        """Get HTTP pool"""
        return self.pools.get('http')
    
    async def _monitor_pools(self):
        """Monitor pool statistics"""
        while self._initialized:
            try:
                await asyncio.sleep(30)  # Log every 30 seconds
                
                # Collect stats
                all_stats = {}
                for name, pool in self.pools.items():
                    all_stats[name] = pool.get_stats()
                
                # Log to W&B
                weave.log({
                    'connection_pools': all_stats
                })
                
                # Log summary
                total_requests = sum(
                    stats.get('total_requests', 0)
                    for stats in all_stats.values()
                )
                
                total_failures = sum(
                    stats.get('failed', 0) or stats.get('failed_requests', 0)
                    for stats in all_stats.values()
                )
                
                self.logger.info(
                    f"Pool stats - Requests: {total_requests}, "
                    f"Failures: {total_failures}"
                )
                
            except Exception as e:
                self.logger.error(f"Error monitoring pools: {e}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        return {
            name: pool.get_stats()
            for name, pool in self.pools.items()
        }

# Connection pool singleton
_pool_manager = None

async def get_pool_manager() -> ConnectionPoolManager:
    """Get or create pool manager instance"""
    global _pool_manager
    
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    
    return _pool_manager

# Example usage
async def demo_connection_pooling():
    """Demonstrate connection pooling"""
    
    # Configuration
    config = {
        'database': {
            'dsn': 'postgresql://user:password@localhost/lab_assistant',
            'min_size': 10,
            'max_size': 20
        },
        'redis': {
            'url': 'redis://localhost:6379',
            'min_connections': 5,
            'max_connections': 10
        },
        'http': {
            'connector_limit': 100,
            'connector_limit_per_host': 30
        }
    }
    
    # Get pool manager
    pool_manager = await get_pool_manager()
    
    try:
        # Initialize pools
        await pool_manager.initialize(config)
        
        # Database operations
        if pool_manager.database:
            print("Testing database pool...")
            
            # Execute query
            result = await pool_manager.database.execute(
                "SELECT $1::text as message",
                "Hello from pool!"
            )
            print(f"Database result: {result}")
            
            # Fetch data
            rows = await pool_manager.database.fetch(
                "SELECT generate_series(1, 5) as num"
            )
            print(f"Fetched {len(rows)} rows")
        
        # Redis operations
        if pool_manager.redis:
            print("\nTesting Redis pool...")
            
            # Set and get
            await pool_manager.redis.set('test_key', 'test_value', expire=60)
            value = await pool_manager.redis.get('test_key')
            print(f"Redis value: {value}")
            
            # Pub/sub
            channel = await pool_manager.redis.subscribe('test_channel')
            await pool_manager.redis.publish('test_channel', 'Hello Redis!')
        
        # HTTP operations
        if pool_manager.http:
            print("\nTesting HTTP pool...")
            
            # Make concurrent requests
            tasks = []
            for i in range(5):
                task = pool_manager.http.get(
                    f"https://jsonplaceholder.typicode.com/posts/{i+1}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            print(f"Made {len(results)} HTTP requests")
        
        # Get statistics
        print("\nPool Statistics:")
        stats = pool_manager.get_all_stats()
        for name, pool_stats in stats.items():
            print(f"\n{name.upper()}:")
            for key, value in pool_stats.items():
                print(f"  {key}: {value}")
        
    finally:
        # Close pools
        await pool_manager.close()
        print("\nAll pools closed")

if __name__ == "__main__":
    # Note: This is a demo. In production, configure with actual connection strings
    print("Connection Pool Demo")
    print("Note: This requires actual database/Redis servers to be running")
    
    # Run demo
    # asyncio.run(demo_connection_pooling())