"""
Advanced multi-tier caching layer with TTL, LRU, and distributed support
"""
import asyncio
import time
import json
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps
import threading
import weakref
from enum import Enum
import logging
import numpy as np
import pandas as pd
import weave

class CacheLevel(Enum):
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"

@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    ttl: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl <= 0:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_memory = 0
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache"""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            entry = self.cache.pop(key)
            
            # Check expiration
            if entry.is_expired():
                self.current_memory -= entry.size_bytes
                self.misses += 1
                return None
            
            # Update and reinsert
            entry.access()
            self.cache[key] = entry
            self.hits += 1
            
            return entry
    
    def put(self, key: str, value: Any, ttl: float = 3600):
        """Put value in cache"""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache.pop(key)
                self.current_memory -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Evict if necessary
            self._evict_if_needed(size_bytes)
            
            # Add to cache
            self.cache[key] = entry
            self.current_memory += size_bytes
    
    def _evict_if_needed(self, needed_bytes: int):
        """Evict entries if needed"""
        while (len(self.cache) >= self.max_size or 
               self.current_memory + needed_bytes > self.max_memory_bytes):
            
            if not self.cache:
                break
            
            # Remove least recently used
            key, entry = self.cache.popitem(last=False)
            self.current_memory -= entry.size_bytes
            self.evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, dict)):
                return len(pickle.dumps(value))
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            else:
                return len(pickle.dumps(value))
        except:
            return 1000  # Default estimate
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'memory_mb': self.current_memory / (1024 * 1024),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }

class MultiTierCache:
    """Multi-tier caching system with memory, Redis, and disk"""
    
    def __init__(self, 
                 memory_size: int = 1000,
                 memory_mb: int = 100,
                 redis_client: Optional[Any] = None,
                 disk_path: Optional[str] = None):
        
        # Memory cache (L1)
        self.memory_cache = LRUCache(memory_size, memory_mb)
        
        # Redis cache (L2)
        self.redis_client = redis_client
        self.redis_prefix = "lab_cache:"
        
        # Disk cache (L3)
        self.disk_path = disk_path
        if disk_path:
            from pathlib import Path
            self.disk_cache_dir = Path(disk_path) / "cache"
            self.disk_cache_dir.mkdir(exist_ok=True)
        
        # Cache policies
        self.policies = {
            'memory_ttl': 300,  # 5 minutes
            'redis_ttl': 3600,  # 1 hour
            'disk_ttl': 86400,  # 24 hours
        }
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'redis_hits': 0,
            'disk_hits': 0,
            'total_misses': 0
        }
        
        self.logger = logging.getLogger('multi_tier_cache')
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy"""
        # Check memory cache
        entry = self.memory_cache.get(key)
        if entry:
            self.stats['memory_hits'] += 1
            return entry.value
        
        # Check Redis cache
        if self.redis_client:
            value = await self._get_from_redis(key)
            if value is not None:
                self.stats['redis_hits'] += 1
                # Promote to memory cache
                self.memory_cache.put(key, value, self.policies['memory_ttl'])
                return value
        
        # Check disk cache
        if self.disk_path:
            value = self._get_from_disk(key)
            if value is not None:
                self.stats['disk_hits'] += 1
                # Promote to higher tiers
                self.memory_cache.put(key, value, self.policies['memory_ttl'])
                if self.redis_client:
                    await self._put_to_redis(key, value)
                return value
        
        self.stats['total_misses'] += 1
        return None
    
    async def put(self, key: str, value: Any, 
                  levels: List[CacheLevel] = None):
        """Put value in specified cache levels"""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
        
        # Store in memory
        if CacheLevel.MEMORY in levels:
            self.memory_cache.put(key, value, self.policies['memory_ttl'])
        
        # Store in Redis
        if CacheLevel.REDIS in levels and self.redis_client:
            await self._put_to_redis(key, value)
        
        # Store on disk
        if CacheLevel.DISK in levels and self.disk_path:
            self._put_to_disk(key, value)
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            redis_key = f"{self.redis_prefix}{key}"
            data = await self.redis_client.get(redis_key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
        return None
    
    async def _put_to_redis(self, key: str, value: Any):
        """Put value in Redis"""
        try:
            redis_key = f"{self.redis_prefix}{key}"
            data = pickle.dumps(value)
            await self.redis_client.set(
                redis_key, data, 
                expire=self.policies['redis_ttl']
            )
        except Exception as e:
            self.logger.error(f"Redis put error: {e}")
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        try:
            file_path = self._get_disk_path(key)
            if file_path.exists():
                # Check if expired
                age = time.time() - file_path.stat().st_mtime
                if age > self.policies['disk_ttl']:
                    file_path.unlink()
                    return None
                
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Disk get error: {e}")
        return None
    
    def _put_to_disk(self, key: str, value: Any):
        """Put value in disk cache"""
        try:
            file_path = self._get_disk_path(key)
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self.logger.error(f"Disk put error: {e}")
    
    def _get_disk_path(self, key: str):
        """Get disk file path for key"""
        # Use hash to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.disk_cache_dir / f"{key_hash[:2]}" / f"{key_hash}.cache"
    
    def clear(self, levels: List[CacheLevel] = None):
        """Clear specified cache levels"""
        if levels is None:
            levels = list(CacheLevel)
        
        if CacheLevel.MEMORY in levels:
            self.memory_cache.clear()
        
        # Clear Redis and disk would be implemented similarly
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        
        total_hits = (self.stats['memory_hits'] + 
                     self.stats['redis_hits'] + 
                     self.stats['disk_hits'])
        
        total_requests = total_hits + self.stats['total_misses']
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'memory': memory_stats,
            'tier_hits': {
                'memory': self.stats['memory_hits'],
                'redis': self.stats['redis_hits'],
                'disk': self.stats['disk_hits']
            },
            'total_misses': self.stats['total_misses'],
            'overall_hit_rate': overall_hit_rate
        }

class CacheManager:
    """Global cache manager with advanced features"""
    
    def __init__(self):
        self.caches = {}
        self.default_cache = None
        self.logger = logging.getLogger('cache_manager')
        
        # Cache key patterns
        self.patterns = {}
        
        # Background tasks
        self._cleanup_task = None
        self._stats_task = None
        
        # Initialize W&B
        weave.init('cache-manager')
    
    def create_cache(self, name: str, **kwargs) -> MultiTierCache:
        """Create named cache"""
        cache = MultiTierCache(**kwargs)
        self.caches[name] = cache
        
        if self.default_cache is None:
            self.default_cache = cache
        
        return cache
    
    def get_cache(self, name: str = None) -> MultiTierCache:
        """Get cache by name"""
        if name is None:
            return self.default_cache
        return self.caches.get(name)
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        # Hash for consistent length
        key_str = ":".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def start(self):
        """Start background tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._stats_task = asyncio.create_task(self._stats_loop())
    
    async def stop(self):
        """Stop background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._stats_task:
            self._stats_task.cancel()
    
    async def _cleanup_loop(self):
        """Periodic cache cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                for name, cache in self.caches.items():
                    # Clean expired entries
                    expired = 0
                    for key, entry in list(cache.memory_cache.cache.items()):
                        if entry.is_expired():
                            cache.memory_cache.cache.pop(key, None)
                            expired += 1
                    
                    if expired > 0:
                        self.logger.info(f"Cleaned {expired} expired entries from {name}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    async def _stats_loop(self):
        """Log cache statistics"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Collect stats
                all_stats = {}
                for name, cache in self.caches.items():
                    all_stats[name] = cache.get_stats()
                
                # Log to W&B
                weave.log({
                    'cache_stats': all_stats
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stats error: {e}")

# Caching decorators
def cached(cache_name: str = None, 
          ttl: float = 300,
          key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache
            cache_manager = get_cache_manager()
            cache = cache_manager.get_cache(cache_name)
            
            if cache is None:
                return await func(*args, **kwargs)
            
            # Generate key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache_manager.cache_key(func.__name__, *args, **kwargs)
            
            # Check cache
            value = await cache.get(key)
            if value is not None:
                return value
            
            # Compute value
            value = await func(*args, **kwargs)
            
            # Store in cache
            await cache.put(key, value)
            
            return value
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            cache_manager = get_cache_manager()
            cache = cache_manager.get_cache(cache_name)
            
            if cache is None:
                return func(*args, **kwargs)
            
            # Generate key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache_manager.cache_key(func.__name__, *args, **kwargs)
            
            # Check memory cache only for sync
            entry = cache.memory_cache.get(key)
            if entry:
                return entry.value
            
            # Compute value
            value = func(*args, **kwargs)
            
            # Store in memory cache
            cache.memory_cache.put(key, value, ttl)
            
            return value
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def cache_invalidate(cache_name: str = None, pattern: str = None):
    """Invalidate cache entries"""
    cache_manager = get_cache_manager()
    cache = cache_manager.get_cache(cache_name)
    
    if cache and pattern:
        # Invalidate matching keys
        to_remove = []
        for key in cache.memory_cache.cache:
            if pattern in key:
                to_remove.append(key)
        
        for key in to_remove:
            cache.memory_cache.cache.pop(key, None)

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager

# Example cached functions
@cached(ttl=60)
def calculate_expensive_metric(data: List[float]) -> float:
    """Example expensive calculation"""
    time.sleep(1)  # Simulate expensive operation
    return np.mean(data) * np.std(data)

@cached(cache_name='chemistry', ttl=3600)
async def fetch_chemical_data(compound: str) -> Dict[str, Any]:
    """Example async cached function"""
    await asyncio.sleep(0.5)  # Simulate API call
    return {
        'compound': compound,
        'molecular_weight': 100.0,
        'properties': ['stable', 'non-toxic']
    }

# Example usage
async def demo_caching():
    """Demonstrate caching functionality"""
    
    # Create cache manager
    cache_manager = get_cache_manager()
    
    # Create specialized caches
    cache_manager.create_cache(
        'default',
        memory_size=1000,
        memory_mb=50
    )
    
    cache_manager.create_cache(
        'chemistry',
        memory_size=500,
        memory_mb=25
    )
    
    # Start background tasks
    await cache_manager.start()
    
    try:
        print("Testing synchronous caching...")
        
        # First call - cache miss
        start = time.time()
        result1 = calculate_expensive_metric([1, 2, 3, 4, 5])
        duration1 = time.time() - start
        print(f"First call: {result1:.2f} (took {duration1:.2f}s)")
        
        # Second call - cache hit
        start = time.time()
        result2 = calculate_expensive_metric([1, 2, 3, 4, 5])
        duration2 = time.time() - start
        print(f"Second call: {result2:.2f} (took {duration2:.2f}s)")
        
        print("\nTesting asynchronous caching...")
        
        # Async calls
        compound = "H2O"
        data1 = await fetch_chemical_data(compound)
        print(f"Chemical data: {data1}")
        
        # Get cache statistics
        print("\nCache Statistics:")
        for name, cache in cache_manager.caches.items():
            stats = cache.get_stats()
            print(f"\n{name.upper()} cache:")
            print(f"  Memory hit rate: {stats['memory']['hit_rate']:.1%}")
            print(f"  Memory usage: {stats['memory']['memory_mb']:.1f} MB")
            print(f"  Entries: {stats['memory']['size']}")
        
    finally:
        await cache_manager.stop()

if __name__ == "__main__":
    asyncio.run(demo_caching())