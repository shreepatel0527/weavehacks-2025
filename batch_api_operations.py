"""
Batch API operations for improved performance
"""
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict
import weave
from concurrent.futures import ThreadPoolExecutor
import functools

@dataclass
class APIRequest:
    """Single API request"""
    id: str
    method: str
    endpoint: str
    data: Optional[Dict] = None
    headers: Optional[Dict] = None
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class APIResponse:
    """API response with metadata"""
    request_id: str
    status_code: int
    data: Any
    headers: Dict
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

class BatchAPIClient:
    """Efficient batch API client with connection pooling"""
    
    def __init__(self, 
                 base_url: str,
                 max_concurrent: int = 10,
                 batch_size: int = 50,
                 batch_timeout: float = 0.1):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Request queue and batching
        self.request_queue = asyncio.Queue()
        self.pending_requests = {}
        self.batch_buffer = []
        self.last_batch_time = time.time()
        
        # Connection pool
        self.connector = None
        self.session = None
        
        # Response callbacks
        self.response_callbacks = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency_ms': 0,
            'batches_sent': 0
        }
        
        # Logging
        self.logger = logging.getLogger('batch_api')
        
        # Start background tasks
        self._running = False
        self._tasks = []
    
    async def start(self):
        """Start the batch API client"""
        if self._running:
            return
        
        self._running = True
        
        # Create connection pool
        self.connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout
        )
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._batch_processor()),
            asyncio.create_task(self._request_executor())
        ]
        
        self.logger.info("Batch API client started")
    
    async def stop(self):
        """Stop the batch API client"""
        self._running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Close session
        if self.session:
            await self.session.close()
        
        # Close connector
        if self.connector:
            await self.connector.close()
        
        self.logger.info("Batch API client stopped")
    
    async def request(self, 
                     method: str,
                     endpoint: str,
                     data: Optional[Dict] = None,
                     headers: Optional[Dict] = None,
                     priority: int = 0) -> APIResponse:
        """Make an API request (will be batched)"""
        # Create request
        request = APIRequest(
            id=f"{time.time()}_{endpoint}",
            method=method,
            endpoint=endpoint,
            data=data,
            headers=headers,
            priority=priority
        )
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request.id] = future
        
        # Add to queue
        await self.request_queue.put(request)
        
        # Update stats
        self.stats['total_requests'] += 1
        
        # Wait for response
        return await future
    
    async def batch_request(self, requests: List[Dict]) -> List[APIResponse]:
        """Make multiple requests as a batch"""
        tasks = []
        
        for req in requests:
            task = self.request(
                method=req.get('method', 'GET'),
                endpoint=req['endpoint'],
                data=req.get('data'),
                headers=req.get('headers'),
                priority=req.get('priority', 0)
            )
            tasks.append(task)
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for resp in responses:
            if isinstance(resp, Exception):
                results.append(APIResponse(
                    request_id='error',
                    status_code=500,
                    data=None,
                    headers={},
                    latency_ms=0,
                    error=str(resp)
                ))
            else:
                results.append(resp)
        
        return results
    
    async def _batch_processor(self):
        """Process requests into batches"""
        while self._running:
            try:
                # Check if we should send batch
                now = time.time()
                time_since_last = now - self.last_batch_time
                
                should_send = (
                    len(self.batch_buffer) >= self.batch_size or
                    (len(self.batch_buffer) > 0 and time_since_last >= self.batch_timeout)
                )
                
                if should_send:
                    # Send batch
                    if self.batch_buffer:
                        await self._send_batch(self.batch_buffer[:])
                        self.batch_buffer.clear()
                        self.last_batch_time = now
                        self.stats['batches_sent'] += 1
                
                # Get new requests (with timeout to allow batch checking)
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=0.01
                    )
                    self.batch_buffer.append(request)
                except asyncio.TimeoutError:
                    pass
                
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_batch(self, batch: List[APIRequest]):
        """Send a batch of requests"""
        # Sort by priority
        batch.sort(key=lambda r: r.priority, reverse=True)
        
        # Execute requests concurrently
        tasks = []
        for request in batch:
            task = self._execute_request(request)
            tasks.append(task)
        
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_task(task) for task in tasks]
        await asyncio.gather(*limited_tasks, return_exceptions=True)
    
    async def _execute_request(self, request: APIRequest):
        """Execute a single request"""
        start_time = time.time()
        
        try:
            # Build URL
            url = f"{self.base_url}{request.endpoint}"
            
            # Make request
            async with self.session.request(
                method=request.method,
                url=url,
                json=request.data,
                headers=request.headers
            ) as response:
                # Get response data
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = await response.text()
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Create response
                api_response = APIResponse(
                    request_id=request.id,
                    status_code=response.status,
                    data=data,
                    headers=dict(response.headers),
                    latency_ms=latency_ms
                )
                
                # Update stats
                self.stats['successful_requests'] += 1
                self.stats['total_latency_ms'] += latency_ms
                
                # Log to W&B
                weave.log({
                    'api_request': {
                        'endpoint': request.endpoint,
                        'status': response.status,
                        'latency_ms': latency_ms
                    }
                })
                
        except Exception as e:
            # Handle error
            api_response = APIResponse(
                request_id=request.id,
                status_code=500,
                data=None,
                headers={},
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
            
            # Update stats
            self.stats['failed_requests'] += 1
            
            # Retry logic
            if request.retry_count < request.max_retries:
                request.retry_count += 1
                await self.request_queue.put(request)
                return
        
        # Resolve future
        if request.id in self.pending_requests:
            future = self.pending_requests.pop(request.id)
            if not future.done():
                future.set_result(api_response)
        
        # Call callbacks
        await self._call_response_callbacks(api_response)
    
    async def _request_executor(self):
        """Execute requests from the queue"""
        # This is handled by the batch processor
        pass
    
    async def _call_response_callbacks(self, response: APIResponse):
        """Call registered callbacks for response"""
        callbacks = self.response_callbacks.get(response.request_id, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(response)
                else:
                    callback(response)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def register_callback(self, request_id: str, callback: Callable):
        """Register callback for request response"""
        self.response_callbacks[request_id].append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = dict(self.stats)
        
        # Calculate averages
        if stats['successful_requests'] > 0:
            stats['avg_latency_ms'] = (
                stats['total_latency_ms'] / stats['successful_requests']
            )
        else:
            stats['avg_latency_ms'] = 0
        
        # Success rate
        total = stats['successful_requests'] + stats['failed_requests']
        if total > 0:
            stats['success_rate'] = stats['successful_requests'] / total
        else:
            stats['success_rate'] = 0
        
        return stats

class CachedBatchAPIClient(BatchAPIClient):
    """Batch API client with response caching"""
    
    def __init__(self, *args, cache_ttl: float = 60.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = cache_ttl
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, method: str, endpoint: str, data: Optional[Dict]) -> str:
        """Generate cache key"""
        data_str = json.dumps(data, sort_keys=True) if data else ""
        return f"{method}:{endpoint}:{data_str}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is valid"""
        if key not in self.cache:
            return False
        
        timestamp = self.cache_timestamps.get(key, 0)
        return (time.time() - timestamp) < self.cache_ttl
    
    async def request(self, 
                     method: str,
                     endpoint: str,
                     data: Optional[Dict] = None,
                     headers: Optional[Dict] = None,
                     priority: int = 0,
                     use_cache: bool = True) -> APIResponse:
        """Make request with caching"""
        
        # Check cache
        if use_cache and method == 'GET':
            cache_key = self._get_cache_key(method, endpoint, data)
            
            if self._is_cache_valid(cache_key):
                self.cache_hits += 1
                cached_response = self.cache[cache_key]
                
                # Log cache hit
                weave.log({
                    'api_cache': {
                        'type': 'hit',
                        'endpoint': endpoint
                    }
                })
                
                return cached_response
            else:
                self.cache_misses += 1
        
        # Make request
        response = await super().request(method, endpoint, data, headers, priority)
        
        # Cache successful GET responses
        if use_cache and method == 'GET' and response.status_code == 200:
            cache_key = self._get_cache_key(method, endpoint, data)
            self.cache[cache_key] = response
            self.cache_timestamps[cache_key] = time.time()
        
        return response
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_timestamps.clear()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_ttl': self.cache_ttl
        }

# Convenience functions for common API patterns
class LabAPIClient(CachedBatchAPIClient):
    """Specialized API client for lab operations"""
    
    def __init__(self, base_url: str):
        super().__init__(
            base_url=base_url,
            max_concurrent=20,
            batch_size=100,
            batch_timeout=0.05,
            cache_ttl=30.0
        )
    
    async def batch_sensor_update(self, sensor_data: List[Dict]) -> List[APIResponse]:
        """Batch update sensor readings"""
        requests = [
            {
                'method': 'POST',
                'endpoint': '/api/sensors',
                'data': data,
                'priority': 1
            }
            for data in sensor_data
        ]
        
        return await self.batch_request(requests)
    
    async def batch_log_events(self, events: List[Dict]) -> List[APIResponse]:
        """Batch log events"""
        requests = [
            {
                'method': 'POST',
                'endpoint': '/api/events',
                'data': event,
                'priority': 0
            }
            for event in events
        ]
        
        return await self.batch_request(requests)
    
    async def batch_get_status(self, component_ids: List[str]) -> List[APIResponse]:
        """Batch get component status"""
        requests = [
            {
                'method': 'GET',
                'endpoint': f'/api/components/{comp_id}/status',
                'priority': 2
            }
            for comp_id in component_ids
        ]
        
        return await self.batch_request(requests)

# Example usage
async def demo_batch_api():
    """Demonstrate batch API operations"""
    client = LabAPIClient("https://api.lab-assistant.com")
    
    try:
        # Start client
        await client.start()
        
        # Batch sensor updates
        sensor_data = [
            {'sensor_id': f'temp_{i}', 'value': 23.5 + i * 0.1}
            for i in range(50)
        ]
        
        print("Sending batch sensor updates...")
        start_time = time.time()
        responses = await client.batch_sensor_update(sensor_data)
        elapsed = time.time() - start_time
        
        print(f"Sent {len(responses)} updates in {elapsed:.2f}s")
        print(f"Average: {elapsed/len(responses)*1000:.1f}ms per request")
        
        # Get statistics
        stats = client.get_statistics()
        cache_stats = client.get_cache_statistics()
        
        print(f"\nStatistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"  Batches sent: {stats['batches_sent']}")
        print(f"\nCache Statistics:")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"  Cache size: {cache_stats['cache_size']}")
        
    finally:
        # Stop client
        await client.stop()

if __name__ == "__main__":
    # Initialize W&B
    weave.init('batch-api-demo')
    
    # Run demo
    asyncio.run(demo_batch_api())