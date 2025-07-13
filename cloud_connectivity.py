"""
Cloud connectivity for remote monitoring and collaboration
"""
import asyncio
import websockets
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import weave
from pathlib import Path
import aiohttp
import jwt
import hashlib
import redis
from azure.storage.blob import BlobServiceClient
import boto3

class MessageType(Enum):
    EXPERIMENT_UPDATE = "experiment_update"
    SENSOR_DATA = "sensor_data"
    SAFETY_ALERT = "safety_alert"
    VOICE_COMMAND = "voice_command"
    VIDEO_FRAME = "video_frame"
    COLLABORATION = "collaboration"
    SYSTEM_STATUS = "system_status"

@dataclass
class CloudMessage:
    message_type: MessageType
    payload: Dict[str, Any]
    sender_id: str
    experiment_id: str
    timestamp: datetime
    priority: int = 0

class CloudConnector:
    """Manages cloud connectivity for remote monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.websocket = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.subscribers = {}
        
        # Authentication
        self.auth_token = None
        self.user_id = config.get('user_id')
        self.api_key = config.get('api_key')
        
        # Cloud storage
        self.storage_backend = config.get('storage_backend', 'azure')
        self.storage_client = self._init_storage_client()
        
        # Redis for real-time pub/sub
        self.redis_client = self._init_redis()
        
        # Connection management
        self.reconnect_interval = 5
        self.max_reconnect_attempts = 10
        self.connection_thread = None
        
        # Initialize W&B
        weave.init('cloud-connectivity')
    
    def _init_storage_client(self):
        """Initialize cloud storage client"""
        if self.storage_backend == 'azure':
            return BlobServiceClient.from_connection_string(
                self.config.get('azure_connection_string')
            )
        elif self.storage_backend == 'aws':
            return boto3.client(
                's3',
                aws_access_key_id=self.config.get('aws_access_key'),
                aws_secret_access_key=self.config.get('aws_secret_key')
            )
        return None
    
    def _init_redis(self):
        """Initialize Redis client for real-time messaging"""
        try:
            return redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                decode_responses=True
            )
        except:
            return None
    
    async def connect(self):
        """Establish cloud connection"""
        # Authenticate first
        await self._authenticate()
        
        # Connect to WebSocket
        ws_url = self.config.get('websocket_url')
        
        try:
            self.websocket = await websockets.connect(
                ws_url,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"}
            )
            self.is_connected = True
            
            # Start message handlers
            asyncio.create_task(self._handle_incoming())
            asyncio.create_task(self._handle_outgoing())
            
            # Log connection
            weave.log({
                'cloud_connection': {
                    'status': 'connected',
                    'user_id': self.user_id,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            print(f"Connected to cloud service at {ws_url}")
            
        except Exception as e:
            print(f"Cloud connection failed: {e}")
            self.is_connected = False
            raise
    
    async def _authenticate(self):
        """Authenticate with cloud service"""
        auth_url = self.config.get('auth_url')
        
        async with aiohttp.ClientSession() as session:
            payload = {
                'user_id': self.user_id,
                'api_key': self.api_key,
                'client_type': 'lab_assistant',
                'version': '3.0'
            }
            
            async with session.post(auth_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    self.auth_token = data.get('token')
                else:
                    raise Exception(f"Authentication failed: {response.status}")
    
    async def _handle_incoming(self):
        """Handle incoming messages from cloud"""
        while self.is_connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Create CloudMessage
                cloud_msg = CloudMessage(
                    message_type=MessageType(data['type']),
                    payload=data['payload'],
                    sender_id=data['sender_id'],
                    experiment_id=data['experiment_id'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    priority=data.get('priority', 0)
                )
                
                # Distribute to subscribers
                await self._distribute_message(cloud_msg)
                
            except websockets.exceptions.ConnectionClosed:
                self.is_connected = False
                await self._reconnect()
            except Exception as e:
                print(f"Error handling incoming message: {e}")
    
    async def _handle_outgoing(self):
        """Handle outgoing messages to cloud"""
        while self.is_connected:
            try:
                # Get message from queue
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    
                    # Send to WebSocket
                    await self.websocket.send(json.dumps({
                        'type': message.message_type.value,
                        'payload': message.payload,
                        'sender_id': message.sender_id,
                        'experiment_id': message.experiment_id,
                        'timestamp': message.timestamp.isoformat(),
                        'priority': message.priority
                    }))
                    
                    # Also publish to Redis if available
                    if self.redis_client:
                        channel = f"experiment:{message.experiment_id}"
                        self.redis_client.publish(
                            channel, 
                            json.dumps(asdict(message), default=str)
                        )
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error sending message: {e}")
    
    async def _distribute_message(self, message: CloudMessage):
        """Distribute message to subscribers"""
        # Get subscribers for this message type
        subscribers = self.subscribers.get(message.message_type, [])
        
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                print(f"Subscriber error: {e}")
    
    async def _reconnect(self):
        """Attempt to reconnect to cloud service"""
        for attempt in range(self.max_reconnect_attempts):
            print(f"Reconnection attempt {attempt + 1}...")
            
            try:
                await asyncio.sleep(self.reconnect_interval)
                await self.connect()
                return
            except:
                continue
        
        print("Max reconnection attempts reached")
    
    def send_message(self, message: CloudMessage):
        """Queue message for sending"""
        self.message_queue.put(message)
    
    def subscribe(self, message_type: MessageType, callback: Callable):
        """Subscribe to message type"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        self.subscribers[message_type].append(callback)
    
    @weave.op()
    async def upload_data(self, data: bytes, filename: str, 
                         experiment_id: str) -> str:
        """Upload data to cloud storage"""
        try:
            if self.storage_backend == 'azure':
                container_name = f"experiment-{experiment_id}"
                blob_client = self.storage_client.get_blob_client(
                    container=container_name, 
                    blob=filename
                )
                blob_client.upload_blob(data, overwrite=True)
                return blob_client.url
                
            elif self.storage_backend == 'aws':
                bucket_name = f"lab-experiments-{self.user_id}"
                key = f"{experiment_id}/{filename}"
                self.storage_client.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=data
                )
                return f"s3://{bucket_name}/{key}"
            
            return ""
            
        except Exception as e:
            print(f"Upload error: {e}")
            return ""
    
    async def download_data(self, url: str) -> bytes:
        """Download data from cloud storage"""
        try:
            if self.storage_backend == 'azure' and url.startswith('https://'):
                blob_client = BlobServiceClient.from_blob_url(url)
                return blob_client.download_blob().readall()
                
            elif self.storage_backend == 'aws' and url.startswith('s3://'):
                # Parse S3 URL
                parts = url.replace('s3://', '').split('/')
                bucket = parts[0]
                key = '/'.join(parts[1:])
                
                response = self.storage_client.get_object(
                    Bucket=bucket,
                    Key=key
                )
                return response['Body'].read()
            
            return b""
            
        except Exception as e:
            print(f"Download error: {e}")
            return b""

class CollaborationManager:
    """Manages real-time collaboration features"""
    
    def __init__(self, cloud_connector: CloudConnector):
        self.cloud = cloud_connector
        self.active_users = {}
        self.shared_state = {}
        self.locks = {}
        
        # Subscribe to collaboration messages
        self.cloud.subscribe(
            MessageType.COLLABORATION, 
            self.handle_collaboration_message
        )
    
    def handle_collaboration_message(self, message: CloudMessage):
        """Handle collaboration messages"""
        action = message.payload.get('action')
        
        if action == 'user_joined':
            self.handle_user_joined(message)
        elif action == 'user_left':
            self.handle_user_left(message)
        elif action == 'state_update':
            self.handle_state_update(message)
        elif action == 'lock_request':
            self.handle_lock_request(message)
        elif action == 'lock_release':
            self.handle_lock_release(message)
    
    def handle_user_joined(self, message: CloudMessage):
        """Handle user joining experiment"""
        user_info = message.payload.get('user_info')
        self.active_users[message.sender_id] = {
            'name': user_info.get('name'),
            'role': user_info.get('role'),
            'joined_at': message.timestamp
        }
        
        print(f"User {user_info.get('name')} joined the experiment")
    
    def handle_user_left(self, message: CloudMessage):
        """Handle user leaving experiment"""
        if message.sender_id in self.active_users:
            user = self.active_users.pop(message.sender_id)
            print(f"User {user['name']} left the experiment")
            
            # Release any locks held by user
            self.release_user_locks(message.sender_id)
    
    def handle_state_update(self, message: CloudMessage):
        """Handle shared state updates"""
        state_key = message.payload.get('key')
        state_value = message.payload.get('value')
        
        self.shared_state[state_key] = {
            'value': state_value,
            'updated_by': message.sender_id,
            'timestamp': message.timestamp
        }
    
    def handle_lock_request(self, message: CloudMessage):
        """Handle resource lock requests"""
        resource = message.payload.get('resource')
        
        if resource not in self.locks:
            # Grant lock
            self.locks[resource] = {
                'owner': message.sender_id,
                'timestamp': message.timestamp
            }
            
            # Notify requester
            self.cloud.send_message(CloudMessage(
                message_type=MessageType.COLLABORATION,
                payload={
                    'action': 'lock_granted',
                    'resource': resource
                },
                sender_id=self.cloud.user_id,
                experiment_id=message.experiment_id,
                timestamp=datetime.now()
            ))
    
    def handle_lock_release(self, message: CloudMessage):
        """Handle lock releases"""
        resource = message.payload.get('resource')
        
        if resource in self.locks and self.locks[resource]['owner'] == message.sender_id:
            del self.locks[resource]
    
    def release_user_locks(self, user_id: str):
        """Release all locks held by a user"""
        to_release = []
        for resource, lock_info in self.locks.items():
            if lock_info['owner'] == user_id:
                to_release.append(resource)
        
        for resource in to_release:
            del self.locks[resource]
    
    @weave.op()
    def broadcast_update(self, key: str, value: Any, experiment_id: str):
        """Broadcast state update to all users"""
        message = CloudMessage(
            message_type=MessageType.COLLABORATION,
            payload={
                'action': 'state_update',
                'key': key,
                'value': value
            },
            sender_id=self.cloud.user_id,
            experiment_id=experiment_id,
            timestamp=datetime.now()
        )
        
        self.cloud.send_message(message)

class RemoteMonitoringService:
    """Service for remote experiment monitoring"""
    
    def __init__(self, cloud_connector: CloudConnector):
        self.cloud = cloud_connector
        self.monitoring_interval = 5  # seconds
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Data buffers
        self.sensor_buffer = []
        self.event_buffer = []
        self.buffer_size = 100
        
        # Subscribe to relevant messages
        self.cloud.subscribe(
            MessageType.SYSTEM_STATUS,
            self.handle_system_status
        )
    
    def start_monitoring(self, experiment_id: str):
        """Start remote monitoring"""
        self.is_monitoring = True
        self.experiment_id = experiment_id
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop remote monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            # Collect and send sensor data
            self.send_sensor_data()
            
            # Send buffered events
            self.send_buffered_events()
            
            time.sleep(self.monitoring_interval)
    
    @weave.op()
    def send_sensor_data(self):
        """Send sensor data to cloud"""
        # In production, would collect actual sensor data
        sensor_data = {
            'temperature': 23.5,
            'pressure': 101.3,
            'stirring_rpm': 1100,
            'ph': 7.0
        }
        
        message = CloudMessage(
            message_type=MessageType.SENSOR_DATA,
            payload=sensor_data,
            sender_id=self.cloud.user_id,
            experiment_id=self.experiment_id,
            timestamp=datetime.now()
        )
        
        self.cloud.send_message(message)
        
        # Buffer for batch upload
        self.sensor_buffer.append(sensor_data)
        
        if len(self.sensor_buffer) >= self.buffer_size:
            self.upload_sensor_batch()
    
    def send_buffered_events(self):
        """Send buffered events"""
        if self.event_buffer:
            for event in self.event_buffer:
                message = CloudMessage(
                    message_type=MessageType.EXPERIMENT_UPDATE,
                    payload=event,
                    sender_id=self.cloud.user_id,
                    experiment_id=self.experiment_id,
                    timestamp=datetime.now()
                )
                
                self.cloud.send_message(message)
            
            self.event_buffer.clear()
    
    async def upload_sensor_batch(self):
        """Upload batch of sensor data"""
        if not self.sensor_buffer:
            return
        
        # Convert to JSON
        batch_data = json.dumps({
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'data': self.sensor_buffer
        }).encode('utf-8')
        
        # Upload to cloud storage
        filename = f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        url = await self.cloud.upload_data(
            batch_data, 
            filename, 
            self.experiment_id
        )
        
        # Clear buffer
        self.sensor_buffer.clear()
        
        print(f"Uploaded sensor batch to: {url}")
    
    def handle_system_status(self, message: CloudMessage):
        """Handle system status updates"""
        status = message.payload.get('status')
        print(f"System status from {message.sender_id}: {status}")
    
    def add_event(self, event: Dict[str, Any]):
        """Add event to buffer"""
        self.event_buffer.append(event)
    
    @weave.op()
    def send_alert(self, alert_type: str, description: str, 
                   severity: str = 'medium'):
        """Send alert to cloud"""
        message = CloudMessage(
            message_type=MessageType.SAFETY_ALERT,
            payload={
                'alert_type': alert_type,
                'description': description,
                'severity': severity,
                'location': 'Lab Station 1'
            },
            sender_id=self.cloud.user_id,
            experiment_id=self.experiment_id,
            timestamp=datetime.now(),
            priority=2 if severity == 'high' else 1
        )
        
        self.cloud.send_message(message)
        
        # Log alert
        weave.log({
            'cloud_alert': {
                'type': alert_type,
                'severity': severity,
                'description': description
            }
        })

# Example usage
async def demo_cloud_connectivity():
    """Demonstrate cloud connectivity features"""
    
    # Configuration
    config = {
        'user_id': 'researcher_001',
        'api_key': 'demo_api_key',
        'websocket_url': 'wss://lab-cloud.example.com/ws',
        'auth_url': 'https://lab-cloud.example.com/auth',
        'storage_backend': 'azure',
        'azure_connection_string': 'DefaultEndpointsProtocol=https;...',
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    # Create cloud connector
    cloud = CloudConnector(config)
    
    try:
        # Connect to cloud
        await cloud.connect()
        
        # Create collaboration manager
        collab = CollaborationManager(cloud)
        
        # Create remote monitoring service
        monitor = RemoteMonitoringService(cloud)
        
        # Start monitoring
        experiment_id = 'exp_20240115_001'
        monitor.start_monitoring(experiment_id)
        
        # Simulate collaboration
        collab.broadcast_update(
            'experiment_phase', 
            'synthesis_started',
            experiment_id
        )
        
        # Simulate events
        monitor.add_event({
            'type': 'color_change',
            'description': 'Solution turned yellow',
            'timestamp': datetime.now().isoformat()
        })
        
        # Send alert
        monitor.send_alert(
            'temperature_warning',
            'Temperature rising above threshold',
            'medium'
        )
        
        # Run for demonstration
        await asyncio.sleep(30)
        
    except Exception as e:
        print(f"Demo error: {e}")
    
    finally:
        monitor.stop_monitoring()
        print("Cloud connectivity demo complete")

if __name__ == "__main__":
    # Note: This is a demonstration
    # In production, would connect to actual cloud services
    print("Cloud connectivity module loaded")
    print("Note: Actual cloud connection requires valid credentials")