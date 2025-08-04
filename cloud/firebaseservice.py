# firebase_service.py - Firebase Integration Service
import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    FIREBASE_AVAILABLE = True
except ImportError:
    print("Firebase SDK not available. Install with: pip install firebase-admin")
    FIREBASE_AVAILABLE = False

@dataclass
class BrailleDetectionResult:
    """Data class for braille detection results"""
    session_id: str
    user_id: str
    filename: str
    detected_text: str
    explanation: str
    confidence: float
    raw_detections: List[str]
    timestamp: datetime
    processing_status: str = "completed"
    image_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class ChatMessage:
    """Data class for chat messages"""
    message_id: str
    user_id: str
    thread_id: str
    user_message: str
    assistant_response: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class FirebaseService:
    """Firebase service for storing braille detection results and chat data"""
    
    def __init__(self):
        self.db = None
        self.bucket = None
        self.app = None
        self._connected = False
        
        if FIREBASE_AVAILABLE:
            self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # Try to get credentials from environment
                cred_json = os.getenv('FIREBASE_CREDENTIALS')
                project_id = os.getenv('FIREBASE_PROJECT_ID')
                storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')
                
                if cred_json and project_id:
                    # Parse credentials from environment variable
                    cred_dict = json.loads(cred_json)
                    cred = credentials.Certificate(cred_dict)
                    
                    # Initialize with storage bucket if provided
                    config = {'projectId': project_id}
                    if storage_bucket:
                        config['storageBucket'] = storage_bucket
                    
                    self.app = firebase_admin.initialize_app(cred, config)
                else:
                    print("Firebase credentials not found in environment variables")
                    print("Set FIREBASE_CREDENTIALS, FIREBASE_PROJECT_ID, and FIREBASE_STORAGE_BUCKET")
                    return
            else:
                self.app = firebase_admin.get_app()
            
            # Initialize Firestore
            self.db = firestore.client()
            
            # Initialize Storage (if bucket is configured)
            try:
                self.bucket = storage.bucket()
                print(f"✓ Firebase Storage initialized: {self.bucket.name}")
            except Exception as e:
                print(f"Firebase Storage not available: {e}")
                self.bucket = None
            
            self._connected = True
            print("✓ Firebase initialized successfully")
            
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if Firebase is connected"""
        return self._connected and self.db is not None
    
    async def store_braille_detection(self, result: BrailleDetectionResult) -> bool:
        """Store braille detection result in Firestore"""
        if not self.is_connected():
            print("Firebase not connected, skipping storage")
            return False
        
        try:
            # Store in detections collection
            doc_ref = self.db.collection('braille_detections').document(result.session_id)
            await asyncio.to_thread(doc_ref.set, result.to_dict())
            
            # Update user's detection count
            user_ref = self.db.collection('users').document(result.user_id)
            await asyncio.to_thread(
                user_ref.update, 
                {
                    'last_detection': result.timestamp,
                    'total_detections': firestore.Increment(1)
                }
            )
            
            print(f"✓ Stored braille detection: {result.session_id}")
            return True
            
        except Exception as e:
            print(f"Error storing braille detection: {e}")
            return False
    
    async def get_user_detections(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's braille detection history"""
        if not self.is_connected():
            return []
        
        try:
            query = (self.db.collection('braille_detections')
                    .where('user_id', '==', user_id)
                    .order_by('timestamp', direction=firestore.Query.DESCENDING)
                    .limit(limit))
            
            docs = await asyncio.to_thread(query.get)
            
            detections = []
            for doc in docs:
                detection = doc.to_dict()
                detection['id'] = doc.id
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error getting user detections: {e}")
            return []
    
    async def upload_image(self, local_path: str, user_id: str, session_id: str, filename: str) -> Optional[str]:
        """Upload image to Firebase Storage"""
        if not self.bucket:
            print("Firebase Storage not available")
            return None
        
        try:
            # Create unique blob name
            blob_name = f"braille_images/{user_id}/{session_id}_{filename}"
            blob = self.bucket.blob(blob_name)
            
            # Upload file
            await asyncio.to_thread(blob.upload_from_filename, local_path)
            
            # Make blob publicly readable (optional - configure based on your needs)
            await asyncio.to_thread(blob.make_public)
            
            print(f"✓ Uploaded image: {blob_name}")
            return blob.public_url
            
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None
    
    async def create_chat_thread(self, user_id: str) -> str:
        """Create new chat thread"""
        if not self.is_connected():
            return str(uuid.uuid4())
        
        try:
            thread_id = str(uuid.uuid4())
            thread_data = {
                'thread_id': thread_id,
                'user_id': user_id,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'message_count': 0
            }
            
            doc_ref = self.db.collection('chat_threads').document(thread_id)
            await asyncio.to_thread(doc_ref.set, thread_data)
            
            print(f"✓ Created chat thread: {thread_id}")
            return thread_id
            
        except Exception as e:
            print(f"Error creating chat thread: {e}")
            return str(uuid.uuid4())
    
    async def add_chat_message(self, user_id: str, thread_id: str, user_message: str, assistant_response: str) -> bool:
        """Add message to chat thread"""
        if not self.is_connected():
            return False
        
        try:
            message_id = str(uuid.uuid4())
            message = ChatMessage(
                message_id=message_id,
                user_id=user_id,
                thread_id=thread_id,
                user_message=user_message,
                assistant_response=assistant_response,
                timestamp=datetime.now()
            )
            
            # Store message
            doc_ref = self.db.collection('chat_messages').document(message_id)
            await asyncio.to_thread(doc_ref.set, message.to_dict())
            
            # Update thread
            thread_ref = self.db.collection('chat_threads').document(thread_id)
            await asyncio.to_thread(
                thread_ref.update,
                {
                    'updated_at': datetime.now(),
                    'message_count': firestore.Increment(1),
                    'last_message': user_message[:100] + '...' if len(user_message) > 100 else user_message
                }
            )
            
            print(f"✓ Added chat message: {message_id}")
            return True
            
        except Exception as e:
            print(f"Error adding chat message: {e}")
            return False
    
    async def get_user_chat_threads(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's chat threads"""
        if not self.is_connected():
            return []
        
        try:
            query = (self.db.collection('chat_threads')
                    .where('user_id', '==', user_id)
                    .order_by('updated_at', direction=firestore.Query.DESCENDING)
                    .limit(limit))
            
            docs = await asyncio.to_thread(query.get)
            
            threads = []
            for doc in docs:
                thread = doc.to_dict()
                thread['id'] = doc.id
                # Convert timestamps
                if 'created_at' in thread:
                    thread['created_at'] = thread['created_at'].isoformat()
                if 'updated_at' in thread:
                    thread['updated_at'] = thread['updated_at'].isoformat()
                threads.append(thread)
            
            return threads
            
        except Exception as e:
            print(f"Error getting chat threads: {e}")
            return []
    
    async def get_chat_thread(self, user_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get specific chat thread with messages"""
        if not self.is_connected():
            return None
        
        try:
            # Get thread info
            thread_ref = self.db.collection('chat_threads').document(thread_id)
            thread_doc = await asyncio.to_thread(thread_ref.get)
            
            if not thread_doc.exists:
                return None
            
            thread_data = thread_doc.to_dict()
            
            # Verify user ownership
            if thread_data.get('user_id') != user_id:
                return None
            
            # Get messages
            messages_query = (self.db.collection('chat_messages')
                            .where('thread_id', '==', thread_id)
                            .order_by('timestamp', direction=firestore.Query.ASCENDING))
            
            message_docs = await asyncio.to_thread(messages_query.get)
            
            messages = []
            for doc in message_docs:
                message = doc.to_dict()
                message['id'] = doc.id
                messages.append(message)
            
            thread_data['messages'] = messages
            thread_data['id'] = thread_id
            
            # Convert timestamps
            if 'created_at' in thread_data:
                thread_data['created_at'] = thread_data['created_at'].isoformat()
            if 'updated_at' in thread_data:
                thread_data['updated_at'] = thread_data['updated_at'].isoformat()
            
            return thread_data
            
        except Exception as e:
            print(f"Error getting chat thread: {e}")
            return None
    
    async def delete_chat_thread(self, user_id: str, thread_id: str) -> bool:
        """Delete chat thread and its messages"""
        if not self.is_connected():
            return False
        
        try:
            # Verify ownership
            thread_ref = self.db.collection('chat_threads').document(thread_id)
            thread_doc = await asyncio.to_thread(thread_ref.get)
            
            if not thread_doc.exists:
                return False
            
            thread_data = thread_doc.to_dict()
            if thread_data.get('user_id') != user_id:
                return False
            
            # Delete messages
            messages_query = self.db.collection('chat_messages').where('thread_id', '==', thread_id)
            message_docs = await asyncio.to_thread(messages_query.get)
            
            for doc in message_docs:
                await asyncio.to_thread(doc.reference.delete)
            
            # Delete thread
            await asyncio.to_thread(thread_ref.delete)
            
            print(f"✓ Deleted chat thread: {thread_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting chat thread: {e}")
            return False
    
    async def create_user_profile(self, user_id: str, email: Optional[str] = None, display_name: Optional[str] = None) -> bool:
        """Create or update user profile"""
        if not self.is_connected():
            return False
        
        try:
            user_data = {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'total_detections': 0,
                'total_chat_threads': 0
            }
            
            if email:
                user_data['email'] = email
            if display_name:
                user_data['display_name'] = display_name
            
            doc_ref = self.db.collection('users').document(user_id)
            await asyncio.to_thread(doc_ref.set, user_data, merge=True)
            
            print(f"✓ Created/updated user profile: {user_id}")
            return True
            
        except Exception as e:
            print(f"Error creating user profile: {e}")
            return False
    
    async def update_user_activity(self, user_id: str) -> bool:
        """Update user's last activity timestamp"""
        if not self.is_connected():
            return False
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            await asyncio.to_thread(
                user_ref.update,
                {'last_activity': datetime.now()}
            )
            return True
            
        except Exception as e:
            print(f"Error updating user activity: {e}")
            return False

# Global instance
firebase_service = FirebaseService()