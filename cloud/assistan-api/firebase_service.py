# firebase_service.py - Firebase Database Service
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    from google.cloud.firestore_v1.base_query import FieldFilter
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Firebase SDK not installed. Run: pip install firebase-admin")

@dataclass
class BrailleDetectionResult:
    """Structure for braille detection results"""
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

@dataclass
class ChatMessage:
    """Structure for chat messages"""
    message_id: str
    user_message: str
    assistant_response: str
    timestamp: datetime
    thread_id: str

@dataclass
class ChatThread:
    """Structure for chat threads"""
    thread_id: str
    user_id: str
    title: str
    messages: List[ChatMessage]
    created_at: datetime
    last_updated: datetime

class FirebaseService:
    """Firebase database service for braille detection app"""
    
    def __init__(self):
        self.db = None
        self.bucket = None
        self.initialized = False
        
        if FIREBASE_AVAILABLE:
            self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            if not firebase_admin._apps:
                # Try different credential sources
                if os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"):
                    # From JSON string (Vercel environment)
                    service_account_info = json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"))
                    cred = credentials.Certificate(service_account_info)
                elif os.getenv("FIREBASE_PRIVATE_KEY"):
                    # From individual environment variables
                    service_account_info = {
                        "type": "service_account",
                        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                        "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
                        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                    cred = credentials.Certificate(service_account_info)
                elif os.path.exists("serviceAccountKey.json"):
                    # From local file (development)
                    cred = credentials.Certificate("serviceAccountKey.json")
                else:
                    print("No Firebase credentials found")
                    return
                
                firebase_admin.initialize_app(cred, {
                    'storageBucket': f"{os.getenv('FIREBASE_PROJECT_ID', 'your-project')}.appspot.com"
                })
            
            self.db = firestore.client()
            self.bucket = storage.bucket()
            self.initialized = True
            print("Firebase initialized successfully")
            
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
            self.initialized = False
    
    def is_connected(self) -> bool:
        """Check if Firebase is connected"""
        return self.initialized and self.db is not None
    
    # Braille Detection Methods
    async def store_braille_detection(self, detection_result: BrailleDetectionResult) -> bool:
        """Store braille detection result in Firebase"""
        if not self.is_connected():
            print("Firebase not connected - cannot store detection")
            return False
        
        try:
            # Store in users/{user_id}/braille_detections/{session_id}
            doc_ref = (self.db.collection('users')
                      .document(detection_result.user_id)
                      .collection('braille_detections')
                      .document(detection_result.session_id))
            
            # Convert to dict and handle timestamp
            data = asdict(detection_result)
            data['timestamp'] = firestore.SERVER_TIMESTAMP
            
            doc_ref.set(data)
            print(f"Stored braille detection: {detection_result.session_id}")
            return True
            
        except Exception as e:
            print(f"Error storing braille detection: {e}")
            return False
    
    async def get_user_detections(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's recent braille detections"""
        if not self.is_connected():
            return []
        
        try:
            docs = (self.db.collection('users')
                   .document(user_id)
                   .collection('braille_detections')
                   .order_by('timestamp', direction=firestore.Query.DESCENDING)
                   .limit(limit)
                   .stream())
            
            detections = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                detections.append(data)
            
            return detections
            
        except Exception as e:
            print(f"Error getting user detections: {e}")
            return []
    
    # Chat Thread Methods
    async def create_chat_thread(self, user_id: str, title: str = None) -> str:
        """Create new chat thread"""
        if not self.is_connected():
            return str(uuid.uuid4())  # Return local ID if Firebase unavailable
        
        thread_id = str(uuid.uuid4())
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        try:
            thread_data = {
                'thread_id': thread_id,
                'user_id': user_id,
                'title': title,
                'messages': [],
                'created_at': firestore.SERVER_TIMESTAMP,
                'last_updated': firestore.SERVER_TIMESTAMP,
                'message_count': 0
            }
            
            doc_ref = (self.db.collection('users')
                      .document(user_id)
                      .collection('chat_threads')
                      .document(thread_id))
            
            doc_ref.set(thread_data)
            print(f"Created chat thread: {thread_id}")
            return thread_id
            
        except Exception as e:
            print(f"Error creating chat thread: {e}")
            return thread_id  # Return ID anyway for local use
    
    async def add_chat_message(self, user_id: str, thread_id: str, 
                              user_message: str, assistant_response: str) -> bool:
        """Add message to chat thread"""
        if not self.is_connected():
            print("Firebase not connected - message not stored")
            return False
        
        try:
            message_id = str(uuid.uuid4())
            message_data = {
                'message_id': message_id,
                'user_message': user_message,
                'assistant_response': assistant_response,
                'timestamp': firestore.SERVER_TIMESTAMP
            }
            
            # Add to thread's messages array and update thread
            thread_ref = (self.db.collection('users')
                         .document(user_id)
                         .collection('chat_threads')
                         .document(thread_id))
            
            # Use transaction to ensure consistency
            @firestore.transactional
            def update_thread(transaction):
                thread_doc = thread_ref.get(transaction=transaction)
                if thread_doc.exists:
                    current_messages = thread_doc.get('messages') or []
                    current_messages.append(message_data)
                    
                    transaction.update(thread_ref, {
                        'messages': current_messages,
                        'last_updated': firestore.SERVER_TIMESTAMP,
                        'message_count': len(current_messages)
                    })
                else:
                    # Create thread if it doesn't exist
                    thread_data = {
                        'thread_id': thread_id,
                        'user_id': user_id,
                        'title': f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        'messages': [message_data],
                        'created_at': firestore.SERVER_TIMESTAMP,
                        'last_updated': firestore.SERVER_TIMESTAMP,
                        'message_count': 1
                    }
                    transaction.set(thread_ref, thread_data)
            
            transaction = self.db.transaction()
            update_thread(transaction)
            
            print(f"Added message to thread: {thread_id}")
            return True
            
        except Exception as e:
            print(f"Error adding chat message: {e}")
            return False
    
    async def get_chat_thread(self, user_id: str, thread_id: str) -> Optional[Dict]:
        """Get chat thread with messages"""
        if not self.is_connected():
            return None
        
        try:
            doc_ref = (self.db.collection('users')
                      .document(user_id)
                      .collection('chat_threads')
                      .document(thread_id))
            
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            
            return None
            
        except Exception as e:
            print(f"Error getting chat thread: {e}")
            return None
    
    async def get_user_chat_threads(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get user's chat threads (without full message history)"""
        if not self.is_connected():
            return []
        
        try:
            docs = (self.db.collection('users')
                   .document(user_id)
                   .collection('chat_threads')
                   .order_by('last_updated', direction=firestore.Query.DESCENDING)
                   .limit(limit)
                   .stream())
            
            threads = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                # Don't include full messages array in list view
                if 'messages' in data:
                    data['messages'] = data['messages'][-1:] if data['messages'] else []  # Just last message
                threads.append(data)
            
            return threads
            
        except Exception as e:
            print(f"Error getting user chat threads: {e}")
            return []
    
    async def delete_chat_thread(self, user_id: str, thread_id: str) -> bool:
        """Delete chat thread"""
        if not self.is_connected():
            return False
        
        try:
            doc_ref = (self.db.collection('users')
                      .document(user_id)
                      .collection('chat_threads')
                      .document(thread_id))
            
            doc_ref.delete()
            print(f"Deleted chat thread: {thread_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting chat thread: {e}")
            return False
    
    # Image Storage Methods
    async def upload_image(self, image_path: str, user_id: str, session_id: str, 
                          filename: str) -> Optional[str]:
        """Upload image to Firebase Storage"""
        if not self.bucket:
            print("Firebase Storage not available")
            return None
        
        try:
            blob_name = f"braille_images/{user_id}/{session_id}_{filename}"
            blob = self.bucket.blob(blob_name)
            
            blob.upload_from_filename(image_path)
            blob.make_public()
            
            print(f"Uploaded image: {blob_name}")
            return blob.public_url
            
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None
    
    # User Management Methods
    async def create_user_profile(self, user_id: str, email: str = None, 
                                 display_name: str = None) -> bool:
        """Create or update user profile"""
        if not self.is_connected():
            return False
        
        try:
            user_data = {
                'user_id': user_id,
                'email': email,
                'display_name': display_name,
                'created_at': firestore.SERVER_TIMESTAMP,
                'last_active': firestore.SERVER_TIMESTAMP,
                'detection_count': 0,
                'chat_thread_count': 0
            }
            
            doc_ref = self.db.collection('users').document(user_id)
            doc_ref.set(user_data, merge=True)  # Merge to avoid overwriting existing data
            
            print(f"Created/updated user profile: {user_id}")
            return True
            
        except Exception as e:
            print(f"Error creating user profile: {e}")
            return False
    
    async def update_user_activity(self, user_id: str) -> bool:
        """Update user's last activity timestamp"""
        if not self.is_connected():
            return False
        
        try:
            doc_ref = self.db.collection('users').document(user_id)
            doc_ref.update({
                'last_active': firestore.SERVER_TIMESTAMP
            })
            return True
            
        except Exception as e:
            print(f"Error updating user activity: {e}")
            return False

# Global Firebase service instance
firebase_service = FirebaseService()