# api/index.py - Enhanced FastAPI with Firebase Integration
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import os
import uuid
import tempfile
from typing import Optional, List
from datetime import datetime
import asyncio

# Import your modules
try:
    from detector import BrailleDetector
    from assistant import BrailleAssistant
    from firebase_service import firebase_service, BrailleDetectionResult
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure detector.py, assistant.py, and firebase_service.py are in the same directory")

# Initialize FastAPI
app = FastAPI(
    title="Braille Detection API with Firebase",
    description="AI-powered Braille detection and chat with Firebase database",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detector = None
assistant = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global detector, assistant
    
    try:
        print("Initializing Braille Detector...")
        detector = BrailleDetector()
        print("Braille Detector initialized")
        
        print("Initializing AI Assistant...")
        assistant = BrailleAssistant()
        print("AI Assistant initialized")
        
        print(f"Firebase connection: {'✓' if firebase_service.is_connected() else '✗'}")
        
    except Exception as e:
        print(f"Startup initialization error: {e}")

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "message": "Braille Detection API with Firebase",
        "status": "active",
        "version": "1.0.0",
        "services": {
            "braille_detector": "ready" if detector else "failed",
            "ai_assistant": "ready" if assistant else "failed",
            "firebase_connection": "connected" if firebase_service.is_connected() else "disconnected"
        },
        "endpoints": {
            "detect_braille": "/detect-braille",
            "chat": "/chat",
            "chat_threads": "/chat-threads",
            "user_detections": "/user-detections"
        }
    }

@app.post("/detect-braille")
async def detect_braille(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Detect Braille characters in uploaded image
    
    - **file**: Image file (jpg, png, etc.)
    - **user_id**: User identifier from frontend authentication
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not detector:
        raise HTTPException(status_code=503, detail="Braille detector not available")
    
    session_id = str(uuid.uuid4())
    temp_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"Processing image for user {user_id}, session {session_id}")
        
        # 1. Run Braille detection
        result = detector.detect_braille(temp_path)
        if not result:
            raise HTTPException(status_code=422, detail="No braille characters detected in image")
        
        predictions = detector.extract_predictions(result)
        if not predictions:
            raise HTTPException(status_code=422, detail="No valid braille predictions found")
        
        detected_strings = detector.organize_text_by_rows(predictions)
        
        # 2. Process with AI assistant
        braille_result = None
        if assistant:
            try:
                braille_result = assistant.process_braille_strings(detected_strings)
            except Exception as e:
                print(f"AI processing error: {e}")
                # Fallback result
                braille_result = type('Result', (), {
                    'text': ' '.join(detected_strings),
                    'explanation': f'Basic detection completed. AI processing failed: {str(e)}',
                    'confidence': 0.5
                })()
        else:
            # Fallback if assistant not available
            braille_result = type('Result', (), {
                'text': ' '.join(detected_strings),
                'explanation': 'Braille characters detected. AI assistant not available for enhanced processing.',
                'confidence': 0.6
            })()
        
        # 3. Upload image to Firebase Storage (optional)
        image_url = None
        if firebase_service.is_connected():
            image_url = await firebase_service.upload_image(
                temp_path, user_id, session_id, file.filename
            )
        
        # 4. Store in Firebase database
        detection_result = BrailleDetectionResult(
            session_id=session_id,
            user_id=user_id,
            filename=file.filename or "uploaded_image.jpg",
            detected_text=braille_result.text,
            explanation=braille_result.explanation,
            confidence=braille_result.confidence,
            raw_detections=detected_strings,
            timestamp=datetime.now(),
            processing_status="completed",
            image_url=image_url
        )
        
        stored = await firebase_service.store_braille_detection(detection_result)
        
        # 5. Update user activity
        if firebase_service.is_connected():
            await firebase_service.update_user_activity(user_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "result": {
                "text": braille_result.text,
                "explanation": braille_result.explanation,
                "confidence": round(braille_result.confidence, 3),
                "detected_characters": len(predictions),
                "rows": len(detected_strings)
            },
            "raw_detections": detected_strings,
            "image_url": image_url,
            "stored_in_database": stored,
            "processing_time": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Detection processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.post("/chat")
async def chat(
    message: str = Form(...),
    user_id: str = Form(...),
    thread_id: Optional[str] = Form(None)
):
    """
    Chat with AI assistant
    
    - **message**: User's message/question
    - **user_id**: User identifier
    - **thread_id**: Optional conversation thread ID (creates new if not provided)
    """
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not assistant:
        raise HTTPException(status_code=503, detail="AI assistant not available")
    
    # Generate or validate thread_id
    if not thread_id:
        thread_id = await firebase_service.create_chat_thread(user_id)
    
    try:
        print(f"Processing chat for user {user_id}, thread {thread_id}")
        
        # Get AI response
        response_text = assistant.chat(message, thread_id)
        if not response_text:
            response_text = "I apologize, but I couldn't process your message at this time."
        
        # Store conversation in Firebase
        stored = await firebase_service.add_chat_message(
            user_id=user_id,
            thread_id=thread_id,
            user_message=message,
            assistant_response=response_text
        )
        
        # Update user activity
        if firebase_service.is_connected():
            await firebase_service.update_user_activity(user_id)
        
        return {
            "success": True,
            "thread_id": thread_id,
            "response": response_text,
            "stored_in_database": stored,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Chat processing error: {e}")
        # Still return a response even if storage fails
        return {
            "success": True,
            "thread_id": thread_id,
            "response": f"I encountered an issue: {str(e)}",
            "stored_in_database": False,
            "error": str(e)
        }

@app.get("/chat-threads")
async def get_chat_threads(
    user_id: str = Query(...),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get user's chat threads
    
    - **user_id**: User identifier
    - **limit**: Maximum number of threads to return (1-100)
    """
    
    try:
        threads = await firebase_service.get_user_chat_threads(user_id, limit)
        return {
            "success": True,
            "threads": threads,
            "count": len(threads)
        }
        
    except Exception as e:
        print(f"Error getting chat threads: {e}")
        return {
            "success": False,
            "threads": [],
            "count": 0,
            "error": str(e)
        }

@app.get("/chat-threads/{thread_id}")
async def get_chat_thread(
    thread_id: str,
    user_id: str = Query(...)
):
    """
    Get specific chat thread with full message history
    
    - **thread_id**: Thread identifier
    - **user_id**: User identifier
    """
    
    try:
        thread = await firebase_service.get_chat_thread(user_id, thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Chat thread not found")
        
        return {
            "success": True,
            "thread": thread
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting chat thread: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving thread: {str(e)}")

@app.delete("/chat-threads/{thread_id}")
async def delete_chat_thread(
    thread_id: str,
    user_id: str = Query(...)
):
    """
    Delete chat thread
    
    - **thread_id**: Thread identifier  
    - **user_id**: User identifier
    """
    
    try:
        deleted = await firebase_service.delete_chat_thread(user_id, thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found or could not be deleted")
        
        return {
            "success": True,
            "message": "Thread deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting thread: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting thread: {str(e)}")

@app.get("/user-detections")
async def get_user_detections(
    user_id: str = Query(...),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get user's braille detection history
    
    - **user_id**: User identifier
    - **limit**: Maximum number of detections to return (1-50)
    """
    
    try:
        detections = await firebase_service.get_user_detections(user_id, limit)
        return {
            "success": True,
            "detections": detections,
            "count": len(detections)
        }
        
    except Exception as e:
        print(f"Error getting user detections: {e}")
        return {
            "success": False,
            "detections": [],
            "count": 0,
            "error": str(e)
        }

@app.post("/users")
async def create_user_profile(
    user_id: str = Form(...),
    email: Optional[str] = Form(None),
    display_name: Optional[str] = Form(None)
):
    """
    Create or update user profile
    
    - **user_id**: User identifier
    - **email**: User email (optional)
    - **display_name**: User display name (optional)
    """
    
    try:
        created = await firebase_service.create_user_profile(
            user_id=user_id,
            email=email,
            display_name=display_name
        )
        
        return {
            "success": created,
            "user_id": user_id,
            "message": "User profile created/updated successfully" if created else "Failed to create user profile"
        }
        
    except Exception as e:
        print(f"Error creating user profile: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "detector": "ready" if detector else "failed",
            "assistant": "ready" if assistant else "failed",
            "firebase": "connected" if firebase_service.is_connected() else "disconnected"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

# Vercel handler - required for deployment
handler = Mangum(app)