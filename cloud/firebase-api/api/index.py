# firebase_api.py - Main Firebase Integration API
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import os
import uuid
import tempfile
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import json

# Import Firebase service
try:
    from firebase_service import firebase_service, BrailleDetectionResult
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure firebase_service.py is in the same directory")

# Initialize FastAPI
app = FastAPI(
    title="Braille Detection Firebase API",
    description="Main API that orchestrates Detector and Assistant services with Firebase storage",
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

# Service URLs - Configure these based on your deployment
DETECTOR_API_URL = os.getenv("DETECTOR_API_URL", "http://localhost:8001")
ASSISTANT_API_URL = os.getenv("ASSISTANT_API_URL", "http://localhost:8002")

@app.on_event("startup")
async def startup_event():
    """Initialize Firebase service on startup"""
    print(f"Firebase connection: {'✓' if firebase_service.is_connected() else '✗'}")
    print(f"Detector API URL: {DETECTOR_API_URL}")
    print(f"Assistant API URL: {ASSISTANT_API_URL}")

@app.get("/")
async def root():
    """API status endpoint"""
    # Check service health
    detector_healthy = await check_service_health(DETECTOR_API_URL)
    assistant_healthy = await check_service_health(ASSISTANT_API_URL)
    
    return {
        "service": "Braille Detection Firebase API",
        "status": "active",
        "version": "1.0.0",
        "services": {
            "firebase_connection": "connected" if firebase_service.is_connected() else "disconnected",
            "detector_api": "healthy" if detector_healthy else "unhealthy",
            "assistant_api": "healthy" if assistant_healthy else "unhealthy"
        },
        "endpoints": {
            "detect_braille": "/detect-braille",
            "chat": "/chat",
            "chat_threads": "/chat-threads",
            "user_detections": "/user-detections",
            "users": "/users"
        },
        "external_services": {
            "detector_api_url": DETECTOR_API_URL,
            "assistant_api_url": ASSISTANT_API_URL
        }
    }

async def check_service_health(service_url: str) -> bool:
    """Check if external service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{service_url}/health")
            return response.status_code == 200
    except:
        return False

@app.post("/detect-braille")
async def detect_braille(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    min_confidence: float = Form(0.4),
    store_in_database: bool = Form(True)
):
    """
    Complete Braille detection pipeline: detect → process → store
    
    - **file**: Image file (jpg, png, etc.)
    - **user_id**: User identifier from frontend authentication
    - **min_confidence**: Minimum confidence threshold for detection
    - **store_in_database**: Whether to store results in Firebase
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    session_id = str(uuid.uuid4())
    temp_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"Processing complete braille detection pipeline for user {user_id}, session {session_id}")
        
        # Step 1: Call Detector API
        detector_result = await call_detector_api(temp_path, min_confidence, file.filename)
        if not detector_result["success"]:
            raise HTTPException(status_code=422, detail="Braille detection failed")
        
        detected_strings = detector_result["detection_results"]["organized_text"]
        if not detected_strings:
            raise HTTPException(status_code=422, detail="No braille characters detected")
        
        # Step 2: Call Assistant API
        assistant_result = await call_assistant_api(detected_strings, session_id)
        
        # Step 3: Upload image to Firebase Storage (if enabled)
        image_url = None
        if store_in_database and firebase_service.is_connected():
            image_url = await firebase_service.upload_image(
                temp_path, user_id, session_id, file.filename
            )
        
        # Step 4: Store complete result in Firebase Database
        stored = False
        if store_in_database:
            detection_result = BrailleDetectionResult(
                session_id=session_id,
                user_id=user_id,
                filename=file.filename or "uploaded_image.jpg",
                detected_text=assistant_result.get("processing_result", {}).get("interpreted_text", " ".join(detected_strings)),
                explanation=assistant_result.get("processing_result", {}).get("explanation", "No explanation available"),
                confidence=assistant_result.get("processing_result", {}).get("confidence", 0.5),
                raw_detections=detected_strings,
                timestamp=datetime.now(),
                processing_status="completed",
                image_url=image_url
            )
            
            stored = await firebase_service.store_braille_detection(detection_result)
            
            # Update user activity
            if firebase_service.is_connected():
                await firebase_service.update_user_activity(user_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "result": {
                "interpreted_text": assistant_result.get("processing_result", {}).get("interpreted_text", " ".join(detected_strings)),
                "explanation": assistant_result.get("processing_result", {}).get("explanation", "Processing completed"),
                "confidence": assistant_result.get("processing_result", {}).get("confidence", 0.5),
                "detected_characters": detector_result["detection_results"]["statistics"]["total_detections"],
                "rows": len(detected_strings)
            },
            "raw_detections": detected_strings,
            "detection_statistics": detector_result["detection_results"]["statistics"],
            "image_url": image_url,
            "stored_in_database": stored,
            "processing_timestamp": datetime.now().isoformat(),
            "services_used": {
                "detector_api": detector_result["success"],
                "assistant_api": assistant_result["success"],
                "firebase_storage": image_url is not None,
                "firebase_database": stored
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Complete detection pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing pipeline failed: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

async def call_detector_api(image_path: str, min_confidence: float, filename: str) -> Dict[str, Any]:
    """Call the Detector API service"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(image_path, 'rb') as f:
                files = {"file": (filename, f, "image/jpeg")}
                data = {
                    "min_confidence": min_confidence,
                    "create_annotated": True
                }
                
                response = await client.post(f"{DETECTOR_API_URL}/detect", files=files, data=data)
                response.raise_for_status()
                return response.json()
                
    except httpx.RequestError as e:
        print(f"Detector API request error: {e}")
        return {"success": False, "error": f"Detector service unavailable: {str(e)}"}
    except httpx.HTTPStatusError as e:
        print(f"Detector API HTTP error: {e}")
        return {"success": False, "error": f"Detector service error: {e.response.status_code}"}
    except Exception as e:
        print(f"Detector API unexpected error: {e}")
        return {"success": False, "error": f"Detector service failed: {str(e)}"}

async def call_assistant_api(detected_strings: List[str], session_id: str) -> Dict[str, Any]:
    """Call the Assistant API service"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = {
                "detected_strings": detected_strings,
                "session_id": session_id
            }
            
            response = await client.post(f"{ASSISTANT_API_URL}/process-braille", data=data)
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as e:
        print(f"Assistant API request error: {e}")
        # Return fallback result
        return {
            "success": True,
            "processing_result": {
                "interpreted_text": " ".join(detected_strings),
                "explanation": f"Basic text assembly completed. AI service unavailable: {str(e)}",
                "confidence": 0.3
            }
        }
    except httpx.HTTPStatusError as e:
        print(f"Assistant API HTTP error: {e}")
        return {
            "success": True,
            "processing_result": {
                "interpreted_text": " ".join(detected_strings),
                "explanation": f"Basic text assembly completed. AI processing error: {e.response.status_code}",
                "confidence": 0.3
            }
        }
    except Exception as e:
        print(f"Assistant API unexpected error: {e}")
        return {
            "success": True,
            "processing_result": {
                "interpreted_text": " ".join(detected_strings),
                "explanation": f"Basic text assembly completed. AI processing failed: {str(e)}",
                "confidence": 0.3
            }
        }

@app.post("/chat")
async def chat(
    message: str = Form(...),
    user_id: str = Form(...),
    thread_id: Optional[str] = Form(None),
    context: Optional[str] = Form(None)
):
    """
    Chat with AI assistant and store conversation in Firebase
    
    - **message**: User's message/question
    - **user_id**: User identifier
    - **thread_id**: Optional conversation thread ID (creates new if not provided)
    - **context**: Optional context for the conversation
    """
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Generate or validate thread_id
    if not thread_id:
        thread_id = await firebase_service.create_chat_thread(user_id)
    
    try:
        print(f"Processing chat for user {user_id}, thread {thread_id}")
        
        # Call Assistant API
        assistant_result = await call_assistant_chat_api(message, thread_id, context)
        
        response_text = "I apologize, but I couldn't process your message at this time."
        if assistant_result["success"]:
            response_text = assistant_result["chat_result"]["assistant_response"]
        else:
            response_text = assistant_result.get("fallback_response", response_text)
        
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
            "timestamp": datetime.now().isoformat(),
            "services_used": {
                "assistant_api": assistant_result["success"],
                "firebase_database": stored
            },
            "metadata": assistant_result.get("metadata", {})
        }
        
    except Exception as e:
        print(f"Chat processing error: {e}")
        # Still return a response even if something fails
        return {
            "success": True,
            "thread_id": thread_id,
            "response": f"I encountered an issue: {str(e)}",
            "stored_in_database": False,
            "error": str(e)
        }

async def call_assistant_chat_api(message: str, thread_id: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Call the Assistant API chat endpoint"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = {
                "message": message,
                "thread_id": thread_id
            }
            if context:
                data["context"] = context
            
            response = await client.post(f"{ASSISTANT_API_URL}/chat", data=data)
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as e:
        print(f"Assistant chat API request error: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_response": "I'm having trouble connecting to the AI service. Please try again later."
        }
    except httpx.HTTPStatusError as e:
        print(f"Assistant chat API HTTP error: {e}")
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}",
            "fallback_response": "The AI service is currently experiencing issues. Please try again."
        }
    except Exception as e:
        print(f"Assistant chat API unexpected error: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_response": "An unexpected error occurred while processing your message."
        }

@app.get("/chat-threads")
async def get_chat_threads(
    user_id: str = Query(...),
    limit: int = Query(20, ge=1, le=100)
):
   
    try:
        threads = await firebase_service.get_user_chat_threads(user_id, limit)
        return {
            "success": True,
            "threads": threads,
            "count": len(threads),
            "user_id": user_id
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
    Get specific chat thread with full message history from Firebase
    
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
    Delete chat thread from Firebase
    
    - **thread_id**: Thread identifier  
    - **user_id**: User identifier
    """
    
    try:
        deleted = await firebase_service.delete_chat_thread(user_id, thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found or could not be deleted")
        
        return {
            "success": True,
            "message": "Thread deleted successfully",
            "thread_id": thread_id
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
    Get user's braille detection history from Firebase
    
    - **user_id**: User identifier
    - **limit**: Maximum number of detections to return (1-50)
    """
    
    try:
        detections = await firebase_service.get_user_detections(user_id, limit)
        return {
            "success": True,
            "detections": detections,
            "count": len(detections),
            "user_id": user_id
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
    Create or update user profile in Firebase
    
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

@app.post("/batch-detect")
async def batch_detect_braille(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    min_confidence: float = Form(0.4),
    store_in_database: bool = Form(True)
):
    """
    Batch process multiple braille images
    
    - **files**: List of image files (max 5)
    - **user_id**: User identifier
    - **min_confidence**: Minimum confidence threshold
    - **store_in_database**: Whether to store results in Firebase
    """
    
    if len(files) > 5:  # Limit batch size for this main API
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each file individually
            result = await detect_braille(file, user_id, min_confidence, store_in_database)
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": True,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    successful_results = [r for r in results if r['success']]
    
    return {
        "success": True,
        "batch_results": results,
        "summary": {
            "total_files": len(files),
            "successful_detections": len(successful_results),
            "failed_detections": len(files) - len(successful_results),
            "user_id": user_id
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    
    # Check external services
    detector_healthy = await check_service_health(DETECTOR_API_URL)
    assistant_healthy = await check_service_health(ASSISTANT_API_URL)
    
    return {
        "service": "Braille Detection Firebase API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "firebase_connection": "connected" if firebase_service.is_connected() else "disconnected",
            "detector_api": "healthy" if detector_healthy else "unhealthy",
            "assistant_api": "healthy" if assistant_healthy else "unhealthy"
        },
        "external_services": {
            "detector_api_url": DETECTOR_API_URL,
            "assistant_api_url": ASSISTANT_API_URL
        }
    }

@app.get("/service-status")
async def get_service_status():
    """Detailed service status information"""
    
    services_status = {}
    
    # Check Detector API
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{DETECTOR_API_URL}/detector-info")
            if response.status_code == 200:
                services_status["detector"] = {
                    "status": "healthy",
                    "info": response.json()
                }
            else:
                services_status["detector"] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        services_status["detector"] = {"status": "unavailable", "error": str(e)}
    
    # Check Assistant API
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ASSISTANT_API_URL}/assistant-info")
            if response.status_code == 200:
                services_status["assistant"] = {
                    "status": "healthy",
                    "info": response.json()
                }
            else:
                services_status["assistant"] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        services_status["assistant"] = {"status": "unavailable", "error": str(e)}
    
    return {
        "main_api": "healthy",
        "firebase": "connected" if firebase_service.is_connected() else "disconnected",
        "external_services": services_status,
        "timestamp": datetime.now().isoformat()
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

# Vercel handler
handler = Mangum(app)