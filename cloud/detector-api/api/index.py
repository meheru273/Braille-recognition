# braille-detector-api/api/index.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import os
import uuid
import tempfile
from typing import Optional, List
from datetime import datetime

# Import your modules
try:
    from detector import BrailleDetector
    from firebase_service import firebase_service, BrailleDetectionResult
except ImportError as e:
    print(f"Import error: {e}")

# Initialize FastAPI
app = FastAPI(
    title="Braille Detection API",
    description="AI-powered Braille detection service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global detector
    
    try:
        print("Initializing Braille Detector...")
        detector = BrailleDetector()
        print("Braille Detector initialized")
        print(f"Firebase connection: {'✓' if firebase_service.is_connected() else '✗'}")
    except Exception as e:
        print(f"Startup initialization error: {e}")

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "service": "Braille Detection API",
        "status": "active",
        "version": "1.0.0",
        "detector": "ready" if detector else "failed",
        "firebase": "connected" if firebase_service.is_connected() else "disconnected"
    }

@app.post("/detect-braille")
async def detect_braille(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """Detect Braille characters in uploaded image"""
    
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
        
        # Run Braille detection
        result = await detector.detect_braille(temp_path)
        if not result:
            raise HTTPException(status_code=422, detail="No braille characters detected in image")
        
        predictions = detector.extract_predictions(result)
        if not predictions:
            raise HTTPException(status_code=422, detail="No valid braille predictions found")
        
        detected_strings = detector.organize_text_by_rows(predictions)
        
        # Basic text processing (no heavy AI here)
        processed_text = ' '.join(detected_strings)
        confidence = min(0.9, len([s for s in detected_strings if s.strip()]) / max(1, len(detected_strings)))
        
        # Upload image to Firebase Storage
        image_url = None
        if firebase_service.is_connected():
            image_url = await firebase_service.upload_image(
                temp_path, user_id, session_id, file.filename
            )
        
        # Store in Firebase database
        detection_result = BrailleDetectionResult(
            session_id=session_id,
            user_id=user_id,
            filename=file.filename or "uploaded_image.jpg",
            detected_text=processed_text,
            explanation="Raw braille detection completed. Use AI assistant for detailed explanation.",
            confidence=confidence,
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
                "text": processed_text,
                "confidence": round(confidence, 3),
                "detected_characters": len(predictions),
                "rows": len(detected_strings),
                "explanation": "Use the AI assistant for detailed explanation of this text."
            },
            "raw_detections": detected_strings,
            "image_url": image_url,
            "stored_in_database": stored
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

@app.get("/user-detections")
async def get_user_detections(
    user_id: str = Query(...),
    limit: int = Query(10, ge=1, le=50)
):
    """Get user's braille detection history"""
    
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
    """Create or update user profile"""
    
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
        "service": "detector",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detector": "ready" if detector else "failed",
        "firebase": "connected" if firebase_service.is_connected() else "disconnected"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "service": "detector"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "service": "detector"}
    )

# Vercel handler
handler = Mangum(app)