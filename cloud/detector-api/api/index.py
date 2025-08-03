# detector_api.py - Braille Detection Microservice
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import os
import uuid
import tempfile
from typing import List, Dict, Any
from datetime import datetime
import asyncio

# Import detector module
try:
    from detector import BrailleDetector
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure detector.py is in the same directory")

# Initialize FastAPI
app = FastAPI(
    title="Braille Detector API",
    description="Microservice for Braille character detection using Computer Vision",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize Braille detector on startup"""
    global detector
    
    try:
        print("Initializing Braille Detector...")
        detector = BrailleDetector()
        print("Braille Detector initialized successfully")
        
    except Exception as e:
        print(f"Detector initialization error: {e}")

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "service": "Braille Detector API",
        "status": "active",
        "version": "1.0.0",
        "detector_ready": detector is not None,
        "endpoints": {
            "detect": "/detect",
            "health": "/health"
        }
    }

@app.post("/detect")
async def detect_braille_characters(
    file: UploadFile = File(...),
    min_confidence: float = Form(0.4),
    create_annotated: bool = Form(True)
):
    """
    Detect Braille characters in uploaded image
    
    - **file**: Image file (jpg, png, etc.)
    - **min_confidence**: Minimum confidence threshold (0.0-1.0)
    - **create_annotated**: Whether to create annotated image
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not detector:
        raise HTTPException(status_code=503, detail="Braille detector not available")
    
    session_id = str(uuid.uuid4())
    temp_path = None
    annotated_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"Processing image for session {session_id}")
        
        # Run Braille detection
        detection_result = detector.detect_braille(temp_path)
        if not detection_result:
            raise HTTPException(status_code=422, detail="No braille characters detected in image")
        
        # Extract predictions
        predictions = detector.extract_predictions(detection_result)
        if not predictions:
            raise HTTPException(status_code=422, detail="No valid braille predictions found")
        
        # Organize into text rows
        detected_strings = detector.organize_text_by_rows(predictions, min_confidence)
        
        # Create annotated image if requested
        annotated_image_created = False
        if create_annotated and predictions:
            annotated_path = temp_path.replace('.jpg', '_annotated.png')
            annotated_image_created = detector.create_annotated_image(
                temp_path, predictions, annotated_path, min_confidence
            )
        
        # Calculate statistics
        high_confidence_predictions = [p for p in predictions if p['confidence'] >= min_confidence]
        avg_confidence = sum(p['confidence'] for p in high_confidence_predictions) / len(high_confidence_predictions) if high_confidence_predictions else 0
        
        return {
            "success": True,
            "session_id": session_id,
            "detection_results": {
                "raw_predictions": predictions,
                "organized_text": detected_strings,
                "statistics": {
                    "total_detections": len(predictions),
                    "high_confidence_detections": len(high_confidence_predictions),
                    "average_confidence": round(avg_confidence, 3),
                    "rows_detected": len(detected_strings),
                    "min_confidence_used": min_confidence
                },
                "annotated_image_created": annotated_image_created,
                "processing_timestamp": datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Detection processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        for path in [temp_path, annotated_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

@app.post("/detect-batch")
async def detect_braille_batch(
    files: List[UploadFile] = File(...),
    min_confidence: float = Form(0.4)
):
    """
    Batch detect Braille characters in multiple images
    
    - **files**: List of image files
    - **min_confidence**: Minimum confidence threshold
    """
    
    if not detector:
        raise HTTPException(status_code=503, detail="Braille detector not available")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for i, file in enumerate(files):
        if not file.content_type or not file.content_type.startswith('image/'):
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": False,
                "error": "File must be an image"
            })
            continue
        
        temp_path = None
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
            
            # Run detection
            detection_result = detector.detect_braille(temp_path)
            predictions = detector.extract_predictions(detection_result) if detection_result else []
            detected_strings = detector.organize_text_by_rows(predictions, min_confidence)
            
            high_confidence_predictions = [p for p in predictions if p['confidence'] >= min_confidence]
            avg_confidence = sum(p['confidence'] for p in high_confidence_predictions) / len(high_confidence_predictions) if high_confidence_predictions else 0
            
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": True,
                "detected_text": detected_strings,
                "statistics": {
                    "total_detections": len(predictions),
                    "high_confidence_detections": len(high_confidence_predictions),
                    "average_confidence": round(avg_confidence, 3),
                    "rows_detected": len(detected_strings)
                }
            })
            
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
        
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    successful_results = [r for r in results if r['success']]
    
    return {
        "success": True,
        "batch_results": results,
        "summary": {
            "total_files": len(files),
            "successful_detections": len(successful_results),
            "failed_detections": len(files) - len(successful_results)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "Braille Detector API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detector_ready": detector is not None,
        "dependencies": {
            "roboflow_client": "available" if detector and detector.client else "unavailable",
            "pil": "available",
            "inference_sdk": "available"
        }
    }

@app.get("/detector-info")
async def get_detector_info():
    """Get detector configuration information"""
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    return {
        "detector_config": {
            "workspace_name": detector.workspace_name,
            "workflow_id": detector.workflow_id,
            "available_classes": list(detector.class_colors.keys()),
            "total_classes": len(detector.class_colors)
        },
        "color_mapping": detector.class_colors
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