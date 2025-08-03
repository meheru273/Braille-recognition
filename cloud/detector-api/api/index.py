# detector_api.py - Optimized Braille Detection for Vercel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import os
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from PIL import Image
import io

# Initialize FastAPI with reduced metadata
app = FastAPI(
    title="Braille Detector API",
    version="1.0.0"
)

# Minimal CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Lazy loading for detector
detector = None

def get_detector():
    """Lazy load detector to reduce cold start time"""
    global detector
    if detector is None:
        try:
            from detector import BrailleDetector
            detector = BrailleDetector()
        except ImportError as e:
            raise HTTPException(status_code=503, detail=f"Detector unavailable: {e}")
    return detector

# Validate image size and format
def validate_image(file: UploadFile) -> None:
    """Validate image before processing"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (limit to 4MB to stay under Vercel's 4.5MB limit)
    if hasattr(file, 'size') and file.size > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Maximum 4MB allowed")

def optimize_image(image_content: bytes) -> bytes:
    """Optimize image to reduce processing load"""
    try:
        with Image.open(io.BytesIO(image_content)) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (max 1920x1080)
            max_size = (1920, 1080)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Compress image
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            return output.getvalue()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {e}")

@app.get("/")
async def root():
    """Minimal API status endpoint"""
    return {
        "service": "Braille Detector API",
        "status": "active",
        "version": "1.0.0"
    }

@app.post("/detect")
async def detect_braille_characters(
    file: UploadFile = File(...),
    min_confidence: float = Form(0.4),
    create_annotated: bool = Form(False)  # Default to False to save processing
):
    """
    Detect Braille characters in uploaded image (optimized for Vercel)
    """
    
    validate_image(file)
    session_id = str(uuid.uuid4())[:8]  # Shorter session ID
    temp_path = None
    
    try:
        # Get detector (lazy loaded)
        braille_detector = get_detector()
        
        # Read and optimize image
        content = await file.read()
        optimized_content = optimize_image(content)
        
        # Use memory-based temporary file to avoid disk I/O issues
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(optimized_content)
            temp_path = temp_file.name
        
        # Run detection with timeout and fallback methods
        try:
            detection_result = await asyncio.wait_for(
                asyncio.to_thread(braille_detector.detect_braille_with_fallback, temp_path),
                timeout=25  # Leave 5 seconds buffer for Vercel's 30s limit
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Detection timeout")
        
        if not detection_result:
            return {
                "success": True,
                "session_id": session_id,
                "detection_results": {
                    "organized_text": [],
                    "statistics": {
                        "total_detections": 0,
                        "message": "No braille characters detected"
                    }
                }
            }
        
        # Extract predictions
        predictions = braille_detector.extract_predictions(detection_result)
        if not predictions:
            return {
                "success": True,
                "session_id": session_id,
                "detection_results": {
                    "organized_text": [],
                    "statistics": {
                        "total_detections": 0,
                        "message": "No valid predictions found"
                    }
                }
            }
        
        # Organize text
        detected_strings = braille_detector.organize_text_by_rows(predictions, min_confidence)
        
        # Calculate basic statistics
        high_confidence_predictions = [p for p in predictions if p['confidence'] >= min_confidence]
        avg_confidence = (
            sum(p['confidence'] for p in high_confidence_predictions) / len(high_confidence_predictions)
            if high_confidence_predictions else 0
        )
        
        # Minimal response to reduce bandwidth
        response = {
            "success": True,
            "session_id": session_id,
            "detection_results": {
                "organized_text": detected_strings,
                "statistics": {
                    "total_detections": len(predictions),
                    "high_confidence_detections": len(high_confidence_predictions),
                    "average_confidence": round(avg_confidence, 2),
                    "rows_detected": len(detected_strings)
                }
            }
        }
        
        # Only include raw predictions if explicitly requested
        if create_annotated and len(predictions) < 50:  # Limit to prevent large responses
            response["detection_results"]["raw_predictions"] = predictions[:50]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.post("/detect-batch")
async def detect_braille_batch(
    files: List[UploadFile] = File(...),
    min_confidence: float = Form(0.4)
):
    """
    Batch detect (limited to 3 files for Vercel)
    """
    
    if len(files) > 3:  # Reduced limit for Vercel
        raise HTTPException(status_code=400, detail="Maximum 3 files allowed per batch")
    
    braille_detector = get_detector()
    results = []
    
    for i, file in enumerate(files):
        try:
            validate_image(file)
            
            content = await file.read()
            optimized_content = optimize_image(content)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(optimized_content)
                temp_path = temp_file.name
            
            try:
                detection_result = await asyncio.wait_for(
                    asyncio.to_thread(braille_detector.detect_braille_with_fallback, temp_path),
                    timeout=20  # Shorter timeout for batch
                )
                
                predictions = braille_detector.extract_predictions(detection_result) if detection_result else []
                detected_strings = braille_detector.organize_text_by_rows(predictions, min_confidence)
                
                high_confidence_predictions = [p for p in predictions if p['confidence'] >= min_confidence]
                avg_confidence = (
                    sum(p['confidence'] for p in high_confidence_predictions) / len(high_confidence_predictions)
                    if high_confidence_predictions else 0
                )
                
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": True,
                    "detected_text": detected_strings,
                    "statistics": {
                        "total_detections": len(predictions),
                        "high_confidence_detections": len(high_confidence_predictions),
                        "average_confidence": round(avg_confidence, 2)
                    }
                })
                
            except asyncio.TimeoutError:
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": "Processing timeout"
                })
            
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
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
            "successful_detections": len(successful_results)
        }
    }

@app.get("/health")
async def health_check():
    """Minimal health check"""
    try:
        detector_ready = detector is not None
        return {
            "status": "healthy",
            "detector_ready": detector_ready
        }
    except:
        return {
            "status": "healthy",
            "detector_ready": False
        }

# Minimal error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

