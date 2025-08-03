from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from mangum import Mangum
from detector import BrailleDetector
import os
import uuid
import tempfile
import datetime
from fastapi.responses import JSONResponse

app = FastAPI(title="Braille Detection API")

# Initialize detector
detector = BrailleDetector()

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "message": "Braille Detection API",
        "status": "active",
        "version": "1.0.0",
        "services": {
            "braille_detector": "ready"
        }
    }

@app.post("/detect-braille")
async def detect_braille(
    file: UploadFile = File(...),
    user_id: str = Form(None)
):
    """Detect Braille characters in uploaded image"""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_path = temp_file.name
        
        # Save uploaded file
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Run detection
        result = detector.detect_braille(temp_path)
        if not result:
            os.unlink(temp_path)
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Detection failed"}
            )
        
        # Extract and organize predictions
        predictions = detector.extract_predictions(result)
        detected_strings = detector.organize_text_by_rows(predictions)
        
        # Clean up
        os.unlink(temp_path)
        
        # Return results
        return {
            "success": True,
            "text": " ".join(detected_strings),
            "rows": detected_strings,
            "raw_detections": predictions,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Detection error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "detector",
        "dependencies": {
            "roboflow": "connected" if os.getenv("ROBOFLOW_API_KEY") else "missing"
        }
    }

handler = Mangum(app)