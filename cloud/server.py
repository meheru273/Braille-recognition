# server.py - Main FastAPI server for Vercel deployment
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import uuid
from typing import Optional
import json

from detector import BrailleDetector
from assistant import BrailleAssistant

app = FastAPI(title="Braille Detection & Chat API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detector = BrailleDetector()
assistant = BrailleAssistant()

# Create temp directories
os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/outputs", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Braille Detection & Chat API", "status": "active"}

@app.post("/detect-braille")
async def detect_braille_endpoint(file: UploadFile = File(...)):
    """
    Image Input -> Detector -> Assistant (Output: annotated image + text)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Generate unique IDs
        session_id = str(uuid.uuid4())
        
        # Save uploaded image
        input_path = f"/tmp/uploads/{session_id}_{file.filename}"
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Step 1: Run detector
        result = detector.detect_braille(input_path)
        if not result:
            raise HTTPException(status_code=500, detail="Detection failed")
        
        predictions = detector.extract_predictions(result)
        if not predictions:
            return JSONResponse({
                "success": False,
                "message": "No braille characters detected",
                "session_id": session_id
            })
        
        # Step 2: Create annotated image
        annotated_path = f"/tmp/outputs/{session_id}_annotated.png"
        annotation_success = detector.create_annotated_image(
            input_path, predictions, annotated_path
        )
        
        # Step 3: Organize text and send to assistant
        detected_strings = detector.organize_text_by_rows(predictions)
        braille_result = assistant.process_braille_strings(detected_strings)
        
        # Prepare response
        response_data = {
            "success": True,
            "session_id": session_id,
            "detection_stats": {
                "total_characters": len(predictions),
                "detected_rows": len(detected_strings),
                "average_confidence": sum(p['confidence'] for p in predictions) / len(predictions)
            },
            "braille_result": {
                "text": braille_result.text,
                "explanation": braille_result.explanation,
                "confidence": braille_result.confidence
            },
            "annotated_image_available": annotation_success,
            "raw_detections": detected_strings
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/get-annotated-image/{session_id}")
async def get_annotated_image(session_id: str):
    """Get the annotated image for a session"""
    image_path = f"/tmp/outputs/{session_id}_annotated.png"
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Annotated image not found")
    
    return FileResponse(
        image_path, 
        media_type="image/png",
        filename=f"annotated_{session_id}.png"
    )

@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    thread_id: Optional[str] = Form(None)
):
    """
    Text Input -> Assistant (Output: text response)
    """
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Use provided thread_id or generate new one
        if not thread_id:
            thread_id = str(uuid.uuid4())
        
        # Send directly to assistant
        response = assistant.chat(message, thread_id)
        
        return JSONResponse({
            "success": True,
            "thread_id": thread_id,
            "message": message,
            "response": response
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/batch-process")
async def batch_process_endpoint(
    files: list[UploadFile] = File(...),
    chat_messages: Optional[str] = Form(None)  # JSON string of messages
):
    """
    Process multiple images and chat messages in one request
    """
    results = {
        "braille_results": [],
        "chat_results": [],
        "session_id": str(uuid.uuid4())
    }
    
    try:
        # Process images
        for file in files:
            if file.content_type.startswith('image/'):
                # Save file
                file_id = str(uuid.uuid4())
                input_path = f"/tmp/uploads/{file_id}_{file.filename}"
                with open(input_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Process through detector -> assistant
                result = detector.detect_braille(input_path)
                if result:
                    predictions = detector.extract_predictions(result)
                    if predictions:
                        # Create annotated image
                        annotated_path = f"/tmp/outputs/{file_id}_annotated.png"
                        detector.create_annotated_image(input_path, predictions, annotated_path)
                        
                        # Process with assistant
                        detected_strings = detector.organize_text_by_rows(predictions)
                        braille_result = assistant.process_braille_strings(detected_strings)
                        
                        results["braille_results"].append({
                            "file_id": file_id,
                            "filename": file.filename,
                            "text": braille_result.text,
                            "explanation": braille_result.explanation,
                            "confidence": braille_result.confidence,
                            "detected_characters": len(predictions)
                        })
        
        # Process chat messages
        if chat_messages:
            try:
                messages = json.loads(chat_messages)
                thread_id = str(uuid.uuid4())
                
                for msg in messages:
                    if isinstance(msg, str):
                        response = assistant.chat(msg, thread_id)
                        results["chat_results"].append({
                            "message": msg,
                            "response": response
                        })
                        
            except json.JSONDecodeError:
                results["chat_error"] = "Invalid JSON format for chat messages"
        
        return JSONResponse(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector": "ready",
        "assistant": "ready",
        "version": "1.0"
    }

@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up temporary files for a session"""
    try:
        # Remove uploaded files
        for file_path in [
            f"/tmp/uploads/{session_id}_*",
            f"/tmp/outputs/{session_id}_*"
        ]:
            import glob
            for f in glob.glob(file_path):
                if os.path.exists(f):
                    os.remove(f)
        
        return {"success": True, "message": f"Session {session_id} cleaned up"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# ====================== DEPLOYMENT CONFIGURATION ======================

# For Vercel deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Vercel handler
def handler(request, response):
    return app