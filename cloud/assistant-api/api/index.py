# api/index.py - Vercel-compatible entry point (JSON-based)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import asyncio
import sys
import os

# Import lightweight assistant
try:
    # Add parent directory to path so we can import assistant.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from assistant import BrailleAssistant
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current dir: {os.getcwd()}")
    print(f"Files in current dir: {os.listdir('.')}")
    if os.path.exists('..'):
        print(f"Files in parent dir: {os.listdir('..')}")
    BrailleAssistant = None

# Pydantic models for request/response
class BrailleProcessRequest(BaseModel):
    detected_strings: List[str]
    session_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class TextProcessRequest(BaseModel):
    text: str
    task: str = "explain"
    max_length: Optional[int] = 400

class BatchProcessRequest(BaseModel):
    texts: List[str]
    task: str = "explain"

# Initialize FastAPI
app = FastAPI(
    title="AI Assistant API",
    description="Lightweight AI Assistant for text processing and chat",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global assistant instance
assistant = None

def get_assistant():
    """Lazy load assistant to reduce cold start time"""
    global assistant
    if assistant is None:
        if BrailleAssistant is None:
            raise HTTPException(
                status_code=503, 
                detail="Assistant module not available. Check assistant.py import."
            )
        try:
            assistant = BrailleAssistant()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Assistant initialization failed: {e}")
    return assistant

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "service": "AI Assistant API",
        "status": "active",
        "version": "2.0.0",
        "type": "lightweight",
        "assistant_available": BrailleAssistant is not None,
        "request_format": "JSON",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API status"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/capabilities", "method": "GET", "description": "Get capabilities"},
            {"path": "/process-braille", "method": "POST", "description": "Process braille strings"},
            {"path": "/chat", "method": "POST", "description": "Chat with assistant"},
            {"path": "/process-text", "method": "POST", "description": "Process text"},
            {"path": "/batch-process", "method": "POST", "description": "Batch process texts"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        assistant_ready = assistant is not None
        assistant_available = BrailleAssistant is not None
        
        return {
            "service": "AI Assistant API",
            "status": "healthy",
            "version": "2.0.0",
            "type": "lightweight",
            "assistant_available": assistant_available,
            "assistant_ready": assistant_ready,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "AI Assistant API",
            "status": "degraded",
            "version": "2.0.0",
            "type": "lightweight",
            "assistant_available": False,
            "assistant_ready": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/capabilities")
async def get_capabilities():
    """Get assistant capabilities"""
    return {
        "version": "2.0.0",
        "type": "lightweight",
        "request_format": "JSON",
        "capabilities": {
            "braille_processing": True,
            "general_chat": True,
            "text_processing": True,
            "batch_processing": True,
            "wikipedia_search": True
        },
        "supported_tasks": ["explain", "summarize", "correct", "enhance", "analyze"],
        "limits": {
            "max_batch_size": 5,
            "max_text_length": 2000,
            "max_response_length": 500
        },
        "performance": {
            "bundle_size": "~3MB (JSON-based, no multipart)",
            "cold_start": "2-6 seconds",
            "processing_timeout": "20-25 seconds"
        }
    }

@app.post("/process-braille")
async def process_braille_text(request: BrailleProcessRequest):
    """Process detected braille strings into readable text with explanation"""
    
    if not request.detected_strings:
        raise HTTPException(status_code=400, detail="No braille strings provided")
    
    session_id = request.session_id or str(uuid.uuid4())[:8]
    
    try:
        ai_assistant = get_assistant()
        
        # Process with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(ai_assistant.process_braille_strings, request.detected_strings),
            timeout=25
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "processing_result": {
                "interpreted_text": result.text,
                "explanation": result.explanation,
                "confidence": round(result.confidence, 3),
                "input_strings": request.detected_strings[:10],  # Limit for response size
                "processing_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "input_count": len(request.detected_strings),
                "processing_method": "lightweight_ai"
            }
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Processing timeout")
    except Exception as e:
        # Fallback result
        fallback_text = ' '.join(request.detected_strings)
        return {
            "success": True,
            "session_id": session_id,
            "processing_result": {
                "interpreted_text": fallback_text,
                "explanation": f"Basic text assembly. AI processing failed: {str(e)}",
                "confidence": 0.3,
                "input_strings": request.detected_strings[:10],
                "processing_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "input_count": len(request.detected_strings),
                "processing_method": "fallback",
                "error": str(e)
            }
        }

@app.post("/chat")
async def chat_with_assistant(request: ChatRequest):
    """Chat with AI assistant"""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    thread_id = request.thread_id or f"chat_{uuid.uuid4()}"[:16]
    
    try:
        ai_assistant = get_assistant()
        
        # Process with timeout
        response_text = await asyncio.wait_for(
            asyncio.to_thread(ai_assistant.chat, request.message, thread_id),
            timeout=20
        )
        
        return {
            "success": True,
            "thread_id": thread_id,
            "chat_result": {
                "user_message": request.message,
                "assistant_response": response_text,
                "response_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "message_length": len(request.message),
                "response_length": len(response_text)
            }
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Chat timeout")
    except Exception as e:
        return {
            "success": False,
            "thread_id": thread_id,
            "error": str(e),
            "fallback_response": "I encountered an issue. Please try again."
        }

@app.post("/process-text")
async def process_general_text(request: TextProcessRequest):
    """Process general text with specific tasks"""
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
    if request.task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Task must be one of: {valid_tasks}")
    
    try:
        ai_assistant = get_assistant()
        
        # Process with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(ai_assistant.process_text, request.text, request.task, request.max_length),
            timeout=20
        )
        
        return {
            "success": True,
            "processing_result": {
                "original_text": request.text,
                "processed_text": response,
                "task_performed": request.task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "original_length": len(request.text),
                "processed_length": len(response),
                "task": request.task
            }
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Processing timeout")
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original_text": request.text,
            "task": request.task
        }

@app.post("/batch-process")
async def batch_process_texts(request: BatchProcessRequest):
    """Process multiple texts in batch (limited to 5 for performance)"""
    
    if len(request.texts) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 texts allowed per batch")
    
    valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
    if request.task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Task must be one of: {valid_tasks}")
    
    try:
        ai_assistant = get_assistant()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Assistant unavailable: {e}")
    
    results = []
    
    for i, text in enumerate(request.texts):
        if not text.strip():
            results.append({
                "index": i,
                "success": False,
                "error": "Empty text"
            })
            continue
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(ai_assistant.process_text, text, request.task, 300),
                timeout=15
            )
            
            results.append({
                "index": i,
                "success": True,
                "original_text": text[:100] + "..." if len(text) > 100 else text,
                "processed_text": response,
                "task": request.task
            })
            
        except asyncio.TimeoutError:
            results.append({
                "index": i,
                "success": False,
                "error": "Processing timeout"
            })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    successful_results = [r for r in results if r.get('success', False)]
    
    return {
        "success": True,
        "batch_results": results,
        "summary": {
            "total_texts": len(request.texts),
            "successful_processing": len(successful_results),
            "task_performed": request.task
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found", 
            "available_endpoints": ["/", "/health", "/capabilities", "/process-braille", "/chat", "/process-text", "/batch-process"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# For Vercel - Mangum handler
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    # Fallback if mangum isn't available
    def handler(event, context):
        return {"statusCode": 500, "body": "Mangum not available"}

# Export app for direct usage
__all__ = ["app", "handler"]