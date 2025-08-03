# api/index.py - Fixed Vercel-compatible entry point
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

# Import lightweight assistant with better error handling
BrailleAssistant = None
try:
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from assistant import BrailleAssistant
except ImportError as e:
    print(f"Import error: {e}")
    BrailleAssistant = None

# Pydantic models
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

# Create FastAPI app with minimal config for Vercel
app = FastAPI(
    title="AI Assistant API",
    description="Lightweight AI Assistant for text processing and chat",
    version="2.0.1"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global assistant instance
assistant_instance = None

def get_assistant():
    """Lazy load assistant"""
    global assistant_instance
    if assistant_instance is None:
        if BrailleAssistant is None:
            raise HTTPException(
                status_code=503, 
                detail="Assistant module not available"
            )
        try:
            assistant_instance = BrailleAssistant()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Assistant initialization failed: {str(e)}"
            )
    return assistant_instance

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "service": "AI Assistant API",
        "status": "active",
        "version": "2.0.1",
        "assistant_available": BrailleAssistant is not None,
        "endpoints": [
            "/health", "/capabilities", "/process-braille", 
            "/chat", "/process-text", "/batch-process"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.1",
        "assistant_available": BrailleAssistant is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/capabilities")
async def get_capabilities():
    """Get assistant capabilities"""
    return {
        "version": "2.0.1",
        "capabilities": {
            "braille_processing": True,
            "general_chat": True,
            "text_processing": True,
            "batch_processing": True
        },
        "supported_tasks": ["explain", "summarize", "correct", "enhance", "analyze"],
        "limits": {
            "max_batch_size": 5,
            "max_text_length": 2000
        }
    }

@app.post("/process-braille")
async def process_braille_text(request: BrailleProcessRequest):
    """Process detected braille strings"""
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
                "input_strings": request.detected_strings[:10],
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
        # Fallback response
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
    """Process multiple texts in batch"""
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

# Vercel handler - Fixed version
try:
    from mangum import Mangum
    # Use lifespan="off" to avoid issues with Vercel
    handler = Mangum(app, lifespan="off", api_gateway_base_path="/")
except ImportError:
    # Create a simple ASGI handler if mangum fails
    async def handler(scope, receive, send):
        if scope["type"] == "http":
            await app(scope, receive, send)
        else:
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"application/json"]],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"error": "Unsupported request type"}',
            })

# Alternative handler for Vercel compatibility
def lambda_handler(event, context):
    """Alternative handler for better Vercel compatibility"""
    try:
        from mangum import Mangum
        mangum_handler = Mangum(app, lifespan="off")
        return mangum_handler(event, context)
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": f'{{"error": "Handler error: {str(e)}"}}'
        }

# Export both handlers
__all__ = ["app", "handler", "lambda_handler"]