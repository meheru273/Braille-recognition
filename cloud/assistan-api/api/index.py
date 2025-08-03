# assistant_api.py - Lightweight AI Assistant Microservice
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import asyncio

# Import lightweight assistant
try:
    from assistant import BrailleAssistant, BrailleResult
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure assistant.py is in the same directory")

# Initialize FastAPI
app = FastAPI(
    title="AI Assistant API",
    description="Lightweight AI Assistant for text processing and chat",
    version="2.0.0"
)

# Minimal CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Lazy load assistant
assistant = None

def get_assistant():
    """Lazy load assistant to reduce cold start time"""
    global assistant
    if assistant is None:
        try:
            assistant = BrailleAssistant()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Assistant unavailable: {e}")
    return assistant

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "service": "AI Assistant API",
        "status": "active",
        "version": "2.0.0",
        "type": "lightweight"
    }

@app.post("/process-braille")
async def process_braille_text(
    detected_strings: List[str] = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Process detected braille strings into readable text with explanation"""
    
    if not detected_strings:
        raise HTTPException(status_code=400, detail="No braille strings provided")
    
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    
    try:
        ai_assistant = get_assistant()
        
        # Process with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(ai_assistant.process_braille_strings, detected_strings),
            timeout=25
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "processing_result": {
                "interpreted_text": result.text,
                "explanation": result.explanation,
                "confidence": round(result.confidence, 3),
                "input_strings": detected_strings[:10],  # Limit for response size
                "processing_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "input_count": len(detected_strings),
                "processing_method": "lightweight_ai"
            }
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Processing timeout")
    except Exception as e:
        # Fallback result
        fallback_text = ' '.join(detected_strings)
        return {
            "success": True,
            "session_id": session_id,
            "processing_result": {
                "interpreted_text": fallback_text,
                "explanation": f"Basic text assembly. AI processing failed.",
                "confidence": 0.3,
                "input_strings": detected_strings[:10],
                "processing_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "input_count": len(detected_strings),
                "processing_method": "fallback",
                "error": str(e)
            }
        }

@app.post("/chat")
async def chat_with_assistant(
    message: str = Form(...),
    thread_id: Optional[str] = Form(None)
):
    """Chat with AI assistant"""
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not thread_id:
        thread_id = f"chat_{uuid.uuid4()}"[:16]
    
    try:
        ai_assistant = get_assistant()
        
        # Process with timeout
        response_text = await asyncio.wait_for(
            asyncio.to_thread(ai_assistant.chat, message, thread_id),
            timeout=20
        )
        
        return {
            "success": True,
            "thread_id": thread_id,
            "chat_result": {
                "user_message": message,
                "assistant_response": response_text,
                "response_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "message_length": len(message),
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
async def process_general_text(
    text: str = Form(...),
    task: str = Form("explain"),
    max_length: Optional[int] = Form(400)
):
    """Process general text with specific tasks"""
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Task must be one of: {valid_tasks}")
    
    try:
        ai_assistant = get_assistant()
        
        # Process with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(ai_assistant.process_text, text, task, max_length),
            timeout=20
        )
        
        return {
            "success": True,
            "processing_result": {
                "original_text": text,
                "processed_text": response,
                "task_performed": task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "original_length": len(text),
                "processed_length": len(response),
                "task": task
            }
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Processing timeout")
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original_text": text,
            "task": task
        }

@app.post("/batch-process")
async def batch_process_texts(
    texts: List[str] = Form(...),
    task: str = Form("explain")
):
    """Process multiple texts in batch (limited to 5 for performance)"""
    
    if len(texts) > 5:  # Reduced for lightweight version
        raise HTTPException(status_code=400, detail="Maximum 5 texts allowed per batch")
    
    valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Task must be one of: {valid_tasks}")
    
    ai_assistant = get_assistant()
    results = []
    
    for i, text in enumerate(texts):
        if not text.strip():
            results.append({
                "index": i,
                "success": False,
                "error": "Empty text"
            })
            continue
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(ai_assistant.process_text, text, task, 300),
                timeout=15
            )
            
            results.append({
                "index": i,
                "success": True,
                "original_text": text[:100] + "..." if len(text) > 100 else text,
                "processed_text": response,
                "task": task
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
    
    successful_results = [r for r in results if r['success']]
    
    return {
        "success": True,
        "batch_results": results,
        "summary": {
            "total_texts": len(texts),
            "successful_processing": len(successful_results),
            "task_performed": task
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        assistant_ready = assistant is not None
        return {
            "service": "AI Assistant API",
            "status": "healthy",
            "version": "2.0.0",
            "type": "lightweight",
            "assistant_ready": assistant_ready
        }
    except:
        return {
            "service": "AI Assistant API",
            "status": "healthy",
            "version": "2.0.0",
            "type": "lightweight",
            "assistant_ready": False
        }

@app.get("/capabilities")
async def get_capabilities():
    """Get assistant capabilities"""
    return {
        "version": "2.0.0",
        "type": "lightweight",
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
            "bundle_size": "~5MB (vs 80MB+ with LangChain)",
            "cold_start": "3-8 seconds",
            "processing_timeout": "20-25 seconds"
        }
    }

# Minimal error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Vercel handler
handler = Mangum(app)