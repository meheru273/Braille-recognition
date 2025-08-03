# assistant_api.py - AI Assistant Microservice
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from typing import List, Dict, Optional
from datetime import datetime
import uuid

# Import assistant module
try:
    from assistant import BrailleAssistant, BrailleResult
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure assistant.py is in the same directory")

# Initialize FastAPI
app = FastAPI(
    title="AI Assistant API",
    description="Microservice for AI-powered text processing and chat assistance",
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

# Initialize assistant
assistant = None

@app.on_event("startup")
async def startup_event():
    """Initialize AI assistant on startup"""
    global assistant
    
    try:
        print("Initializing AI Assistant...")
        assistant = BrailleAssistant()
        print("AI Assistant initialized successfully")
        
    except Exception as e:
        print(f"Assistant initialization error: {e}")

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "service": "AI Assistant API",
        "status": "active",
        "version": "1.0.0",
        "assistant_ready": assistant is not None,
        "endpoints": {
            "process_braille": "/process-braille",
            "chat": "/chat",
            "process_text": "/process-text",
            "health": "/health"
        }
    }

@app.post("/process-braille")
async def process_braille_text(
    detected_strings: List[str] = Form(...),
    session_id: Optional[str] = Form(None)
):
    """
    Process detected braille strings into readable text with explanation
    
    - **detected_strings**: List of detected braille character strings
    - **session_id**: Optional session identifier for tracking
    """
    
    if not assistant:
        raise HTTPException(status_code=503, detail="AI assistant not available")
    
    if not detected_strings:
        raise HTTPException(status_code=400, detail="No braille strings provided")
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        print(f"Processing braille strings for session {session_id}")
        
        # Process braille strings
        result = assistant.process_braille_strings(detected_strings)
        
        return {
            "success": True,
            "session_id": session_id,
            "processing_result": {
                "interpreted_text": result.text,
                "explanation": result.explanation,
                "confidence": round(result.confidence, 3),
                "input_strings": detected_strings,
                "processing_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "input_count": len(detected_strings),
                "total_characters": sum(len(s) for s in detected_strings),
                "processing_method": "ai_enhanced"
            }
        }
        
    except Exception as e:
        print(f"Braille processing error: {e}")
        # Return fallback result
        fallback_text = ' '.join(detected_strings)
        return {
            "success": True,
            "session_id": session_id,
            "processing_result": {
                "interpreted_text": fallback_text,
                "explanation": f"Basic text assembly completed. Enhanced processing failed: {str(e)}",
                "confidence": 0.3,
                "input_strings": detected_strings,
                "processing_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "input_count": len(detected_strings),
                "total_characters": sum(len(s) for s in detected_strings),
                "processing_method": "fallback",
                "error": str(e)
            }
        }

@app.post("/chat")
async def chat_with_assistant(
    message: str = Form(...),
    thread_id: Optional[str] = Form(None),
    context: Optional[str] = Form(None)
):
    """
    Chat with AI assistant
    
    - **message**: User's message/question
    - **thread_id**: Optional conversation thread ID
    - **context**: Optional context for the conversation
    """
    
    if not assistant:
        raise HTTPException(status_code=503, detail="AI assistant not available")
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not thread_id:
        thread_id = f"chat_{uuid.uuid4()}"
    
    try:
        print(f"Processing chat message for thread {thread_id}")
        
        # Add context to message if provided
        if context:
            enhanced_message = f"Context: {context}\n\nUser question: {message}"
        else:
            enhanced_message = message
        
        # Get AI response
        response_text = assistant.chat(enhanced_message, thread_id)
        
        if not response_text:
            response_text = "I apologize, but I couldn't process your message at this time."
        
        return {
            "success": True,
            "thread_id": thread_id,
            "chat_result": {
                "user_message": message,
                "assistant_response": response_text,
                "context_used": context is not None,
                "response_timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "message_length": len(message),
                "response_length": len(response_text),
                "has_context": context is not None
            }
        }
        
    except Exception as e:
        print(f"Chat processing error: {e}")
        return {
            "success": False,
            "thread_id": thread_id,
            "error": str(e),
            "fallback_response": "I encountered an issue processing your message. Please try again."
        }

@app.post("/process-text")
async def process_general_text(
    text: str = Form(...),
    task: str = Form("explain"),
    max_length: Optional[int] = Form(500)
):
    """
    Process general text with specific tasks
    
    - **text**: Input text to process
    - **task**: Processing task (explain, summarize, correct, enhance)
    - **max_length**: Maximum response length
    """
    
    if not assistant:
        raise HTTPException(status_code=503, detail="AI assistant not available")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Task must be one of: {valid_tasks}")
    
    try:
        # Create task-specific prompt
        if task == "explain":
            prompt = f"Explain this text clearly and concisely: '{text}'"
        elif task == "summarize":
            prompt = f"Provide a brief summary of this text: '{text}'"
        elif task == "correct":
            prompt = f"Correct any errors in this text and improve clarity: '{text}'"
        elif task == "enhance":
            prompt = f"Enhance and improve this text while maintaining its meaning: '{text}'"
        elif task == "analyze":
            prompt = f"Analyze the content and meaning of this text: '{text}'"
        
        if max_length:
            prompt += f"\n\nKeep response under {max_length} characters."
        
        response = assistant.chat(prompt, f"text_processing_{uuid.uuid4()}")
        
        # Truncate if needed
        if max_length and len(response) > max_length:
            response = response[:max_length-3] + "..."
        
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
                "task": task,
                "truncated": max_length and len(response) >= max_length-3
            }
        }
        
    except Exception as e:
        print(f"Text processing error: {e}")
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
    """
    Process multiple texts in batch
    
    - **texts**: List of texts to process
    - **task**: Processing task for all texts
    """
    
    if not assistant:
        raise HTTPException(status_code=503, detail="AI assistant not available")
    
    if len(texts) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 20 texts allowed per batch")
    
    valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Task must be one of: {valid_tasks}")
    
    results = []
    
    for i, text in enumerate(texts):
        if not text.strip():
            results.append({
                "index": i,
                "success": False,
                "error": "Empty text",
                "original_text": text
            })
            continue
        
        try:
            # Create task-specific prompt
            if task == "explain":
                prompt = f"Explain this text briefly: '{text}'"
            elif task == "summarize":
                prompt = f"Summarize: '{text}'"
            elif task == "correct":
                prompt = f"Correct errors in: '{text}'"
            elif task == "enhance":
                prompt = f"Enhance: '{text}'"
            elif task == "analyze":
                prompt = f"Analyze: '{text}'"
            
            response = assistant.chat(prompt, f"batch_{uuid.uuid4()}_{i}")
            
            results.append({
                "index": i,
                "success": True,
                "original_text": text,
                "processed_text": response,
                "task": task
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e),
                "original_text": text,
                "task": task
            })
    
    successful_results = [r for r in results if r['success']]
    
    return {
        "success": True,
        "batch_results": results,
        "summary": {
            "total_texts": len(texts),
            "successful_processing": len(successful_results),
            "failed_processing": len(texts) - len(successful_results),
            "task_performed": task
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "AI Assistant API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "assistant_ready": assistant is not None,
        "capabilities": {
            "braille_processing": True,
            "general_chat": True,
            "text_processing": True,
            "batch_processing": True
        }
    }

@app.get("/assistant-info")
async def get_assistant_info():
    """Get assistant configuration information"""
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not available")
    
    return {
        "assistant_config": {
            "model_type": "llm_with_tools" if hasattr(assistant, 'llm_with_tools') else "basic_llm",
            "has_wikipedia_access": True,
            "supported_tasks": ["explain", "summarize", "correct", "enhance", "analyze"],
            "max_batch_size": 20
        },
        "available_tools": ["wikipedia_search"],
        "processing_capabilities": {
            "braille_text_interpretation": True,
            "context_aware_chat": True,
            "multi_task_processing": True
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

# Vercel handler
handler = Mangum(app)