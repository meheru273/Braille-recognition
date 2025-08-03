# api/index.py - Native Vercel handler without Mangum
from http.server import BaseHTTPRequestHandler
import json
import urllib.parse
import sys
import os
from datetime import datetime
import uuid
import asyncio
from typing import Dict, Any, List

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Lazy load assistant
assistant = None

def get_assistant():
    """Lazy load assistant"""
    global assistant
    if assistant is None:
        try:
            from assistant import BrailleAssistant
            assistant = BrailleAssistant()
        except ImportError as e:
            raise Exception(f"Assistant unavailable: {e}")
        except Exception as e:
            raise Exception(f"Assistant initialization failed: {e}")
    return assistant

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == "/" or self.path == "/api" or self.path == "/api/":
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "service": "AI Assistant API",
                    "status": "active",
                    "version": "2.0.0"
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif self.path == "/health" or self.path == "/api/health":
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                assistant_ready = assistant is not None
                response = {
                    "status": "healthy",
                    "assistant_ready": assistant_ready
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif self.path == "/capabilities" or self.path == "/api/capabilities":
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "version": "2.0.0",
                    "capabilities": {
                        "braille_processing": True,
                        "general_chat": True,
                        "text_processing": True,
                        "batch_processing": True
                    },
                    "supported_tasks": ["explain", "summarize", "correct", "enhance", "analyze"],
                    "limits": {
                        "max_batch_size": 3,
                        "max_text_length": 2000
                    }
                }
                self.wfile.write(json.dumps(response).encode())
                
            else:
                self.send_error(404, "Endpoint not found")
                
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                except json.JSONDecodeError:
                    self.send_error(400, "Invalid JSON")
                    return
            else:
                data = {}
            
            # Route requests
            if self.path == "/process-braille" or self.path == "/api/process-braille":
                response = self.handle_process_braille(data)
            elif self.path == "/chat" or self.path == "/api/chat":
                response = self.handle_chat(data)
            elif self.path == "/process-text" or self.path == "/api/process-text":
                response = self.handle_process_text(data)
            elif self.path == "/batch-process" or self.path == "/api/batch-process":
                response = self.handle_batch_process(data)
            else:
                self.send_error(404, "Endpoint not found")
                return
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def handle_process_braille(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle braille processing"""
        detected_strings = data.get('detected_strings', [])
        if not detected_strings:
            raise Exception("No braille strings provided")
        
        session_id = data.get('session_id') or str(uuid.uuid4())[:8]
        
        try:
            ai_assistant = get_assistant()
            result = ai_assistant.process_braille_strings(detected_strings)
            
            return {
                "success": True,
                "session_id": session_id,
                "processing_result": {
                    "interpreted_text": result.text,
                    "explanation": result.explanation,
                    "confidence": round(result.confidence, 3),
                    "input_strings": detected_strings[:10],
                    "processing_timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "input_count": len(detected_strings),
                    "processing_method": "lightweight_ai"
                }
            }
            
        except Exception as e:
            # Fallback response
            fallback_text = ' '.join(detected_strings)
            return {
                "success": True,
                "session_id": session_id,
                "processing_result": {
                    "interpreted_text": fallback_text,
                    "explanation": f"Basic text assembly. AI processing failed: {str(e)}",
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
    
    def handle_chat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat requests"""
        message = data.get('message', '').strip()
        if not message:
            raise Exception("Message cannot be empty")
        
        thread_id = data.get('thread_id') or f"chat_{uuid.uuid4()}"[:16]
        
        try:
            ai_assistant = get_assistant()
            response_text = ai_assistant.chat(message, thread_id)
            
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
            
        except Exception as e:
            return {
                "success": False,
                "thread_id": thread_id,
                "error": str(e),
                "fallback_response": "I encountered an issue. Please try again."
            }
    
    def handle_process_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text processing"""
        text = data.get('text', '').strip()
        if not text:
            raise Exception("Text cannot be empty")
        
        task = data.get('task', 'explain')
        max_length = data.get('max_length', 400)
        
        valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
        if task not in valid_tasks:
            raise Exception(f"Task must be one of: {valid_tasks}")
        
        try:
            ai_assistant = get_assistant()
            response = ai_assistant.process_text(text, task, max_length)
            
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
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_text": text,
                "task": task
            }
    
    def handle_batch_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch processing"""
        texts = data.get('texts', [])
        task = data.get('task', 'explain')
        
        if len(texts) > 3:
            raise Exception("Maximum 3 texts allowed per batch")
        
        valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
        if task not in valid_tasks:
            raise Exception(f"Task must be one of: {valid_tasks}")
        
        try:
            ai_assistant = get_assistant()
        except Exception as e:
            raise Exception(f"Assistant unavailable: {e}")
        
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
                response = ai_assistant.process_text(text, task, 300)
                
                results.append({
                    "index": i,
                    "success": True,
                    "original_text": text[:100] + "..." if len(text) > 100 else text,
                    "processed_text": response,
                    "task": task
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
                "total_texts": len(texts),
                "successful_processing": len(successful_results),
                "task_performed": task
            }
        }

# Vercel handler function
def handler(request, response):
    """Main Vercel handler"""
    try:
        # Create a simple handler that works with Vercel
        method = request.get('httpMethod') or request.get('method', 'GET')
        path = request.get('path') or request.get('rawPath', '/')
        body = request.get('body', '')
        
        if method == 'GET':
            if path in ['/', '/api', '/api/', '/health', '/api/health']:
                if 'health' in path:
                    assistant_ready = assistant is not None
                    result = {
                        "status": "healthy",
                        "assistant_ready": assistant_ready
                    }
                else:
                    result = {
                        "service": "AI Assistant API",
                        "status": "active",
                        "version": "2.0.0"
                    }
                
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(result)
                }
            
            elif path in ['/capabilities', '/api/capabilities']:
                result = {
                    "version": "2.0.0",
                    "capabilities": {
                        "braille_processing": True,
                        "general_chat": True,
                        "text_processing": True,
                        "batch_processing": True
                    },
                    "supported_tasks": ["explain", "summarize", "correct", "enhance", "analyze"],
                    "limits": {
                        "max_batch_size": 3,
                        "max_text_length": 2000
                    }
                }
                
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(result)
                }
        
        elif method == 'POST':
            try:
                data = json.loads(body) if body else {}
            except:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({"error": "Invalid JSON"})
                }
            
            # Handle different endpoints
            if path in ['/process-braille', '/api/process-braille']:
                try:
                    detected_strings = data.get('detected_strings', [])
                    if not detected_strings:
                        raise Exception("No braille strings provided")
                    
                    session_id = data.get('session_id') or str(uuid.uuid4())[:8]
                    
                    try:
                        ai_assistant = get_assistant()
                        result = ai_assistant.process_braille_strings(detected_strings)
                        
                        response_data = {
                            "success": True,
                            "session_id": session_id,
                            "processing_result": {
                                "interpreted_text": result.text,
                                "explanation": result.explanation,
                                "confidence": round(result.confidence, 3),
                                "input_strings": detected_strings[:10],
                                "processing_timestamp": datetime.now().isoformat()
                            },
                            "metadata": {
                                "input_count": len(detected_strings),
                                "processing_method": "lightweight_ai"
                            }
                        }
                        
                    except Exception as e:
                        fallback_text = ' '.join(detected_strings)
                        response_data = {
                            "success": True,
                            "session_id": session_id,
                            "processing_result": {
                                "interpreted_text": fallback_text,
                                "explanation": f"Basic text assembly. AI processing failed: {str(e)}",
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
                    
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps(response_data)
                    }
                    
                except Exception as e:
                    return {
                        'statusCode': 500,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({"error": str(e)})
                    }
        
            # Handle chat endpoint
            elif path in ['/chat', '/api/chat']:
                try:
                    message = data.get('message', '').strip()
                    if not message:
                        raise Exception("Message cannot be empty")
                    
                    thread_id = data.get('thread_id') or f"chat_{uuid.uuid4()}"[:16]
                    
                    try:
                        ai_assistant = get_assistant()
                        response_text = ai_assistant.chat(message, thread_id)
                        
                        response_data = {
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
                        
                    except Exception as e:
                        response_data = {
                            "success": False,
                            "thread_id": thread_id,
                            "error": str(e),
                            "fallback_response": "I encountered an issue. Please try again."
                        }
                    
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps(response_data)
                    }
                    
                except Exception as e:
                    return {
                        'statusCode': 500,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({"error": str(e)})
                    }
            
            # Handle process-text endpoint
            elif path in ['/process-text', '/api/process-text']:
                try:
                    text = data.get('text', '').strip()
                    if not text:
                        raise Exception("Text cannot be empty")
                    
                    task = data.get('task', 'explain')
                    max_length = data.get('max_length', 400)
                    
                    valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
                    if task not in valid_tasks:
                        raise Exception(f"Task must be one of: {valid_tasks}")
                    
                    try:
                        ai_assistant = get_assistant()
                        response = ai_assistant.process_text(text, task, max_length)
                        
                        response_data = {
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
                        
                    except Exception as e:
                        response_data = {
                            "success": False,
                            "error": str(e),
                            "original_text": text,
                            "task": task
                        }
                    
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps(response_data)
                    }
                    
                except Exception as e:
                    return {
                        'statusCode': 500,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({"error": str(e)})
                    }
            
            # Handle batch-process endpoint
            elif path in ['/batch-process', '/api/batch-process']:
                try:
                    texts = data.get('texts', [])
                    task = data.get('task', 'explain')
                    
                    if len(texts) > 3:
                        raise Exception("Maximum 3 texts allowed per batch")
                    
                    valid_tasks = ["explain", "summarize", "correct", "enhance", "analyze"]
                    if task not in valid_tasks:
                        raise Exception(f"Task must be one of: {valid_tasks}")
                    
                    try:
                        ai_assistant = get_assistant()
                    except Exception as e:
                        raise Exception(f"Assistant unavailable: {e}")
                    
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
                            response = ai_assistant.process_text(text, task, 300)
                            
                            results.append({
                                "index": i,
                                "success": True,
                                "original_text": text[:100] + "..." if len(text) > 100 else text,
                                "processed_text": response,
                                "task": task
                            })
                            
                        except Exception as e:
                            results.append({
                                "index": i,
                                "success": False,
                                "error": str(e)
                            })
                    
                    successful_results = [r for r in results if r.get('success', False)]
                    
                    response_data = {
                        "success": True,
                        "batch_results": results,
                        "summary": {
                            "total_texts": len(texts),
                            "successful_processing": len(successful_results),
                            "task_performed": task
                        }
                    }
                    
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps(response_data)
                    }
                    
                except Exception as e:
                    return {
                        'statusCode': 500,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({"error": str(e)})
                    }
        
        # Default 404
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": "Endpoint not found"})
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Handler error: {str(e)}"})
        }