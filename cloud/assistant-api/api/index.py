# api/index.py - Native Vercel handler
import json
import os
import sys
from datetime import datetime
import uuid

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

def handler(request):
    """Main Vercel handler"""
    try:
        # Extract method, path, and body from request
        method = request.method
        path = request.path
        body = request.body
        
        # Decode body if needed
        if isinstance(body, bytes):
            try:
                body = body.decode('utf-8')
            except:
                body = ""

        # Parse JSON body
        data = {}
        if body and body.strip():
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({"error": "Invalid JSON"})
                }

        # Route requests
        if method == 'GET':
            if path in ['/', '/api', '/api/']:
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        "service": "AI Assistant API",
                        "status": "active",
                        "version": "2.0.0"
                    })
                }
            
            elif path in ['/health', '/api/health']:
                assistant_ready = assistant is not None
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        "status": "healthy",
                        "assistant_ready": assistant_ready
                    })
                }
            
            elif path in ['/capabilities', '/api/capabilities']:
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
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
                    })
                }
            else:
                return {
                    'statusCode': 404,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({"error": "Endpoint not found"})
                }
                
        elif method == 'POST':
            # Handle different endpoints
            if path in ['/process-braille', '/api/process-braille']:
                detected_strings = data.get('detected_strings', [])
                if not detected_strings:
                    return {
                        'statusCode': 400,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({"error": "No braille strings provided"})
                    }
                
                session_id = data.get('session_id') or str(uuid.uuid4())[:8]
                
                try:
                    ai_assistant = get_assistant()
                    result = ai_assistant.process_braille_strings(detected_strings)
                    
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps({
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
                        })
                    }
                    
                except Exception as e:
                    fallback_text = ' '.join(detected_strings)
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps({
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
                        })
                    }
            
            # Similar handling for other POST endpoints (/chat, /process-text, /batch-process)
            # ... (keep your existing implementation for these endpoints)
            
            else:
                return {
                    'statusCode': 404,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({"error": "Endpoint not found"})
                }
                
        elif method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': ''
            }
            
        else:
            return {
                'statusCode': 405,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({"error": "Method not allowed"})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Handler error: {str(e)}"})
        }