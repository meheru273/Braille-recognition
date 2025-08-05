# api/index.py - Clean API Layer using Controller Pattern
"""
API layer that handles HTTP requests and delegates to the controller
Clean separation following MVC pattern
"""

import json
import os
import base64
from http.server import BaseHTTPRequestHandler
import traceback
import logging
from datetime import datetime
from typing import Dict, Any

# Import our controller
from connector import BrailleController, create_braille_controller

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global controller instance (initialized on first request)
_controller_instance = None

def get_controller() -> BrailleController:
    """Get or create controller instance"""
    global _controller_instance
    
    if _controller_instance is None:
        logger.info("🏭 Creating new BrailleController instance")
        # Configure controller with environment variables
        assistant_config = {
            'api_key': os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        }
        _controller_instance = create_braille_controller(
            detector_config={},
            assistant_config=assistant_config
        )
        logger.info("✅ Controller instance created")
    
    return _controller_instance

class handler(BaseHTTPRequestHandler):
    """
    Clean API handler that delegates to controller
    Handles HTTP concerns only (routing, parsing, response formatting)
    """
    
    def __init__(self, *args, **kwargs):
        logger.info("🌐 Initializing API handler")
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        logger.info(f"📥 GET {self.path}")
        
        try:
            if self.path == '/' or self.path == '/index.html':
                self._serve_web_interface()
            elif self.path == '/health' or self.path == '/api/health':
                self._handle_health_check()
            elif self.path == '/api/status':
                self._handle_detailed_status()
            else:
                self._send_json_response({'error': 'Endpoint not found'}, 404)
                
        except Exception as e:
            logger.error(f"GET error: {str(e)}")
            self._send_json_response({'error': f'Server error: {str(e)}'}, 500)
    
    def do_POST(self):
        """Handle POST requests"""
        logger.info(f"📤 POST {self.path}")
        
        try:
            # Parse request body
            request_data = self._parse_request_body()
            if request_data is None:
                return  # Error already sent
            
            # Route to appropriate handler
            if self.path == '/api/detect':
                self._handle_detect_only(request_data)
            elif self.path == '/api/process':
                self._handle_process_only(request_data)
            elif self.path == '/api/detect-and-process':
                self._handle_detect_and_process(request_data)
            elif self.path == '/api/chat':
                self._handle_chat(request_data)
            elif self.path == '/api/process-text':
                self._handle_text_processing(request_data)
            else:
                self._send_json_response({'error': 'Endpoint not found'}, 404)
                
        except Exception as e:
            logger.error(f"POST error: {str(e)}")
            logger.error(traceback.format_exc())
            self._send_json_response({'error': f'Server error: {str(e)}'}, 500)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        logger.info(f"🔄 OPTIONS {self.path}")
        self._send_cors_headers()
        self.end_headers()
    
    # ==========================================
    # REQUEST HANDLERS
    # ==========================================
    
    def _handle_health_check(self):
        """Simple health check"""
        try:
            controller = get_controller()
            status = controller.get_system_status()
            
            health_response = {
                'status': 'healthy' if status['overall_health'] else 'degraded',
                'timestamp': datetime.now().isoformat(),
                'detector_ok': status['detector']['api_configured'],
                'assistant_ok': status['assistant']['api_available'],
                'version': '2.0.0'
            }
            
            self._send_json_response(health_response)
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            self._send_json_response({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500)
    
    def _handle_detailed_status(self):
        """Detailed system status"""
        try:
            controller = get_controller()
            status = controller.get_system_status()
            self._send_json_response(status)
            
        except Exception as e:
            logger.error(f"Status error: {str(e)}")
            self._send_json_response({'error': str(e)}, 500)
    
    def _handle_detect_only(self, data: Dict[str, Any]):
        """Handle detection-only requests"""
        logger.info("🔍 Processing detection-only request")
        
        # Validate and extract image
        image_bytes = self._extract_image_bytes(data)
        if image_bytes is None:
            return  # Error already sent
        
        try:
            controller = get_controller()
            result = controller.detect_only(image_bytes)
            
            # Format response
            response = {
                'success': result.success,
                'detection': {
                    'predictions': result.predictions,
                    'text_rows': result.text_rows,
                    'detection_count': result.detection_count
                }
            }
            
            if result.error:
                response['detection']['error'] = result.error
            
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Detection handler error: {str(e)}")
            self._send_json_response({'error': f'Detection failed: {str(e)}'}, 500)
    
    def _handle_process_only(self, data: Dict[str, Any]):
        """Handle processing-only requests"""
        logger.info("🤖 Processing process-only request")
        
        text_rows = data.get('text_rows', [])
        if not isinstance(text_rows, list):
            self._send_json_response({'error': 'text_rows must be a list'}, 400)
            return
        
        try:
            controller = get_controller()
            result = controller.process_only(text_rows)
            
            response = {
                'success': result.success,
                'processing': {
                    'text': result.text,
                    'explanation': result.explanation,
                    'confidence': result.confidence
                }
            }
            
            if result.error:
                response['processing']['error'] = result.error
            
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Processing handler error: {str(e)}")
            self._send_json_response({'error': f'Processing failed: {str(e)}'}, 500)
    
    def _handle_detect_and_process(self, data: Dict[str, Any]):
        """Handle complete detection + processing requests"""
        logger.info("🚀 Processing complete detect-and-process request")
        
        # Validate and extract image
        image_bytes = self._extract_image_bytes(data)
        if image_bytes is None:
            return  # Error already sent
        
        try:
            controller = get_controller()
            result = controller.detect_and_process(image_bytes)
            
            # Format response
            response = {
                'success': result.detection.success and result.processing.success,
                'detection': {
                    'success': result.detection.success,
                    'predictions': result.detection.predictions,
                    'text_rows': result.detection.text_rows,
                    'detection_count': result.detection.detection_count
                },
                'processing': {
                    'success': result.processing.success,
                    'text': result.processing.text,
                    'explanation': result.processing.explanation,
                    'confidence': result.processing.confidence
                },
                'metadata': {
                    'timestamp': result.timestamp.isoformat(),
                    'total_time_ms': result.total_time_ms
                }
            }
            
            # Add errors if any
            if result.detection.error:
                response['detection']['error'] = result.detection.error
            if result.processing.error:
                response['processing']['error'] = result.processing.error
            
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Complete processing handler error: {str(e)}")
            self._send_json_response({'error': f'Complete processing failed: {str(e)}'}, 500)
    
    def _handle_chat(self, data: Dict[str, Any]):
        """Handle chat requests"""
        logger.info("💬 Processing chat request")
        
        message = data.get('message', '').strip()
        if not message:
            self._send_json_response({'error': 'Message is required'}, 400)
            return
        
        context = data.get('context', {})
        thread_id = data.get('thread_id', 'default')
        
        try:
            controller = get_controller()
            response_text = controller.chat_with_context(message, context, thread_id)
            
            self._send_json_response({
                'success': True,
                'response': response_text,
                'thread_id': thread_id
            })
            
        except Exception as e:
            logger.error(f"Chat handler error: {str(e)}")
            self._send_json_response({'error': f'Chat failed: {str(e)}'}, 500)
    
    def _handle_text_processing(self, data: Dict[str, Any]):
        """Handle arbitrary text processing requests"""
        logger.info("📝 Processing text processing request")
        
        text = data.get('text', '').strip()
        task = data.get('task', 'explain')
        
        if not text:
            self._send_json_response({'error': 'Text is required'}, 400)
            return
        
        try:
            controller = get_controller()
            result = controller.process_text_only(text, task)
            
            self._send_json_response({
                'success': True,
                'result': result,
                'task': task
            })
            
        except Exception as e:
            logger.error(f"Text processing handler error: {str(e)}")
            self._send_json_response({'error': f'Text processing failed: {str(e)}'}, 500)
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def _parse_request_body(self) -> Dict[str, Any]:
        """Parse and validate request body"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json_response({'error': 'Empty request body'}, 400)
                return None
            
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            logger.info(f"Request data keys: {list(data.keys())}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {str(e)}")
            self._send_json_response({'error': 'Invalid JSON in request body'}, 400)
            return None
        except Exception as e:
            logger.error(f"Request parsing error: {str(e)}")
            self._send_json_response({'error': f'Request parsing failed: {str(e)}'}, 400)
            return None
    
    def _extract_image_bytes(self, data: Dict[str, Any]) -> bytes:
        """Extract and validate image bytes from request data"""
        image_data = data.get('image')
        if not image_data:
            self._send_json_response({'error': 'Image data is required'}, 400)
            return None
        
        try:
            # Handle data URL format
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            logger.info(f"Image decoded: {len(image_bytes)} bytes")
            return image_bytes
            
        except Exception as e:
            logger.error(f"Image decoding error: {str(e)}")
            self._send_json_response({'error': f'Invalid image data: {str(e)}'}, 400)
            return None
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response with proper headers"""
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2, default=str)
        self.wfile.write(response_json.encode())
        
        logger.info(f"Response sent: {status_code} ({len(response_json)} bytes)")
    
    def _send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def _serve_web_interface(self):
        """Serve the web interface"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Braille Detection System v2.0</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; margin: 20px 0; border: 1px solid rgba(255,255,255,0.2); }
        .btn { background: linear-gradient(45deg, #ff6b6b, #ee5522); color: white; padding: 12px 24px; border: none; border-radius: 25px; cursor: pointer; margin: 8px; font-size: 16px; transition: all 0.3s ease; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-secondary { background: linear-gradient(45deg, #4ecdc4, #44a08d); }
        .btn-info { background: linear-gradient(45deg, #667eea, #764ba2); }
        .result-card { background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.3); border-radius: 10px; padding: 20px; margin: 15px 0; }
        .success { border-left: 4px solid #4ecdc4; }
        .error { border-left: 4px solid #ff6b6b; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-ok { background: #4ecdc4; }
        .status-error { background: #ff6b6b; }
        .status-warning { background: #feca57; }
        input[type="file"] { background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.3); padding: 10px; border-radius: 8px; width: 100%; }
        .image-preview { max-width: 300px; max-height: 200px; border-radius: 8px; margin: 10px 0; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-item { text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; }
        .stat-number { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Braille Detection System</h1>
            <p>Advanced MVC Architecture • Version