# api/index.py - Merged Braille Detection + Assistant API
import json
import os
import base64
import requests
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
import traceback
import io
import tempfile

# ============================================================================
# BRAILLE ASSISTANT CLASSES
# ============================================================================

@dataclass
class BrailleResult:
    """Result from braille processing"""
    text: str
    explanation: str
    confidence: float

class LightweightLLM:
    """Enhanced LLM client with better error handling and fallbacks"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Configure based on API key type
        if self.api_key and self.api_key.startswith("gsk_"):  # Groq
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = "llama-3.1-8b-instant"
            self.provider = "groq"
        elif self.api_key:  # OpenAI
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-3.5-turbo"
            self.provider = "openai"
        else:
            self.provider = "fallback"
    
    def generate_response(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Generate response with fallback for no API key"""
        
        # Fallback mode if no API key
        if not self.api_key:
            return self._fallback_response(messages)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=25
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                return self._fallback_response(messages)
                
        except Exception:
            return self._fallback_response(messages)
    
    def _fallback_response(self, messages: List[Dict]) -> str:
        """Provide intelligent fallback responses without API"""
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "").lower()
                break
        
        # Simple pattern matching for common queries
        if "hello" in user_message or "hi" in user_message:
            return "Hello! I'm your Braille Recognition Assistant. I can help you process braille text and answer questions. How can I assist you today?"
        
        elif "help" in user_message:
            return "I can help you with:\n1. Processing braille text into readable format\n2. Detecting braille from images\n3. Explaining topics and concepts\n4. General conversation\n\nWhat would you like to do?"
        
        elif "braille" in user_message:
            return "I can process braille characters and convert them to readable text. You can also upload images for braille detection!"
        
        elif any(word in user_message for word in ["what", "explain", "tell me"]):
            return "I'd be happy to help explain something. Could you be more specific about what you'd like to know?"
        
        elif "thank" in user_message:
            return "You're welcome! Is there anything else I can help you with?"
        
        else:
            return f"I understand you're asking about something. I'm currently in limited mode, but I can still help with braille processing and basic questions."

class BrailleAssistant:
    """Enhanced Braille Assistant with better error handling"""
    
    def __init__(self, api_key: str = None):
        self.llm = LightweightLLM(api_key)
        self.conversation_memory = {}
    
    def process_braille_strings(self, detected_strings: List[str]) -> BrailleResult:
        """Process braille detection results with fallback"""
        
        if not detected_strings:
            return BrailleResult(
                text="",
                explanation="No braille characters detected.",
                confidence=0.0
            )
        
        try:
            raw_text = " ".join(detected_strings).strip()
            
            if not self.llm.api_key:
                processed_text = self._fallback_braille_processing(raw_text)
                explanation = f"Processed braille text: {processed_text}. (Using basic processing)"
                confidence = 0.6
            else:
                process_prompt = [
                    {
                        "role": "system", 
                        "content": "You are a braille text interpreter. Convert detected braille characters into meaningful text."
                    },
                    {
                        "role": "user", 
                        "content": f"Braille characters detected: '{raw_text}'\n\nConvert to readable text:"
                    }
                ]
                
                processed_text = self.llm.generate_response(process_prompt, max_tokens=200)
                
                if not processed_text or len(processed_text.strip()) < 2:
                    processed_text = raw_text
                
                explanation = self._generate_explanation(processed_text)
                confidence = min(0.9, len([s for s in detected_strings if s.strip()]) / max(1, len(detected_strings)))
            
            return BrailleResult(
                text=processed_text,
                explanation=explanation,
                confidence=confidence
            )
            
        except Exception as e:
            fallback_text = " ".join(detected_strings)
            return BrailleResult(
                text=fallback_text,
                explanation=f"Basic text assembly: {fallback_text}",
                confidence=0.3
            )
    
    def _fallback_braille_processing(self, text: str) -> str:
        """Basic braille processing without API"""
        cleaned = text.strip()
        
        if len(cleaned) > 10:
            words = []
            current_word = ""
            
            for char in cleaned:
                if char.isspace() or char in ".,!?":
                    if current_word:
                        words.append(current_word)
                        current_word = ""
                    if char in ".,!?":
                        words.append(char)
                else:
                    current_word += char
            
            if current_word:
                words.append(current_word)
            
            return " ".join(words)
        
        return cleaned
    
    def _generate_explanation(self, text: str) -> str:
        """Generate explanation with fallback"""
        try:
            if not self.llm.api_key:
                return f"This appears to be braille text that reads: '{text}'. For detailed explanations, please configure an API key."
            
            explain_prompt = [
                {
                    "role": "system",
                    "content": "Provide brief, helpful explanations about topics."
                },
                {
                    "role": "user",
                    "content": f'Explain this topic in 2-3 sentences: "{text}"'
                }
            ]
            
            explanation = self.llm.generate_response(explain_prompt, max_tokens=150)
            return explanation or f"This text discusses: {text}"
            
        except Exception as e:
            return f"This appears to be about: {text}"
    
    def chat(self, user_message: str, thread_id: str = "default") -> str:
        """Enhanced chat with better fallback handling"""
        
        if not user_message.strip():
            return "Please provide a message."
        
        try:
            history = self.conversation_memory.get(thread_id, [])
            
            if not history:
                system_msg = "You are a helpful AI assistant specializing in braille recognition and general assistance. Provide clear, concise, and helpful responses."
                if not self.llm.api_key:
                    system_msg += " You are currently operating in fallback mode with limited capabilities."
                
                history = [{"role": "system", "content": system_msg}]
            
            history.append({"role": "user", "content": user_message})
            
            if len(history) > 7:
                history = [history[0]] + history[-6:]
            
            response = self.llm.generate_response(history, max_tokens=300)
            
            history.append({"role": "assistant", "content": response})
            self.conversation_memory[thread_id] = history
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error. Please try rephrasing your question."

# ============================================================================
# BRAILLE DETECTOR CLASSES
# ============================================================================

class BrailleDetector:
    """Lightweight Braille Detection using direct HTTP requests"""
    
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            print("Warning: ROBOFLOW_API_KEY not found - detection will be disabled")
            
        self.workspace_name = "braille-to-text-0xo2p"
        self.workflow_id = "custom-workflow"
        self.base_url = "https://detect.roboflow.com"
    
    def _encode_image_from_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        try:
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')
            return encoded_string
        except Exception as e:
            raise Exception(f"Failed to encode image: {e}")
    
    def detect_braille_from_bytes(self, image_bytes: bytes) -> Optional[Dict]:
           
        if not self.api_key:
            error_msg = "ROBOFLOW_API_KEY not configured"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}

        try:
            encoded_image = self._encode_image_from_bytes(image_bytes)
            print(f"DEBUG: Image encoded. Length: {len(encoded_image)} chars")

            # --- Potentially try different base URLs ---
            # Primary: detect.roboflow.com (Standard Serverless Inference)
            # Fallback: serverless.roboflow.com (Sometimes used, though less common for direct workflow calls now)
            urls_to_try = [
                f"{self.base_url.rstrip('/')}/{self.workspace_name}/{self.workflow_id}".rstrip('/'),
                f"https://serverless.roboflow.com/{self.workspace_name}/workflows/{self.workflow_id}" # Fallback structure
            ]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            print(f"DEBUG: Headers: {headers}")

            payload = {
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": encoded_image
                    }
                }
            }
            print(f"DEBUG: Payload prepared.")

            params = {"api_key": self.api_key}
            print(f"DEBUG: Params: {params}")

            last_error_response_text = ""
            last_status_code = None

            # Try different potential URLs
            for i, url in enumerate(urls_to_try):
                print(f"ATTEMPT {i+1}: Trying API URL: {url}")
                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        params=params,
                        timeout=30
                    )
                    print(f"ATTEMPT {i+1}: Response Status Code: {response.status_code}")

                    if response.status_code == 200:
                        result_data = response.json()
                        print(f"ATTEMPT {i+1}: SUCCESS! API request successful.")
                        return result_data
                    else:
                        # Capture error details for this attempt
                        last_status_code = response.status_code
                        last_error_response_text = response.text
                        print(f"ATTEMPT {i+1}: FAILED. Response Text: {last_error_response_text}")
                        try:
                            error_json = response.json()
                            print(f"ATTEMPT {i+1}: Parsed Error JSON: {error_json}")
                        except:
                            print(f"ATTEMPT {i+1}: Response body is not valid JSON.")
                        # Don't return yet, try the next URL if available

                except requests.exceptions.RequestException as req_err:
                    print(f"ATTEMPT {i+1}: Request Exception: {req_err}")
                    last_error_response_text = str(req_err)
                    last_status_code = "Request Error"

            # If we get here, all attempts failed
            error_msg = f"All API attempts failed. Last error (Status {last_status_code}): {last_error_response_text}"
            print(f"ERROR: {error_msg}")
            return {
                "error": "Detection failed",
                "detail": error_msg,
                "status_code": last_status_code
            }

        except Exception as e:
            error_msg = f"Unexpected error during detection setup: {str(e)}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            return {"error": error_msg}

        
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions with robust error handling"""
        if not result or "error" in result:
            return []
            
        try:
            predictions = []
            
            # Handle different response structures
            if isinstance(result, list) and len(result) > 0:
                if "predictions" in result[0]:
                    pred_data = result[0]["predictions"]
                    if isinstance(pred_data, dict) and "predictions" in pred_data:
                        predictions = pred_data["predictions"]
                    elif isinstance(pred_data, list):
                        predictions = pred_data
            
            elif isinstance(result, dict):
                if "predictions" in result:
                    pred_data = result["predictions"]
                    if isinstance(pred_data, dict) and "predictions" in pred_data:
                        predictions = pred_data["predictions"]
                    elif isinstance(pred_data, list):
                        predictions = pred_data
                elif "outputs" in result:
                    outputs = result["outputs"]
                    if isinstance(outputs, list) and len(outputs) > 0:
                        if "predictions" in outputs[0]:
                            predictions = outputs[0]["predictions"]
            
            # Validate prediction structure
            valid_predictions = []
            for pred in predictions:
                if isinstance(pred, dict) and all(key in pred for key in ['x', 'y', 'width', 'height', 'confidence', 'class']):
                    try:
                        pred['x'] = float(pred['x'])
                        pred['y'] = float(pred['y'])
                        pred['width'] = float(pred['width'])
                        pred['height'] = float(pred['height'])
                        pred['confidence'] = float(pred['confidence'])
                        valid_predictions.append(pred)
                    except (ValueError, TypeError):
                        continue
            
            return valid_predictions
            
        except Exception as e:
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.4) -> List[str]:
        """Organize detected characters into rows"""
        if not predictions:
            return []
        
        try:
            # Filter by confidence
            filtered_predictions = [
                pred for pred in predictions 
                if pred.get('confidence', 0) >= min_confidence
            ]
            
            if not filtered_predictions:
                return []
            
            # Sort by Y coordinate
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p.get('y', 0))
            
            rows = []
            current_group = [sorted_by_y[0]]
            
            # Group predictions into rows
            for i in range(1, len(sorted_by_y)):
                current_pred = sorted_by_y[i]
                prev_pred = sorted_by_y[i-1]
                
                # Calculate dynamic threshold for row grouping
                avg_height = (current_pred.get('height', 30) + prev_pred.get('height', 30)) / 2
                threshold = max(15, avg_height * 0.5)
                
                y_diff = abs(current_pred.get('y', 0) - prev_pred.get('y', 0))
                
                if y_diff <= threshold:
                    current_group.append(current_pred)
                else:
                    # Process current group
                    if current_group:
                        current_group.sort(key=lambda p: p.get('x', 0))
                        row_text = ''.join([p.get('class', '') for p in current_group])
                        if row_text.strip():
                            rows.append(row_text)
                    current_group = [current_pred]
            
            # Process final group
            if current_group:
                current_group.sort(key=lambda p: p.get('x', 0))
                row_text = ''.join([p.get('class', '') for p in current_group])
                if row_text.strip():
                    rows.append(row_text)
            
            return rows
            
        except Exception as e:
            return []

# ============================================================================
# MERGED API HANDLER
# ============================================================================

class handler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Initialize components
        self.assistant = BrailleAssistant()
        self.detector = BrailleDetector()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            if path == '/' or path == '/index.html':
                self.serve_html()
            elif path == '/health':
                self.send_json_response({
                    'status': 'healthy',
                    'features': ['braille_detection', 'ai_assistant', 'chat'],
                    'roboflow_configured': bool(self.detector.api_key),
                    'ai_configured': bool(self.assistant.llm.api_key)
                })
            elif path.startswith('/favicon'):
                self.send_response(404)
                self.end_headers()
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Not found'}).encode())
                
        except Exception as e:
            self.send_error_response(f"GET error: {str(e)}")
    
    def do_POST(self):
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self.send_error_response("Invalid JSON", 400)
                return
            
            # Route to appropriate handler
            if path == '/api/chat':
                self.handle_chat(data)
            elif path == '/api/process-braille':
                self.handle_braille_processing(data)
            elif path == '/api/detect-braille':
                self.handle_braille_detection(data)
            elif path == '/api/detect-and-process':
                self.handle_detect_and_process(data)
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())
                
        except Exception as e:
            self.send_error_response(f"POST error: {str(e)}")
    
    def handle_chat(self, data):
        """Handle chat requests"""
        try:
            message = data.get('message', '').strip()
            if not message:
                self.send_error_response('Message is required', 400)
                return
            
            response = self.assistant.chat(message)
            self.send_json_response({'response': response})
            
        except Exception as e:
            self.send_error_response(f'Chat processing error: {str(e)}')
    
    def handle_braille_processing(self, data):
        """Handle braille text processing"""
        try:
            braille_strings = data.get('braille_strings', [])
            if not braille_strings:
                self.send_error_response('Braille strings are required', 400)
                return
            
            result = self.assistant.process_braille_strings(braille_strings)
            
            self.send_json_response({
                'text': result.text,
                'explanation': result.explanation,
                'confidence': result.confidence
            })
            
        except Exception as e:
            self.send_error_response(f'Braille processing error: {str(e)}')
    
    def handle_braille_detection(self, data):
        """Handle braille detection from image"""
        try:
            image_data = data.get('image')
            if not image_data:
                self.send_error_response('Image data is required', 400)
                return
            
            # Decode base64 image
            try:
                if image_data.startswith('data:image'):
                    # Remove data URL prefix
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                self.send_error_response(f'Invalid image data: {str(e)}', 400)
                return
            
            # Run detection
            detection_result = self.detector.detect_braille_from_bytes(image_bytes)
            
            if "error" in detection_result:
                self.send_error_response(detection_result["error"])
                return
            
            # Extract predictions
            predictions = self.detector.extract_predictions(detection_result)
            text_rows = self.detector.organize_text_by_rows(predictions)
            
            self.send_json_response({
                'predictions': predictions,
                'text_rows': text_rows,
                'detection_count': len(predictions)
            })
            
        except Exception as e:
            self.send_error_response(f'Braille detection error: {str(e)}')
    
    def handle_detect_and_process(self, data):
        """Handle end-to-end braille detection and processing"""
        try:
            image_data = data.get('image')
            if not image_data:
                self.send_error_response('Image data is required', 400)
                return
            
            # Decode base64 image
            try:
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                self.send_error_response(f'Invalid image data: {str(e)}', 400)
                return
            
            # Run detection
            detection_result = self.detector.detect_braille_from_bytes(image_bytes)
            
            if "error" in detection_result:
                self.send_error_response(detection_result["error"])
                return
            
            # Extract and organize text
            predictions = self.detector.extract_predictions(detection_result)
            text_rows = self.detector.organize_text_by_rows(predictions)
            
            # Process with assistant
            if text_rows:
                processing_result = self.assistant.process_braille_strings(text_rows)
                
                self.send_json_response({
                    'detection': {
                        'predictions': predictions,
                        'text_rows': text_rows,
                        'detection_count': len(predictions)
                    },
                    'processing': {
                        'text': processing_result.text,
                        'explanation': processing_result.explanation,
                        'confidence': processing_result.confidence
                    }
                })
            else:
                self.send_json_response({
                    'detection': {
                        'predictions': [],
                        'text_rows': [],
                        'detection_count': 0
                    },
                    'processing': {
                        'text': '',
                        'explanation': 'No braille characters detected in the image.',
                        'confidence': 0.0
                    }
                })
            
        except Exception as e:
            self.send_error_response(f'Detection and processing error: {str(e)}')
    
    def serve_html(self):
        """Serve the enhanced web interface"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complete Braille Recognition System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container { 
            max-width: 1000px; margin: 0 auto; background: white; 
            border-radius: 15px; padding: 30px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header { text-align: center; margin-bottom: 30px; color: #333; }
        .header h1 { 
            font-size: 2.5em; margin-bottom: 10px; 
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .input-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #555; }
        input, textarea, select { 
            width: 100%; padding: 12px; border: 2px solid #e0e0e0; 
            border-radius: 8px; font-size: 16px; transition: border-color 0.3s;
        }
        input:focus, textarea:focus, select:focus { 
            outline: none; border-color: #667eea; 
        }
        .btn { 
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: white; padding: 12px 24px; border: none; 
            border-radius: 8px; cursor: pointer; font-size: 16px; 
            font-weight: 600; transition: transform 0.2s; margin: 5px;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .result { 
            margin-top: 20px; padding: 20px; background: #f8f9fa; 
            border-radius: 8px; border-left: 4px solid #667eea;
        }
        .chat-container { 
            max-height: 400px; overflow-y: auto; border: 2px solid #e0e0e0; 
            border-radius: 8px; padding: 15px; margin-bottom: 15px; background: #fafafa;
        }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 8px; }
        .user-message { background: #667eea; color: white; margin-left: 20px; }
        .assistant-message { background: white; border: 1px solid #e0e0e0; margin-right: 20px; }
        .loading { display: none; text-align: center; color: #667eea; font-weight: 600; }
        .tabs { display: flex; margin-bottom: 20px; border-bottom: 2px solid #e0e0e0; }
        .tab { 
            padding: 12px 24px; cursor: pointer; border: none; 
            background: none; font-size: 16px; color: #666; transition: all 0.3s;
        }
        .tab.active { color: #667eea; border-bottom: 2px solid #667eea; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .upload-area {
            border: 2px dashed #667eea; border-radius: 8px; padding: 40px;
            text-align: center; cursor: pointer; transition: all 0.3s;
        }
        .upload-area:hover { background: #f0f7ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #4a90e2; }
        .image-preview { max-width: 300px; max-height: 200px; margin: 10px auto; display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔤 Complete Braille Recognition System</h1>
            <p>AI-powered braille detection, processing, and intelligent assistance</p>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('detection')">Braille Detection</button>
            <button class="tab" onclick="switchTab('processing')">Text Processing</button>
            <button class="tab" onclick="switchTab('chat')">AI Chat</button>
        </div>

        <div id="detection-tab" class="tab-content active">
            <div class="upload-area" onclick="document.getElementById('imageInput').click()" 
                 ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                <p>📸 Click to upload or drag & drop braille image</p>
                <input type="file" id="imageInput" accept="image/*" style="display: none;" onchange="handleImageUpload(event)">
            </div>
            <img id="imagePreview" class="image-preview" style="display: none;">
            <button class="btn" onclick="detectBraille()" id="detectBtn" disabled>🔍 Detect Braille</button>
            <button class="btn" onclick="detectAndProcess()" id="processBtn" disabled>🚀 Detect & Process</button>
            <div class="loading" id="detection-loading">Processing image...</div>
            <div id="detection-result" class="result" style="display: none;">
                <h3>Detection Results:</h3>
                <div id="detection-output"></div>
            </div>
        </div>

        <div id="processing-tab" class="tab-content">
            <div class="input-group">
                <label for="braille-input">Detected Braille Text (comma-separated):</label>
                <textarea id="braille-input" rows="4" placeholder="Enter detected braille characters, e.g: hello, world"></textarea>
            </div>
            <button class="btn" onclick="processBraille()">🔍 Process Braille</button>
            <div class="loading" id="braille-loading">Processing braille text...</div>
            <div id="braille-result" class="result" style="display: none;">
                <h3>Processing Results:</h3>
                <div id="braille-output"></div>
            </div>
        </div>

        <div id="chat-tab" class="tab-content">
            <div id="chat-messages" class="chat-container">
                <div class="message assistant-message">
                    <strong>Assistant:</strong> Hello! I'm your AI assistant. I can help with braille processing and answer questions.
                </div>
            </div>
            <div class="input-group">
                <input type="text" id="chat-input" placeholder="Type your message..." onkeypress="handleChatKeyPress(event)">
            </div>
            <button class="btn" onclick="sendMessage()">💬 Send Message</button>
            <button class="btn" onclick="clearChat()" style="background: #dc3545;">🗑️ Clear Chat</button>
            <div class="loading" id="chat-loading">Thinking...</div>
        </div>
    </div>

    <script>
        let currentImage = null;

        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }

        function handleDragOver(event) {
            event.preventDefault();
            event.currentTarget.classList.add('dragover');
        }

        function handleDragLeave(event) {
            event.currentTarget.classList.remove('dragover');
        }

        function handleDrop(event) {
            event.preventDefault();
            event.currentTarget.classList.remove('dragover');
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                handleImageFile(files[0]);
            }
        }

        function handleImageUpload(event) {
            const file = event.target.files[0];
            if (file) {
                handleImageFile(file);
            }
        }

        function handleImageFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file.');
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                currentImage = e.target.result;
                const preview = document.getElementById('imagePreview');
                preview.src = currentImage;
                preview.style.display = 'block';
                
                document.getElementById('detectBtn').disabled = false;
                document.getElementById('processBtn').disabled = false;
            };
            reader.readAsDataURL(file);
        }

        async function detectBraille() {
            if (!currentImage) {
                alert('Please upload an image first.');
                return;
            }

            const resultDiv = document.getElementById('detection-result');
            const outputDiv = document.getElementById('detection-output');
            const loading = document.getElementById('detection-loading');

            loading.style.display = 'block';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/api/detect-braille', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: currentImage })
                });

                const data = await response.json();

                if (response.ok) {
                    outputDiv.innerHTML = `
                        <div style="margin-bottom: 15px;">
                            <strong>Detections Found:</strong> ${data.detection_count}
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong>Text Rows:</strong>
                            <ul style="margin-left: 20px;">
                                ${data.text_rows.map(row => `<li>${row}</li>`).join('')}
                            </ul>
                        </div>
                        <div>
                            <strong>Raw Predictions:</strong> ${data.predictions.length} characters detected
                        </div>
                    `;
                    resultDiv.style.display = 'block';
                } else {
                    outputDiv.innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                outputDiv.innerHTML = `<div style="color: red;">Network error: ${error.message}</div>`;
                resultDiv.style.display = 'block';
            }

            loading.style.display = 'none';
        }

        async function detectAndProcess() {
            if (!currentImage) {
                alert('Please upload an image first.');
                return;
            }

            const resultDiv = document.getElementById('detection-result');
            const outputDiv = document.getElementById('detection-output');
            const loading = document.getElementById('detection-loading');

            loading.style.display = 'block';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/api/detect-and-process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: currentImage })
                });

                const data = await response.json();

                if (response.ok) {
                    outputDiv.innerHTML = `
                        <div style="margin-bottom: 20px; padding: 15px; background: #e3f2fd; border-radius: 5px;">
                            <h4>🎯 Detection Results</h4>
                            <p><strong>Characters Found:</strong> ${data.detection.detection_count}</p>
                            <p><strong>Text Rows:</strong> ${data.detection.text_rows.join(', ')}</p>
                        </div>
                        <div style="padding: 15px; background: #f3e5f5; border-radius: 5px;">
                            <h4>🤖 AI Processing Results</h4>
                            <p><strong>Processed Text:</strong> ${data.processing.text}</p>
                            <p><strong>Explanation:</strong> ${data.processing.explanation}</p>
                            <p><strong>Confidence:</strong> ${(data.processing.confidence * 100).toFixed(1)}%</p>
                        </div>
                    `;
                    resultDiv.style.display = 'block';
                } else {
                    outputDiv.innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                outputDiv.innerHTML = `<div style="color: red;">Network error: ${error.message}</div>`;
                resultDiv.style.display = 'block';
            }

            loading.style.display = 'none';
        }

        async function processBraille() {
            const input = document.getElementById('braille-input').value.trim();
            const resultDiv = document.getElementById('braille-result');
            const outputDiv = document.getElementById('braille-output');
            const loading = document.getElementById('braille-loading');

            if (!input) {
                alert('Please enter some braille text to process.');
                return;
            }

            loading.style.display = 'block';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/api/process-braille', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        braille_strings: input.split(',').map(s => s.trim())
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    outputDiv.innerHTML = `
                        <div style="margin-bottom: 15px;">
                            <strong>Processed Text:</strong> ${data.text}
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong>Explanation:</strong> ${data.explanation}
                        </div>
                        <div>
                            <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
                        </div>
                    `;
                    resultDiv.style.display = 'block';
                } else {
                    outputDiv.innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                outputDiv.innerHTML = `<div style="color: red;">Network error: ${error.message}</div>`;
                resultDiv.style.display = 'block';
            }

            loading.style.display = 'none';
        }

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            const messagesDiv = document.getElementById('chat-messages');
            const loading = document.getElementById('chat-loading');

            if (!message) return;

            const userMessageDiv = document.createElement('div');
            userMessageDiv.className = 'message user-message';
            userMessageDiv.innerHTML = `<strong>You:</strong> ${message}`;
            messagesDiv.appendChild(userMessageDiv);

            input.value = '';
            loading.style.display = 'block';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();

                const assistantMessageDiv = document.createElement('div');
                assistantMessageDiv.className = 'message assistant-message';
                
                if (response.ok) {
                    assistantMessageDiv.innerHTML = `<strong>Assistant:</strong> ${data.response}`;
                } else {
                    assistantMessageDiv.innerHTML = `<strong>Assistant:</strong> <span style="color: red;">Error: ${data.error}</span>`;
                }
                
                messagesDiv.appendChild(assistantMessageDiv);
            } catch (error) {
                const errorMessageDiv = document.createElement('div');
                errorMessageDiv.className = 'message assistant-message';
                errorMessageDiv.innerHTML = `<strong>Assistant:</strong> <span style="color: red;">Network error: ${error.message}</span>`;
                messagesDiv.appendChild(errorMessageDiv);
            }

            loading.style.display = 'none';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function handleChatKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function clearChat() {
            const messagesDiv = document.getElementById('chat-messages');
            messagesDiv.innerHTML = `
                <div class="message assistant-message">
                    <strong>Assistant:</strong> Hello! I'm your AI assistant. I can help with braille processing and answer questions.
                </div>
            `;
        }

        // Check system status on load
        window.onload = async function() {
            try {
                const response = await fetch('/health');
                const status = await response.json();
                
                if (!status.roboflow_configured) {
                    document.getElementById('detectBtn').title = "Roboflow API key not configured";
                    document.getElementById('processBtn').title = "Roboflow API key not configured";
                }
                
                console.log('System status:', status);
            } catch (error) {
                console.log('Could not check system status');
            }
        };
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_error_response(self, error_message, status_code=500):
        self.send_json_response({'error': error_message}, status_code)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging to reduce noise
        pass