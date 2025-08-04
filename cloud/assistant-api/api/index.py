# api/index.py - Simplified Braille Detection API
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
# BRAILLE ASSISTANT CLASSES (UNCHANGED - WORKING FINE)
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
# FIXED BRAILLE DETECTOR CLASSES
# ============================================================================

class BrailleDetector:
    """FIXED Braille Detection using correct detection API endpoint"""
    
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            print("Warning: ROBOFLOW_API_KEY not found - detection will be disabled")
            
        self.workspace_name = "braille-to-text-0xo2p"
        self.model_version = "1"
        # CORRECT endpoint for detection API (not workflows)
        self.base_url = "https://api.roboflow.com"
        
        print(f"BrailleDetector initialized:")
        print(f"  Workspace: {self.workspace_name}")
        print(f"  Model Version: {self.model_version}")
        print(f"  Detection endpoint: {self.base_url}")
    
    def _encode_image_from_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        try:
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')
            return encoded_string
        except Exception as e:
            raise Exception(f"Failed to encode image: {e}")
    
    def detect_braille_from_bytes(self, image_bytes: bytes) -> Optional[Dict]:
        """Run Braille detection using CORRECT detection API endpoint"""
        if not self.api_key:
            return {"error": "ROBOFLOW_API_KEY not configured"}
            
        try:
            # Encode image to base64
            encoded_image = self._encode_image_from_bytes(image_bytes)
            
            # CORRECT detection API endpoint (discovered by debugging script)
            url = f"{self.base_url}/{self.workspace_name}/{self.model_version}/predict"
            
            print(f"Detection API URL: {url}")
            
            # Correct payload format for detection API
            payload = {
                "api_key": self.api_key,
                "image": encoded_image,
                "confidence": 0.3,  # Reasonable confidence threshold
                "overlap": 0.5
            }
            
            # Headers
            headers = {
                "Content-Type": "application/json"
            }
            
            print("Sending detection request to Roboflow...")
            
            # Make the request
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Check for API errors in successful response
                if "error" in result:
                    print(f"API returned error: {result['error']}")
                    return {"error": result["error"]}
                
                predictions = result.get("predictions", [])
                print(f"✅ Detection successful! Found {len(predictions)} predictions")
                
                return result
            else:
                error_text = response.text
                print(f"API Error {response.status_code}: {error_text}")
                
                # Provide specific error guidance
                if response.status_code == 401:
                    return {"error": "Invalid API key or unauthorized access"}
                elif response.status_code == 404:
                    return {"error": f"Model not found. Check workspace '{self.workspace_name}' and version '{self.model_version}'"}
                else:
                    return {"error": f"API request failed: {response.status_code} - {error_text}"}
                
        except Exception as e:
            print(f"Detection error: {e}")
            return {"error": f"Detection error: {str(e)}"}
    
    def try_alternative_versions(self, image_bytes: bytes) -> Optional[Dict]:
        """Try different model versions if the default fails"""
        print("Trying alternative model versions...")
        
        versions_to_try = ["2", "3", "4", "1"]
        
        for version in versions_to_try:
            try:
                print(f"Trying version {version}...")
                
                encoded_image = self._encode_image_from_bytes(image_bytes)
                url = f"{self.base_url}/{self.workspace_name}/{version}/predict"
                
                payload = {
                    "api_key": self.api_key,
                    "image": encoded_image,
                    "confidence": 0.1,  # Lower confidence for testing
                    "overlap": 0.5
                }
                
                response = requests.post(
                    url, 
                    headers={"Content-Type": "application/json"}, 
                    json=payload, 
                    timeout=25
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "error" not in result:
                        predictions = result.get("predictions", [])
                        if predictions:
                            print(f"✅ Version {version} works! Found {len(predictions)} predictions")
                            self.model_version = version
                            return result
                    
            except Exception as e:
                continue
        
        return None
    
    def detect_braille_with_fallback(self, image_bytes: bytes) -> Optional[Dict]:
        """Try detection with fallback strategies"""
        print("=== Starting Braille Detection ===")
        
        # Primary attempt
        result = self.detect_braille_from_bytes(image_bytes)
        
        if result and "error" not in result:
            predictions = result.get("predictions", [])
            if predictions:
                return result
        
        # Try alternative versions
        print("Primary detection failed, trying alternatives...")
        result = self.try_alternative_versions(image_bytes)
        
        if result and "error" not in result:
            return result
        
        # Try with very low confidence
        print("Trying with very low confidence...")
        try:
            encoded_image = self._encode_image_from_bytes(image_bytes)
            url = f"{self.base_url}/{self.workspace_name}/1/predict"
            
            payload = {
                "api_key": self.api_key,
                "image": encoded_image,
                "confidence": 0.01,  # Extremely low
                "overlap": 0.9
            }
            
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    return result
                    
        except Exception:
            pass
        
        return {"error": "All detection methods failed"}
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions from detection response"""
        if not result or "error" in result:
            return []
            
        try:
            predictions = result.get("predictions", [])
            
            if not predictions:
                return []
            
            # Validate predictions
            valid_predictions = []
            required_keys = ['x', 'y', 'width', 'height', 'confidence', 'class']
            
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                    
                if not all(key in pred for key in required_keys):
                    continue
                
                try:
                    cleaned_pred = {
                        'x': float(pred['x']),
                        'y': float(pred['y']),
                        'width': float(pred['width']),
                        'height': float(pred['height']),
                        'confidence': max(0.0, min(1.0, float(pred['confidence']))),
                        'class': str(pred['class']).strip()
                    }
                    
                    if cleaned_pred['width'] > 0 and cleaned_pred['height'] > 0 and cleaned_pred['class']:
                        valid_predictions.append(cleaned_pred)
                        
                except (ValueError, TypeError):
                    continue
            
            return valid_predictions
            
        except Exception:
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.2) -> List[str]:
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
                # Try with lower confidence
                filtered_predictions = [
                    pred for pred in predictions 
                    if pred.get('confidence', 0) >= 0.05
                ]
            
            if not filtered_predictions:
                filtered_predictions = predictions
            
            # Sort by Y coordinate
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p.get('y', 0))
            
            if not sorted_by_y:
                return []
            
            rows = []
            current_group = [sorted_by_y[0]]
            
            # Group into rows
            for i in range(1, len(sorted_by_y)):
                current_pred = sorted_by_y[i]
                prev_pred = sorted_by_y[i-1]
                
                # Calculate threshold for row grouping
                avg_height = (current_pred.get('height', 20) + prev_pred.get('height', 20)) / 2
                threshold = max(8, avg_height * 0.7)
                
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
            
        except Exception:
            return []

# ============================================================================
# SIMPLIFIED API HANDLER
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
                    'roboflow_configured': bool(self.detector.api_key),
                    'detection_endpoint': f"{self.detector.base_url}/{self.detector.workspace_name}/{self.detector.model_version}/predict"
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
            if path == '/api/detect-and-process':
                self.handle_detect_and_process(data)
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())
                
        except Exception as e:
            self.send_error_response(f"POST error: {str(e)}")
    
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
            
            # Run detection with improved method
            detection_result = self.detector.detect_braille_with_fallback(image_bytes)
            
            if "error" in detection_result:
                self.send_json_response({
                    'detection': {
                        'predictions': [],
                        'text_rows': [],
                        'detection_count': 0,
                        'error': detection_result["error"]
                    },
                    'processing': {
                        'text': '',
                        'explanation': f'Detection failed: {detection_result["error"]}',
                        'confidence': 0.0
                    }
                })
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
                        'predictions': predictions,
                        'text_rows': [],
                        'detection_count': len(predictions)
                    },
                    'processing': {
                        'text': '',
                        'explanation': 'No braille text could be organized from the detected characters.' if predictions else 'No braille characters detected in the image.',
                        'confidence': 0.0
                    }
                })
            
        except Exception as e:
            self.send_error_response(f'Detection and processing error: {str(e)}')
    
    def serve_html(self):
        """Serve the simplified web interface"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Braille Recognition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 10px; 
            padding: 30px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { 
            text-align: center; 
            color: #333; 
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #4CAF50; 
            border-radius: 8px; 
            padding: 40px;
            text-align: center; 
            cursor: pointer; 
            margin-bottom: 20px;
            transition: background-color 0.3s;
        }
        .upload-area:hover { 
            background: #f0f8f0; 
        }
        .upload-area.dragover { 
            background: #e8f5e8; 
            border-color: #45a049; 
        }
        .image-preview { 
            max-width: 100%; 
            max-height: 300px; 
            margin: 20px auto; 
            display: block; 
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .btn { 
            background: #4CAF50; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 6px; 
            cursor: pointer; 
            font-size: 16px; 
            width: 100%;
            margin-top: 10px;
        }
        .btn:hover { 
            background: #45a049; 
        }
        .btn:disabled { 
            background: #cccccc; 
            cursor: not-allowed; 
        }
        .result { 
            margin-top: 20px; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 8px; 
            border-left: 4px solid #4CAF50;
        }
        .loading { 
            display: none; 
            text-align: center; 
            color: #4CAF50; 
            font-weight: bold; 
            margin: 20px 0;
        }
        .status { 
            background: #e8f5e8; 
            border: 1px solid #4CAF50; 
            border-radius: 6px;
            padding: 10px; 
            margin-bottom: 20px; 
            text-align: center;
            font-size: 14px;
        }
        .status.error {
            background: #ffe8e8; 
            border-color: #f44336;
            color: #d32f2f;
        }
        .detected-text {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        .detected-text h4 {
            margin-bottom: 10px;
            color: #856404;
        }
        .text-row {
            background: white;
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            font-family: monospace;
            border-left: 3px solid #ffc107;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔤 Braille Recognition</h1>
        
        <div id="status" class="status" style="display: none;">
            <span id="status-text">Checking system...</span>
        </div>

        <div class="upload-area" onclick="document.getElementById('imageInput').click()" 
             ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
            <p>📸 Click to upload or drag & drop braille image</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;" onchange="handleImageUpload(event)">
        </div>
        
        <img id="imagePreview" class="image-preview" style="display: none;">
        
        <button class="btn" onclick="processImage()" id="processBtn" disabled>🔍 Detect Braille</button>
        
        <div class="loading" id="loading">Processing image...</div>
        
        <div id="result" class="result" style="display: none;">
            <div id="output"></div>
        </div>
    </div>

    <script>
        let currentImage = null;

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
                
                document.getElementById('processBtn').disabled = false;
            };
            reader.readAsDataURL(file);
        }

        async function processImage() {
            if (!currentImage) {
                alert('Please upload an image first.');
                return;
            }

            const resultDiv = document.getElementById('result');
            const outputDiv = document.getElementById('output');
            const loading = document.getElementById('loading');

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
                    let outputHTML = '';
                    
                    // Show detected text rows
                    if (data.detection.text_rows && data.detection.text_rows.length > 0) {
                        outputHTML += `
                            <div class="detected-text">
                                <h4>📝 Detected Braille Text:</h4>
                                ${data.detection.text_rows.map(row => 
                                    `<div class="text-row">${row}</div>`
                                ).join('')}
                            </div>
                        `;
                    }
                    
                    // Show processing results
                    outputHTML += `
                        <div style="margin-top: 20px;">
                            <h4>🤖 AI Processing:</h4>
                            <p><strong>Processed Text:</strong> ${data.processing.text || 'No text processed'}</p>
                            <p><strong>Explanation:</strong> ${data.processing.explanation}</p>
                            <p><strong>Confidence:</strong> ${(data.processing.confidence * 100).toFixed(1)}%</p>
                        </div>
                        <div style="margin-top: 15px; font-size: 14px; color: #666;">
                            <strong>Detection Info:</strong> ${data.detection.detection_count} characters detected
                            ${data.detection.error ? `<br><span style="color: red;">Error: ${data.detection.error}</span>` : ''}
                        </div>
                    `;
                    
                    outputDiv.innerHTML = outputHTML;
                    resultDiv.style.display = 'block';
                } else {
                    outputDiv.innerHTML = `<div style="color: red;">❌ Error: ${data.error}</div>`;
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                outputDiv.innerHTML = `<div style="color: red;">❌ Network error: ${error.message}</div>`;
                resultDiv.style.display = 'block';
            }

            loading.style.display = 'none';
        }

        // Check system status on load
        window.onload = async function() {
            try {
                const response = await fetch('/health');
                const status = await response.json();
                
                const statusDiv = document.getElementById('status');
                const statusText = document.getElementById('status-text');
                
                if (status.roboflow_configured) {
                    statusText.textContent = '✅ System Ready - Braille detection enabled';
                    statusDiv.classList.remove('error');
                } else {
                    statusText.textContent = '❌ Roboflow API key not configured - Detection disabled';
                    statusDiv.classList.add('error');
                    document.getElementById('processBtn').disabled = true;
                }
                
                statusDiv.style.display = 'block';
                
            } catch (error) {
                console.log('Could not check system status:', error);
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