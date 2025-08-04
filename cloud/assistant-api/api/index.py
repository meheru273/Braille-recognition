# api/index.py - Complete Working Minimal Braille System
import json
import os
import base64
import requests
from http.server import BaseHTTPRequestHandler
import traceback
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# ============================================================================
# GLOBAL DEBUG STORAGE (for web interface debugging)
# ============================================================================
DEBUG_LOGS = []

def debug_log(message, level="INFO"):
    """Enhanced logging that stores logs for web interface"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    
    # Store for web interface (keep last 100 logs)
    DEBUG_LOGS.append(log_entry)
    if len(DEBUG_LOGS) > 100:
        DEBUG_LOGS.pop(0)
    
    return log_entry

def get_debug_logs():
    """Get all debug logs for web interface"""
    return "\n".join(DEBUG_LOGS)

# ============================================================================
# BRAILLE RESULT DATACLASS
# ============================================================================

@dataclass
class BrailleResult:
    """Result from braille processing"""
    text: str
    explanation: str
    confidence: float

# ============================================================================
# SIMPLE AI ASSISTANT
# ============================================================================

class SimpleAssistant:
    """Minimal assistant with API and fallback"""
    
    def __init__(self):
        debug_log("Initializing SimpleAssistant", "INIT")
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            if self.api_key.startswith("gsk_"):
                self.base_url = "https://api.groq.com/openai/v1"
                self.model = "llama-3.1-8b-instant"
                self.provider = "groq"
            else:
                self.base_url = "https://api.openai.com/v1"
                self.model = "gpt-3.5-turbo"
                self.provider = "openai"
            debug_log(f"✅ AI API configured: {self.provider}", "INIT")
        else:
            debug_log("⚠️ No AI API key - using fallback mode", "INIT")
    
    def process_braille_strings(self, detected_strings: List[str]) -> BrailleResult:
        """Process braille strings with debugging"""
        debug_log(f"Processing {len(detected_strings)} braille strings", "PROCESS")
        debug_log(f"Input strings: {detected_strings}", "PROCESS")
        
        if not detected_strings:
            debug_log("No strings to process", "PROCESS")
            return BrailleResult("", "No braille characters detected.", 0.0)
        
        try:
            raw_text = " ".join(detected_strings).strip()
            debug_log(f"Combined raw text: '{raw_text}'", "PROCESS")
            
            if self.api_key:
                processed_text = self._process_with_api(raw_text)
                explanation = self._get_explanation(processed_text)
                confidence = 0.8
                debug_log("✅ Processed with AI API", "PROCESS")
            else:
                processed_text = self._fallback_process(raw_text)
                explanation = f"Basic processing: {processed_text} (No AI API available)"
                confidence = 0.5
                debug_log("⚠️ Used fallback processing", "PROCESS")
            
            result = BrailleResult(processed_text, explanation, confidence)
            debug_log(f"Final result: {result}", "PROCESS")
            return result
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            debug_log(error_msg, "ERROR")
            traceback.print_exc()
            
            fallback_text = " ".join(detected_strings)
            return BrailleResult(fallback_text, f"Error processing: {error_msg}", 0.3)
    
    def _process_with_api(self, text: str) -> str:
        """Process text using AI API"""
        debug_log(f"Using AI API to process: '{text}'", "API")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a braille text processor. Clean up and correct braille text."},
                    {"role": "user", "content": f"Clean and correct this braille text: '{text}'"}
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                processed = result["choices"][0]["message"]["content"].strip()
                debug_log(f"✅ AI processed result: '{processed}'", "API")
                return processed
            else:
                debug_log(f"AI API error: {response.status_code}", "ERROR")
                return self._fallback_process(text)
                
        except Exception as e:
            debug_log(f"AI API exception: {str(e)}", "ERROR")
            return self._fallback_process(text)
    
    def _fallback_process(self, text: str) -> str:
        """Basic text processing without AI"""
        debug_log(f"Fallback processing: '{text}'", "FALLBACK")
        
        # Basic cleanup
        cleaned = text.strip()
        
        # Simple word separation
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
            
            result = " ".join(words)
        else:
            result = cleaned
        
        debug_log(f"Fallback result: '{result}'", "FALLBACK")
        return result
    
    def _get_explanation(self, text: str) -> str:
        """Get explanation for processed text"""
        if self.api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": f"Briefly explain what this text is about: '{text}'"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.3
                }
                
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                    
            except Exception as e:
                debug_log(f"Explanation API error: {str(e)}", "ERROR")
        
        return f"This appears to be braille text that reads: '{text}'"

# ============================================================================
# BRAILLE DETECTOR
# ============================================================================

class BrailleDetector:
    """Fixed Braille Detector with debugging"""
    
    def __init__(self):
        debug_log("Initializing BrailleDetector", "INIT")
        
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        self.workspace_name = "braille-to-text-0xo2p"
        self.model_version = "1"
        self.base_url = "https://api.roboflow.com"
        
        debug_log(f"API Key present: {bool(self.api_key)}", "INIT")
        debug_log(f"Workspace: {self.workspace_name}", "INIT")
        debug_log(f"Model Version: {self.model_version}", "INIT")
        
        if not self.api_key:
            debug_log("❌ NO ROBOFLOW API KEY FOUND", "ERROR")
    
    def detect_braille(self, image_bytes: bytes) -> Dict:
        """Detect braille with comprehensive debugging"""
        debug_log("=== STARTING BRAILLE DETECTION ===", "DETECTION")
        debug_log(f"Image size: {len(image_bytes)} bytes", "DETECTION")
        
        if not self.api_key:
            error_result = {"error": "ROBOFLOW_API_KEY not configured"}
            debug_log("❌ No API key - detection disabled", "ERROR")
            return error_result
        
        try:
            # Step 1: Encode image
            debug_log("Step 1: Encoding image to base64", "DETECTION")
            start_time = time.time()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            encode_time = time.time() - start_time
            debug_log(f"✅ Image encoded in {encode_time:.3f}s", "DETECTION")
            
            # Step 2: Prepare request
            url = f"{self.base_url}/{self.workspace_name}/{self.model_version}/predict"
            debug_log(f"Detection URL: {url}", "DETECTION")
            
            payload = {
                "api_key": self.api_key,
                "image": encoded_image,
                "confidence": 0.3,
                "overlap": 0.5
            }
            
            headers = {"Content-Type": "application/json"}
            
            # Step 3: Make API call
            debug_log("Step 3: Making API request", "DETECTION")
            request_start = time.time()
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            request_time = time.time() - request_start
            debug_log(f"API request completed in {request_time:.3f}s", "DETECTION")
            debug_log(f"Response status: {response.status_code}", "DETECTION")
            
            # Step 4: Process response
            if response.status_code == 200:
                debug_log("✅ HTTP 200 - Success", "DETECTION")
                try:
                    result = response.json()
                    debug_log("✅ JSON parsing successful", "DETECTION")
                    
                    if "error" in result:
                        debug_log(f"❌ API returned error: {result['error']}", "ERROR")
                        return {"error": result["error"]}
                    
                    predictions = result.get("predictions", [])
                    debug_log(f"✅ Found {len(predictions)} predictions", "DETECTION")
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON response: {str(e)}"
                    debug_log(f"❌ {error_msg}", "ERROR")
                    return {"error": error_msg}
            
            else:
                debug_log(f"❌ HTTP Error {response.status_code}", "ERROR")
                debug_log(f"Response text: {response.text}", "ERROR")
                
                if response.status_code == 401:
                    return {"error": "Invalid API key or unauthorized"}
                elif response.status_code == 404:
                    return {"error": f"Model not found: {self.workspace_name}/v{self.model_version}"}
                else:
                    return {"error": f"API error {response.status_code}: {response.text}"}
        
        except requests.exceptions.Timeout:
            error_msg = "Request timeout - API took too long"
            debug_log(f"❌ {error_msg}", "ERROR")
            return {"error": error_msg}
        
        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            debug_log(f"❌ {error_msg}", "ERROR")
            traceback.print_exc()
            return {"error": error_msg}
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions with debugging"""
        debug_log("=== EXTRACTING PREDICTIONS ===", "EXTRACT")
        
        if not result or "error" in result:
            debug_log("❌ No valid result to extract from", "ERROR")
            return []
        
        try:
            predictions = result.get("predictions", [])
            debug_log(f"Raw predictions count: {len(predictions)}", "EXTRACT")
            
            if not predictions:
                debug_log("❌ No predictions found", "EXTRACT")
                return []
            
            valid_predictions = []
            required_keys = ['x', 'y', 'width', 'height', 'confidence', 'class']
            
            for i, pred in enumerate(predictions):
                debug_log(f"Processing prediction {i+1}/{len(predictions)}", "EXTRACT")
                
                if not isinstance(pred, dict):
                    debug_log(f"❌ Prediction {i+1} is not a dict", "ERROR")
                    continue
                
                missing_keys = [key for key in required_keys if key not in pred]
                if missing_keys:
                    debug_log(f"❌ Prediction {i+1} missing keys: {missing_keys}", "ERROR")
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
                        debug_log(f"✅ Prediction {i+1} validated: '{cleaned_pred['class']}' (conf: {cleaned_pred['confidence']:.3f})", "EXTRACT")
                    else:
                        debug_log(f"❌ Prediction {i+1} has invalid dimensions or empty class", "ERROR")
                        
                except (ValueError, TypeError) as e:
                    debug_log(f"❌ Prediction {i+1} conversion failed: {str(e)}", "ERROR")
                    continue
            
            debug_log(f"✅ Extracted {len(valid_predictions)} valid predictions", "EXTRACT")
            return valid_predictions
            
        except Exception as e:
            debug_log(f"❌ Error extracting predictions: {str(e)}", "ERROR")
            traceback.print_exc()
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict]) -> List[str]:
        """Organize predictions into text rows"""
        debug_log(f"=== ORGANIZING {len(predictions)} PREDICTIONS INTO ROWS ===", "ORGANIZE")
        
        if not predictions:
            debug_log("❌ No predictions to organize", "ORGANIZE")
            return []
        
        try:
            # Sort by Y coordinate
            sorted_by_y = sorted(predictions, key=lambda p: p.get('y', 0))
            debug_log(f"Sorted predictions by Y coordinate", "ORGANIZE")
            
            rows = []
            current_group = [sorted_by_y[0]]
            
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
                            debug_log(f"✅ Row completed: '{row_text}'", "ORGANIZE")
                    current_group = [current_pred]
            
            # Process final group
            if current_group:
                current_group.sort(key=lambda p: p.get('x', 0))
                row_text = ''.join([p.get('class', '') for p in current_group])
                if row_text.strip():
                    rows.append(row_text)
                    debug_log(f"✅ Final row: '{row_text}'", "ORGANIZE")
            
            debug_log(f"✅ Organized into {len(rows)} text rows", "ORGANIZE")
            return rows
            
        except Exception as e:
            debug_log(f"❌ Error organizing text: {str(e)}", "ERROR")
            traceback.print_exc()
            return []

# ============================================================================
# API HANDLER
# ============================================================================

class handler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        debug_log("Initializing API handler", "HANDLER")
        self.detector = BrailleDetector()
        self.assistant = SimpleAssistant()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        debug_log(f"GET request: {self.path}", "REQUEST")
        
        try:
            if self.path == '/' or self.path == '/index.html':
                self.serve_html()
            elif self.path == '/health':
                self.send_json_response({
                    'status': 'healthy',
                    'roboflow_configured': bool(self.detector.api_key),
                    'ai_configured': bool(self.assistant.api_key),
                    'debug_mode': True
                })
            elif self.path == '/debug-logs':
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(get_debug_logs().encode())
            else:
                self.send_json_response({'error': 'Not found'}, 404)
                
        except Exception as e:
            debug_log(f"GET error: {str(e)}", "ERROR")
            self.send_json_response({'error': f'GET error: {str(e)}'}, 500)
    
    def do_POST(self):
        debug_log(f"POST request: {self.path}", "REQUEST")
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            
            try:
                data = json.loads(body) if body else {}
                debug_log(f"Request data keys: {list(data.keys())}", "REQUEST")
            except json.JSONDecodeError as e:
                debug_log(f"Invalid JSON: {str(e)}", "ERROR")
                self.send_json_response({'error': 'Invalid JSON'}, 400)
                return
            
            # Route requests
            if self.path == '/api/detect-braille':
                self.handle_detection(data)
            elif self.path == '/api/detect-and-process':
                self.handle_detect_and_process(data)
            else:
                debug_log(f"Unknown endpoint: {self.path}", "ERROR")
                self.send_json_response({'error': 'Endpoint not found'}, 404)
                
        except Exception as e:
            debug_log(f"POST error: {str(e)}", "ERROR")
            self.send_json_response({'error': f'POST error: {str(e)}'}, 500)
    
    def handle_detection(self, data):
        """Handle braille detection"""
        debug_log("=== HANDLING DETECTION REQUEST ===", "HANDLER")
        
        try:
            image_data = data.get('image')
            if not image_data:
                debug_log("❌ No image data provided", "ERROR")
                self.send_json_response({'error': 'Image data required'}, 400)
                return
            
            # Decode image
            try:
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                debug_log(f"✅ Image decoded: {len(image_bytes)} bytes", "HANDLER")
                
            except Exception as e:
                debug_log(f"Image decoding error: {str(e)}", "ERROR")
                self.send_json_response({'error': f'Invalid image data: {str(e)}'}, 400)
                return
            
            # Run detection
            detection_result = self.detector.detect_braille(image_bytes)
            
            if "error" in detection_result:
                debug_log(f"❌ Detection failed: {detection_result['error']}", "ERROR")
                self.send_json_response(detection_result)
                return
            
            # Extract and organize
            predictions = self.detector.extract_predictions(detection_result)
            text_rows = self.detector.organize_text_by_rows(predictions)
            
            response_data = {
                'predictions': predictions,
                'text_rows': text_rows,
                'detection_count': len(predictions),
                'debug': True
            }
            
            debug_log(f"✅ Detection complete: {len(predictions)} predictions, {len(text_rows)} rows", "HANDLER")
            self.send_json_response(response_data)
            
        except Exception as e:
            debug_log(f"Detection handler error: {str(e)}", "ERROR")
            self.send_json_response({'error': f'Detection error: {str(e)}'}, 500)
    
    def handle_detect_and_process(self, data):
        """Handle detection + AI processing"""
        debug_log("=== HANDLING DETECT AND PROCESS REQUEST ===", "HANDLER")
        
        try:
            image_data = data.get('image')
            if not image_data:
                debug_log("❌ No image data provided", "ERROR")
                self.send_json_response({'error': 'Image data required'}, 400)
                return
            
            # Decode image
            try:
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                debug_log(f"✅ Image decoded: {len(image_bytes)} bytes", "HANDLER")
                
            except Exception as e:
                debug_log(f"Image decoding error: {str(e)}", "ERROR")
                self.send_json_response({'error': f'Invalid image data: {str(e)}'}, 400)
                return
            
            # Step 1: Run detection
            detection_result = self.detector.detect_braille(image_bytes)
            
            if "error" in detection_result:
                self.send_json_response({
                    'detection': {'error': detection_result['error'], 'predictions': [], 'text_rows': []},
                    'processing': {'text': '', 'explanation': f'Detection failed: {detection_result["error"]}', 'confidence': 0.0}
                })
                return
            
            # Step 2: Extract and organize
            predictions = self.detector.extract_predictions(detection_result)
            text_rows = self.detector.organize_text_by_rows(predictions)
            
            # Step 3: Process with AI
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
                        'explanation': 'No braille text could be organized from detected characters.' if predictions else 'No braille characters detected.',
                        'confidence': 0.0
                    }
                })
            
        except Exception as e:
            debug_log(f"Detect and process error: {str(e)}", "ERROR")
            self.send_json_response({'error': f'Processing error: {str(e)}'}, 500)
    
    def serve_html(self):
        """Serve minimal HTML interface with real-time debugging"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Working Braille Detection System</title>
    <style>
        body { font-family: monospace; padding: 20px; background: #1a1a1a; color: #00ff00; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00ff00; font-size: 2em; }
        .section { background: #000; border: 1px solid #00ff00; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .btn { background: #004400; color: #00ff00; padding: 10px 20px; border: 1px solid #00ff00; border-radius: 4px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #006600; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result { background: #002200; border: 1px solid #00ff00; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .error { border-color: #ff0000; color: #ff0000; background: #220000; }
        .debug { background: #000; color: #00ff00; padding: 15px; border: 1px solid #00ff00; border-radius: 4px; font-size: 12px; overflow-y: auto; max-height: 400px; white-space: pre-wrap; }
        input[type="file"] { background: #002200; color: #00ff00; border: 1px solid #00ff00; padding: 5px; }
        img { max-width: 300px; max-height: 200px; border: 1px solid #00ff00; }
        .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
        .status.ok { background: #002200; border: 1px solid #00ff00; }
        .status.error { background: #220000; border: 1px solid #ff0000; color: #ff0000; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 WORKING BRAILLE DETECTION SYSTEM</h1>
            <p>Complete system with real-time debugging</p>
            <div id="status" class="status">Checking system...</div>
        </div>
        
        <div class="section">
            <h3>📸 Image Upload & Detection</h3>
            <input type="file" id="imageInput" accept="image/*" onchange="handleImageUpload(event)">
            <br><br>
            <img id="imagePreview" style="display: none;">
            <br><br>
            <button class="btn" onclick="detectOnly()" id="detectBtn" disabled>🔍 Detect Only</button>
            <button class="btn" onclick="detectAndProcess()" id="processBtn" disabled>🚀 Detect + AI Process</button>
            <button class="btn" onclick="checkSystem()">🏥 Check System</button>
            <button class="btn" onclick="refreshDebug()">🔄 Refresh Debug</button>
        </div>
        
        <div id="results" class="section" style="display: none;">
            <h3>📊 Results</h3>
            <div id="resultsContent"></div>
        </div>
        
        <div class="section">
            <h3>🐛 Real-Time Debug Output</h3>
            <div id="debug" class="debug">Initializing debug console...\n</div>
        </div>
    </div>

    <script>
        let currentImage = null;
        let debugUpdateInterval = null;
        
        function log(message, level = 'INFO') {
            const debug = document.getElementById('debug');
            const timestamp = new Date().toISOString();
            const logEntry = `[${timestamp}] [${level}] ${message}\n`;
            debug.innerHTML += logEntry;
            debug.scrollTop = debug.scrollHeight;
            console.log(`[${level}] ${message}`);
        }
        
        function handleImageUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            log(`📁 File selected: ${file.name} (${file.size} bytes)`, 'UPLOAD');
            
            if (!file.type.startsWith('image/')) {
                log('❌ ERROR: Not an image file', 'ERROR');
                alert('Please select an image file.');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImage = e.target.result;
                log(`✅ Image loaded successfully`, 'UPLOAD');
                
                const preview = document.getElementById('imagePreview');
                preview.src = currentImage;
                preview.style.display = 'block';
                
                document.getElementById('detectBtn').disabled = false;
                document.getElementById('processBtn').disabled = false;
                log('🎯 Detection buttons enabled', 'UPLOAD');
            };
            reader.readAsDataURL(file);
        }
        
        async function detectOnly() {
            if (!currentImage) {
                log('❌ ERROR: No image selected', 'ERROR');
                alert('Please upload an image first.');
                return;
            }
            
            log('🚀 Starting detection only...', 'DETECT');
            showResults('⏳ Detecting braille characters...');
            
            try {
                const startTime = Date.now();
                const response = await fetch('/api/detect-braille', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: currentImage })
                });
                const requestTime = Date.now() - startTime;
                
                log(`📡 Detection response received in ${requestTime}ms`, 'DETECT');
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`✅ Detection successful: ${data.detection_count} predictions`, 'DETECT');
                    
                    showResults(`
                        <div class="result">
                            <h4>✅ Detection Results</h4>
                            <p><strong>Characters Found:</strong> ${data.detection_count}</p>
                            <p><strong>Text Rows:</strong> ${data.text_rows.length}</p>
                            <div><strong>Organized Text:</strong></div>
                            ${data.text_rows.length > 0 ? 
                                '<ul>' + data.text_rows.map(row => `<li>'${row}'</li>`).join('') + '</ul>' : 
                                '<p>No organized text found</p>'
                            }
                            <details>
                                <summary>Raw Predictions (${data.predictions.length})</summary>
                                <pre>${JSON.stringify(data.predictions, null, 2)}</pre>
                            </details>
                        </div>
                    `);
                } else {
                    log(`❌ Detection failed: ${data.error}`, 'ERROR');
                    showResults(`<div class="result error"><h4>❌ Detection Failed</h4><p>${data.error}</p></div>`);
                }
                
            } catch (error) {
                log(`💥 Detection exception: ${error.message}`, 'ERROR');
                showResults(`<div class="result error"><h4>💥 Network Error</h4><p>${error.message}</p></div>`);
            }
        }
        
        async function detectAndProcess() {
            if (!currentImage) {
                log('❌ ERROR: No image selected', 'ERROR');
                alert('Please upload an image first.');
                return;
            }
            
            log('🚀 Starting detection + AI processing...', 'PROCESS');
            showResults('⏳ Detecting and processing with AI...');
            
            try {
                const startTime = Date.now();
                const response = await fetch('/api/detect-and-process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: currentImage })
                });
                const requestTime = Date.now() - startTime;
                
                log(`📡 Processing response received in ${requestTime}ms`, 'PROCESS');
                
                const data = await response.json();
                
                if (response.ok) {
                    log(`✅ Processing complete`, 'PROCESS');
                    
                    showResults(`
                        <div class="result">
                            <h4>🎯 Detection Results</h4>
                            <p><strong>Characters Found:</strong> ${data.detection.detection_count}</p>
                            <p><strong>Text Rows:</strong> ${data.detection.text_rows.length}</p>
                            ${data.detection.error ? `<p class="error">Detection Error: ${data.detection.error}</p>` : ''}
                        </div>
                        <div class="result">
                            <h4>🤖 AI Processing Results</h4>
                            <p><strong>Processed Text:</strong> ${data.processing.text || 'None'}</p>
                            <p><strong>Explanation:</strong> ${data.processing.explanation}</p>
                            <p><strong>Confidence:</strong> ${(data.processing.confidence * 100).toFixed(1)}%</p>
                        </div>
                    `);
                } else {
                    log(`❌ Processing failed: ${data.error}`, 'ERROR');
                    showResults(`<div class="result error"><h4>❌ Processing Failed</h4><p>${data.error}</p></div>`);
                }
                
            } catch (error) {
                log(`💥 Processing exception: ${error.message}`, 'ERROR');
                showResults(`<div class="result error"><h4>💥 Network Error</h4><p>${error.message}</p></div>`);
            }
        }
        
        async function checkSystem() {
            log('🏥 Checking system health...', 'SYSTEM');
            
            try {
                const response = await fetch('/health');
                const status = await response.json();
                
                log(`✅ System status: ${JSON.stringify(status)}`, 'SYSTEM');
                
                const statusDiv = document.getElementById('status');
                if (status.roboflow_configured && status.ai_configured) {
                    statusDiv.className = 'status ok';
                    statusDiv.innerHTML = '✅ System Ready - Full AI capabilities';
                } else if (status.roboflow_configured) {
                    statusDiv.className = 'status ok';
                    statusDiv.innerHTML = '⚠️ Detection OK - AI in fallback mode';
                } else {
                    statusDiv.className = 'status error';
                    statusDiv.innerHTML = '❌ Roboflow API key not configured';
                }
                
            } catch (error) {
                log(`💥 Health check failed: ${error.message}`, 'ERROR');
                const statusDiv = document.getElementById('status');
                statusDiv.className = 'status error';
                statusDiv.innerHTML = '💥 System check failed';
            }
        }
        
        async function refreshDebug() {
            try {
                const response = await fetch('/debug-logs');
                const logs = await response.text();
                document.getElementById('debug').innerHTML = logs;
                log('🔄 Debug logs refreshed', 'DEBUG');
            } catch (error) {
                log(`❌ Failed to refresh debug logs: ${error.message}`, 'ERROR');
            }
        }
        
        function showResults(content) {
            const resultsDiv = document.getElementById('results');
            const contentDiv = document.getElementById('resultsContent');
            contentDiv.innerHTML = content;
            resultsDiv.style.display = 'block';
        }
        
        // Auto-refresh debug logs every 5 seconds
        function startDebugRefresh() {
            debugUpdateInterval = setInterval(async () => {
                try {
                    const response = await fetch('/debug-logs');
                    const logs = await response.text();
                    const debugDiv = document.getElementById('debug');
                    const shouldScroll = debugDiv.scrollTop + debugDiv.clientHeight >= debugDiv.scrollHeight - 10;
                    debugDiv.innerHTML = logs;
                    if (shouldScroll) {
                        debugDiv.scrollTop = debugDiv.scrollHeight;
                    }
                } catch (error) {
                    // Silently fail debug refresh
                }
            }, 2000);
        }
        
        function stopDebugRefresh() {
            if (debugUpdateInterval) {
                clearInterval(debugUpdateInterval);
                debugUpdateInterval = null;
            }
        }
        
        // Initialize
        window.onload = function() {
            log('🚀 Application initialized', 'INIT');
            checkSystem();
            startDebugRefresh();
        };
        
        // Stop refresh when page unloads
        window.onbeforeunload = function() {
            stopDebugRefresh();
        };
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def send_json_response(self, data, status_code=200):
        debug_log(f"Sending JSON response (status: {status_code})", "RESPONSE")
        
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_OPTIONS(self):
        debug_log("OPTIONS request received", "REQUEST")
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        debug_log(f"HTTP: {format % args}", "HTTP")

# ============================================================================
# STARTUP
# ============================================================================

debug_log("🚀 Starting Complete Braille Detection System", "STARTUP")
debug_log(f"Environment check:", "STARTUP")
debug_log(f"  ROBOFLOW_API_KEY: {'✅ SET' if os.getenv('ROBOFLOW_API_KEY') else '❌ NOT SET'}", "STARTUP")
debug_log(f"  GROQ_API_KEY: {'✅ SET' if os.getenv('GROQ_API_KEY') else '❌ NOT SET'}", "STARTUP")
debug_log(f"  OPENAI_API_KEY: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}", "STARTUP")
debug_log("✅ System ready for requests", "STARTUP")