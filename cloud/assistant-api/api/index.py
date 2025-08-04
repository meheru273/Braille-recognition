# api/index.py - Minimal Braille Detection with Intense Debugging
import json
import os
import base64
import requests
from http.server import BaseHTTPRequestHandler
import traceback
import time
from datetime import datetime

# ============================================================================
# LOGGING & DEBUGGING UTILITIES
# ============================================================================

def debug_log(message, level="INFO"):
    """Enhanced logging with timestamp and level"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")

def debug_dict(data, title="Debug Data"):
    """Pretty print dictionary data"""
    debug_log(f"=== {title} ===", "DEBUG")
    try:
        print(json.dumps(data, indent=2, default=str))
    except:
        print(str(data))
    debug_log(f"=== End {title} ===", "DEBUG")

def debug_exception(e, context=""):
    """Log detailed exception information"""
    debug_log(f"EXCEPTION in {context}: {str(e)}", "ERROR")
    debug_log(f"Exception type: {type(e).__name__}", "ERROR")
    debug_log("Traceback:", "ERROR")
    traceback.print_exc()

# ============================================================================
# MINIMAL BRAILLE DETECTOR WITH INTENSE DEBUGGING
# ============================================================================

class BrailleDetector:
    """Minimal Braille Detector with maximum debugging"""
    
    def __init__(self):
        debug_log("Initializing BrailleDetector", "INIT")
        
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        self.workspace_name = "braille-to-text-0xo2p"
        self.model_version = "1"
        self.base_url = "https://api.roboflow.com"
        
        debug_log(f"API Key present: {bool(self.api_key)}", "INIT")
        debug_log(f"Workspace: {self.workspace_name}", "INIT")
        debug_log(f"Model Version: {self.model_version}", "INIT")
        debug_log(f"Base URL: {self.base_url}", "INIT")
        
        if self.api_key:
            debug_log(f"API Key prefix: {self.api_key[:8]}...", "INIT")
        else:
            debug_log("❌ NO API KEY FOUND", "ERROR")
    
    def detect_braille(self, image_bytes):
        """Detect braille with maximum debugging"""
        debug_log("=== STARTING BRAILLE DETECTION ===", "DETECTION")
        debug_log(f"Image size: {len(image_bytes)} bytes", "DETECTION")
        
        if not self.api_key:
            debug_log("❌ No API key - detection disabled", "ERROR")
            return {"error": "ROBOFLOW_API_KEY not configured"}
        
        try:
            # Step 1: Encode image
            debug_log("Step 1: Encoding image to base64", "DETECTION")
            start_time = time.time()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            encode_time = time.time() - start_time
            debug_log(f"✅ Image encoded in {encode_time:.3f}s (length: {len(encoded_image)})", "DETECTION")
            
            # Step 2: Prepare request
            debug_log("Step 2: Preparing API request", "DETECTION")
            url = f"{self.base_url}/{self.workspace_name}/{self.model_version}/predict"
            debug_log(f"Detection URL: {url}", "DETECTION")
            
            payload = {
                "api_key": self.api_key,
                "image": encoded_image,
                "confidence": 0.3,
                "overlap": 0.5
            }
            
            headers = {"Content-Type": "application/json"}
            
            debug_log("Request payload prepared:", "DETECTION")
            debug_dict({
                "url": url,
                "confidence": payload["confidence"],
                "overlap": payload["overlap"],
                "image_length": len(encoded_image),
                "api_key_prefix": self.api_key[:8] + "..."
            }, "Request Details")
            
            # Step 3: Make API call
            debug_log("Step 3: Making API request to Roboflow", "DETECTION")
            request_start = time.time()
            
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
            request_time = time.time() - request_start
            debug_log(f"API request completed in {request_time:.3f}s", "DETECTION")
            debug_log(f"Response status: {response.status_code}", "DETECTION")
            debug_log(f"Response headers: {dict(response.headers)}", "DETECTION")
            
            # Step 4: Process response
            debug_log("Step 4: Processing API response", "DETECTION")
            
            if response.status_code == 200:
                debug_log("✅ HTTP 200 - Success", "DETECTION")
                try:
                    result = response.json()
                    debug_log("✅ JSON parsing successful", "DETECTION")
                    debug_dict(result, "API Response")
                    
                    # Check for API errors
                    if "error" in result:
                        debug_log(f"❌ API returned error: {result['error']}", "ERROR")
                        return {"error": result["error"]}
                    
                    predictions = result.get("predictions", [])
                    debug_log(f"✅ Found {len(predictions)} predictions", "DETECTION")
                    
                    if predictions:
                        debug_log("Sample prediction:", "DETECTION")
                        debug_dict(predictions[0] if predictions else {}, "First Prediction")
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    debug_exception(e, "JSON parsing")
                    debug_log(f"Raw response: {response.text[:500]}...", "ERROR")
                    return {"error": f"Invalid JSON response: {str(e)}"}
            
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
            debug_log("❌ Request timeout", "ERROR")
            return {"error": "Request timeout - API took too long to respond"}
        
        except requests.exceptions.ConnectionError as e:
            debug_exception(e, "Connection error")
            return {"error": f"Connection error: {str(e)}"}
        
        except Exception as e:
            debug_exception(e, "Braille detection")
            return {"error": f"Detection failed: {str(e)}"}
    
    def extract_predictions(self, result):
        """Extract and validate predictions with debugging"""
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
                debug_dict(pred, f"Raw Prediction {i+1}")
                
                if not isinstance(pred, dict):
                    debug_log(f"❌ Prediction {i+1} is not a dict: {type(pred)}", "ERROR")
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
                        debug_log(f"✅ Prediction {i+1} validated", "EXTRACT")
                        debug_dict(cleaned_pred, f"Cleaned Prediction {i+1}")
                    else:
                        debug_log(f"❌ Prediction {i+1} has invalid dimensions or empty class", "ERROR")
                        
                except (ValueError, TypeError) as e:
                    debug_log(f"❌ Prediction {i+1} conversion failed: {str(e)}", "ERROR")
                    continue
            
            debug_log(f"✅ Extracted {len(valid_predictions)} valid predictions", "EXTRACT")
            return valid_predictions
            
        except Exception as e:
            debug_exception(e, "Prediction extraction")
            return []

# ============================================================================
# MINIMAL API HANDLER WITH DEBUGGING
# ============================================================================

class handler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        debug_log("Initializing API handler", "HANDLER")
        self.detector = BrailleDetector()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        debug_log(f"GET request: {self.path}", "REQUEST")
        
        try:
            if self.path == '/' or self.path == '/index.html':
                self.serve_minimal_html()
            elif self.path == '/health':
                self.send_json_response({
                    'status': 'healthy',
                    'roboflow_configured': bool(self.detector.api_key),
                    'detection_endpoint': f"{self.detector.base_url}/{self.detector.workspace_name}/{self.detector.model_version}/predict",
                    'debug_mode': True
                })
            else:
                debug_log(f"❌ 404 - Path not found: {self.path}", "ERROR")
                self.send_json_response({'error': 'Not found'}, 404)
                
        except Exception as e:
            debug_exception(e, "GET handler")
            self.send_json_response({'error': f'GET error: {str(e)}'}, 500)
    
    def do_POST(self):
        debug_log(f"POST request: {self.path}", "REQUEST")
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            debug_log(f"Content length: {content_length}", "REQUEST")
            
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            debug_log(f"Body length: {len(body)}", "REQUEST")
            
            try:
                data = json.loads(body) if body else {}
                debug_log("✅ JSON parsing successful", "REQUEST")
                debug_dict(data, "Request Data")
            except json.JSONDecodeError as e:
                debug_log(f"❌ Invalid JSON: {str(e)}", "ERROR")
                self.send_json_response({'error': 'Invalid JSON'}, 400)
                return
            
            # Route to handler
            if self.path == '/api/detect-braille':
                self.handle_braille_detection(data)
            else:
                debug_log(f"❌ Unknown endpoint: {self.path}", "ERROR")
                self.send_json_response({'error': 'Endpoint not found'}, 404)
                
        except Exception as e:
            debug_exception(e, "POST handler")
            self.send_json_response({'error': f'POST error: {str(e)}'}, 500)
    
    def handle_braille_detection(self, data):
        """Handle braille detection with intense debugging"""
        debug_log("=== HANDLING BRAILLE DETECTION ===", "HANDLER")
        
        try:
            image_data = data.get('image')
            if not image_data:
                debug_log("❌ No image data provided", "ERROR")
                self.send_json_response({'error': 'Image data required'}, 400)
                return
            
            debug_log(f"Image data length: {len(image_data)}", "HANDLER")
            
            # Decode base64 image
            try:
                if image_data.startswith('data:image'):
                    debug_log("Removing data URL prefix", "HANDLER")
                    image_data = image_data.split(',')[1]
                
                debug_log("Decoding base64 image", "HANDLER")
                image_bytes = base64.b64decode(image_data)
                debug_log(f"✅ Image decoded: {len(image_bytes)} bytes", "HANDLER")
                
            except Exception as e:
                debug_exception(e, "Image decoding")
                self.send_json_response({'error': f'Invalid image data: {str(e)}'}, 400)
                return
            
            # Run detection
            debug_log("Starting braille detection", "HANDLER")
            detection_result = self.detector.detect_braille(image_bytes)
            
            if "error" in detection_result:
                debug_log(f"❌ Detection failed: {detection_result['error']}", "ERROR")
                self.send_json_response({
                    'error': detection_result['error'],
                    'debug': True
                })
                return
            
            # Extract predictions
            debug_log("Extracting predictions", "HANDLER")
            predictions = self.detector.extract_predictions(detection_result)
            
            response_data = {
                'predictions': predictions,
                'detection_count': len(predictions),
                'raw_response': detection_result,
                'debug': {
                    'timestamp': datetime.now().isoformat(),
                    'image_size': len(image_bytes),
                    'api_endpoint': f"{self.detector.base_url}/{self.detector.workspace_name}/{self.detector.model_version}/predict"
                }
            }
            
            debug_log(f"✅ Detection complete: {len(predictions)} predictions", "HANDLER")
            debug_dict(response_data, "Final Response")
            
            self.send_json_response(response_data)
            
        except Exception as e:
            debug_exception(e, "Braille detection handler")
            self.send_json_response({
                'error': f'Detection handler error: {str(e)}',
                'debug': True
            }, 500)
    
    def serve_minimal_html(self):
        """Serve minimal HTML interface"""
        debug_log("Serving minimal HTML interface", "HTML")
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Minimal Braille Detection - Debug Mode</title>
    <style>
        body { font-family: monospace; padding: 20px; background: #f0f0f0; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .debug { background: #000; color: #0f0; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #0056b3; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 10px 0; border-left: 4px solid #007bff; }
        .error { border-left-color: #dc3545; }
        input[type="file"] { margin: 10px 0; }
        img { max-width: 300px; max-height: 200px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Minimal Braille Detection - Debug Mode</h1>
        <p>Ultra-minimal interface with maximum debugging output</p>
        
        <div>
            <input type="file" id="imageInput" accept="image/*" onchange="handleImageUpload(event)">
            <button class="btn" onclick="detectBraille()" id="detectBtn" disabled>Detect Braille</button>
            <button class="btn" onclick="checkHealth()">Check System</button>
            <button class="btn" onclick="clearDebug()">Clear Debug</button>
        </div>
        
        <img id="imagePreview" style="display: none;">
        
        <div id="result"></div>
        
        <h3>🐛 Debug Output:</h3>
        <div id="debug" class="debug">Ready for debugging...\n</div>
    </div>

    <script>
        let currentImage = null;
        
        function log(message) {
            const debug = document.getElementById('debug');
            const timestamp = new Date().toISOString();
            debug.innerHTML += `[${timestamp}] ${message}\n`;
            debug.scrollTop = debug.scrollHeight;
            console.log(message);
        }
        
        function handleImageUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            log(`📁 File selected: ${file.name} (${file.size} bytes, ${file.type})`);
            
            if (!file.type.startsWith('image/')) {
                log('❌ ERROR: Not an image file');
                alert('Please select an image file.');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImage = e.target.result;
                log(`✅ Image loaded: ${currentImage.length} characters`);
                
                const preview = document.getElementById('imagePreview');
                preview.src = currentImage;
                preview.style.display = 'block';
                
                document.getElementById('detectBtn').disabled = false;
                log('🎯 Detection button enabled');
            };
            reader.readAsDataURL(file);
        }
        
        async function detectBraille() {
            if (!currentImage) {
                log('❌ ERROR: No image selected');
                alert('Please upload an image first.');
                return;
            }
            
            log('🚀 Starting braille detection...');
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '⏳ Processing...';
            
            try {
                log('📡 Sending POST request to /api/detect-braille');
                log(`📊 Payload size: ${JSON.stringify({image: currentImage}).length} bytes`);
                
                const startTime = Date.now();
                const response = await fetch('/api/detect-braille', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: currentImage })
                });
                const requestTime = Date.now() - startTime;
                
                log(`📡 Response received in ${requestTime}ms (Status: ${response.status})`);
                
                const data = await response.json();
                log('📋 Response data:');
                log(JSON.stringify(data, null, 2));
                
                if (response.ok) {
                    resultDiv.innerHTML = `
                        <div class="result">
                            <h3>✅ Detection Results</h3>
                            <p><strong>Predictions Found:</strong> ${data.detection_count}</p>
                            <p><strong>API Endpoint:</strong> ${data.debug?.api_endpoint || 'N/A'}</p>
                            <p><strong>Image Size:</strong> ${data.debug?.image_size || 'N/A'} bytes</p>
                            <details>
                                <summary>Raw Predictions (${data.predictions.length})</summary>
                                <pre>${JSON.stringify(data.predictions, null, 2)}</pre>
                            </details>
                            <details>
                                <summary>Raw API Response</summary>
                                <pre>${JSON.stringify(data.raw_response, null, 2)}</pre>
                            </details>
                        </div>
                    `;
                    log(`✅ SUCCESS: Found ${data.detection_count} predictions`);
                } else {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>❌ Detection Failed</h3>
                            <p><strong>Error:</strong> ${data.error}</p>
                        </div>
                    `;
                    log(`❌ FAILED: ${data.error}`);
                }
                
            } catch (error) {
                log(`💥 EXCEPTION: ${error.message}`);
                resultDiv.innerHTML = `
                    <div class="result error">
                        <h3>💥 Network Error</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }
        
        async function checkHealth() {
            log('🏥 Checking system health...');
            
            try {
                const response = await fetch('/health');
                const status = await response.json();
                
                log('🏥 Health check response:');
                log(JSON.stringify(status, null, 2));
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `
                    <div class="result">
                        <h3>🏥 System Status</h3>
                        <p><strong>Status:</strong> ${status.status}</p>
                        <p><strong>Roboflow Configured:</strong> ${status.roboflow_configured ? '✅' : '❌'}</p>
                        <p><strong>Detection Endpoint:</strong> ${status.detection_endpoint}</p>
                        <p><strong>Debug Mode:</strong> ${status.debug_mode ? '✅' : '❌'}</p>
                    </div>
                `;
                
            } catch (error) {
                log(`💥 Health check failed: ${error.message}`);
            }
        }
        
        function clearDebug() {
            document.getElementById('debug').innerHTML = 'Debug cleared...\n';
            log('🧹 Debug output cleared');
        }
        
        // Initialize
        window.onload = function() {
            log('🚀 Application initialized');
            checkHealth();
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
        debug_dict(data, "Response Data")
        
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
        # Custom logging
        debug_log(f"HTTP: {format % args}", "HTTP")

# ============================================================================
# STARTUP DEBUG
# ============================================================================

debug_log("🚀 Starting Minimal Braille Detection API with Intense Debugging", "STARTUP")
debug_log(f"Environment variables:", "STARTUP")
debug_log(f"  ROBOFLOW_API_KEY: {'SET' if os.getenv('ROBOFLOW_API_KEY') else 'NOT SET'}", "STARTUP")
debug_log("✅ API ready for requests", "STARTUP")