# api/index.py - Braille Detection API using inference_sdk
import json
import os
import base64
import requests
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any, Optional
import sys
import traceback

# Try to import inference_sdk - if it fails, we'll use HTTP requests
try:
    from inference_sdk import InferenceHTTPClient
    INFERENCE_SDK_AVAILABLE = True
    print("✅ inference_sdk available - using optimized detection")
except ImportError:
    INFERENCE_SDK_AVAILABLE = False
    print("⚠️ inference_sdk not available - using HTTP requests")

# ============================================================================
# BRAILLE DETECTOR CLASSES
# ============================================================================

class BrailleDetector:
    """Braille Detection using inference_sdk or HTTP requests"""
    
    def __init__(self):
        # Get API key from environment
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            print("ERROR: ROBOFLOW_API_KEY environment variable not set")
            print("Please set your Roboflow API key:")
            print("1. Go to https://roboflow.com/account")
            print("2. Copy your API key")
            print("3. Set ROBOFLOW_API_KEY environment variable")
        else:
            print(f"✅ Using Roboflow API key: {self.api_key[:5]}...{self.api_key[-5:]}")
            
        self.workspace_name = "braille-to-text-0xo2p"
        self.workflow_id = "custom-workflow"
        self.api_url = "https://serverless.roboflow.com"
        
        # Initialize inference client if available
        self.client = None
        if INFERENCE_SDK_AVAILABLE and self.api_key:
            try:
                self.client = InferenceHTTPClient(
                    api_url=self.api_url,
                    api_key=self.api_key
                )
                print("✅ Inference client initialized successfully")
            except Exception as e:
                print(f"⚠️ Failed to initialize inference client: {e}")
                self.client = None
    
    def _encode_image_from_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        try:
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')
            return encoded_string
        except Exception as e:
            raise Exception(f"Failed to encode image: {e}")
    
    def detect_braille_from_bytes(self, image_bytes: bytes) -> Optional[Dict]:
        """
        Run Braille detection using image bytes.
        Uses inference_sdk if available, otherwise HTTP requests.
        """
        print("--- START detect_braille_from_bytes ---")
        
        if not self.api_key:
            error_msg = "Roboflow API key not configured. Please set ROBOFLOW_API_KEY environment variable."
            print(f"ERROR: {error_msg}")
            return {"error": "Detection configuration error", "detail": error_msg}

        try:
            print(f"1. Processing image bytes (size: {len(image_bytes)} bytes)...")
            
            # Use inference_sdk if available
            if self.client and INFERENCE_SDK_AVAILABLE:
                print("2. Using inference_sdk...")
                return self._detect_with_inference_sdk(image_bytes)
            else:
                print("2. Using HTTP requests (inference_sdk not available)...")
                return self._detect_with_http_requests(image_bytes)

        except Exception as e:
            error_msg = f"Unexpected error in detect_braille_from_bytes: {e}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            return {"error": "Detection internal error", "detail": error_msg}
    
    def _detect_with_inference_sdk(self, image_bytes: bytes) -> Dict:
        """Use inference_sdk for detection"""
        try:
            # Convert bytes to base64 string
            encoded_image = self._encode_image_from_bytes(image_bytes)
            print(f"   -> Encoded image length: {len(encoded_image)} characters")
            
            # Run workflow using inference_sdk
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={
                    "image": encoded_image
                },
                use_cache=True
            )
            
            print(f"3. Inference SDK Response:")
            print(f"   -> Response type: {type(result)}")
            if isinstance(result, dict):
                print(f"   -> Response keys: {list(result.keys())}")
            elif isinstance(result, list):
                print(f"   -> Response list length: {len(result)}")
            
            print("--- END detect_braille_from_bytes (SUCCESS with inference_sdk) ---")
            return result
            
        except Exception as e:
            print(f"ERROR in inference_sdk detection: {e}")
            # Fallback to HTTP requests
            print("   -> Falling back to HTTP requests...")
            return self._detect_with_http_requests(image_bytes)
    
    def _detect_with_http_requests(self, image_bytes: bytes) -> Dict:
        """Use HTTP requests for detection (fallback)"""
        try:
            encoded_image = self._encode_image_from_bytes(image_bytes)
            print(f"   -> Encoded image length: {len(encoded_image)} characters")

            # Use the correct endpoint structure for workflows
            url = f"{self.api_url}/{self.workspace_name}/workflows/{self.workflow_id}"
            
            headers = {"Content-Type": "application/json"}
            
            # Use the correct payload structure for workflows
            payload = {
                "images": {
                    "image": encoded_image
                },
                "use_cache": True
            }
            
            params = {"api_key": self.api_key}

            print(f"3. Making HTTP request to: {url}")
            masked_key = self.api_key[:5] + '*' * (len(self.api_key) - 10) + self.api_key[-5:] if len(self.api_key) > 10 else '*' * len(self.api_key)
            print(f"   -> API Key: {masked_key}")

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                params=params,
                timeout=30
            )
            
            print(f"4. HTTP Response:")
            print(f"   -> Status Code: {response.status_code}")

            if response.status_code == 200:
                result_data = response.json()
                print(f"   -> Successfully parsed response")
                print(f"   -> Response type: {type(result_data)}")
                if isinstance(result_data, dict):
                    print(f"   -> Response keys: {list(result_data.keys())}")
                elif isinstance(result_data, list):
                    print(f"   -> Response list length: {len(result_data)}")
                print("--- END detect_braille_from_bytes (SUCCESS with HTTP) ---")
                return result_data
            else:
                error_detail = response.text
                print(f"   -> Error response: {error_detail[:500]}...")
                
                try:
                    error_json = response.json()
                    error_detail = error_json.get('message', error_json.get('error', error_detail))
                except:
                    pass

                final_error_msg = f"Roboflow API returned status {response.status_code}. Details: {error_detail}"
                print(f"ERROR: {final_error_msg}")
                return {
                    "error": "Detection failed",
                    "detail": final_error_msg,
                    "status_code": response.status_code
                }

        except Exception as e:
            error_msg = f"HTTP request error: {e}"
            print(f"ERROR: {error_msg}")
            return {"error": "HTTP request failed", "detail": error_msg}
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions with robust error handling"""
        print(f"--- START extract_predictions ---")
        print(f"Input result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Input result keys: {list(result.keys())}")
        elif isinstance(result, list):
            print(f"Input result length: {len(result)}")
        
        if not result or "error" in result:
            print("No result or error in result")
            print("--- END extract_predictions (EMPTY) ---")
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
            
            print(f"Extracted {len(valid_predictions)} valid predictions")
            print("--- END extract_predictions (SUCCESS) ---")
            return valid_predictions
            
        except Exception as e:
            print(f"Error in extract_predictions: {e}")
            print("--- END extract_predictions (ERROR) ---")
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.1) -> List[str]:
        """Organize detected characters into rows"""
        print(f"--- START organize_text_by_rows ---")
        print(f"Input predictions count: {len(predictions)}")
        print(f"Min confidence threshold: {min_confidence}")
        
        if not predictions:
            print("No predictions to organize")
            print("--- END organize_text_by_rows (EMPTY) ---")
            return []
        
        try:
            # Filter by confidence
            print(f"Confidence levels in predictions:")
            for i, pred in enumerate(predictions[:5]):  # Show first 5
                print(f"  Prediction {i}: confidence={pred.get('confidence', 0)}, class={pred.get('class', 'unknown')}")
            
            filtered_predictions = [
                pred for pred in predictions 
                if pred.get('confidence', 0) >= min_confidence
            ]
            
            print(f"Predictions after confidence filtering: {len(filtered_predictions)}")
            
            if not filtered_predictions:
                print("No predictions meet confidence threshold")
                print("--- END organize_text_by_rows (NO CONFIDENT PREDICTIONS) ---")
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
            
            print(f"Organized into {len(rows)} text rows")
            print("--- END organize_text_by_rows (SUCCESS) ---")
            return rows
            
        except Exception as e:
            print(f"Error in organize_text_by_rows: {e}")
            print("--- END organize_text_by_rows (ERROR) ---")
            return []

# ============================================================================
# API HANDLER
# ============================================================================

class handler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Initialize detector
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
                    'features': ['braille_detection'],
                    'roboflow_configured': bool(self.detector.api_key),
                    'inference_sdk_available': INFERENCE_SDK_AVAILABLE,
                    'inference_client_ready': bool(self.detector.client),
                    'api_key_length': len(self.detector.api_key) if self.detector.api_key else 0
                })
            elif path == '/test-api':
                if not self.detector.api_key:
                    self.send_json_response({
                        'error': 'API key not configured',
                        'message': 'Please set ROBOFLOW_API_KEY environment variable'
                    }, 400)
                else:
                    self.send_json_response({
                        'status': 'API key configured',
                        'key_length': len(self.detector.api_key),
                        'workspace': self.detector.workspace_name,
                        'workflow': self.detector.workflow_id,
                        'inference_sdk_available': INFERENCE_SDK_AVAILABLE,
                        'inference_client_ready': bool(self.detector.client),
                        'api_url': self.detector.api_url
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
            if path == '/api/detect-braille':
                self.handle_braille_detection(data)
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())
                
        except Exception as e:
            self.send_error_response(f"POST error: {str(e)}")
    
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
                'detection_count': len(predictions),
                'detection_method': 'inference_sdk' if INFERENCE_SDK_AVAILABLE and self.detector.client else 'http_requests'
            })
            
        except Exception as e:
            self.send_error_response(f'Braille detection error: {str(e)}')
    
    def serve_html(self):
        """Serve the web interface"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Braille Detection API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container { 
            max-width: 800px; margin: 0 auto; background: white; 
            border-radius: 15px; padding: 30px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header { text-align: center; margin-bottom: 30px; color: #333; }
        .header h1 { 
            font-size: 2.5em; margin-bottom: 10px; 
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .upload-area {
            border: 2px dashed #667eea; border-radius: 8px; padding: 40px;
            text-align: center; cursor: pointer; transition: all 0.3s;
        }
        .upload-area:hover { background: #f0f7ff; }
        .upload-area.dragover { background: #e6f3ff; border-color: #4a90e2; }
        .image-preview { max-width: 300px; max-height: 200px; margin: 10px auto; display: block; }
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
        .loading { display: none; text-align: center; color: #667eea; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔤 Braille Detection API</h1>
            <p>AI-powered braille detection using Roboflow</p>
        </div>

        <div class="upload-area" onclick="document.getElementById('imageInput').click()" 
             ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
            <p>📸 Click to upload or drag & drop braille image</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;" onchange="handleImageUpload(event)">
        </div>
        <img id="imagePreview" class="image-preview" style="display: none;">
        <button class="btn" onclick="detectBraille()" id="detectBtn" disabled>🔍 Detect Braille</button>
        <div class="loading" id="detection-loading">Processing image...</div>
        <div id="detection-result" class="result" style="display: none;">
            <h3>Detection Results:</h3>
            <div id="detection-output"></div>
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
                
                document.getElementById('detectBtn').disabled = false;
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
                            <strong>Detection Method:</strong> ${data.detection_method}
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

        // Check system status on load
        window.onload = async function() {
            try {
                const response = await fetch('/health');
                const status = await response.json();
                
                if (!status.roboflow_configured) {
                    document.getElementById('detectBtn').title = "Roboflow API key not configured";
                    document.getElementById('detectBtn').disabled = true;
                    
                    // Show warning message
                    const warningDiv = document.createElement('div');
                    warningDiv.style.cssText = 'background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin-bottom: 20px;';
                    warningDiv.innerHTML = `
                        <strong>⚠️ Configuration Required:</strong><br>
                        Roboflow API key not configured. Please set the ROBOFLOW_API_KEY environment variable.<br>
                        <a href="https://roboflow.com/account" target="_blank">Get your API key here</a>
                    `;
                    document.querySelector('.container').insertBefore(warningDiv, document.querySelector('.upload-area'));
                } else {
                    const sdkStatus = status.inference_sdk_available ? '✅ Available' : '⚠️ Not available';
                    const clientStatus = status.inference_client_ready ? '✅ Ready' : '⚠️ Not ready';
                    
                    const statusDiv = document.createElement('div');
                    statusDiv.style.cssText = 'background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 14px;';
                    statusDiv.innerHTML = `
                        <strong>🔧 System Status:</strong><br>
                        • Roboflow API: ✓ Configured (Key length: ${status.api_key_length})<br>
                        • Inference SDK: ${sdkStatus}<br>
                        • Inference Client: ${clientStatus}
                    `;
                    document.querySelector('.container').insertBefore(statusDiv, document.querySelector('.upload-area'));
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

