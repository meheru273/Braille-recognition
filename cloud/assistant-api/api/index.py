# api/index.py - Simplified Braille Detection API
import json
import os
import base64
import requests
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BrailleResult:
    """Result from braille processing"""
    text: str
    explanation: str
    confidence: float

class BrailleDetector:
    """Simplified Braille Detection with debugging"""
    
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        self.workspace_name = "braille-to-text-0xo2p"
        self.model_version = "1"
        self.base_url = "https://detect.roboflow.com"  # Try detect endpoint
        
        print(f"BrailleDetector initialized:")
        print(f"  API Key: {'Present' if self.api_key else 'Missing'}")
        print(f"  Workspace: {self.workspace_name}")
        print(f"  Model Version: {self.model_version}")
    
    def detect_braille_from_bytes(self, image_bytes: bytes) -> Dict:
        """Detect braille with multiple endpoint attempts and detailed debugging"""
        debug_info = {
            "api_key_present": bool(self.api_key),
            "image_size": len(image_bytes),
            "attempts": []
        }
        
        if not self.api_key:
            debug_info["error"] = "ROBOFLOW_API_KEY not configured"
            return {"error": "ROBOFLOW_API_KEY not configured", "debug": debug_info}
        
        print(f"🔍 Starting detection with image size: {len(image_bytes)} bytes")
        
        # Try different endpoints
        endpoints_to_try = [
            f"https://detect.roboflow.com/{self.workspace_name}/{self.model_version}",
            f"https://api.roboflow.com/{self.workspace_name}/{self.model_version}/predict",
            f"https://infer.roboflow.com/{self.workspace_name}/{self.model_version}"
        ]
        
        for endpoint_idx, endpoint in enumerate(endpoints_to_try):
            print(f"\n📡 Attempt {endpoint_idx + 1}/3: {endpoint}")
            
            try:
                # Encode image
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                print(f"   Encoded image length: {len(encoded_image)} chars")
                
                # Try different payload formats
                payloads_to_try = [
                    # Format 1: Standard JSON payload
                    {
                        "api_key": self.api_key,
                        "image": encoded_image,
                        "confidence": 0.1,
                        "overlap": 0.5
                    },
                    # Format 2: With model specification
                    {
                        "api_key": self.api_key,
                        "image": encoded_image,
                        "confidence": 0.05,
                        "overlap": 0.3,
                        "format": "json"
                    },
                    # Format 3: URL parameters
                    None  # Will use URL params
                ]
                
                for payload_idx, payload in enumerate(payloads_to_try):
                    attempt_info = {
                        "endpoint": endpoint,
                        "payload_format": payload_idx + 1,
                        "timestamp": str(datetime.now()) if 'datetime' in globals() else "unknown"
                    }
                    
                    try:
                        print(f"   🔄 Payload format {payload_idx + 1}/3")
                        
                        if payload is None:
                            # Try with URL parameters
                            url_with_params = f"{endpoint}?api_key={self.api_key}&confidence=0.05&overlap=0.3"
                            print(f"      URL: {url_with_params[:100]}...")
                            
                            response = requests.post(
                                url_with_params,
                                data=encoded_image,
                                headers={"Content-Type": "application/x-www-form-urlencoded"},
                                timeout=30
                            )
                            attempt_info["method"] = "URL_PARAMS"
                        else:
                            # Try with JSON payload
                            print(f"      JSON payload with confidence: {payload['confidence']}")
                            response = requests.post(
                                endpoint,
                                json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=30
                            )
                            attempt_info["method"] = "JSON_PAYLOAD"
                        
                        attempt_info["status_code"] = response.status_code
                        attempt_info["response_length"] = len(response.text)
                        
                        print(f"      Status: {response.status_code}")
                        print(f"      Response length: {len(response.text)} chars")
                        
                        if response.status_code == 200:
                            try:
                                result = response.json()
                                predictions = result.get("predictions", [])
                                
                                attempt_info["predictions_count"] = len(predictions)
                                attempt_info["success"] = True
                                
                                print(f"      ✅ JSON parsed successfully")
                                print(f"      Predictions found: {len(predictions)}")
                                
                                if predictions:
                                    # Log first few predictions for debugging
                                    for i, pred in enumerate(predictions[:3]):
                                        print(f"         Pred {i+1}: class='{pred.get('class', 'N/A')}', conf={pred.get('confidence', 0):.3f}")
                                    
                                    debug_info["attempts"] = debug_info.get("attempts", []) + [attempt_info]
                                    result["debug"] = debug_info
                                    print(f"🎉 SUCCESS! Returning {len(predictions)} predictions")
                                    return result
                                else:
                                    print(f"      ⚠️ No predictions in successful response")
                                    attempt_info["issue"] = "no_predictions"
                            except json.JSONDecodeError as e:
                                print(f"      ❌ JSON decode error: {e}")
                                print(f"      Raw response: {response.text[:200]}...")
                                attempt_info["issue"] = f"json_decode_error: {e}"
                        else:
                            print(f"      ❌ HTTP Error {response.status_code}")
                            print(f"      Error response: {response.text[:300]}...")
                            attempt_info["issue"] = f"http_error_{response.status_code}"
                            attempt_info["error_text"] = response.text[:500]
                            
                    except requests.exceptions.Timeout:
                        print(f"      ⏰ Request timeout")
                        attempt_info["issue"] = "timeout"
                    except requests.exceptions.ConnectionError as e:
                        print(f"      🌐 Connection error: {e}")
                        attempt_info["issue"] = f"connection_error: {e}"
                    except Exception as e:
                        print(f"      💥 Unexpected error: {e}")
                        attempt_info["issue"] = f"unexpected_error: {e}"
                    
                    debug_info["attempts"] = debug_info.get("attempts", []) + [attempt_info]
                        
            except Exception as e:
                print(f"   💥 Endpoint {endpoint} completely failed: {e}")
                debug_info["attempts"] = debug_info.get("attempts", []) + [{
                    "endpoint": endpoint,
                    "issue": f"endpoint_failed: {e}"
                }]
        
        print(f"\n❌ All detection methods failed after {len(debug_info['attempts'])} attempts")
        return {"error": "All detection endpoints failed", "debug": debug_info}
    
    def organize_text_by_rows(self, predictions: List[Dict]) -> List[str]:
        """Organize detected characters into readable rows with debugging"""
        print(f"\n📝 Organizing {len(predictions)} predictions into rows")
        
        if not predictions:
            print("   ⚠️ No predictions to organize")
            return []
        
        try:
            # Log all predictions for debugging
            print("   Raw predictions:")
            for i, pred in enumerate(predictions):
                print(f"     {i+1}: class='{pred.get('class', 'N/A')}', x={pred.get('x', 0):.1f}, y={pred.get('y', 0):.1f}, conf={pred.get('confidence', 0):.3f}")
            
            # Sort by Y position first, then X position
            sorted_predictions = sorted(predictions, key=lambda p: (p.get('y', 0), p.get('x', 0)))
            print(f"   Sorted by position (y, x)")
            
            # Group into rows based on Y position
            rows = []
            current_row = []
            last_y = None
            row_threshold = 30  # pixels
            
            print(f"   Grouping into rows (threshold: {row_threshold}px)")
            
            for i, pred in enumerate(sorted_predictions):
                y_pos = pred.get('y', 0)
                x_pos = pred.get('x', 0)
                char_class = pred.get('class', '')
                
                if last_y is None or abs(y_pos - last_y) < row_threshold:
                    # Same row
                    current_row.append(pred)
                    print(f"     Added '{char_class}' to current row (y_diff: {abs(y_pos - (last_y or y_pos)):.1f})")
                else:
                    # New row - process current row first
                    if current_row:
                        # Sort current row by X position and join characters
                        current_row.sort(key=lambda p: p.get('x', 0))
                        row_chars = [p.get('class', '') for p in current_row]
                        row_text = ''.join(row_chars).strip()
                        
                        if row_text:
                            rows.append(row_text)
                            print(f"     ✅ Completed row {len(rows)}: '{row_text}' ({len(current_row)} chars)")
                        else:
                            print(f"     ⚠️ Skipped empty row")
                    
                    # Start new row
                    current_row = [pred]
                    print(f"     🆕 Started new row with '{char_class}' (y_diff: {abs(y_pos - (last_y or 0)):.1f})")
                
                last_y = y_pos
            
            # Process final row
            if current_row:
                current_row.sort(key=lambda p: p.get('x', 0))
                row_chars = [p.get('class', '') for p in current_row]
                row_text = ''.join(row_chars).strip()
                
                if row_text:
                    rows.append(row_text)
                    print(f"     ✅ Final row {len(rows)}: '{row_text}' ({len(current_row)} chars)")
            
            print(f"   📋 Final result: {len(rows)} text rows")
            for i, row in enumerate(rows):
                print(f"     Row {i+1}: '{row}'")
            
            return rows
            
        except Exception as e:
            print(f"   ❌ Error organizing text: {e}")
            print(f"   🔄 Fallback: returning raw character classes")
            # Fallback: just return all classes
            fallback_result = [pred.get('class', '') for pred in predictions if pred.get('class', '').strip()]
            print(f"   📋 Fallback result: {fallback_result}")
            return fallback_result

class BrailleAssistant:
    """Simple braille text processor"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
    def process_braille_strings(self, detected_strings: List[str]) -> BrailleResult:
        """Process detected braille strings"""
        if not detected_strings:
            return BrailleResult(
                text="",
                explanation="No braille characters detected.",
                confidence=0.0
            )
        
        raw_text = " ".join(detected_strings).strip()
        
        # Basic processing without API
        processed_text = self._clean_text(raw_text)
        explanation = f"Detected braille text: '{processed_text}'"
        confidence = 0.7 if processed_text else 0.1
        
        return BrailleResult(
            text=processed_text,
            explanation=explanation,
            confidence=confidence
        )
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Remove extra spaces and clean up
        cleaned = ' '.join(text.split())
        return cleaned

class handler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.detector = BrailleDetector()
        self.assistant = BrailleAssistant()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.serve_html()
        elif path == '/health':
            self.send_json_response({
                'status': 'healthy',
                'roboflow_configured': bool(self.detector.api_key)
            })
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/detect-and-process':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            self.handle_detect_and_process(data)
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_detect_and_process(self, data):
        """Handle detection and processing with detailed debugging"""
        debug_log = []
        
        try:
            debug_log.append("🚀 Starting detect_and_process")
            
            image_data = data.get('image')
            if not image_data:
                debug_log.append("❌ No image data provided")
                self.send_error_response('Image data is required', 400)
                return
            
            debug_log.append(f"📸 Image data received: {len(str(image_data))} chars")
            
            # Decode image
            try:
                if image_data.startswith('data:image'):
                    header, encoded = image_data.split(',', 1)
                    debug_log.append(f"🔍 Image header: {header}")
                    image_data = encoded
                
                image_bytes = base64.b64decode(image_data)
                debug_log.append(f"✅ Image decoded: {len(image_bytes)} bytes")
                
            except Exception as e:
                debug_log.append(f"❌ Image decode failed: {e}")
                self.send_error_response(f'Invalid image data: {str(e)}', 400)
                return
            
            # Run detection
            debug_log.append("🔍 Starting braille detection...")
            detection_result = self.detector.detect_braille_from_bytes(image_bytes)
            
            # Extract debug info from detection
            detection_debug = detection_result.get("debug", {})
            debug_log.append(f"🔍 Detection completed. Attempts made: {len(detection_debug.get('attempts', []))}")
            
            if "error" in detection_result:
                debug_log.append(f"❌ Detection failed: {detection_result['error']}")
                
                # Send detailed debug info in error response
                self.send_json_response({
                    'success': False,
                    'error': detection_result["error"],
                    'detected_strings': [],
                    'processed_text': '',
                    'explanation': f'Detection failed: {detection_result["error"]}',
                    'debug': {
                        'log': debug_log,
                        'detection_debug': detection_debug,
                        'image_size': len(image_bytes),
                        'api_configured': bool(self.detector.api_key)
                    }
                })
                return
            
            # Extract predictions and organize
            predictions = detection_result.get("predictions", [])
            debug_log.append(f"📊 Raw predictions: {len(predictions)}")
            
            # Log prediction details
            if predictions:
                confidence_values = [p.get('confidence', 0) for p in predictions]
                debug_log.append(f"📈 Confidence range: {min(confidence_values):.3f} - {max(confidence_values):.3f}")
                
                classes_found = [p.get('class', '') for p in predictions]
                unique_classes = list(set(classes_found))
                debug_log.append(f"🔤 Unique classes: {unique_classes}")
            
            text_rows = self.detector.organize_text_by_rows(predictions)
            debug_log.append(f"📝 Organized into {len(text_rows)} text rows")
            
            # Process with assistant
            debug_log.append("🤖 Processing with AI assistant...")
            processing_result = self.assistant.process_braille_strings(text_rows)
            debug_log.append(f"✅ AI processing complete. Confidence: {processing_result.confidence:.2f}")
            
            # Prepare comprehensive response
            response_data = {
                'success': True,
                'detected_strings': text_rows,
                'processed_text': processing_result.text,
                'explanation': processing_result.explanation,
                'confidence': processing_result.confidence,
                'detection_count': len(predictions),
                'debug': {
                    'log': debug_log,
                    'detection_debug': detection_debug,
                    'image_size': len(image_bytes),
                    'api_configured': bool(self.detector.api_key),
                    'predictions_sample': predictions[:5] if predictions else [],  # First 5 predictions
                    'text_organization': {
                        'raw_predictions': len(predictions),
                        'organized_rows': len(text_rows),
                        'processing_confidence': processing_result.confidence
                    }
                }
            }
            
            debug_log.append("📤 Sending successful response")
            print("\n".join(debug_log))  # Print to server console
            
            self.send_json_response(response_data)
            
        except Exception as e:
            debug_log.append(f"💥 Unexpected error: {str(e)}")
            print("\n".join(debug_log))  # Print to server console
            
            import traceback
            debug_log.append(f"📋 Traceback: {traceback.format_exc()}")
            
            self.send_json_response({
                'success': False,
                'error': f'Processing error: {str(e)}',
                'detected_strings': [],
                'processed_text': '',
                'explanation': f'System error occurred: {str(e)}',
                'debug': {
                    'log': debug_log,
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
            })
    
    def serve_html(self):
        """Serve simple HTML interface"""
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
            background: #f0f2f5;
            padding: 20px;
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px; 
            padding: 30px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        h1 { 
            text-align: center; 
            color: #2c3e50; 
            margin-bottom: 30px;
            font-size: 2.2em;
        }
        .status { 
            background: #d4edda; 
            border: 1px solid #c3e6cb; 
            border-radius: 8px;
            padding: 12px; 
            margin-bottom: 20px; 
            text-align: center;
        }
        .status.error {
            background: #f8d7da; 
            border-color: #f5c6cb;
            color: #721c24;
        }
        .upload-area {
            border: 3px dashed #3498db; 
            border-radius: 12px; 
            padding: 50px 20px;
            text-align: center; 
            cursor: pointer; 
            margin-bottom: 20px;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }
        .upload-area:hover { 
            background: #e3f2fd; 
            border-color: #2196f3;
        }
        .upload-area.dragover { 
            background: #bbdefb; 
            border-color: #1976d2; 
        }
        .upload-text {
            font-size: 18px;
            color: #34495e;
            margin-bottom: 10px;
        }
        .upload-hint {
            font-size: 14px;
            color: #7f8c8d;
        }
        .image-preview { 
            max-width: 100%; 
            max-height: 300px; 
            margin: 20px auto; 
            display: block; 
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .btn { 
            background: linear-gradient(135deg, #3498db, #2980b9); 
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 16px; 
            font-weight: 600;
            width: 100%;
            margin-top: 15px;
            transition: all 0.3s ease;
        }
        .btn:hover { 
            background: linear-gradient(135deg, #2980b9, #1f639a);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        }
        .btn:disabled { 
            background: #bdc3c7; 
            cursor: not-allowed; 
            transform: none;
            box-shadow: none;
        }
        .loading { 
            display: none; 
            text-align: center; 
            color: #3498db; 
            font-weight: 600; 
            margin: 20px 0;
            font-size: 16px;
        }
        .result { 
            margin-top: 25px; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 10px; 
            border-left: 5px solid #3498db;
        }
        .detected-strings {
            margin: 15px 0;
        }
        .detected-strings h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .string-item {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 10px;
            margin: 8px 0;
            font-family: 'Courier New', monospace;
            font-size: 16px;
            border-left: 4px solid #f39c12;
        }
        .processing-result {
            margin-top: 20px;
            padding: 15px;
            background: #e8f5e8;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
        }
        .error-message {
            color: #e74c3c;
            background: #fadbd8;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
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
            <div class="upload-text">📸 Click to upload braille image</div>
            <div class="upload-hint">Or drag and drop your image here</div>
            <input type="file" id="imageInput" accept="image/*" style="display: none;" onchange="handleImageUpload(event)">
        </div>
        
        <img id="imagePreview" class="image-preview" style="display: none;">
        
        <button class="btn" onclick="processImage()" id="processBtn" disabled>
            🔍 Detect Braille Text
        </button>
        
        <div class="loading" id="loading">
            <div>🔄 Processing your image...</div>
            <div style="font-size: 14px; margin-top: 5px;">This may take a few seconds</div>
        </div>
        
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

                if (data.success) {
                    let outputHTML = '';
                    
                    // Show detected strings
                    if (data.detected_strings && data.detected_strings.length > 0) {
                        outputHTML += `
                            <div class="detected-strings">
                                <h4>📝 Detected Braille Text:</h4>
                                ${data.detected_strings.map(str => 
                                    `<div class="string-item">${str}</div>`
                                ).join('')}
                            </div>
                        `;
                    }
                    
                    // Show processing results
                    outputHTML += `
                        <div class="processing-result">
                            <h4>🤖 Processed Result:</h4>
                            <p><strong>Text:</strong> ${data.processed_text || 'No text processed'}</p>
                            <p><strong>Explanation:</strong> ${data.explanation}</p>
                            <p><strong>Detection Count:</strong> ${data.detection_count} characters</p>
                            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                        </div>
                    `;
                    
                    // Add debug information
                    if (data.debug) {
                        outputHTML += `
                            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #6c757d;">
                                <h4>🔧 Debug Information:</h4>
                                <details style="margin-top: 10px;">
                                    <summary style="cursor: pointer; font-weight: bold;">Click to expand debug details</summary>
                                    <div style="margin-top: 10px; font-family: monospace; font-size: 12px;">
                                        <p><strong>Image Size:</strong> ${data.debug.image_size} bytes</p>
                                        <p><strong>API Configured:</strong> ${data.debug.api_configured ? 'Yes' : 'No'}</p>
                                        <p><strong>Detection Attempts:</strong> ${data.debug.detection_debug?.attempts?.length || 0}</p>
                                        
                                        ${data.debug.predictions_sample && data.debug.predictions_sample.length > 0 ? `
                                            <p><strong>Sample Predictions:</strong></p>
                                            <ul style="margin-left: 20px;">
                                                ${data.debug.predictions_sample.map(pred => 
                                                    `<li>Class: "${pred.class}", Confidence: ${(pred.confidence * 100).toFixed(1)}%, Position: (${pred.x.toFixed(1)}, ${pred.y.toFixed(1)})</li>`
                                                ).join('')}
                                            </ul>
                                        ` : ''}
                                        
                                        <p><strong>Processing Log:</strong></p>
                                        <ul style="margin-left: 20px; max-height: 200px; overflow-y: auto;">
                                            ${data.debug.log.map(entry => `<li>${entry}</li>`).join('')}
                                        </ul>
                                        
                                        ${data.debug.detection_debug?.attempts ? `
                                            <p><strong>Detection Attempts:</strong></p>
                                            <ul style="margin-left: 20px; max-height: 150px; overflow-y: auto;">
                                                ${data.debug.detection_debug.attempts.map((attempt, i) => 
                                                    `<li>Attempt ${i+1}: ${attempt.endpoint} - ${attempt.method || 'N/A'} - Status: ${attempt.status_code || 'N/A'} - ${attempt.issue || 'Success'}</li>`
                                                ).join('')}
                                            </ul>
                                        ` : ''}
                                    </div>
                                </details>
                            </div>
                        `;
                    }
                    
                    outputDiv.innerHTML = outputHTML;
                } else {
                    let errorHTML = `
                        <div class="error-message">
                            <h4>❌ Detection Failed</h4>
                            <p>${data.error}</p>
                        </div>
                    `;
                    
                    // Add debug information for errors too
                    if (data.debug) {
                        errorHTML += `
                            <div style="margin-top: 15px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                                <h4>🔧 Debug Information:</h4>
                                <details style="margin-top: 10px;">
                                    <summary style="cursor: pointer; font-weight: bold;">Click to see what went wrong</summary>
                                    <div style="margin-top: 10px; font-family: monospace; font-size: 12px;">
                                        <p><strong>Image Size:</strong> ${data.debug.image_size || 'Unknown'} bytes</p>
                                        <p><strong>API Configured:</strong> ${data.debug.api_configured ? 'Yes' : 'No'}</p>
                                        
                                        <p><strong>Debug Log:</strong></p>
                                        <ul style="margin-left: 20px; max-height: 200px; overflow-y: auto;">
                                            ${data.debug.log?.map(entry => `<li>${entry}</li>`).join('') || '<li>No debug log available</li>'}
                                        </ul>
                                        
                                        ${data.debug.detection_debug?.attempts ? `
                                            <p><strong>Detection Attempts:</strong></p>
                                            <ul style="margin-left: 20px; max-height: 150px; overflow-y: auto;">
                                                ${data.debug.detection_debug.attempts.map((attempt, i) => 
                                                    `<li>Attempt ${i+1}: ${attempt.endpoint} - ${attempt.method || 'N/A'} - Status: ${attempt.status_code || 'Failed'} - Issue: ${attempt.issue || 'Unknown'}</li>`
                                                ).join('')}
                                            </ul>
                                        ` : ''}
                                        
                                        ${data.debug.error_type ? `<p><strong>Error Type:</strong> ${data.debug.error_type}</p>` : ''}
                                    </div>
                                </details>
                            </div>
                        `;
                    }
                    
                    outputDiv.innerHTML = errorHTML;
                } Detection Failed</h4>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
                
                resultDiv.style.display = 'block';
                
            } catch (error) {
                outputDiv.innerHTML = `
                    <div class="error-message">
                        <h4>❌ Network Error</h4>
                        <p>${error.message}</p>
                    </div>
                `;
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
                    statusText.textContent = '❌ Roboflow API key missing - Detection disabled';
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
        pass