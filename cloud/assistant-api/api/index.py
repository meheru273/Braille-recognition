# api/index.py - Fixed Braille Detection System for Vercel (No SDK Dependencies)
"""
Braille Detection System - Fixed for Vercel serverless deployment
Uses direct HTTP requests instead of inference_sdk
"""

from flask import Flask, request, jsonify, render_template_string
import json
import os
import base64
import requests
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import io
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BrailleResult:
    text: str
    explanation: str
    confidence: float

@dataclass
class CompleteResult:
    success: bool
    detected_text: str
    explanation: str
    confidence: float
    detection_count: int
    timestamp: datetime
    total_time_ms: int
    error: Optional[str] = None

# ============================================================================
# AI ASSISTANT
# ============================================================================

class SimpleAssistant:
    def __init__(self, api_key: str = None):
        logger.info("Initializing SimpleAssistant")
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            if self.api_key.startswith("gsk_"):
                self.base_url = "https://api.groq.com/openai/v1"
                self.model = "llama-3.1-8b-instant"
                self.provider = "groq"
            else:
                self.base_url = "https://api.openai.com/v1"
                self.model = "gpt-3.5-turbo"
                self.provider = "openai"
            logger.info(f"✅ AI API configured: {self.provider}")
        else:
            self.provider = "fallback"
            logger.info("⚠️ No AI API key - using fallback mode")
    
    def process_braille_strings(self, detected_strings: List[str]) -> BrailleResult:
        logger.info(f"Processing {len(detected_strings)} braille strings")
        
        if not detected_strings:
            return BrailleResult("", "No braille characters detected.", 0.0)
        
        try:
            raw_text = " ".join(detected_strings).strip()
            
            if self.api_key:
                processed_text = self._process_with_api(raw_text)
                explanation = self._get_explanation(processed_text)
                confidence = 0.8
            else:
                processed_text = self._fallback_process(raw_text)
                explanation = f"Basic processing: {processed_text} (No AI API available)"
                confidence = 0.5
            
            return BrailleResult(processed_text, explanation, confidence)
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            fallback_text = " ".join(detected_strings)
            return BrailleResult(fallback_text, f"Error processing: {str(e)}", 0.3)
    
    def _process_with_api(self, text: str) -> str:
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
            
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return self._fallback_process(text)
                
        except Exception as e:
            logger.error(f"AI API exception: {str(e)}")
            return self._fallback_process(text)
    
    def _fallback_process(self, text: str) -> str:
        return text.strip()
    
    def _get_explanation(self, text: str) -> str:
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
                
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=8)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                    
            except Exception as e:
                logger.error(f"Explanation API error: {str(e)}")
        
        return f"This appears to be braille text that reads: '{text}'"

# ============================================================================
# BRAILLE DETECTOR - DIRECT HTTP VERSION
# ============================================================================

class BrailleDetector:
    def __init__(self):
        logger.info("Initializing BrailleDetector with direct HTTP requests")
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        self.workspace_name = "braille-to-text-0xo2p"
        self.model_version = "1"
        
        # Different API endpoints to try
        self.endpoints = [
            f"https://detect.roboflow.com/{self.workspace_name}/{self.model_version}",
            f"https://api.roboflow.com/{self.workspace_name}/{self.model_version}/predict"
        ]
        
        if not self.api_key:
            logger.warning("❌ NO ROBOFLOW API KEY FOUND")
        else:
            logger.info("✅ Roboflow API key configured")
    
    def detect_braille(self, image_bytes: bytes) -> Dict:
        logger.info("=== STARTING BRAILLE DETECTION ===")
        
        if not self.api_key:
            return {"error": "ROBOFLOW_API_KEY not configured"}
        
        try:
            # Convert image bytes to base64
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            logger.info("Image encoded to base64")
            
            # Try different API endpoints
            for i, endpoint in enumerate(self.endpoints):
                try:
                    logger.info(f"Attempting endpoint {i+1}: {endpoint}")
                    result = self._try_endpoint(endpoint, encoded_image)
                    if result and "error" not in result:
                        logger.info(f"✅ Success with endpoint {i+1}")
                        return result
                    elif "error" in result:
                        logger.warning(f"Endpoint {i+1} returned error: {result['error']}")
                except Exception as e:
                    logger.warning(f"Endpoint {i+1} failed: {str(e)}")
                    continue
            
            # If all endpoints failed
            return {"error": "All API endpoints failed"}
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return {"error": f"Detection failed: {str(e)}"}
    
    def _try_endpoint(self, endpoint: str, encoded_image: str) -> Dict:
        """Try a specific API endpoint"""
        
        # Method 1: detect.roboflow.com format
        if "detect.roboflow.com" in endpoint:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            params = {
                "api_key": self.api_key,
                "confidence": 0.3,
                "overlap": 0.5
            }
            response = requests.post(endpoint, params=params, data=encoded_image, headers=headers, timeout=25)
            
        # Method 2: api.roboflow.com format  
        else:
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "api_key": self.api_key,
                "image": encoded_image,
                "confidence": 0.3,
                "overlap": 0.5
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=25)
        
        if response.status_code == 200:
            result = response.json()
            if "predictions" in result:
                return result
            else:
                return {"error": f"Unexpected response format: {result}"}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        if not result or "error" in result:
            logger.warning("No valid result to extract predictions from")
            return []
        
        try:
            predictions = result.get("predictions", [])
            logger.info(f"Extracted {len(predictions)} predictions")
            
            valid_predictions = []
            for pred in predictions:
                if isinstance(pred, dict) and all(key in pred for key in ['x', 'y', 'width', 'height', 'confidence', 'class']):
                    try:
                        valid_predictions.append({
                            'x': float(pred['x']),
                            'y': float(pred['y']),
                            'width': float(pred['width']),
                            'height': float(pred['height']),
                            'confidence': float(pred['confidence']),
                            'class': str(pred['class']).strip()
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid prediction data: {pred}, error: {e}")
                        continue
                else:
                    logger.warning(f"Missing required keys in prediction: {pred}")
            
            logger.info(f"Validated {len(valid_predictions)} predictions")
            return valid_predictions
            
        except Exception as e:
            logger.error(f"Error extracting predictions: {str(e)}")
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict]) -> List[str]:
        if not predictions:
            logger.info("No predictions to organize")
            return []
        
        try:
            # Filter by confidence
            min_confidence = 0.3
            filtered_predictions = [pred for pred in predictions if pred['confidence'] >= min_confidence]
            logger.info(f"Filtered to {len(filtered_predictions)} predictions with confidence >= {min_confidence}")
            
            if not filtered_predictions:
                return []
            
            # Sort by Y coordinate (top to bottom)
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p['y'])
            
            # Group into rows
            rows = []
            current_group = [sorted_by_y[0]]
            
            for i in range(1, len(sorted_by_y)):
                current_pred = sorted_by_y[i]
                prev_pred = sorted_by_y[i-1]
                
                # Calculate threshold for same row
                avg_height = (current_pred['height'] + prev_pred['height']) / 2
                threshold = max(8, avg_height * 0.7)
                y_diff = abs(current_pred['y'] - prev_pred['y'])
                
                if y_diff <= threshold:
                    # Same row
                    current_group.append(current_pred)
                else:
                    # New row
                    if current_group:
                        # Sort current group by X coordinate (left to right)
                        current_group.sort(key=lambda p: p['x'])
                        row_text = ''.join([p['class'] for p in current_group])
                        if row_text.strip():
                            rows.append(row_text)
                    current_group = [current_pred]
            
            # Handle last group
            if current_group:
                current_group.sort(key=lambda p: p['x'])
                row_text = ''.join([p['class'] for p in current_group])
                if row_text.strip():
                    rows.append(row_text)
            
            logger.info(f"Organized into {len(rows)} text rows: {rows}")
            return rows
            
        except Exception as e:
            logger.error(f"Error organizing text: {str(e)}")
            return []

# ============================================================================
# CONTROLLER
# ============================================================================

class BrailleController:
    def __init__(self):
        self.detector = BrailleDetector()
        self.assistant = SimpleAssistant()
        logger.info("BrailleController initialized")
    
    def detect_and_process(self, image_bytes: bytes) -> CompleteResult:
        start_time = datetime.now()
        logger.info("Starting complete detection and processing pipeline")
        
        try:
            # Step 1: Detection
            detection_result = self.detector.detect_braille(image_bytes)
            
            if "error" in detection_result:
                logger.error(f"Detection failed: {detection_result['error']}")
                return CompleteResult(
                    success=False,
                    detected_text="",
                    explanation=f"Detection failed: {detection_result['error']}",
                    confidence=0.0,
                    detection_count=0,
                    timestamp=datetime.now(),
                    total_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    error=detection_result['error']
                )
            
            # Step 2: Extract predictions
            predictions = self.detector.extract_predictions(detection_result)
            
            # Step 3: Organize text
            text_rows = self.detector.organize_text_by_rows(predictions)
            
            # Step 4: AI Processing
            if text_rows:
                braille_result = self.assistant.process_braille_strings(text_rows)
                detected_text = braille_result.text
                explanation = braille_result.explanation
                confidence = braille_result.confidence
                logger.info(f"Successfully processed: '{detected_text}'")
            else:
                detected_text = ""
                explanation = "No braille characters detected in the image."
                confidence = 0.0
                logger.info("No braille text detected")
            
            total_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return CompleteResult(
                success=True,
                detected_text=detected_text,
                explanation=explanation,
                confidence=confidence,
                detection_count=len(predictions),
                timestamp=datetime.now(),
                total_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            logger.error(traceback.format_exc())
            return CompleteResult(
                success=False,
                detected_text="",
                explanation=f"Processing error: {str(e)}",
                confidence=0.0,
                detection_count=0,
                timestamp=datetime.now(),
                total_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                error=str(e)
            )

# Global controller instance
controller = BrailleController()

# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Braille Recognition System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .upload-section {
            border: 2px dashed #ccc;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        .upload-btn:hover {
            transform: translateY(-2px);
        }
        .file-input { display: none; }
        .status {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
        .status.loading { background: #fff3cd; color: #856404; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }
        .detected-text {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 15px;
            font-family: monospace;
            font-size: 1.2em;
        }
        .explanation {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
        }
        .hidden { display: none; }
        .preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .stats {
            background: #f1f3f4;
            padding: 10px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔤 Braille Recognition</h1>
            <p>Upload an image containing braille text for recognition</p>
        </div>
        
        <div class="upload-section">
            <p>📁 Drop an image here or click to browse</p>
            <br>
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                Choose File
            </button>
            <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFile(event)">
        </div>
        
        <div id="status" class="status hidden"></div>
        <div id="preview" class="hidden">
            <img id="previewImage" class="preview" alt="Preview">
        </div>
        <div id="results" class="results hidden">
            <div>
                <h3>🔤 Detected Text</h3>
                <div id="detectedText" class="detected-text"></div>
            </div>
            <div>
                <h3>💡 Explanation</h3>
                <div id="explanation" class="explanation"></div>
            </div>
            <div id="stats" class="stats"></div>
        </div>
    </div>

    <script>
        function handleFile(event) {
            const file = event.target.files[0];
            if (file && file.type.startsWith('image/')) {
                processFile(file);
            } else {
                showStatus('Please select an image file', 'error');
            }
        }

        function processFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('previewImage').src = e.target.result;
                document.getElementById('preview').classList.remove('hidden');
            };
            reader.readAsDataURL(file);

            showStatus('Processing image...', 'loading');
            
            const formData = new FormData();
            formData.append('image', file);

            fetch('/api/detect-and-process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayResults(data);
                    showStatus('Processing complete!', 'success');
                } else {
                    showStatus('Error: ' + (data.error || 'Unknown error'), 'error');
                }
            })
            .catch(error => {
                showStatus('Network error: ' + error.message, 'error');
            });
        }

        function displayResults(data) {
            document.getElementById('detectedText').textContent = data.detected_text || 'No text detected';
            document.getElementById('explanation').textContent = data.explanation || 'No explanation available';
            
            const stats = `Detected ${data.detection_count} characters • Confidence: ${(data.confidence * 100).toFixed(1)}% • Processing time: ${data.total_time_ms}ms`;
            document.getElementById('stats').textContent = stats;
            
            document.getElementById('results').classList.remove('hidden');
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.classList.remove('hidden');
            
            if (type === 'success') {
                setTimeout(() => status.classList.add('hidden'), 3000);
            }
        }

        const uploadSection = document.querySelector('.upload-section');
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#667eea';
        });
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.style.borderColor = '#ccc';
        });
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#ccc';
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                processFile(files[0]);
            }
        });
    </script>
</body>
</html>'''

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect-and-process', methods=['POST'])
def detect_and_process():
    """API endpoint for braille detection and processing"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'}), 400
        
        image_bytes = image_file.read()
        result = controller.detect_and_process(image_bytes)
        
        response_data = {
            'success': result.success,
            'detected_text': result.detected_text,
            'explanation': result.explanation,
            'confidence': result.confidence,
            'detection_count': result.detection_count,
            'total_time_ms': result.total_time_ms,
            'timestamp': result.timestamp.isoformat()
        }
        
        if result.error:
            response_data['error'] = result.error
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'roboflow_configured': bool(os.getenv("ROBOFLOW_API_KEY")),
        'ai_configured': bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))
    })

# For local development
if __name__ == '__main__':
    app.run(debug=True)