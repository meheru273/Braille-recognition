# api/index.py - Simplified Braille Detection System
"""
Simplified Braille Detection System - Detection to Assistant path only
All components included in a single file for Vercel deployment
"""

import json
import os
import base64
import requests
from http.server import BaseHTTPRequestHandler
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BrailleResult:
    """Result from braille processing"""
    text: str
    explanation: str
    confidence: float

@dataclass
class CompleteResult:
    """Combined result from detection + processing"""
    success: bool
    detected_text: str
    explanation: str
    confidence: float
    detection_count: int
    timestamp: datetime
    total_time_ms: int
    error: Optional[str] = None

# ============================================================================
# AI ASSISTANT (Embedded)
# ============================================================================

class SimpleAssistant:
    """AI Assistant with API and fallback capabilities"""
    
    def __init__(self, api_key: str = None):
        logger.info("Initializing SimpleAssistant")
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.conversation_memory = {}
        
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
        """Process braille strings"""
        logger.info(f"Processing {len(detected_strings)} braille strings")
        
        if not detected_strings:
            return BrailleResult("", "No braille characters detected.", 0.0)
        
        try:
            raw_text = " ".join(detected_strings).strip()
            logger.info(f"Combined raw text: '{raw_text}'")
            
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
        """Process text using AI API"""
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
                logger.info(f"✅ AI processed result: '{processed}'")
                return processed
            else:
                return self._fallback_process(text)
                
        except Exception as e:
            logger.error(f"AI API exception: {str(e)}")
            return self._fallback_process(text)
    
    def _fallback_process(self, text: str) -> str:
        """Basic text processing without AI"""
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
                logger.error(f"Explanation API error: {str(e)}")
        
        return f"This appears to be braille text that reads: '{text}'"
    
    def chat(self, user_message: str, thread_id: str = "default") -> str:
        """Chat with context"""
        try:
            history = self.conversation_memory.get(thread_id, [])
            
            if not history:
                system_msg = "You are a helpful AI assistant specializing in braille recognition and general assistance."
                if not self.api_key:
                    system_msg += " You are currently operating in fallback mode."
                history = [{"role": "system", "content": system_msg}]
            
            history.append({"role": "user", "content": user_message})
            
            if len(history) > 7:
                history = [history[0]] + history[-6:]
            
            if self.api_key:
                response = self._generate_response(history)
            else:
                response = self._fallback_chat_response(user_message)
            
            history.append({"role": "assistant", "content": response})
            self.conversation_memory[thread_id] = history
            
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate AI response"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.3
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return self._fallback_chat_response(messages[-1]["content"])
                
        except Exception as e:
            return self._fallback_chat_response(messages[-1]["content"])
    
    def _fallback_chat_response(self, message: str) -> str:
        """Fallback chat responses"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your Braille Recognition Assistant. I can help with braille processing and answer questions."
        elif "help" in message_lower:
            return "I can help with:\n1. Processing braille text\n2. Explaining concepts\n3. General conversation\n\nWhat would you like to do?"
        elif "braille" in message_lower:
            return "I can process braille characters and convert them to readable text. Please provide braille characters or ask specific questions."
        elif any(word in message_lower for word in ["what", "explain", "tell me"]):
            return f"I'd be happy to explain. However, I'm currently in limited mode. Could you provide more details about '{message[:30]}...'?"
        else:
            return f"I understand you're asking about: {message[:50]}{'...' if len(message) > 50 else ''}. I'm in limited mode but happy to help with basic questions."

# ============================================================================
# BRAILLE DETECTOR (Embedded)
# ============================================================================

class BrailleDetector:
    """Braille Detection using Roboflow API"""
    
    def __init__(self):
        logger.info("Initializing BrailleDetector")
        
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        self.workspace_name = "braille-to-text-0xo2p"
        self.model_version = "1"
        self.base_url = "https://api.roboflow.com"
        
        logger.info(f"API Key present: {bool(self.api_key)}")
        
        if not self.api_key:
            logger.warning("❌ NO ROBOFLOW API KEY FOUND")
    
    def detect_braille(self, image_bytes: bytes) -> Dict:
        """Detect braille in image"""
        logger.info("=== STARTING BRAILLE DETECTION ===")
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        if not self.api_key:
            return {"error": "ROBOFLOW_API_KEY not configured"}
        
        try:
            # Encode image
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare request
            url = f"{self.base_url}/{self.workspace_name}/{self.model_version}/predict"
            
            payload = {
                "api_key": self.api_key,
                "image": encoded_image,
                "confidence": 0.3,
                "overlap": 0.5
            }
            
            headers = {"Content-Type": "application/json"}
            
            # Make API call
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    if "error" in result:
                        logger.error(f"API returned error: {result['error']}")
                        return {"error": result["error"]}
                    
                    predictions = result.get("predictions", [])
                    logger.info(f"✅ Found {len(predictions)} predictions")
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON response: {str(e)}"
                    logger.error(error_msg)
                    return {"error": error_msg}
            
            else:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                
                if response.status_code == 401:
                    return {"error": "Invalid API key or unauthorized"}
                elif response.status_code == 404:
                    return {"error": f"Model not found: {self.workspace_name}/v{self.model_version}"}
                else:
                    return {"error": f"API error {response.status_code}: {response.text}"}
        
        except requests.exceptions.Timeout:
            return {"error": "Request timeout - API took too long"}
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return {"error": f"Detection failed: {str(e)}"}
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract and validate predictions"""
        if not result or "error" in result:
            return []
        
        try:
            predictions = result.get("predictions", [])
            valid_predictions = []
            required_keys = ['x', 'y', 'width', 'height', 'confidence', 'class']
            
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                
                missing_keys = [key for key in required_keys if key not in pred]
                if missing_keys:
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
            
            logger.info(f"✅ Extracted {len(valid_predictions)} valid predictions")
            return valid_predictions
            
        except Exception as e:
            logger.error(f"Error extracting predictions: {str(e)}")
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict]) -> List[str]:
        """Organize predictions into text rows"""
        if not predictions:
            return []
        
        try:
            # Sort by Y coordinate
            sorted_by_y = sorted(predictions, key=lambda p: p.get('y', 0))
            
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
                    current_group = [current_pred]
            
            # Process final group
            if current_group:
                current_group.sort(key=lambda p: p.get('x', 0))
                row_text = ''.join([p.get('class', '') for p in current_group])
                if row_text.strip():
                    rows.append(row_text)
            
            logger.info(f"✅ Organized into {len(rows)} text rows")
            return rows
            
        except Exception as e:
            logger.error(f"Error organizing text: {str(e)}")
            return []

# ============================================================================
# BRAILLE CONTROLLER (Simplified)
# ============================================================================

class BrailleController:
    """Simplified controller for detection -> assistant pipeline"""
    
    def __init__(self):
        logger.info("Initializing simplified BrailleController")
        
        try:
            self.detector = BrailleDetector()
            logger.info("✅ BrailleDetector initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BrailleDetector: {e}")
            raise
        
        try:
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.assistant = SimpleAssistant(api_key=api_key)
            logger.info("✅ SimpleAssistant initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SimpleAssistant: {e}")
            raise
        
        logger.info("🎯 BrailleController ready")
    
    def detect_and_process(self, image_bytes: bytes) -> CompleteResult:
        """Complete pipeline: detect braille + process with AI"""
        logger.info("🚀 Starting detect and process operation")
        start_time = datetime.now()
        
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
            
            # Step 3: Organize into text rows
            text_rows = self.detector.organize_text_by_rows(predictions)
            
            # Step 4: Process with AI
            if text_rows:
                braille_result = self.assistant.process_braille_strings(text_rows)
                detected_text = braille_result.text
                explanation = braille_result.explanation
                confidence = braille_result.confidence
            else:
                detected_text = ""
                explanation = "No braille characters detected in the image."
                confidence = 0.0
            
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
    
    def chat(self, message: str, context: Dict = None, thread_id: str = "default") -> str:
        """Chat with optional context"""
        try:
            enhanced_message = message
            if context and context.get('detected_text'):
                enhanced_message = f"Context: Recently detected braille text: '{context['detected_text']}'\n\nUser question: {message}"
            
            return self.assistant.chat(enhanced_message, thread_id)
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

# ============================================================================
# GLOBAL CONTROLLER INSTANCE
# ============================================================================

_controller_instance = None

def get_controller() -> BrailleController:
    """Get or create controller instance"""
    global _controller_instance
    
    if _controller_instance is None:
        logger.info("🏭 Creating new BrailleController instance")
        _controller_instance = BrailleController()
        logger.info("✅ Controller instance created")
    
    return _controller_instance

# ============================================================================
# WEB INTERFACE HTML
# ============================================================================

def get_web_interface_html() -> str:
    """Return the complete web interface HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Braille Recognition System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
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
        
        .upload-section.dragover {
            border-color: #667eea;
            background: #f0f2ff;
        }
        
        .file-input {
            display: none;
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
        
        .status {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
        
        .status.loading {
            background: #fff3cd;
            color: #856404;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }
        
        .result-section {
            margin-bottom: 20px;
        }
        
        .result-section h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .detected-text {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 15px;
        }
        
        .explanation {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
        }
        
        .confidence {
            display: inline-block;
            padding: 5px 10px;
            background: #4caf50;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        .preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 15px 0;
        }
        
        .hidden {
            display: none;
        }
        
        .chat-section {
            margin-top: 30px;
            border-top: 1px solid #eee;
            padding-top: 30px;
        }
        
        .chat-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 1em;
            margin-bottom: 15px;
        }
        
        .chat-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
        }
        
        .chat-response {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            border-left: 4px solid #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔤 Braille Recognition</h1>
            <p>Upload an image containing braille text for recognition and processing</p>
        </div>
        
        <div class="upload-section" id="uploadSection">
            <p>📁 Drag and drop an image here or</p>
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
            <div class="result-section">
                <h3>🔤 Detected Text</h3>
                <div id="detectedText" class="detected-text"></div>
            </div>
            
            <div class="result-section">
                <h3>💡 Explanation</h3>
                <div id="explanation" class="explanation"></div>
                <div id="confidence" class="confidence"></div>
            </div>
        </div>
        
        <div class="chat-section">
            <h3>💬 Ask Questions</h3>
            <input type="text" id="chatInput" class="chat-input" placeholder="Ask about the detected text or braille in general...">
            <button class="chat-btn" onclick="sendChat()">Send</button>
            <div id="chatResponse" class="chat-response hidden"></div>
        </div>
    </div>

    <script>
        let detectedContext = null;

        // File handling
        function handleFile(event) {
            const file = event.target.files[0];
            if (file) {
                processFile(file);
            }
        }

        // Drag and drop
        const uploadSection = document.getElementById('uploadSection');
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                processFile(files[0]);
            }
        });

        function processFile(file) {
            if (!file.type.startsWith('image/')) {
                showStatus('Please select an image file', 'error');
                return;
            }

            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('previewImage').src = e.target.result;
                document.getElementById('preview').classList.remove('hidden');
            };
            reader.readAsDataURL(file);

            // Process image
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
                    detectedContext = { detected_text: data.detected_text };
                    showStatus('Processing complete!', 'success');
                } else {
                    showStatus('Processing failed: ' + (data.error || 'Unknown error'), 'error');
                }
            })
            .catch(error => {
                showStatus('Error: ' + error.message, 'error');
            });
        }

        function displayResults(data) {
            document.getElementById('detectedText').textContent = data.detected_text || 'No text detected';
            document.getElementById('explanation').textContent = data.explanation || 'No explanation available';
            document.getElementById('confidence').textContent = `Confidence: ${Math.round(data.confidence * 100)}%`;
            document.getElementById('results').classList.remove('hidden');
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.classList.remove('hidden');
            
            if (type === 'success') {
                setTimeout(() => {
                    status.classList.add('hidden');
                }, 3000);
            }