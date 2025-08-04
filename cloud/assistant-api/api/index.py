# api/index.py - Final Fixed Braille Detection + Assistant API
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
                    'ai_configured': bool(self.assistant.llm.api_key),
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
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                self.send_error_response(f'Invalid image data: {str(e)}', 400)
                return
            
            # Run detection with improved method
            detection_result = self.detector.detect_braille_with_fallback(image_bytes)
            
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
                'raw_response': detection_result  # For debugging
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
        """Serve the enhanced web interface"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fixed Braille Recognition System</title>
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
        .status-banner {
            background: #e8f5e8; border: 1px solid #4caf50; border-radius: 8px;
            padding: 12px; margin-bottom: 20px; text-align: center;
        }
        .status-banner.error {
            background: #ffe8e8; border-color: #f44336;
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
        .debug-info { font-size: 12px; color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔤 Fixed Braille Recognition System</h1>
            <p>AI-powered braille detection with corrected API endpoints</p>
        </div>
        
        <div id="status-banner" class="status-banner" style="display: none;">
            <span id="status-text">Checking system status...</span>
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
                <div id="debug-output" class="debug-info"></div>
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
                    <strong>Assistant:</strong> Hello! I'm your AI assistant with FIXED braille detection. Upload an image to test!
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
            const debugDiv = document.getElementById('debug-output');
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
                            <strong>✅ Detections Found:</strong> ${data.detection_count}
                        </div>
                        <div style="margin-bottom: 15px;">
                            <strong>📝 Text Rows:</strong>
                            ${data.text_rows.length > 0 ? 
                                '<ul style="margin-left: 20px;">' + 
                                data.text_rows.map(row => `<li>'${row}'</li>`).join('') + 
                                '</ul>' : 
                                '<p>No organized text rows found</p>'
                            }
                        </div>
                        <div>
                            <strong>🔍 Raw Predictions:</strong> ${data.predictions.length} characters detected
                        </div>
                    `;
                    
                    // Show debug info
                    debugDiv.innerHTML = `
                        <strong>Debug Info:</strong><br>
                        - API Response Status: OK<br>
                        - Detection Count: ${data.detection_count}<br>
                        - Organized Rows: ${data.text_rows.length}<br>
                        - Raw Predictions: ${data.predictions.length}
                    `;
                    
                    resultDiv.style.display = 'block';
                } else {
                    outputDiv.innerHTML = `<div style="color: red;">❌ Error: ${data.error}</div>`;
                    debugDiv.innerHTML = `<strong>Debug:</strong> Detection failed - ${data.error}`;
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                outputDiv.innerHTML = `<div style="color: red;">❌ Network error: ${error.message}</div>`;
                debugDiv.innerHTML = `<strong>Debug:</strong> Network error - ${error.message}`;
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
            const debugDiv = document.getElementById('debug-output');
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
                            <p><strong>Text Rows:</strong> ${data.detection.text_rows.length > 0 ? data.detection.text_rows.join(', ') : 'None organized'}</p>
                            ${data.detection.error ? `<p style="color: red;"><strong>Detection Error:</strong> ${data.detection.error}</p>` : ''}
                        </div>
                        <div style="padding: 15px; background: #f3e5f5; border-radius: 5px;">
                            <h4>🤖 AI Processing Results</h4>
                            <p><strong>Processed Text:</strong> ${data.processing.text || 'No text processed'}</p>
                            <p><strong>Explanation:</strong> ${data.processing.explanation}</p>
                            <p><strong>Confidence:</strong> ${(data.processing.confidence * 100).toFixed(1)}%</p>
                        </div>
                    `;
                    
                    debugDiv.innerHTML = `
                        <strong>Debug Info:</strong><br>
                        - Detection: ${data.detection.detection_count} chars found<br>
                        - Organization: ${data.detection.text_rows.length} rows<br>
                        - Processing: ${data.processing.text ? 'Success' : 'No text'}<br>
                        - Confidence: ${(data.processing.confidence * 100).toFixed(1)}%
                    `;
                    
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
                    <strong>Assistant:</strong> Hello! I'm your AI assistant with FIXED braille detection. Upload an image to test!
                </div>
            `;
        }

        // Check system status on load
        window.onload = async function() {
            try {
                const response = await fetch('/health');
                const status = await response.json();
                
                const statusBanner = document.getElementById('status-banner');
                const statusText = document.getElementById('status-text');
                
                if (status.roboflow_configured && status.ai_configured) {
                    statusText.textContent = `✅ System Ready - Detection: ${status.detection_endpoint}`;
                    statusBanner.classList.remove('error');
                } else if (status.roboflow_configured) {
                    statusText.textContent = '⚠️ Roboflow OK, AI Assistant in fallback mode';
                    statusBanner.classList.add('error');
                } else {
                    statusText.textContent = '❌ Roboflow API key not configured - Detection disabled';
                    statusBanner.classList.add('error');
                    document.getElementById('detectBtn').disabled = true;
                    document.getElementById('processBtn').disabled = true;
                }
                
                statusBanner.style.display = 'block';
                
                console.log('System status:', status);
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