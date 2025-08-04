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

# inference_sdk removed to stay within Vercel's 250MB memory limit
INFERENCE_SDK_AVAILABLE = False

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
        # Try to get API key from environment, with fallback
        env_key = os.getenv("ROBOFLOW_API_KEY")
        if env_key:
            self.api_key = env_key
        else:
            # Use the provided API key as fallback
            self.api_key = "RzOXFbriJONcee7MHKN8"
            print("Using fallback API key - consider setting ROBOFLOW_API_KEY environment variable")
            print("Note: This API key appears to be incomplete. A full Roboflow API key is typically 32+ characters.")
        
        if not self.api_key:
            print("ERROR: No Roboflow API key available - detection will be disabled")
            print("To fix this:")
            print("1. Go to https://roboflow.com/account")
            print("2. Copy your API key")
            print("3. Set the ROBOFLOW_API_KEY environment variable")
            print("4. Restart the application")
        else:
            print(f"Using Roboflow API key: {self.api_key[:5]}...{self.api_key[-5:]}")
            
        self.workspace_name = "braille-to-text-0xo2p"
        self.workflow_id = "custom-workflow"
        self.base_url = "https://serverless.roboflow.com"
        
        # inference_sdk removed to stay within Vercel's 250MB memory limit
        self.client = None
    
    def _encode_image_from_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        try:
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')
            return encoded_string
        except Exception as e:
            raise Exception(f"Failed to encode image: {e}")
    

# ... other parts of your BrailleDetector class ...

    def detect_braille_from_bytes(self, image_bytes: bytes) -> Optional[Dict]:
        """
        Run Braille detection using image bytes.
        Uses direct HTTP requests (inference_sdk removed for Vercel memory limit).
        """
        print("--- START detect_braille_from_bytes ---")
        if not self.api_key:
            error_msg = "Roboflow API key not configured. Please set ROBOFLOW_API_KEY environment variable."
            print(f"ERROR: {error_msg}")
            print("--- END detect_braille_from_bytes (FAILURE) ---")
            return {"error": "Detection configuration error", "detail": error_msg}

        try:
            print(f"1. Processing image bytes (size: {len(image_bytes)} bytes)...")
            
            # Use direct HTTP requests (inference_sdk removed for Vercel memory limit)
            print("2. Using direct HTTP requests...")
            encoded_image = self._encode_image_from_bytes(image_bytes)
            print(f"   -> Encoded image length: {len(encoded_image)} characters")

            # Prepare request details - try both endpoint formats
            url_option_1 = f"{self.base_url}/{self.workspace_name}/workflows/{self.workflow_id}"
            url_option_2 = f"https://detect.roboflow.com/{self.workspace_name}/{self.workflow_id}"
            
            headers = {"Content-Type": "application/json"}
            payload = {"image": encoded_image}
            params = {"api_key": self.api_key}

            print(f"3. Trying first endpoint: {url_option_1}")
            masked_key = self.api_key[:5] + '*' * (len(self.api_key) - 10) + self.api_key[-5:] if len(self.api_key) > 10 else '*' * len(self.api_key)
            print(f"   -> API Key: {masked_key}")

            # Try first endpoint
            response = requests.post(
                url_option_1,
                headers=headers,
                json=payload,
                params=params,
                timeout=30
            )
            
            # If first endpoint fails, try alternative
            if response.status_code != 200:
                print(f"   -> First endpoint failed (status {response.status_code}), trying alternative...")
                print(f"   -> Trying alternative endpoint: {url_option_2}")
                
                # For the alternative endpoint, use different payload structure
                alt_payload = {
                    "inputs": {
                        "image": {
                            "type": "base64",
                            "value": encoded_image
                        }
                    }
                }
                
                response = requests.post(
                    url_option_2,
                    headers=headers,
                    json=alt_payload,
                    params=params,
                    timeout=30
                )
            
            print(f"4. Received Response:")
            print(f"   -> Status Code: {response.status_code}")

            if response.status_code == 200:
                result_data = response.json()
                print(f"   -> Successfully parsed response")
                print(f"   -> Response type: {type(result_data)}")
                if isinstance(result_data, dict):
                    print(f"   -> Response keys: {list(result_data.keys())}")
                    # Check for specific keys that indicate success
                    if "predictions" in result_data:
                        pred_count = len(result_data["predictions"]) if isinstance(result_data["predictions"], list) else 0
                        print(f"   -> Found {pred_count} predictions in response")
                    elif "outputs" in result_data:
                        print(f"   -> Found 'outputs' key in response")
                elif isinstance(result_data, list):
                    print(f"   -> Response list length: {len(result_data)}")
                    if len(result_data) > 0 and isinstance(result_data[0], dict):
                        print(f"   -> First item keys: {list(result_data[0].keys())}")
                print("--- END detect_braille_from_bytes (SUCCESS) ---")
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
                print("--- END detect_braille_from_bytes (FAILURE) ---")
                return {
                    "error": "Detection failed",
                    "detail": final_error_msg,
                    "status_code": response.status_code
                }

        except Exception as e:
            error_msg = f"Unexpected error inside detect_braille_from_bytes: {e}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            print("--- END detect_braille_from_bytes (FAILURE) ---")
            return {"error": "Detection internal error", "detail": error_msg}

# ... rest of your BrailleDetector class ...
        
    
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
                    'roboflow_key_length': len(self.detector.api_key) if self.detector.api_key else 0,
                    'inference_sdk_available': False,
                    'inference_client_ready': False
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
                        'inference_sdk_available': False,
                        'inference_client_ready': False,
                        'api_url': self.detector.base_url
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
                    document.getElementById('detectBtn').disabled = true;
                    document.getElementById('processBtn').disabled = true;
                    
                    // Show warning message
                    const warningDiv = document.createElement('div');
                    warningDiv.style.cssText = 'background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin-bottom: 20px;';
                    warningDiv.innerHTML = `
                        <strong>⚠️ Configuration Required:</strong><br>
                        Roboflow API key not configured. Please set the ROBOFLOW_API_KEY environment variable.<br>
                        <a href="https://roboflow.com/account" target="_blank">Get your API key here</a>
                    `;
                    document.querySelector('.container').insertBefore(warningDiv, document.querySelector('.tabs'));
                } else {
                                         // Show SDK status (removed for Vercel memory limit)
                     const sdkStatus = '⚠️ Using HTTP requests (SDK removed for memory optimization)';
                    
                    const statusDiv = document.createElement('div');
                    statusDiv.style.cssText = 'background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 14px;';
                    statusDiv.innerHTML = `
                        <strong>🔧 System Status:</strong><br>
                        • Roboflow API: ✓ Configured (Key length: ${status.roboflow_key_length})<br>
                        • Inference SDK: ${sdkStatus}<br>
                        • AI Assistant: ${status.ai_configured ? '✓ Configured' : '⚠️ Using fallback mode'}
                    `;
                    document.querySelector('.container').insertBefore(statusDiv, document.querySelector('.tabs'));
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