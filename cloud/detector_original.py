# detector_original.py - Original Working Braille Detection Module
import json
import base64
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv

# Try to import inference_sdk, but provide fallback if not available
try:
    from inference_sdk import InferenceHTTPClient
    INFERENCE_SDK_AVAILABLE = True
except ImportError:
    INFERENCE_SDK_AVAILABLE = False
    print("Warning: inference_sdk not available, will use HTTP requests")

# Import configuration
try:
    from config import (
        MODEL_1, MODEL_2, DEFAULT_MODEL, ROBOFLOW_API_URL,
        get_active_model, validate_config, USE_INFERENCE_SDK
    )
except ImportError:
    # Fallback configuration
    MODEL_1 = {
        "workspace": "braille-to-text-0xo2p",
        "workflow_id": "custom-workflow",
        "api_key": "RzOXFbriJONcee7MHKN8"
    }
    MODEL_2 = {
        "workspace": "braille-image",
        "workflow_id": "custom-workflow", 
        "api_key": "NRmMU6uU07XILRg52e7n"
    }
    DEFAULT_MODEL = "MODEL_1"
    ROBOFLOW_API_URL = "https://serverless.roboflow.com"
    USE_INFERENCE_SDK = True

load_dotenv()  # Loads .env file

class BrailleDetector:
    def __init__(self, model_choice: str = None):
        """
        Initialize detector with specified model or default
        
        Args:
            model_choice: "MODEL_1" or "MODEL_2" or None for default
        """
        # Determine which model to use
        if model_choice == "MODEL_1":
            self.model_config = MODEL_1
        elif model_choice == "MODEL_2":
            self.model_config = MODEL_2
        else:
            self.model_config = get_active_model() if 'get_active_model' in globals() else MODEL_1
        
        self.workspace_name = self.model_config["workspace"]
        self.workflow_id = self.model_config["workflow_id"]
        self.api_key = self.model_config["api_key"]
        
        print(f"🔧 Using model: {self.workspace_name}")
        print(f"🔑 API key: {self.api_key[:5]}...{self.api_key[-5:] if len(self.api_key) > 10 else '***'}")
        
        # Initialize client based on availability
        if INFERENCE_SDK_AVAILABLE and USE_INFERENCE_SDK:
            try:
                self.client = InferenceHTTPClient(
                    api_url=ROBOFLOW_API_URL,
                    api_key=self.api_key
                )
                self.use_sdk = True
                print("✅ Using inference_sdk")
            except Exception as e:
                print(f"❌ Failed to initialize inference_sdk: {e}")
                self.use_sdk = False
        else:
            self.use_sdk = False
            print("⚠️ Using HTTP requests (inference_sdk not available)")
        
        # 26 distinct hex colors for different Braille classes
        self.class_colors = {
            'a': '#FF0000', 'b': '#00FF00', 'c': '#0000FF', 'd': '#FFFF00', 'e': '#FF00FF',
            'f': '#00FFFF', 'g': '#FF8000', 'h': '#8000FF', 'i': '#00FF80', 'j': '#FF0080',
            'k': '#80FF00', 'l': '#0080FF', 'm': '#FF8080', 'n': '#80FF80', 'o': '#8080FF',
            'p': '#FFFF80', 'q': '#FF80FF', 'r': '#80FFFF', 's': '#C0C0C0', 't': '#800000',
            'u': '#008000', 'v': '#000080', 'w': '#808000', 'x': '#800080', 'y': '#008080',
            'z': '#404040'
        }
    
    def detect_braille(self, image_path: str) -> Optional[Dict]:
        """Run Braille detection on the input image"""
        try:
            if self.use_sdk:
                # Use original inference_sdk approach
                result = self.client.run_workflow(
                    workspace_name=self.workspace_name,
                    workflow_id=self.workflow_id,
                    images={"image": image_path},
                    use_cache=True
                )
                return result
            else:
                # Fallback to HTTP requests
                return self._detect_braille_http(image_path)
                
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def _detect_braille_http(self, image_path: str) -> Optional[Dict]:
        """Fallback detection using HTTP requests"""
        try:
            import requests
            
            # Encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Prepare request
            url = f"{ROBOFLOW_API_URL}/{self.workspace_name}/workflows/{self.workflow_id}"
            
            payload = {
                "image": image_data,
                "api_key": self.api_key
            }
            
            headers = {"Content-Type": "application/json"}
            
            print(f"🌐 Making HTTP request to: {url}")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ HTTP request failed: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ HTTP detection error: {e}")
            return None
    
    def detect_braille_from_bytes(self, image_bytes: bytes) -> Optional[Dict]:
        """Detect braille from image bytes"""
        try:
            # Save bytes to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            
            # Use the file-based detection
            result = self.detect_braille(temp_path)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            print(f"Error in detect_braille_from_bytes: {e}")
            return None
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions from the result"""
        try:
            if not result:
                return []
            
            # Handle different result structures
            if isinstance(result, list) and len(result) > 0:
                if "predictions" in result[0]:
                    predictions_data = result[0]["predictions"]
                    if "predictions" in predictions_data:
                        return predictions_data["predictions"]
                    elif isinstance(predictions_data, list):
                        return predictions_data
                elif "predictions" in result[0]:
                    return result[0]["predictions"]
            elif isinstance(result, dict):
                if "predictions" in result:
                    return result["predictions"]
                elif "data" in result and "predictions" in result["data"]:
                    return result["data"]["predictions"]
            
            print(f"⚠️ Unexpected result structure: {type(result)}")
            if isinstance(result, dict):
                print(f"   Keys: {list(result.keys())}")
            elif isinstance(result, list) and len(result) > 0:
                print(f"   First item keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'Not a dict'}")
            
            return []
            
        except Exception as e:
            print(f"Error extracting predictions: {e}")
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.4) -> List[str]:
        """Organize detected characters into rows"""
        if not predictions:
            return []
        
        try:
            filtered_predictions = [pred for pred in predictions if pred.get('confidence', 0) >= min_confidence]
            if not filtered_predictions:
                return []
            
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p.get('y', 0))
            rows = []
            current_group = [sorted_by_y[0]]
            
            for i in range(1, len(sorted_by_y)):
                current_pred = sorted_by_y[i]
                prev_pred = sorted_by_y[i-1]
                
                avg_height = (current_pred.get('height', 30) + prev_pred.get('height', 30)) / 2
                threshold = max(15, avg_height * 0.5)
                
                if abs(current_pred.get('y', 0) - prev_pred.get('y', 0)) <= threshold:
                    current_group.append(current_pred)
                else:
                    if current_group:
                        current_group.sort(key=lambda p: p.get('x', 0))
                        row_text = ''.join([p.get('class', '') for p in current_group])
                        if row_text.strip():
                            rows.append(row_text)
                    current_group = [current_pred]
            
            if current_group:
                current_group.sort(key=lambda p: p.get('x', 0))
                row_text = ''.join([p.get('class', '') for p in current_group])
                if row_text.strip():
                    rows.append(row_text)
            
            return rows
            
        except Exception as e:
            print(f"Error organizing text: {e}")
            return []
    
    def create_annotated_image(self, image_path: str, predictions: List[Dict], 
                             output_path: str, min_confidence: float = 0.1) -> bool:
        """Create high-resolution annotated image"""
        try:
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            filtered_predictions = [pred for pred in predictions if pred.get('confidence', 0) >= min_confidence]
            
            for pred in filtered_predictions:
                x, y = pred.get('x', 0), pred.get('y', 0)
                width, height = pred.get('width', 20), pred.get('height', 20)
                confidence = pred.get('confidence', 0)
                class_name = pred.get('class', '')
                
                x1, y1 = int(x - width/2), int(y - height/2)
                x2, y2 = int(x + width/2), int(y + height/2)
                
                color_hex = self.class_colors.get(class_name.lower(), '#FFFFFF')
                color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                label = f"{class_name} {confidence:.2f}"
                label_y = max(0, y1 - 20)
                draw.rectangle([x1, label_y, x1 + 80, label_y + 18], fill=color)
                draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255), font=font)
            
            image.save(output_path, format='PNG', optimize=False)
            return True
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return False

# Test function
def test_detector():
    """Test the detector with both models"""
    print("🧪 Testing Original Braille Detector")
    print("=" * 50)
    
    # Test both models
    for model_name in ["MODEL_1", "MODEL_2"]:
        print(f"\n🔍 Testing {model_name}")
        print("-" * 30)
        
        try:
            detector = BrailleDetector(model_name)
            print(f"✅ {model_name} initialized successfully")
            
            # Test with a sample image if available
            test_image = "../test/before.jpg"
            if os.path.exists(test_image):
                print(f"🖼️ Testing with: {test_image}")
                result = detector.detect_braille(test_image)
                
                if result:
                    print(f"✅ Detection successful")
                    predictions = detector.extract_predictions(result)
                    print(f"📊 Found {len(predictions)} predictions")
                    
                    text_rows = detector.organize_text_by_rows(predictions, min_confidence=0.1)
                    print(f"📝 Organized into {len(text_rows)} text rows")
                    
                    if text_rows:
                        print(f"📄 Text rows: {text_rows}")
                else:
                    print(f"❌ Detection failed")
            else:
                print(f"⚠️ Test image not found: {test_image}")
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")

if __name__ == "__main__":
    test_detector() 