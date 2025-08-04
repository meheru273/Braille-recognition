# detector.py - Fixed Braille Detection Module
import json
import os
import base64
import requests
from typing import List, Dict, Optional, Union
from PIL import Image, ImageDraw
import io

class BrailleDetector:
    def __init__(self):
        """Initialize with correct API configuration"""
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is required")
            
        self.workspace_name = "braille-to-text-0xo2p"
        self.workflow_id = "custom-workflow"
        # Fixed base URL for workflows
        self.base_url = "https://serverless.roboflow.com"
        
        # Color mapping for different Braille classes
        self.class_colors = {
            'a': '#FF0000', 'b': '#00FF00', 'c': '#0000FF', 'd': '#FFFF00', 'e': '#FF00FF',
            'f': '#00FFFF', 'g': '#FF8000', 'h': '#8000FF', 'i': '#00FF80', 'j': '#FF0080',
            'k': '#80FF00', 'l': '#0080FF', 'm': '#FF8080', 'n': '#80FF80', 'o': '#8080FF',
            'p': '#FFFF80', 'q': '#FF80FF', 'r': '#80FFFF', 's': '#C0C0C0', 't': '#800000',
            'u': '#008000', 'v': '#000080', 'w': '#808000', 'x': '#800080', 'y': '#008080',
            'z': '#404040'
        }
    
    def _encode_image_from_path(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image: {e}")
    
    def _encode_image_from_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        try:
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image bytes: {e}")
    
    def detect_braille(self, image_path: str) -> Optional[Dict]:
        """Run Braille detection using correct workflow API format"""
        try:
            # Correct API endpoint for workflows
            url = f"{self.base_url}/{self.workspace_name}/workflows/{self.workflow_id}"
            
            print(f"API URL: {url}")
            
            # Encode image to base64
            encoded_image = self._encode_image_from_path(image_path)
            
            # Correct payload structure for workflows
            payload = {
                "api_key": self.api_key,
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": encoded_image
                    }
                }
            }
            
            # Headers
            headers = {
                "Content-Type": "application/json"
            }
            
            print("Sending request to Roboflow...")
            
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
                print(f"Raw response: {json.dumps(result, indent=2)}")
                return result
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def detect_braille_multipart(self, image_path: str) -> Optional[Dict]:
        """Alternative method using multipart form data"""
        try:
            # URL for multipart upload
            url = f"{self.base_url}/{self.workspace_name}/workflows/{self.workflow_id}"
            
            # Prepare multipart form data
            with open(image_path, 'rb') as image_file:
                files = {
                    'image': ('image.jpg', image_file, 'image/jpeg')
                }
                
                data = {
                    'api_key': self.api_key
                }
                
                print("Trying multipart upload...")
                
                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=30
                )
                
                print(f"Multipart response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Multipart raw response: {json.dumps(result, indent=2)}")
                    return result
                else:
                    print(f"Multipart API Error {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"Multipart detection error: {e}")
            return None
    
    def detect_braille_inference_style(self, image_path: str) -> Optional[Dict]:
        """Try to mimic the inference SDK request format"""
        try:
            # Try the format that matches inference SDK
            url = f"{self.base_url}/{self.workspace_name}/workflows/{self.workflow_id}"
            
            # Encode image
            encoded_image = self._encode_image_from_path(image_path)
            
            # Try different payload formats
            payloads_to_try = [
                # Format 1: Direct workflow format
                {
                    "api_key": self.api_key,
                    "inputs": {
                        "image": encoded_image
                    },
                    "use_cache": True
                },
                # Format 2: With type specification
                {
                    "api_key": self.api_key,
                    "inputs": {
                        "image": {
                            "type": "base64",
                            "value": encoded_image
                        }
                    },
                    "use_cache": True
                },
                # Format 3: Images key (like inference SDK)
                {
                    "api_key": self.api_key,
                    "images": {
                        "image": encoded_image
                    },
                    "use_cache": True
                }
            ]
            
            headers = {
                "Content-Type": "application/json"
            }
            
            for i, payload in enumerate(payloads_to_try):
                try:
                    print(f"Trying payload format {i+1}...")
                    
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    print(f"Format {i+1} status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"Success with format {i+1}!")
                        print(f"Response: {json.dumps(result, indent=2)}")
                        return result
                    else:
                        print(f"Format {i+1} failed: {response.text}")
                        
                except Exception as e:
                    print(f"Format {i+1} exception: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"Inference style detection error: {e}")
            return None
    
    def detect_braille_with_fallback(self, image_path: str) -> Optional[Dict]:
        """Try multiple detection methods with comprehensive fallback"""
        print("=== Starting Braille Detection ===")
        
        methods = [
            ("Primary JSON method", self.detect_braille),
            ("Inference SDK style", self.detect_braille_inference_style),
            ("Multipart form", self.detect_braille_multipart)
        ]
        
        for method_name, method_func in methods:
            print(f"\n--- Trying: {method_name} ---")
            try:
                result = method_func(image_path)
                if result:
                    print(f"✅ Success with {method_name}")
                    return result
                else:
                    print(f"❌ {method_name} returned None")
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
        
        print("\n❌ All detection methods failed")
        return None
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions with comprehensive error handling"""
        if not result:
            print("No result to extract predictions from")
            return []
            
        try:
            print(f"Extracting predictions from: {type(result)}")
            
            predictions = []
            
            # Multiple extraction strategies
            extraction_paths = [
                # Direct predictions
                lambda r: r.get("predictions", []) if isinstance(r, dict) else [],
                # Nested predictions
                lambda r: r.get("predictions", {}).get("predictions", []) if isinstance(r, dict) else [],
                # Outputs structure
                lambda r: r.get("outputs", [{}])[0].get("predictions", []) if isinstance(r, dict) and r.get("outputs") else [],
                # Results structure
                lambda r: r.get("results", []) if isinstance(r, dict) else [],
                # List with predictions
                lambda r: r[0].get("predictions", []) if isinstance(r, list) and len(r) > 0 and isinstance(r[0], dict) else [],
                # Workflow outputs
                lambda r: r.get("workflow_outputs", {}).get("predictions", []) if isinstance(r, dict) else [],
            ]
            
            for i, extract_func in enumerate(extraction_paths):
                try:
                    extracted = extract_func(result)
                    if extracted and isinstance(extracted, list):
                        print(f"✅ Extraction method {i+1} found {len(extracted)} items")
                        predictions = extracted
                        break
                    else:
                        print(f"❌ Extraction method {i+1} found nothing or wrong type")
                except Exception as e:
                    print(f"❌ Extraction method {i+1} failed: {e}")
            
            if not predictions:
                print("No predictions found with any extraction method")
                print(f"Full result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                return []
            
            # Validate and clean predictions
            valid_predictions = []
            required_keys = ['x', 'y', 'width', 'height', 'confidence', 'class']
            
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                    
                # Check if all required keys exist
                if not all(key in pred for key in required_keys):
                    print(f"Skipping prediction missing keys: {pred}")
                    continue
                
                try:
                    # Convert to proper types
                    cleaned_pred = {
                        'x': float(pred['x']),
                        'y': float(pred['y']),
                        'width': float(pred['width']),
                        'height': float(pred['height']),
                        'confidence': float(pred['confidence']),
                        'class': str(pred['class'])
                    }
                    valid_predictions.append(cleaned_pred)
                except (ValueError, TypeError) as e:
                    print(f"Skipping invalid prediction: {pred}, error: {e}")
                    continue
            
            print(f"✅ Found {len(valid_predictions)} valid predictions")
            return valid_predictions
            
        except Exception as e:
            print(f"Error extracting predictions: {e}")
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.4) -> List[str]:
        """Organize detected characters into rows with better debugging"""
        if not predictions:
            print("No predictions to organize")
            return []
        
        try:
            print(f"Organizing {len(predictions)} predictions (min_confidence: {min_confidence})")
            
            # Filter by confidence
            filtered_predictions = [
                pred for pred in predictions 
                if pred.get('confidence', 0) >= min_confidence
            ]
            
            print(f"After confidence filtering: {len(filtered_predictions)} predictions")
            
            if not filtered_predictions:
                # Try with lower confidence
                print("No predictions meet confidence threshold, trying with 0.1")
                filtered_predictions = [
                    pred for pred in predictions 
                    if pred.get('confidence', 0) >= 0.1
                ]
                print(f"With confidence 0.1: {len(filtered_predictions)} predictions")
            
            if not filtered_predictions:
                print("Still no predictions, using all")
                filtered_predictions = predictions
            
            # Sort by Y coordinate first
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p.get('y', 0))
            
            if not sorted_by_y:
                return []
            
            print(f"Y-sorted predictions: {[(p.get('class', '?'), p.get('y', 0), p.get('confidence', 0)) for p in sorted_by_y]}")
            
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
                
                print(f"Comparing {current_pred.get('class', '?')} (y={current_pred.get('y', 0)}) with {prev_pred.get('class', '?')} (y={prev_pred.get('y', 0)}), diff={y_diff}, threshold={threshold}")
                
                if y_diff <= threshold:
                    current_group.append(current_pred)
                else:
                    # Process current group
                    if current_group:
                        current_group.sort(key=lambda p: p.get('x', 0))
                        row_text = ''.join([p.get('class', '') for p in current_group])
                        if row_text.strip():
                            rows.append(row_text)
                            print(f"Added row: '{row_text}'")
                    current_group = [current_pred]
            
            # Process final group
            if current_group:
                current_group.sort(key=lambda p: p.get('x', 0))
                row_text = ''.join([p.get('class', '') for p in current_group])
                if row_text.strip():
                    rows.append(row_text)
                    print(f"Added final row: '{row_text}'")
            
            print(f"Final rows: {rows}")
            return rows
            
        except Exception as e:
            print(f"Error organizing text: {e}")
            return []
    
    def create_annotated_image(self, image_path: str, predictions: List[Dict], 
                             output_path: str, min_confidence: float = 0.1) -> bool:
        """Create annotated image with memory optimization"""
        try:
            print(f"Creating annotated image with {len(predictions)} predictions")
            
            # Limit annotations to prevent memory issues
            filtered_predictions = [
                pred for pred in predictions[:50]  # Limit to 50 predictions
                if pred.get('confidence', 0) >= min_confidence
            ]
            
            print(f"Filtered to {len(filtered_predictions)} predictions for annotation")
            
            if not filtered_predictions:
                print("No predictions to annotate")
                return False
            
            # Process image
            with Image.open(image_path) as image:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                draw = ImageDraw.Draw(image)
                
                for pred in filtered_predictions:
                    try:
                        x = pred.get('x', 0)
                        y = pred.get('y', 0)
                        width = pred.get('width', 20)
                        height = pred.get('height', 20)
                        confidence = pred.get('confidence', 0)
                        class_name = pred.get('class', '')
                        
                        # Calculate bounding box
                        x1, y1 = int(x - width/2), int(y - height/2)
                        x2, y2 = int(x + width/2), int(y + height/2)
                        
                        # Get color
                        color_hex = self.class_colors.get(class_name.lower(), '#FFFFFF')
                        color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                        
                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        
                        # Draw label with confidence
                        label = f"{class_name} ({confidence:.2f})"
                        label_y = max(0, y1 - 15)
                        
                        # Calculate label background size
                        try:
                            bbox = draw.textbbox((0, 0), label)
                            label_width = bbox[2] - bbox[0]
                            label_height = bbox[3] - bbox[1]
                        except:
                            label_width = len(label) * 8
                            label_height = 12
                        
                        draw.rectangle([x1, label_y, x1 + label_width + 4, label_y + label_height + 2], fill=color)
                        draw.text((x1 + 2, label_y + 1), label, fill=(255, 255, 255))
                        
                    except Exception as e:
                        print(f"Error annotating prediction: {e}")
                        continue
                
                # Save optimized image
                image.save(output_path, format='PNG', optimize=True)
                print(f"Annotated image saved to: {output_path}")
                return True
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return False

# Test function
def test_detector():
    """Test the detector with debugging"""
    print("Testing Braille Detector...")
    
    try:
        detector = BrailleDetector()
        print("✅ Detector initialized")
        
        # Test with a sample image (you'll need to provide a real image path)
        test_image = "test_image.jpg"  # Replace with actual image path
        
        if os.path.exists(test_image):
            result = detector.detect_braille_with_fallback(test_image)
            
            if result:
                predictions = detector.extract_predictions(result)
                print(f"Extracted {len(predictions)} predictions")
                
                rows = detector.organize_text_by_rows(predictions)
                print(f"Organized into {len(rows)} rows: {rows}")
                
                # Create annotated image
                if predictions:
                    detector.create_annotated_image(test_image, predictions, "annotated_test.png")
            else:
                print("❌ No detection results")
        else:
            print(f"❌ Test image not found: {test_image}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_detector()