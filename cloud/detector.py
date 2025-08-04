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
        """Initialize with minimal dependencies"""
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is required")
            
        # Updated API endpoints - try multiple formats
        self.endpoints = [
            # Standard Roboflow inference endpoint
            "https://detect.roboflow.com/braille-to-text-0xo2p/1",
            # Alternative format
            "https://api.roboflow.com/braille-to-text-0xo2p/1",
            # Hosted API format
            "https://outline.roboflow.com/braille-to-text-0xo2p/1"
        ]
        
        # Color mapping for different Braille classes
        self.class_colors = {
            'a': '#FF0000', 'b': '#00FF00', 'c': '#0000FF', 'd': '#FFFF00', 'e': '#FF00FF',
            'f': '#00FFFF', 'g': '#FF8000', 'h': '#8000FF', 'i': '#00FF80', 'j': '#FF0080',
            'k': '#80FF00', 'l': '#0080FF', 'm': '#FF8080', 'n': '#80FF80', 'o': '#8080FF',
            'p': '#FFFF80', 'q': '#FF80FF', 'r': '#80FFFF', 's': '#C0C0C0', 't': '#800000',
            'u': '#008000', 'v': '#000080', 'w': '#808000', 'x': '#800080', 'y': '#008080',
            'z': '#404040'
        }
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            raise Exception(f"Failed to encode image: {e}")
    
    def detect_braille_method1(self, image_path: str) -> Optional[Dict]:
        """Method 1: Standard Roboflow API with base64"""
        try:
            url = self.endpoints[0]
            
            # Encode image
            encoded_image = self._encode_image(image_path)
            
            # Prepare request
            payload = {
                "image": encoded_image,
                "api_key": self.api_key,
                "confidence": 0.1,  # Lower confidence threshold
                "overlap": 0.5
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            print(f"Trying Method 1: {url}")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Method 1 success: {len(result.get('predictions', []))} predictions")
                return result
            else:
                print(f"Method 1 failed: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"Method 1 error: {e}")
            return None
    
    def detect_braille_method2(self, image_path: str) -> Optional[Dict]:
        """Method 2: Multipart form upload"""
        try:
            url = self.endpoints[0]
            
            with open(image_path, 'rb') as image_file:
                files = {
                    'file': ('image.jpg', image_file, 'image/jpeg')
                }
                
                data = {
                    'api_key': self.api_key,
                    'confidence': '0.1',
                    'overlap': '0.5'
                }
                
                print(f"Trying Method 2: {url}")
                response = requests.post(url, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Method 2 success: {len(result.get('predictions', []))} predictions")
                    return result
                else:
                    print(f"Method 2 failed: {response.status_code} - {response.text[:200]}")
                    return None
                    
        except Exception as e:
            print(f"Method 2 error: {e}")
            return None
    
    def detect_braille_method3(self, image_path: str) -> Optional[Dict]:
        """Method 3: URL parameter format"""
        try:
            url = f"{self.endpoints[0]}?api_key={self.api_key}&confidence=0.1&overlap=0.5"
            
            encoded_image = self._encode_image(image_path)
            
            payload = encoded_image
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            print(f"Trying Method 3: {url}")
            response = requests.post(url, data=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Method 3 success: {len(result.get('predictions', []))} predictions")
                return result
            else:
                print(f"Method 3 failed: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"Method 3 error: {e}")
            return None
    
    def detect_braille_method4(self, image_path: str) -> Optional[Dict]:
        """Method 4: Try alternative endpoints"""
        for i, endpoint in enumerate(self.endpoints[1:], 2):
            try:
                url = endpoint
                
                encoded_image = self._encode_image(image_path)
                
                payload = {
                    "image": encoded_image,
                    "api_key": self.api_key,
                    "confidence": 0.1,
                    "overlap": 0.5
                }
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                print(f"Trying Method 4.{i}: {url}")
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Method 4.{i} success: {len(result.get('predictions', []))} predictions")
                    return result
                else:
                    print(f"Method 4.{i} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"Method 4.{i} error: {e}")
                continue
        
        return None
    
    def detect_braille_with_fallback(self, image_path: str) -> Optional[Dict]:
        """Try all detection methods with fallback"""
        print(f"Starting Braille detection for: {image_path}")
        
        # Verify image exists and is readable
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        
        try:
            # Test image can be opened
            with Image.open(image_path) as img:
                print(f"Image info: {img.size}, {img.mode}")
        except Exception as e:
            print(f"Cannot open image: {e}")
            return None
        
        # Try each method
        methods = [
            self.detect_braille_method1,
            self.detect_braille_method2, 
            self.detect_braille_method3,
            self.detect_braille_method4
        ]
        
        for i, method in enumerate(methods, 1):
            print(f"\n--- Trying Detection Method {i} ---")
            result = method(image_path)
            if result and result.get('predictions'):
                print(f"✓ Method {i} successful!")
                return result
            else:
                print(f"✗ Method {i} failed or returned no predictions")
        
        print("\n❌ All detection methods failed")
        return None
    
    def detect_braille(self, image_path: str) -> Optional[Dict]:
        """Main detection method - alias for fallback"""
        return self.detect_braille_with_fallback(image_path)
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions with robust error handling"""
        if not result:
            return []
            
        try:
            predictions = []
            
            # Debug: Print result structure
            print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Handle different response structures
            if isinstance(result, dict):
                # Direct predictions
                if "predictions" in result:
                    predictions = result["predictions"]
                # Sometimes wrapped in another layer
                elif "data" in result and "predictions" in result["data"]:
                    predictions = result["data"]["predictions"]
                # Check for other common structures
                elif "detections" in result:
                    predictions = result["detections"]
            elif isinstance(result, list):
                # Sometimes the entire result is a list of predictions
                predictions = result
            
            print(f"Found {len(predictions)} raw predictions")
            
            # Validate and clean predictions
            valid_predictions = []
            for i, pred in enumerate(predictions):
                if not isinstance(pred, dict):
                    continue
                
                # Check for required fields
                required_fields = ['x', 'y', 'width', 'height', 'confidence', 'class']
                if not all(field in pred for field in required_fields):
                    print(f"Prediction {i} missing required fields: {pred.keys()}")
                    continue
                
                try:
                    # Ensure numeric values
                    cleaned_pred = {
                        'x': float(pred['x']),
                        'y': float(pred['y']),
                        'width': float(pred['width']),
                        'height': float(pred['height']),
                        'confidence': float(pred['confidence']),
                        'class': str(pred['class'])
                    }
                    valid_predictions.append(cleaned_pred)
                    print(f"Valid prediction: {cleaned_pred['class']} ({cleaned_pred['confidence']:.2f})")
                except (ValueError, TypeError) as e:
                    print(f"Prediction {i} has invalid numeric values: {e}")
                    continue
            
            print(f"Returning {len(valid_predictions)} valid predictions")
            return valid_predictions
            
        except Exception as e:
            print(f"Error extracting predictions: {e}")
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.4) -> List[str]:
        """Organize detected characters into rows"""
        if not predictions:
            return []
        
        try:
            # Filter by confidence
            filtered_predictions = [
                pred for pred in predictions 
                if pred.get('confidence', 0) >= min_confidence
            ]
            
            print(f"Filtered {len(predictions)} -> {len(filtered_predictions)} predictions (confidence >= {min_confidence})")
            
            if not filtered_predictions:
                return []
            
            # Sort by Y coordinate first
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p.get('y', 0))
            
            rows = []
            current_group = [sorted_by_y[0]]
            
            # Group predictions into rows based on Y coordinate
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
                    # Process current group and start new one
                    if current_group:
                        # Sort by X coordinate within the row
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
            
            print(f"Organized into {len(rows)} rows: {rows}")
            return rows
            
        except Exception as e:
            print(f"Error organizing text: {e}")
            return []
    
    def create_annotated_image(self, image_path: str, predictions: List[Dict], 
                             output_path: str, min_confidence: float = 0.1) -> bool:
        """Create annotated image with bounding boxes"""
        try:
            # Filter predictions by confidence
            filtered_predictions = [
                pred for pred in predictions[:50]  # Limit to 50 for performance
                if pred.get('confidence', 0) >= min_confidence
            ]
            
            if not filtered_predictions:
                print("No predictions to annotate")
                return False
            
            # Open and process image
            with Image.open(image_path) as image:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                draw = ImageDraw.Draw(image)
                
                for i, pred in enumerate(filtered_predictions):
                    try:
                        x = pred.get('x', 0)
                        y = pred.get('y', 0)
                        width = pred.get('width', 20)
                        height = pred.get('height', 20)
                        confidence = pred.get('confidence', 0)
                        class_name = pred.get('class', '')
                        
                        # Calculate bounding box coordinates
                        x1, y1 = int(x - width/2), int(y - height/2)
                        x2, y2 = int(x + width/2), int(y + height/2)
                        
                        # Get color for this class
                        color_hex = self.class_colors.get(class_name.lower(), '#FFFFFF')
                        color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                        
                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        
                        # Draw label with confidence
                        label = f"{class_name} ({confidence:.2f})"
                        label_y = max(0, y1 - 20)
                        
                        # Simple text background
                        text_width = len(label) * 6
                        draw.rectangle([x1, label_y, x1 + text_width, label_y + 15], fill=color)
                        draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255))
                        
                    except Exception as e:
                        print(f"Error annotating prediction {i}: {e}")
                        continue
                
                # Save annotated image
                image.save(output_path, format='PNG', optimize=True)
                print(f"Annotated image saved: {output_path}")
                return True
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return False

# Test function for debugging
def test_detector():
    """Test function to debug detector issues"""
    try:
        detector = BrailleDetector()
        print("Detector initialized successfully")
        print(f"API Key present: {'Yes' if detector.api_key else 'No'}")
        print(f"API Key length: {len(detector.api_key) if detector.api_key else 0}")
        
        # Test with a dummy request to check API key validity
        test_url = detector.endpoints[0]
        response = requests.get(f"{test_url}?api_key={detector.api_key}", timeout=10)
        print(f"API Key test response: {response.status_code}")
        
    except Exception as e:
        print(f"Detector test failed: {e}")

if __name__ == "__main__":
    test_detector()