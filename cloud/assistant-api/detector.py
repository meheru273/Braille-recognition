# detector.py - Lightweight Braille Detection Module (no inference-sdk)
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
            
        self.workspace_name = "braille-to-text-0xo2p"
        self.workflow_id = "custom-workflow"
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
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            raise Exception(f"Failed to encode image: {e}")
    
    def _prepare_image_data(self, image_path: str) -> Dict:
        """Prepare image data for API request"""
        try:
            # Encode image to base64
            encoded_image = self._encode_image(image_path)
            
            # Prepare the request data
            image_data = {
                "type": "base64",
                "value": encoded_image
            }
            
            return {"image": image_data}
            
        except Exception as e:
            raise Exception(f"Failed to prepare image data: {e}")
    
    def detect_braille(self, image_path: str) -> Optional[Dict]:
        """Run Braille detection using direct HTTP requests"""
        try:
            # Prepare the API endpoint
            url = f"{self.base_url}/{self.workspace_name}/workflows/{self.workflow_id}"
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Prepare image data
            image_data = self._prepare_image_data(image_path)
            
            # Prepare request payload
            payload = {
                "api_key": self.api_key,
                "inputs": image_data
            }
            
            # Make the request with timeout
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=25  # 25 second timeout
            )
            
            # Check response status
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
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
    
    def detect_braille_alternative(self, image_path: str) -> Optional[Dict]:
        """Alternative detection method using different API endpoint"""
        try:
            # Alternative endpoint structure
            url = f"{self.base_url}/workflows/{self.workspace_name}/{self.workflow_id}"
            
            # Prepare multipart form data
            with open(image_path, 'rb') as image_file:
                files = {
                    'image': ('image.jpg', image_file, 'image/jpeg')
                }
                
                data = {
                    'api_key': self.api_key
                }
                
                # Make request
                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=25
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Alternative API failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"Alternative detection error: {e}")
            return None
    
    def detect_braille_with_fallback(self, image_path: str) -> Optional[Dict]:
        """Try multiple detection methods with fallback"""
        # Try primary method first
        result = self.detect_braille(image_path)
        if result:
            return result
            
        print("Primary detection failed, trying alternative method...")
        
        # Try alternative method
        result = self.detect_braille_alternative(image_path)
        if result:
            return result
            
        print("All detection methods failed")
        return None
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions with robust error handling"""
        if not result:
            return []
            
        try:
            # Handle different response structures
            predictions = []
            
            # Method 1: Direct predictions array
            if isinstance(result, list) and len(result) > 0:
                if "predictions" in result[0]:
                    pred_data = result[0]["predictions"]
                    if isinstance(pred_data, dict) and "predictions" in pred_data:
                        predictions = pred_data["predictions"]
                    elif isinstance(pred_data, list):
                        predictions = pred_data
            
            # Method 2: Direct predictions in result
            elif isinstance(result, dict):
                if "predictions" in result:
                    pred_data = result["predictions"]
                    if isinstance(pred_data, dict) and "predictions" in pred_data:
                        predictions = pred_data["predictions"]
                    elif isinstance(pred_data, list):
                        predictions = pred_data
                elif "outputs" in result:
                    # Handle outputs structure
                    outputs = result["outputs"]
                    if isinstance(outputs, list) and len(outputs) > 0:
                        if "predictions" in outputs[0]:
                            predictions = outputs[0]["predictions"]
            
            # Validate prediction structure
            valid_predictions = []
            for pred in predictions:
                if isinstance(pred, dict) and all(key in pred for key in ['x', 'y', 'width', 'height', 'confidence', 'class']):
                    # Ensure numeric values
                    try:
                        pred['x'] = float(pred['x'])
                        pred['y'] = float(pred['y'])
                        pred['width'] = float(pred['width'])
                        pred['height'] = float(pred['height'])
                        pred['confidence'] = float(pred['confidence'])
                        valid_predictions.append(pred)
                    except (ValueError, TypeError):
                        continue
            
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
            
            if not filtered_predictions:
                return []
            
            # Sort by Y coordinate
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p.get('y', 0))
            
            if not sorted_by_y:
                return []
            
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
            
            return rows
            
        except Exception as e:
            print(f"Error organizing text: {e}")
            return []
    
    def create_annotated_image(self, image_path: str, predictions: List[Dict], 
                             output_path: str, min_confidence: float = 0.1) -> bool:
        """Create annotated image with memory optimization"""
        try:
            # Limit annotations to prevent memory issues
            filtered_predictions = [
                pred for pred in predictions[:50]  # Limit to 50 predictions
                if pred.get('confidence', 0) >= min_confidence
            ]
            
            if not filtered_predictions:
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
                        
                        # Draw simple label
                        label = f"{class_name}"
                        label_y = max(0, y1 - 12)
                        draw.rectangle([x1, label_y, x1 + 30, label_y + 10], fill=color)
                        draw.text((x1 + 2, label_y + 1), label, fill=(255, 255, 255))
                        
                    except Exception:
                        continue
                
                # Save optimized image
                image.save(output_path, format='PNG', optimize=True)
                return True
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return False