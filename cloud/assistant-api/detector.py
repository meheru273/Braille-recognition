# detector_final_fix.py - Fixed Braille Detection with Correct Endpoint
import json
import os
import base64
import requests
from typing import List, Dict, Optional, Union
from PIL import Image, ImageDraw
import io

class BrailleDetector:
    def __init__(self):
        """Initialize with correct detection API configuration"""
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is required")
            
        self.workspace_name = "braille-to-text-0xo2p"
        self.model_version = "1"
        # CORRECT endpoint format for detection (not workflows)
        self.base_url = "https://api.roboflow.com"
        
        print(f"Initialized BrailleDetector:")
        print(f"  Workspace: {self.workspace_name}")
        print(f"  Model Version: {self.model_version}")
        print(f"  Base URL: {self.base_url}")
    
    def _encode_image_from_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        try:
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image: {e}")
    
    def detect_braille_from_bytes(self, image_bytes: bytes) -> Optional[Dict]:
        """Run Braille detection using correct detection API endpoint"""
        if not self.api_key:
            return {"error": "ROBOFLOW_API_KEY not configured"}
            
        try:
            # Encode image to base64
            encoded_image = self._encode_image_from_bytes(image_bytes)
            
            # CORRECT detection API endpoint (not workflow)
            url = f"{self.base_url}/{self.workspace_name}/{self.model_version}/predict"
            
            print(f"Detection API URL: {url}")
            
            # Correct payload format for detection API
            payload = {
                "api_key": self.api_key,
                "image": encoded_image,
                "confidence": 0.3,  # Lower confidence threshold
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
                
                print(f"Detection successful! Response keys: {list(result.keys())}")
                
                # Log predictions count
                predictions = result.get("predictions", [])
                print(f"Found {len(predictions)} predictions")
                
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
                
        except requests.exceptions.Timeout:
            print("Request timed out")
            return {"error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            print(f"Detection error: {e}")
            return {"error": f"Detection error: {str(e)}"}
    
    def try_alternative_versions(self, image_bytes: bytes) -> Optional[Dict]:
        """Try different model versions if the default fails"""
        print("Trying alternative model versions...")
        
        versions_to_try = ["2", "3", "4", "1"]  # Try different versions
        
        for version in versions_to_try:
            try:
                print(f"Trying version {version}...")
                
                encoded_image = self._encode_image_from_bytes(image_bytes)
                url = f"{self.base_url}/{self.workspace_name}/{version}/predict"
                
                payload = {
                    "api_key": self.api_key,
                    "image": encoded_image,
                    "confidence": 0.1,  # Very low confidence
                    "overlap": 0.5
                }
                
                response = requests.post(
                    url, 
                    headers={"Content-Type": "application/json"}, 
                    json=payload, 
                    timeout=25
                )
                
                print(f"Version {version} status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    if "error" not in result:
                        predictions = result.get("predictions", [])
                        print(f"✅ Version {version} works! Found {len(predictions)} predictions")
                        # Update the working version
                        self.model_version = version
                        return result
                    else:
                        print(f"Version {version} returned error: {result.get('error')}")
                else:
                    print(f"Version {version} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"Version {version} exception: {e}")
                continue
        
        return None
    
    def detect_braille_with_fallback(self, image_bytes: bytes) -> Optional[Dict]:
        """Try detection with multiple fallback strategies"""
        print("=== Starting Braille Detection with Fallback ===")
        
        # Strategy 1: Try the configured version
        print("Strategy 1: Using configured model version")
        result = self.detect_braille_from_bytes(image_bytes)
        
        if result and "error" not in result:
            predictions = result.get("predictions", [])
            if predictions:
                print("✅ Primary detection successful!")
                return result
            else:
                print("Primary detection returned no predictions, trying alternatives...")
        else:
            error_msg = result.get("error", "Unknown error") if result else "No result"
            print(f"Primary detection failed: {error_msg}")
        
        # Strategy 2: Try alternative versions
        print("\nStrategy 2: Trying alternative model versions")
        result = self.try_alternative_versions(image_bytes)
        
        if result and "error" not in result:
            predictions = result.get("predictions", [])
            if predictions:
                print("✅ Alternative version detection successful!")
                return result
        
        # Strategy 3: Try with very low confidence on primary version
        print("\nStrategy 3: Trying with very low confidence threshold")
        try:
            encoded_image = self._encode_image_from_bytes(image_bytes)
            url = f"{self.base_url}/{self.workspace_name}/1/predict"
            
            payload = {
                "api_key": self.api_key,
                "image": encoded_image,
                "confidence": 0.01,  # Extremely low confidence
                "overlap": 0.9
            }
            
            response = requests.post(
                url, 
                headers={"Content-Type": "application/json"}, 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    predictions = result.get("predictions", [])
                    print(f"Low confidence detection found {len(predictions)} predictions")
                    if predictions:
                        return result
            
        except Exception as e:
            print(f"Low confidence detection failed: {e}")
        
        print("\n❌ All detection strategies failed")
        return {"error": "All detection methods failed"}
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions from detection response"""
        if not result or "error" in result:
            print(f"No valid result to extract from: {result}")
            return []
            
        try:
            print(f"Extracting predictions from result keys: {list(result.keys())}")
            
            # Standard detection API response format
            predictions = result.get("predictions", [])
            
            if not predictions:
                print("No predictions found in result")
                return []
            
            print(f"Found {len(predictions)} raw predictions")
            
            # Validate and clean predictions
            valid_predictions = []
            required_keys = ['x', 'y', 'width', 'height', 'confidence', 'class']
            
            for i, pred in enumerate(predictions):
                if not isinstance(pred, dict):
                    print(f"Skipping non-dict prediction {i}: {pred}")
                    continue
                    
                # Check if all required keys exist
                missing_keys = [key for key in required_keys if key not in pred]
                if missing_keys:
                    print(f"Skipping prediction {i} missing keys {missing_keys}: {list(pred.keys())}")
                    continue
                
                try:
                    # Convert to proper types and validate ranges
                    cleaned_pred = {
                        'x': float(pred['x']),
                        'y': float(pred['y']),
                        'width': float(pred['width']),
                        'height': float(pred['height']),
                        'confidence': max(0.0, min(1.0, float(pred['confidence']))),
                        'class': str(pred['class']).strip()
                    }
                    
                    # Basic validation
                    if cleaned_pred['width'] > 0 and cleaned_pred['height'] > 0 and cleaned_pred['class']:
                        valid_predictions.append(cleaned_pred)
                        print(f"✅ Valid prediction {len(valid_predictions)}: {cleaned_pred['class']} (conf: {cleaned_pred['confidence']:.3f})")
                    else:
                        print(f"❌ Invalid prediction {i} dimensions or class: {cleaned_pred}")
                        
                except (ValueError, TypeError) as e:
                    print(f"❌ Skipping invalid prediction {i}: {pred}, error: {e}")
                    continue
            
            print(f"✅ Final extraction result: {len(valid_predictions)} valid predictions")
            return valid_predictions
            
        except Exception as e:
            print(f"❌ Error extracting predictions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.2) -> List[str]:
        """Organize detected characters into rows"""
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
                print("No predictions meet confidence threshold, trying with 0.05")
                filtered_predictions = [
                    pred for pred in predictions 
                    if pred.get('confidence', 0) >= 0.05
                ]
                print(f"With confidence 0.05: {len(filtered_predictions)} predictions")
            
            if not filtered_predictions:
                print("Still no predictions, using all")
                filtered_predictions = predictions
            
            # Sort by Y coordinate first (top to bottom)
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p.get('y', 0))
            
            if not sorted_by_y:
                return []
            
            print("Sorted predictions by Y coordinate:")
            for pred in sorted_by_y:
                print(f"  {pred.get('class', '?')} at y={pred.get('y', 0):.1f}, conf={pred.get('confidence', 0):.3f}")
            
            rows = []
            current_group = [sorted_by_y[0]]
            
            # Group predictions into rows with adaptive threshold
            for i in range(1, len(sorted_by_y)):
                current_pred = sorted_by_y[i]
                prev_pred = sorted_by_y[i-1]
                
                # Calculate dynamic threshold for row grouping
                avg_height = (current_pred.get('height', 20) + prev_pred.get('height', 20)) / 2
                threshold = max(8, avg_height * 0.7)  # More lenient grouping
                
                y_diff = abs(current_pred.get('y', 0) - prev_pred.get('y', 0))
                
                print(f"Comparing {current_pred.get('class', '?')} (y={current_pred.get('y', 0):.1f}) with {prev_pred.get('class', '?')} (y={prev_pred.get('y', 0):.1f})")
                print(f"  Y difference: {y_diff:.1f}, threshold: {threshold:.1f}")
                
                if y_diff <= threshold:
                    current_group.append(current_pred)
                    print(f"  -> Added to current group (now {len(current_group)} chars)")
                else:
                    # Process current group - sort by X coordinate (left to right)
                    if current_group:
                        current_group.sort(key=lambda p: p.get('x', 0))
                        row_text = ''.join([p.get('class', '') for p in current_group])
                        if row_text.strip():
                            rows.append(row_text)
                            print(f"  -> Completed row: '{row_text}' ({len(current_group)} chars)")
                    current_group = [current_pred]
                    print(f"  -> Started new group")
            
            # Process final group
            if current_group:
                current_group.sort(key=lambda p: p.get('x', 0))
                row_text = ''.join([p.get('class', '') for p in current_group])
                if row_text.strip():
                    rows.append(row_text)
                    print(f"✅ Final row: '{row_text}' ({len(current_group)} chars)")
            
            print(f"✅ Organized into {len(rows)} text rows: {rows}")
            return rows
            
        except Exception as e:
            print(f"❌ Error organizing text: {e}")
            import traceback
            traceback.print_exc()
            return []

# Test function
def test_detector():
    """Test the detector with comprehensive debugging"""
    print("=== Testing Fixed Braille Detector ===")
    
    try:
        detector = BrailleDetector()
        print("✅ Detector initialized successfully")
        
        # Create a simple test image
        from PIL import Image
        test_img = Image.new('RGB', (200, 100), 'white')
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        test_img.save(img_buffer, format='JPEG')
        image_bytes = img_buffer.getvalue()
        
        print(f"Created test image: {len(image_bytes)} bytes")
        
        # Test detection
        result = detector.detect_braille_with_fallback(image_bytes)
        
        if result and "error" not in result:
            print("\n=== Detection Results ===")
            predictions = detector.extract_predictions(result)
            print(f"Extracted {len(predictions)} predictions")
            
            if predictions:
                print("\nPredictions:")
                for i, pred in enumerate(predictions[:10]):  # Show first 10
                    print(f"  {i+1}. '{pred.get('class', '?')}' (confidence: {pred.get('confidence', 0):.3f})")
                
                rows = detector.organize_text_by_rows(predictions)
                print(f"\nOrganized into {len(rows)} text rows:")
                for i, row in enumerate(rows):
                    print(f"  Row {i+1}: '{row}'")
            else:
                print("No predictions found (this is normal for a blank test image)")
                
        else:
            error_msg = result.get("error", "Unknown error") if result else "No result"
            print(f"\n❌ Detection failed: {error_msg}")
            
            print("\nDebugging suggestions:")
            print("1. Check your API key is correct")
            print("2. Verify the workspace 'braille-to-text-0xo2p' exists")
            print("3. Ensure your model is trained and deployed")
            print("4. Try with a real braille image")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detector()