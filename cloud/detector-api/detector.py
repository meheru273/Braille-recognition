# detector.py - Optimized Braille Detection Module for Vercel
import json
import os
from typing import List, Dict, Optional
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient

class BrailleDetector:
    def __init__(self):
        """Initialize with minimal memory footprint"""
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is required")
            
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.workspace_name = "braille-to-text-0xo2p"
        self.workflow_id = "custom-workflow"
        
        # Reduced color mapping (only essential colors)
        self.class_colors = {
            'a': '#FF0000', 'b': '#00FF00', 'c': '#0000FF', 'd': '#FFFF00', 'e': '#FF00FF',
            'f': '#00FFFF', 'g': '#FF8000', 'h': '#8000FF', 'i': '#00FF80', 'j': '#FF0080',
            'k': '#80FF00', 'l': '#0080FF', 'm': '#FF8080', 'n': '#80FF80', 'o': '#8080FF',
            'p': '#FFFF80', 'q': '#FF80FF', 'r': '#80FFFF', 's': '#C0C0C0', 't': '#800000',
            'u': '#008000', 'v': '#000080', 'w': '#808000', 'x': '#800080', 'y': '#008080',
            'z': '#404040'
        }
    
    def detect_braille(self, image_path: str) -> Optional[Dict]:
        """Run Braille detection with error handling and optimization"""
        try:
            # Use minimal parameters to reduce processing time
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": image_path},
                use_cache=True  # Enable caching to speed up repeated requests
            )
            return result
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions with validation"""
        if not result:
            return []
            
        try:
            if isinstance(result, list) and len(result) > 0:
                predictions_data = result[0].get("predictions", {})
                if isinstance(predictions_data, dict) and "predictions" in predictions_data:
                    predictions = predictions_data["predictions"]
                    # Validate prediction structure
                    valid_predictions = []
                    for pred in predictions:
                        if all(key in pred for key in ['x', 'y', 'width', 'height', 'confidence', 'class']):
                            valid_predictions.append(pred)
                    return valid_predictions
            return []
        except Exception as e:
            print(f"Error extracting predictions: {e}")
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.4) -> List[str]:
        """Organize detected characters into rows with optimization"""
        if not predictions:
            return []
        
        try:
            # Filter by confidence early to reduce processing
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
                
                # Calculate dynamic threshold
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
                        if row_text.strip():  # Only add non-empty rows
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
            # Limit the number of annotations to prevent memory issues
            filtered_predictions = [
                pred for pred in predictions[:100]  # Limit to first 100 predictions
                if pred.get('confidence', 0) >= min_confidence
            ]
            
            if not filtered_predictions:
                return False
            
            # Open and process image
            with Image.open(image_path) as image:
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                draw = ImageDraw.Draw(image)
                
                # Use default font to avoid font loading issues
                try:
                    # Try to load a system font
                    from PIL import ImageFont
                    font = ImageFont.load_default()
                except:
                    font = None
                
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
                        
                        # Draw label with smaller text
                        label = f"{class_name} {confidence:.1f}"
                        label_y = max(0, y1 - 15)
                        
                        # Draw label background
                        if font:
                            bbox = draw.textbbox((x1, label_y), label, font=font)
                            draw.rectangle(bbox, fill=color)
                            draw.text((x1 + 1, label_y + 1), label, fill=(255, 255, 255), font=font)
                        else:
                            draw.rectangle([x1, label_y, x1 + 60, label_y + 12], fill=color)
                            draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255))
                            
                    except Exception as pred_error:
                        print(f"Error drawing prediction: {pred_error}")
                        continue
                
                # Save with optimization
                image.save(output_path, format='PNG', optimize=True)
                return True
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return False