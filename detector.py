# detector.py - Optimized Braille Detection Module
import json
import base64
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

class BrailleDetector:
    def __init__(self):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key= os.getenv("ROBOFLOW_API_KEY")
        )
        self.workspace_name = "braille-to-text-0xo2p"
        self.workflow_id = "custom-workflow"
        
        # Maximum dimensions for processing (to avoid timeout)
        self.max_width = 800
        self.max_height = 600
        
        # 26 distinct hex colors for different Braille classes
        self.class_colors = {
            'a': '#FF0000', 'b': '#00FF00', 'c': '#0000FF', 'd': '#FFFF00', 'e': '#FF00FF',
            'f': '#00FFFF', 'g': '#FF8000', 'h': '#8000FF', 'i': '#00FF80', 'j': '#FF0080',
            'k': '#80FF00', 'l': '#0080FF', 'm': '#FF8080', 'n': '#80FF80', 'o': '#8080FF',
            'p': '#FFFF80', 'q': '#FF80FF', 'r': '#80FFFF', 's': '#C0C0C0', 't': '#800000',
            'u': '#008000', 'v': '#000080', 'w': '#808000', 'x': '#800080', 'y': '#008080',
            'z': '#404040'
        }
    
    def resize_image_for_processing(self, image_path: str) -> Tuple[str, float]:
        """Resize image to optimal size for processing and return scale factor"""
        try:
            image = Image.open(image_path)
            original_width, original_height = image.size
            
            # Calculate scale factor to fit within max dimensions
            scale_width = self.max_width / original_width
            scale_height = self.max_height / original_height
            scale_factor = min(scale_width, scale_height, 1.0)  # Don't upscale
            
            if scale_factor < 1.0:
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                # Use high-quality resampling
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save resized image
                resized_path = image_path.replace('.jpg', '_resized.jpg').replace('.png', '_resized.jpg')
                resized_image.save(resized_path, 'JPEG', quality=95)
                
                return resized_path, scale_factor
            else:
                return image_path, 1.0
                
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image_path, 1.0
    
    def detect_braille(self, image_path: str) -> Dict:
        """Run Braille detection on the input image"""
        try:
            # Resize image for faster processing
            processed_image_path, scale_factor = self.resize_image_for_processing(image_path)
            
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": processed_image_path},
                use_cache=True
            )
            
            # Clean up resized image if it was created
            if processed_image_path != image_path and os.path.exists(processed_image_path):
                os.unlink(processed_image_path)
            
            # Scale predictions back to original size if image was resized
            if result and scale_factor < 1.0:
                result = self.scale_predictions(result, 1.0 / scale_factor)
            
            return result
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def scale_predictions(self, result: Dict, scale_factor: float) -> Dict:
        """Scale prediction coordinates back to original image size"""
        try:
            if result and len(result) > 0 and "predictions" in result[0]:
                predictions_data = result[0]["predictions"]
                if "predictions" in predictions_data:
                    for pred in predictions_data["predictions"]:
                        pred['x'] *= scale_factor
                        pred['y'] *= scale_factor
                        pred['width'] *= scale_factor
                        pred['height'] *= scale_factor
            return result
        except Exception as e:
            print(f"Error scaling predictions: {e}")
            return result
    
    def extract_predictions(self, result: Dict) -> List[Dict]:
        """Extract predictions from the result"""
        try:
            if result and len(result) > 0 and "predictions" in result[0]:
                predictions_data = result[0]["predictions"]
                if "predictions" in predictions_data:
                    return predictions_data["predictions"]
            return []
        except Exception as e:
            print(f"Error extracting predictions: {e}")
            return []
    
    def organize_text_by_rows(self, predictions: List[Dict], min_confidence: float = 0.4) -> List[str]:
        """Organize detected characters into rows"""
        if not predictions:
            return []
        
        try:
            filtered_predictions = [pred for pred in predictions if pred['confidence'] >= min_confidence]
            if not filtered_predictions:
                return []
            
            sorted_by_y = sorted(filtered_predictions, key=lambda p: p['y'])
            rows = []
            current_group = [sorted_by_y[0]]
            
            for i in range(1, len(sorted_by_y)):
                current_pred = sorted_by_y[i]
                prev_pred = sorted_by_y[i-1]
                
                avg_height = (current_pred['height'] + prev_pred['height']) / 2
                threshold = max(15, avg_height * 0.5)
                
                if abs(current_pred['y'] - prev_pred['y']) <= threshold:
                    current_group.append(current_pred)
                else:
                    if current_group:
                        current_group.sort(key=lambda p: p['x'])
                        row_text = ''.join([p['class'] for p in current_group])
                        rows.append(row_text)
                    current_group = [current_pred]
            
            if current_group:
                current_group.sort(key=lambda p: p['x'])
                row_text = ''.join([p['class'] for p in current_group])
                rows.append(row_text)
            
            return rows
            
        except Exception as e:
            print(f"Error organizing text: {e}")
            return []
    
    def create_annotated_image(self, image_path: str, predictions: List[Dict], 
                             output_path: str, min_confidence: float = 0.1,
                             max_display_size: Tuple[int, int] = (800, 600)) -> bool:
        """Create optimized annotated image with resizing for display"""
        try:
            image = Image.open(image_path)
            original_width, original_height = image.size
            
            # Calculate display scale factor
            scale_width = max_display_size[0] / original_width
            scale_height = max_display_size[1] / original_height
            display_scale = min(scale_width, scale_height, 1.0)
            
            # Resize image for display if needed
            if display_scale < 1.0:
                new_width = int(original_width * display_scale)
                new_height = int(original_height * display_scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            draw = ImageDraw.Draw(image)
            
            # Use smaller font for resized image
            try:
                font_size = max(8, int(12 * display_scale))
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            filtered_predictions = [pred for pred in predictions if pred['confidence'] >= min_confidence]
            
            for pred in filtered_predictions:
                # Scale coordinates for display
                x = pred['x'] * display_scale
                y = pred['y'] * display_scale
                width = pred['width'] * display_scale
                height = pred['height'] * display_scale
                
                confidence = pred['confidence']
                class_name = pred['class']
                
                x1, y1 = int(x - width/2), int(y - height/2)
                x2, y2 = int(x + width/2), int(y + height/2)
                
                color_hex = self.class_colors.get(class_name.lower(), '#FFFFFF')
                color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                
                # Draw bounding box with scaled thickness
                box_thickness = max(1, int(2 * display_scale))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)
                
                # Draw label with scaled dimensions - only show class name
                label = f"{class_name}"
                label_height = int(18 * display_scale)
                label_width = int(40 * display_scale)  # Reduced width since no confidence
                label_y = max(0, y1 - label_height)
                
                draw.rectangle([x1, label_y, x1 + label_width, label_y + label_height], fill=color)
                draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255), font=font)
            
            # Save with optimized settings
            image.save(output_path, format='PNG', optimize=True, compress_level=6)
            return True
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return False