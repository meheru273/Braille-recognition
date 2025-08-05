# detector.py - Braille Detection Module
import json
import base64
from typing import List, Dict
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
        
        # 26 distinct hex colors for different Braille classes
        self.class_colors = {
            'a': '#FF0000', 'b': '#00FF00', 'c': '#0000FF', 'd': '#FFFF00', 'e': '#FF00FF',
            'f': '#00FFFF', 'g': '#FF8000', 'h': '#8000FF', 'i': '#00FF80', 'j': '#FF0080',
            'k': '#80FF00', 'l': '#0080FF', 'm': '#FF8080', 'n': '#80FF80', 'o': '#8080FF',
            'p': '#FFFF80', 'q': '#FF80FF', 'r': '#80FFFF', 's': '#C0C0C0', 't': '#800000',
            'u': '#008000', 'v': '#000080', 'w': '#808000', 'x': '#800080', 'y': '#008080',
            'z': '#404040'
        }
    
    def detect_braille(self, image_path: str) -> Dict:
        """Run Braille detection on the input image"""
        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": image_path},
                use_cache=True
            )
            return result
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
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
                             output_path: str, min_confidence: float = 0.1) -> bool:
        """Create high-resolution annotated image"""
        try:
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            filtered_predictions = [pred for pred in predictions if pred['confidence'] >= min_confidence]
            
            for pred in filtered_predictions:
                x, y = pred['x'], pred['y']
                width, height = pred['width'], pred['height']
                confidence = pred['confidence']
                class_name = pred['class']
                
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
        
