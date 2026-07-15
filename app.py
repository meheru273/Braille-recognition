# app.py - Hugging Face Spaces with Gradio
import gradio as gr
import base64
import tempfile
import os
from detector import BrailleDetector
from assistant import BrailleAssistant
import json
from PIL import Image

# Initialize components
detector = BrailleDetector()
assistant = BrailleAssistant()

def process_braille_image(image, min_confidence=0.4):
    """Process uploaded braille image and return annotated image"""
    try:
        if image is None:
            return "Please upload an image", "", 0.0, [], None
        
        # Save uploaded image temporarily with optimized quality
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            # Save with reduced quality for faster processing
            image.save(temp_file.name, 'JPEG', quality=85, optimize=True)
            temp_image_path = temp_file.name
        
        # Create temporary file for annotated image
        annotated_temp_path = tempfile.mktemp(suffix='.png')
        
        try:
            # Run detection (now with automatic resizing)
            result = detector.detect_braille(temp_image_path)
            
            if not result:
                return "Detection failed", "", 0.0, [], None
            
            predictions = detector.extract_predictions(result)
            
            # Create annotated image with bounding boxes (optimized for display)
            annotated_image = None
            if predictions:
                success = detector.create_annotated_image(
                    temp_image_path, 
                    predictions, 
                    annotated_temp_path,
                    min_confidence=min_confidence,
                    max_display_size=(800, 600)  # Limit display size
                )
                if success:
                    annotated_image = Image.open(annotated_temp_path)
            
            # Process text
            text_rows = detector.organize_text_by_rows(predictions, min_confidence)
            braille_result = assistant.process_braille_strings(text_rows)
            
            return (
                braille_result.text,
                braille_result.explanation,
                braille_result.confidence,
                text_rows,
                annotated_image
            )
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            if os.path.exists(annotated_temp_path):
                os.unlink(annotated_temp_path)
                
    except Exception as e:
        return f"Error: {str(e)}", "", 0.0, [], None

def chat_with_assistant(message, history):
    """Chat interface"""
    try:
        response = assistant.chat(message, "gradio_session")
        history.append([message, response])
        return history, ""
    except Exception as e:
        history.append([message, f"Error: {str(e)}"])
        return history, ""

def process_api_request(request_json):
    """API endpoint simulation"""
    try:
        data = json.loads(request_json)
        task = data.get("task", "detect_braille")
        
        if task == "detect_braille":
            image_base64 = data.get("image_base64", "")
            min_confidence = data.get("min_confidence", 0.4)
            
            if not image_base64:
                return json.dumps({"error": "image_base64 required"})
            
            # Decode image
            image_data = base64.b64decode(image_base64)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_data)
                temp_image_path = temp_file.name
            
            try:
                result = detector.detect_braille(temp_image_path)
                if not result:
                    return json.dumps({"error": "Detection failed"})
                
                predictions = detector.extract_predictions(result)
                text_rows = detector.organize_text_by_rows(predictions, min_confidence)
                braille_result = assistant.process_braille_strings(text_rows)
                
                return json.dumps({
                    "detected_rows": text_rows,
                    "processed_text": braille_result.text,
                    "explanation": braille_result.explanation,
                    "confidence": braille_result.confidence,
                    "total_characters": len(predictions)
                })
                
            finally:
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
        
        elif task == "chat":
            message = data.get("message", "")
            response = assistant.chat(message, "api_session")
            return json.dumps({"response": response})
        
        else:
            return json.dumps({"error": "Unknown task"})
            
    except Exception as e:
        return json.dumps({"error": str(e)})

# Create Gradio interface
with gr.Blocks(title="Braille Detection API", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔤 Braille Detection & AI Assistant")
    gr.Markdown("Upload braille images for detection or chat with the AI assistant")
    
    with gr.Tab("📷 Braille Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Braille Image")
                confidence_slider = gr.Slider(0.1, 1.0, value=0.4, label="Min Confidence")
                detect_btn = gr.Button("🔍 Detect Braille", variant="primary")
            
            with gr.Column():
                annotated_output = gr.Image(label="Detected Braille with Bounding Boxes", type="pil")
        
        with gr.Row():
            with gr.Column():
                text_output = gr.Textbox(label="Detected Text", lines=3)
                explanation_output = gr.Textbox(label="AI Explanation", lines=5)
            with gr.Column():
                confidence_output = gr.Number(label="Confidence Score")
                rows_output = gr.JSON(label="Detected Rows")
        
        detect_btn.click(
            process_braille_image,
            inputs=[image_input, confidence_slider],
            outputs=[text_output, explanation_output, confidence_output, rows_output, annotated_output]
        )
    
    with gr.Tab("💬 AI Chat"):
        chatbot = gr.Chatbot(label="Chat with AI Assistant")
        msg = gr.Textbox(label="Your Message", placeholder="Ask about braille or anything else...")
        clear = gr.Button("Clear Chat")
        
        msg.submit(chat_with_assistant, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: [], None, chatbot, queue=False)
    
    with gr.Tab("🔌 API Testing"):
        gr.Markdown("### Test API Endpoints")
        api_input = gr.Textbox(
            label="JSON Request",
            placeholder='{"task": "chat", "message": "What is braille?"}',
            lines=5
        )
        api_btn = gr.Button("Send API Request")
        api_output = gr.JSON(label="API Response")
        
        api_btn.click(process_api_request, inputs=api_input, outputs=api_output)
        
        gr.Markdown("""
        ### API Examples:
        
        **Chat:**
        ```json
        {"task": "chat", "message": "What is braille?"}
        ```
        
        **Braille Detection:**
        ```json
        {
            "task": "detect_braille",
            "image_base64": "base64_encoded_image_data",
            "min_confidence": 0.4
        }
        ```
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)