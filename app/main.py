import os
from detector import BrailleDetector
from assistant import BrailleAssistant

def test_braille_system():
    """Test the full braille detection and processing system"""
    print("\n===== Starting Braille System Test =====")
    
    # Initialize components
    detector = BrailleDetector()
    assistant = BrailleAssistant()
    
    # Test configuration
    test_image = "test/before.jpg"
    output_image = "test/annotated_result.png"
    
    print(f"\n[1/3] Detecting Braille in: {test_image}")
    
    # Verify test image exists
    if not os.path.exists(test_image):
        print(f" Error: Test image not found at {test_image}")
        print("Please add a test image or update the path")
        return
    
    # Run braille detection
    result = detector.detect_braille(test_image)
    
    if not result:
        print(" Detection failed - no results returned")
        return
    
    predictions = detector.extract_predictions(result)
    print(f" Detected {len(predictions)} braille characters")
    
    # Create annotated image
    if predictions:
        success = detector.create_annotated_image(
            test_image, 
            predictions, 
            output_image
        )
        if success:
            print(f" Annotation saved to: {output_image}")
    
    # Organize characters into text rows
    text_rows = detector.organize_text_by_rows(predictions)
    print("\n[2/3] Detected Text Rows:")
    for i, row in enumerate(text_rows, 1):
        print(f"Row {i}: {row}")
    
    # Process with AI assistant
    print("\n[3/3] Processing with AI Assistant...")
    braille_result = assistant.process_braille_strings(text_rows)
    
    # Print results
    print("\n Processing Results:")
    print(f"Text: {braille_result.text}")
    print(f"Explanation: {braille_result.explanation}")
    print(f"Confidence: {braille_result.confidence:.2f}")
    
    # Test chat functionality
    print("\n Testing Chat Functionality:")
    response = assistant.chat("What does braille look like?")
    print(f"Assistant: {response}")
    
    print("\n===== Test Completed =====")

if __name__ == "__main__":
    test_braille_system()