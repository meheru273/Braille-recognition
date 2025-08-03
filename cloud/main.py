# main.py - Enhanced main function with Firebase integration
import os
import asyncio
from datetime import datetime
from detector import BrailleDetector
from assistant import BrailleAssistant
from firebase_service import firebase_service, BrailleDetectionResult

async def main():
    """Enhanced main function with Firebase database integration"""
    
    print("🔍 Braille Detection System with Firebase")
    print("=" * 50)
    
    # Initialize components
    print("Initializing components...")
    detector = BrailleDetector()
    assistant = BrailleAssistant()
    
    # Check Firebase connection
    if firebase_service.is_connected():
        print("✓ Firebase connected")
    else:
        print("⚠ Firebase not connected - running in offline mode")
    
    # Test configuration
    test_image = "before.jpg"
    output_image = "high_res_annotated.png"
    test_user_id = "test_user_123"  # In production, this comes from authentication
    
    print(f"\n📸 Processing: {test_image}")
    
    # Check if test image exists
    if not os.path.exists(test_image):
        print(f"❌ Error: {test_image} not found!")
        print("Please place your test image in the same directory as this script.")
        return
    
    # Run braille detection
    result = detector.detect_braille(test_image)
    
    if result:
        predictions = detector.extract_predictions(result)
        print(f"✓ Found {len(predictions)} characters")
        
        if predictions:
            # Create high-resolution annotated image
            success = detector.create_annotated_image(test_image, predictions, output_image)
            if success:
                print(f"✓ High-resolution image saved: {output_image}")
            
            # Organize into rows
            rows = detector.organize_text_by_rows(predictions)
            print(f"\n📝 Detected text ({len(rows)} rows):")
            for i, row in enumerate(rows, 1):
                print(f"  Row {i}: '{row}'")
            
            # Calculate statistics
            avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
            unique_chars = sorted(set(p['class'] for p in predictions))
            print(f"\n📊 Statistics:")
            print(f"  • Characters detected: {len(predictions)}")