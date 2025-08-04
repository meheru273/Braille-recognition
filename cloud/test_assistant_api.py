#!/usr/bin/env python3
"""
Test script for Assistant API with working detector
"""

import os
import sys
import json
import base64
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_assistant_api():
    """Test the assistant API with working detector"""
    
    print("🧪 Testing Assistant API with Working Detector")
    print("=" * 60)
    
    try:
        # Import the assistant API components
        sys.path.append('assistant-api')
        from api.index import BrailleDetector, BrailleAssistant
        
        # Test detector initialization
        print("🔧 Testing Detector Initialization...")
        detector = BrailleDetector()
        print(f"✅ Detector initialized successfully")
        print(f"   Model: {detector.workspace_name}")
        print(f"   API Key: {detector.api_key[:5]}...{detector.api_key[-5:]}")
        print(f"   Using SDK: {detector.use_sdk}")
        
        # Test with a sample image
        test_image = "../test/before.jpg"
        if os.path.exists(test_image):
            print(f"\n🖼️ Testing Detection with: {test_image}")
            
            # Read image as bytes
            with open(test_image, 'rb') as f:
                image_bytes = f.read()
            
            # Test detection
            result = detector.detect_braille_from_bytes(image_bytes)
            
            if result and "error" not in result:
                print(f"✅ Detection successful!")
                
                # Extract predictions
                predictions = detector.extract_predictions(result)
                print(f"📊 Found {len(predictions)} predictions")
                
                # Organize text
                text_rows = detector.organize_text_by_rows(predictions, min_confidence=0.1)
                print(f"📝 Organized into {len(text_rows)} text rows")
                
                if text_rows:
                    print(f"📄 Sample text rows:")
                    for i, row in enumerate(text_rows[:3]):  # Show first 3
                        print(f"   {i+1}. {row}")
                
                # Test assistant processing
                print(f"\n🤖 Testing Assistant Processing...")
                assistant = BrailleAssistant()
                processing_result = assistant.process_braille_strings(text_rows)
                
                print(f"✅ Processing successful!")
                print(f"   Text: {processing_result.text}")
                print(f"   Explanation: {processing_result.explanation}")
                print(f"   Confidence: {processing_result.confidence:.2f}")
                
            else:
                print(f"❌ Detection failed")
                if result and "error" in result:
                    print(f"   Error: {result['error']}")
                    print(f"   Detail: {result.get('detail', 'No details')}")
        else:
            print(f"⚠️ Test image not found: {test_image}")
        
        print(f"\n🎉 Assistant API test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_assistant_api() 