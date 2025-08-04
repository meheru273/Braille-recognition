#!/usr/bin/env python3
"""
Test script for Braille Detector
Tests the detector with sample images to verify it's working correctly.
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to the path so we can import the detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import BrailleDetector

def test_detector():
    """Test the detector with sample images"""
    
    print("🔍 Testing Braille Detector")
    print("=" * 50)
    
    # Initialize detector
    try:
        detector = BrailleDetector()
        print(f"✅ Detector initialized successfully")
        print(f"📋 API Key length: {len(detector.api_key)}")
        print(f"🔑 API Key preview: {detector.api_key[:5]}...{detector.api_key[-5:]}")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False
    
    # Test with sample images
    test_images = [
        "../test/before.jpg",
        "../test/before2.jpg", 
        "../test/before4.jpg",
        "../test/before5.jpg"
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Test image not found: {image_path}")
            continue
            
        print(f"\n🖼️  Testing with: {image_path}")
        print("-" * 30)
        
        try:
            # Run detection
            result = detector.detect_braille_with_fallback(image_path)
            
            if result:
                print(f"✅ Detection successful")
                
                # Extract predictions
                predictions = detector.extract_predictions(result)
                print(f"📊 Found {len(predictions)} predictions")
                
                # Organize text
                text_rows = detector.organize_text_by_rows(predictions, min_confidence=0.1)
                print(f"📝 Organized into {len(text_rows)} text rows")
                
                if text_rows:
                    print(f"📄 Text rows: {text_rows}")
                else:
                    print("⚠️  No text rows found (may need to lower confidence threshold)")
                
                # Show some prediction details
                if predictions:
                    print(f"🔍 Sample predictions:")
                    for i, pred in enumerate(predictions[:3]):  # Show first 3
                        print(f"   {i+1}. Class: {pred.get('class', 'unknown')}, "
                              f"Confidence: {pred.get('confidence', 0):.2f}, "
                              f"Position: ({pred.get('x', 0):.0f}, {pred.get('y', 0):.0f})")
                
            else:
                print(f"❌ Detection failed - no result returned")
                
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏁 Testing complete")
    return True

def test_api_key():
    """Test if the API key is valid"""
    print("\n🔑 Testing API Key")
    print("-" * 30)
    
    try:
        detector = BrailleDetector()
        
        # Test with a simple request to check API key validity
        import requests
        
        # Try to access the model info
        test_url = f"https://detect.roboflow.com/braille-to-text-0xo2p/1"
        response = requests.get(f"{test_url}?api_key={detector.api_key}", timeout=10)
        
        print(f"📡 API test response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API key appears to be valid")
        elif response.status_code == 401:
            print("❌ API key is invalid or expired")
        elif response.status_code == 403:
            print("❌ API key doesn't have access to this model")
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Braille Detector Test Suite")
    print("=" * 50)
    
    # Test API key first
    api_key_valid = test_api_key()
    
    if api_key_valid:
        # Test detector functionality
        test_detector()
    else:
        print("\n⚠️  Skipping detector tests due to invalid API key")
        print("Please check your ROBOFLOW_API_KEY environment variable") 