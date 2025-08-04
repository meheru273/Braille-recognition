# main.py - Complete offline testing for Braille Detection APIs
import os
import asyncio
import tempfile
import uuid
from datetime import datetime
from dotenv import load_dotenv
load_dotenv() 

# Import with error handling
try:
    from detector import BrailleDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Detector import failed: {e}")
    DETECTOR_AVAILABLE = False

try:
    from assistant import BrailleAssistant
    ASSISTANT_AVAILABLE = True
except ImportError as e:
    print(f"Assistant import failed: {e}")
    ASSISTANT_AVAILABLE = False

# Firebase import with error handling
try:
    from firebaseservice import firebase_service, BrailleDetectionResult
    FIREBASE_AVAILABLE = True
except ImportError as e:
    print(f"Firebase import failed: {e}")
    FIREBASE_AVAILABLE = False
    # Create dummy objects to prevent crashes
    class DummyFirebaseService:
        def is_connected(self): return False
    class DummyBrailleDetectionResult:
        pass
    firebase_service = DummyFirebaseService()
    BrailleDetectionResult = DummyBrailleDetectionResult

async def test_detector_api():
    """Test the Braille Detector API offline"""
    print("\n[DETECTOR] Testing Braille Detector...")
    print("-" * 40)
    
    if not DETECTOR_AVAILABLE:
        print("[FAIL] Detector module not available")
        return False
    
    try:
        detector = BrailleDetector()
        print("[PASS] Detector initialized successfully")
        
        # Check if we have the required API key
        if not hasattr(detector, 'api_key') or not detector.api_key:
            print("[FAIL] ROBOFLOW_API_KEY not found in environment variables")
            return False
        else:
            print("[PASS] Roboflow API key found")
        
        # Test with a sample image (you need to provide this)
        test_image = "test/before2.jpg"  # Update path as needed
        
        if not os.path.exists(test_image):
            print(f"[WARN] Test image not found: {test_image}")
            print("Please provide a test image to fully test the detector")
            return True  # API is configured but no test image
        
        print(f"[TEST] Testing with image: {test_image}")
        
        # Run detection
        result = detector.detect_braille_with_fallback(test_image)
        
        if result:
            predictions = detector.extract_predictions(result)
            print(f"[PASS] Detection successful - found {len(predictions)} characters")
            
            if predictions:
                # Test text organization
                rows = detector.organize_text_by_rows(predictions)
                print(f"[PASS] Text organized into {len(rows)} rows")
                
                # Test annotation creation
                output_path = "test_output_annotated.png"
                if detector.create_annotated_image(test_image, predictions, output_path):
                    print(f"[PASS] Annotated image created: {output_path}")
                    # Clean up
                    if os.path.exists(output_path):
                        os.remove(output_path)
                
                return True
            else:
                print("[WARN] No valid predictions extracted")
                return True
        else:
            print("[FAIL] Detection failed - API might be down or image invalid")
            return False
            
    except Exception as e:
        print(f"[FAIL] Detector test failed: {e}")
        return False

async def test_assistant_api():
    """Test the Braille Assistant API offline"""
    print("\n[ASSISTANT] Testing Braille Assistant...")
    print("-" * 40)
    
    if not ASSISTANT_AVAILABLE:
        print("[FAIL] Assistant module not available")
        return False
    
    try:
        assistant = BrailleAssistant()
        print("[PASS] Assistant initialized")
        
        # Check if we can get status
        try:
            status = assistant.get_status()
            print(f"[PASS] Assistant status - Mode: {status.get('mode', 'unknown')}")
            print(f"[INFO] API Available: {status.get('api_available', False)}")
            print(f"[INFO] Provider: {status.get('provider', 'unknown')}")
        except Exception as e:
            print(f"[WARN] Could not get assistant status: {e}")
        
        # Test braille processing
        test_braille_strings = ["hello", "world", "test"]
        print(f"\n[TEST] Testing braille processing with: {test_braille_strings}")
        
        try:
            result = assistant.process_braille_strings(test_braille_strings)
            print(f"[PASS] Processed text: '{result.text}'")
            print(f"[PASS] Explanation: {result.explanation[:100]}...")
            print(f"[PASS] Confidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"[WARN] Braille processing failed: {e}")
        
        # Test chat functionality
        print(f"\n[TEST] Testing chat functionality...")
        test_messages = [
            "Hello, how are you?",
            "What is braille?",
            "Help me understand this text"
        ]
        
        for i, message in enumerate(test_messages, 1):
            try:
                print(f"\nTest {i}: '{message}'")
                response = assistant.chat(message, f"test_thread_{i}")
                print(f"Response: {response[:100]}...")
            except Exception as e:
                print(f"Chat test {i} failed: {e}")
        
        # Test text processing
        print(f"\n[TEST] Testing text processing...")
        try:
            test_text = "This is a test sentence for processing."
            processed = assistant.process_text(test_text, "explain")
            print(f"[PASS] Text processing successful: {processed[:100]}...")
        except Exception as e:
            print(f"[WARN] Text processing failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Assistant test failed: {e}")
        return False

async def test_firebase_service():
    """Test Firebase service connection"""
    print("\n[FIREBASE] Testing Firebase Service...")
    print("-" * 40)
    
    if not FIREBASE_AVAILABLE:
        print("[FAIL] Firebase import failed - check firebase_service.py and firebase-admin installation")
        return False
    
    try:
        if firebase_service.is_connected():
            print("[PASS] Firebase connected successfully")
            
            # Test user creation
            test_user_id = "test_user_" + str(uuid.uuid4())[:8]
            print(f"[TEST] Testing user creation with ID: {test_user_id}")
            
            user_created = await firebase_service.create_user_profile(
                user_id=test_user_id,
                email="test@example.com",
                display_name="Test User"
            )
            
            if user_created:
                print("[PASS] User profile created successfully")
            else:
                print("[WARN] User profile creation failed")
            
            # Test chat thread creation
            print("[TEST] Testing chat thread creation...")
            thread_id = await firebase_service.create_chat_thread(test_user_id)
            print(f"[PASS] Chat thread created: {thread_id}")
            
            # Test adding chat message
            message_added = await firebase_service.add_chat_message(
                user_id=test_user_id,
                thread_id=thread_id,
                user_message="Test message",
                assistant_response="Test response"
            )
            
            if message_added:
                print("[PASS] Chat message added successfully")
            else:
                print("[WARN] Chat message addition failed")
            
            # Test braille detection storage
            print("[TEST] Testing braille detection storage...")
            test_result = BrailleDetectionResult(
                session_id=str(uuid.uuid4()),
                user_id=test_user_id,
                filename="test_image.jpg",
                detected_text="test braille text",
                explanation="Test explanation",
                confidence=0.85,
                raw_detections=["test", "braille"],
                timestamp=datetime.now(),
                processing_status="completed"
            )
            
            stored = await firebase_service.store_braille_detection(test_result)
            if stored:
                print("[PASS] Braille detection stored successfully")
            else:
                print("[WARN] Braille detection storage failed")
            
            # Clean up test data
            print("[CLEANUP] Cleaning up test data...")
            await firebase_service.delete_chat_thread(test_user_id, thread_id)
            
            return True
            
        else:
            print("[WARN] Firebase not connected - check your configuration")
            print("Required environment variables:")
            print("  - FIREBASE_CREDENTIALS")
            print("  - FIREBASE_PROJECT_ID") 
            print("  - FIREBASE_STORAGE_BUCKET")
            return False
            
    except Exception as e:
        print(f"[FAIL] Firebase test failed: {e}")
        return False

async def test_complete_pipeline():
    """Test the complete braille detection pipeline"""
    print("\n[PIPELINE] Testing Complete Pipeline...")
    print("-" * 40)
    
    try:
        # Initialize components
        if not DETECTOR_AVAILABLE:
            print("[SKIP] Detector not available, using mock data")
            detector = None
        else:
            detector = BrailleDetector()
        
        if not ASSISTANT_AVAILABLE:
            print("[SKIP] Assistant not available")
            return False
        else:
            assistant = BrailleAssistant()
        
        # Mock detection results for testing
        mock_detected_strings = ["hello", "world", "this", "is", "braille"]
        print(f"[TEST] Using mock detected strings: {mock_detected_strings}")
        
        # Process through assistant
        result = assistant.process_braille_strings(mock_detected_strings)
        print(f"[PASS] Assistant processing complete")
        print(f"  Text: '{result.text}'")
        print(f"  Confidence: {result.confidence:.2f}")
        
        # Store in Firebase if available
        if FIREBASE_AVAILABLE and firebase_service.is_connected():
            test_user_id = "eApEeqUDMHRuBEii9h7Ph1Tx1qi1"  # Your actual user ID
            
            detection_result = BrailleDetectionResult(
                session_id=str(uuid.uuid4()),
                user_id=test_user_id,
                filename="pipeline_test.jpg",
                detected_text=result.text,
                explanation=result.explanation,
                confidence=result.confidence,
                raw_detections=mock_detected_strings,
                timestamp=datetime.now(),
                processing_status="completed"
            )
            
            stored = await firebase_service.store_braille_detection(detection_result)
            if stored:
                print("[PASS] Result stored in Firebase")
            else:
                print("[WARN] Firebase storage failed")
        else:
            print("[SKIP] Firebase not available for storage test")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Pipeline test failed: {e}")
        return False

async def check_environment_variables():
    """Check required environment variables"""
    print("\n[ENV] Checking Environment Variables...")
    print("-" * 40)
    
    required_vars = {
        "ROBOFLOW_API_KEY": "Braille detection API",
        "GROQ_API_KEY": "LLM processing (Groq)",
        "OPENAI_API_KEY": "LLM processing (OpenAI)",
        "FIREBASE_CREDENTIALS": "Firebase authentication", 
        "FIREBASE_PROJECT_ID": "Firebase project",
        "FIREBASE_STORAGE_BUCKET": "Firebase storage"
    }
    
    found_vars = {}
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            found_vars[var] = f"[PASS] {description}"
            # Don't print the actual value for security
            print(f"[PASS] {var}: Set ({len(value)} chars)")
        else:
            found_vars[var] = f"[FAIL] {description}"
            print(f"[FAIL] {var}: Not set")
    
    # Check if we have at least one LLM API key
    has_llm = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    print(f"\n[SUMMARY] Summary:")
    print(f"  LLM API Available: {'[PASS]' if has_llm else '[FAIL]'}")
    print(f"  Detector API Available: {'[PASS]' if os.getenv('ROBOFLOW_API_KEY') else '[FAIL]'}")
    print(f"  Firebase Available: {'[PASS]' if all(os.getenv(v) for v in ['FIREBASE_CREDENTIALS', 'FIREBASE_PROJECT_ID']) else '[FAIL]'}")
    
    return found_vars

async def main():
    """Main testing function"""
    print("Braille Detection System - Offline API Testing")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track test results
    test_results = {}
    
    # Check environment variables first
    print("\n" + "=" * 60)
    env_vars = await check_environment_variables()
    
    # Test each component
    print("\n" + "=" * 60)
    test_results["detector"] = await test_detector_api()
    
    print("\n" + "=" * 60)
    test_results["assistant"] = await test_assistant_api()
    
    print("\n" + "=" * 60)
    test_results["firebase"] = await test_firebase_service()
    
    print("\n" + "=" * 60)
    test_results["pipeline"] = await test_complete_pipeline()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    
    for component, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{component.upper():12} : {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All systems operational!")
    elif passed_tests > 0:
        print("Partial functionality available")
    else:
        print("System not operational - check configuration")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Add any test images or configuration here
    print("Tips for testing:")
    print("  1. Place test braille images in 'test_images/' directory")
    print("  2. Set required environment variables")
    print("  3. Ensure internet connection for API calls")
    print("  4. Check Firebase configuration")
    print("  5. Make sure all required files are in the same directory:")
    print("     - main.py")
    print("     - firebase_service.py")
    print("     - detector.py") 
    print("     - assistant.py")
    print()
    
    # Run the tests
    asyncio.run(main())