# debug_braille.py - Quick debugging script for Braille detection issues
import os
import json
import base64
import requests
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv() 
def check_environment():
    """Check environment configuration"""
    print(" Environment Check:")
    print(f"   ROBOFLOW_API_KEY: {' Set' if os.getenv('ROBOFLOW_API_KEY') else ' Missing'}")
    
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if api_key:
        print(f"   API Key length: {len(api_key)} characters")
        print(f"   API Key preview: {api_key[:8]}...{api_key[-4:]}")
        
        if len(api_key) < 30:
            print("     WARNING: API key seems too short (should be 32+ chars)")
    
    return api_key

def test_simple_request(api_key, workspace="braille-to-text-0xo2p", version="1"):
    """Test the most basic Roboflow API request"""
    print(f"\n Testing Basic API Request:")
    print(f"   Workspace: {workspace}")
    print(f"   Version: {version}")
    
    # Create minimal test image
    test_img = Image.new('RGB', (100, 100), 'white')
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    print(f"   Test image size: {len(img_bytes)} bytes")
    
    # Test endpoint
    url = f"https://detect.roboflow.com/{workspace}/{version}"
    
    payload = {
        "api_key": api_key,
        "image": img_b64
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"   Endpoint: {url}")
    print(f"   Payload keys: {list(payload.keys())}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"   Response Status: {response.status_code}")
        print(f"   Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"    SUCCESS!")
                print(f"   Response keys: {list(data.keys())}")
                
                if 'predictions' in data:
                    print(f"   Predictions count: {len(data['predictions'])}")
                    
                return data
            except json.JSONDecodeError:
                print(f"     Status 200 but invalid JSON")
                print(f"   Raw response: {response.text[:200]}")
        else:
            print(f"    FAILED: {response.status_code}")
            print(f"   Error message: {response.text}")
            
            # Check for common error patterns
            error_text = response.text.lower()
            if 'api key' in error_text:
                print("    Suggestion: Check your API key")
            elif 'workspace' in error_text or 'model' in error_text:
                print("   Suggestion: Check workspace name and model version")
            elif 'quota' in error_text or 'limit' in error_text:
                print("   Suggestion: You may have exceeded API limits")
            
    except requests.exceptions.Timeout:
        print("    Request timed out")
    except requests.exceptions.ConnectionError:
        print("    Connection error - check internet connection")
    except Exception as e:
        print(f"    Unexpected error: {e}")
    
    return None

def check_workspace_info(api_key, workspace="braille-to-text-0xo2p"):
    """Check if workspace and models exist"""
    print(f"\n Checking Workspace: {workspace}")
    
    # Try to get workspace info (this might not work with all API keys)
    info_url = f"https://api.roboflow.com/{workspace}"
    
    try:
        response = requests.get(f"{info_url}?api_key={api_key}", timeout=10)
        
        if response.status_code == 200:
            print("   Workspace accessible")
            try:
                data = response.json()
                if 'versions' in data:
                    print(f"   Available versions: {data['versions']}")
            except:
                pass
        else:
            print(f"    Workspace info not accessible: {response.status_code}")
    except:
        print("    Could not check workspace info")

def test_alternative_endpoints(api_key, workspace="braille-to-text-0xo2p"):
    """Test alternative API endpoints"""
    print(f"\n Testing Alternative Endpoints:")
    
    # Create test image
    test_img = Image.new('RGB', (100, 100), 'white')
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Different endpoints to try
    endpoints = [
        f"https://detect.roboflow.com/{workspace}/1",
        f"https://detect.roboflow.com/{workspace}/2", 
        f"https://detect.roboflow.com/{workspace}/3",
        f"https://api.roboflow.com/{workspace}/1/predict",
        f"https://api.roboflow.com/{workspace}/2/predict"
    ]
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n   Endpoint {i}: {endpoint}")
        
        try:
            payload = {
                "api_key": api_key,
                "image": img_b64,
                "confidence": 0.1
            }
            
            response = requests.post(
                endpoint, 
                json=payload, 
                headers={"Content-Type": "application/json"}, 
                timeout=15
            )
            
            print(f"      Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"       Working endpoint found!")
                try:
                    data = response.json()
                    print(f"      Response keys: {list(data.keys())}")
                    return endpoint, data
                except:
                    print(f"        Valid status but invalid JSON")
            elif response.status_code == 404:
                print(f"       Model/endpoint not found")
            else:
                print(f"       Error: {response.text[:100]}")
                
        except Exception as e:
            print(f"       Exception: {e}")
    
    return None, None

def main():
    """Main debugging function"""
    print(" Braille Detection Debugging Tool")
    print("=" * 50)
    
    # Step 1: Check environment
    api_key = check_environment()
    
    if not api_key:
        print("\n Cannot proceed without API key")
        print("Please set your Roboflow API key:")
        print("export ROBOFLOW_API_KEY='your_api_key_here'")
        return
    
    # Step 2: Check workspace
    check_workspace_info(api_key)
    
    # Step 3: Test basic request
    result = test_simple_request(api_key)
    
    if result:
        print("\n Basic API test successful!")
        print("Your detector should work. The issue might be in your code logic.")
        return
    
    # Step 4: Try alternative endpoints
    print("\n Basic test failed, trying alternatives...")
    working_endpoint, result = test_alternative_endpoints(api_key)
    
    if working_endpoint:
        print(f"\n Found working endpoint: {working_endpoint}")
        print("Update your detector to use this endpoint.")
        return
    
    # Step 5: Final recommendations
    print("\n Debugging Recommendations:")
    print("1. Verify your API key is correct at https://roboflow.com/account")
    print("2. Check if your workspace name is exactly: 'braille-to-text-0xo2p'")
    print("3. Verify your model has been trained and deployed")
    print("4. Check if you have API usage remaining in your Roboflow account")
    print("5. Try with a real braille image instead of test patterns")
    
    print("\n Quick fixes to try:")
    print("- Update workspace name in your code")
    print("- Try different model versions (1, 2, 3, etc.)")
    print("- Use multipart form upload instead of JSON")
    print("- Lower the confidence threshold to 0.01")

def create_test_braille_image():
    """Create a more realistic test image with braille-like patterns"""
    print("\n  Creating realistic braille test image...")
    
    # Create image with braille-like dot patterns
    img = Image.new('RGB', (300, 150), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw braille-like patterns (6-dot cells)
    dot_radius = 3
    cell_width = 15
    cell_height = 20
    
    # Pattern for letter 'A' in braille (dot 1)
    patterns = [
        [1, 0, 0, 0, 0, 0],  # A
        [1, 1, 0, 0, 0, 0],  # B  
        [1, 0, 0, 1, 0, 0],  # C
        [1, 0, 0, 1, 1, 0],  # D
        [1, 0, 0, 0, 1, 0],  # E
    ]
    
    for col, pattern in enumerate(patterns):
        x_base = 50 + col * (cell_width + 10)
        y_base = 50
        
        # Draw 6 dot positions (2 columns, 3 rows)
        dot_positions = [
            (0, 0), (0, 10), (0, 20),  # Left column
            (8, 0), (8, 10), (8, 20)   # Right column  
        ]
        
        for i, (dx, dy) in enumerate(dot_positions):
            if i < len(pattern) and pattern[i]:
                x = x_base + dx
                y = y_base + dy
                draw.ellipse([x-dot_radius, y-dot_radius, 
                            x+dot_radius, y+dot_radius], fill='black')
    
    # Save test image
    img.save('test_braille_pattern.jpg', 'JPEG')
    print("   Saved test image: test_braille_pattern.jpg")
    
    return img

def test_with_real_image(api_key):
    """Test detection with a realistic braille image"""
    print("\n Testing with realistic braille pattern...")
    
    # Create test image
    test_img = create_test_braille_image()
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Test with the standard endpoint
    url = "https://detect.roboflow.com/braille-to-text-0xo2p/1"
    
    payload = {
        "api_key": api_key,
        "image": img_b64,
        "confidence": 0.01,  # Very low confidence
        "overlap": 0.5
    }
    
    try:
        response = requests.post(url, json=payload, 
                               headers={"Content-Type": "application/json"}, 
                               timeout=30)
        
        print(f"   Response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"    Detected {len(predictions)} objects!")
            
            for i, pred in enumerate(predictions[:5]):  # Show first 5
                print(f"      {i+1}: {pred.get('class', '?')} "
                      f"({pred.get('confidence', 0):.2f})")
            
            return True
        else:
            print(f"    Failed: {response.text[:200]}")
            
    except Exception as e:
        print(f"    Error: {e}")
    
    return False

if __name__ == "__main__":
    main()
    
    # Also test with realistic image
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if api_key:
        test_with_real_image(api_key)