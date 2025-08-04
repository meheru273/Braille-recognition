#!/usr/bin/env python3
"""
Setup script for Braille Detection System API Key
"""

import os
import sys
from pathlib import Path

def setup_api_key():
    """Interactive setup for API key"""
    print("🔑 Braille Detection System - API Key Setup")
    print("=" * 50)
    
    # Check current status
    current_key = os.getenv("ROBOFLOW_API_KEY", "RzOXFbriJONcee7MHKN8")
    print(f"Current API key length: {len(current_key)} characters")
    
    if len(current_key) >= 32:
        print("✅ Your API key appears to be valid!")
        print(f"Key preview: {current_key[:5]}...{current_key[-5:]}")
        return True
    
    print("\n❌ Your API key appears to be incomplete or invalid.")
    print("Roboflow API keys are typically 32+ characters long.")
    
    print("\n📋 To get a valid API key:")
    print("1. Go to https://roboflow.com/account")
    print("2. Sign in to your account")
    print("3. Navigate to 'API Keys' section")
    print("4. Copy your full API key")
    
    print("\n🔧 Setup Options:")
    print("1. Set environment variable (recommended)")
    print("2. Update config file directly")
    print("3. Test with current key")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            return setup_environment_variable()
        elif choice == "2":
            return update_config_file()
        elif choice == "3":
            return test_current_key()
        elif choice == "4":
            print("Setup cancelled.")
            return False
        else:
            print("Invalid choice. Please enter 1-4.")

def setup_environment_variable():
    """Set up environment variable"""
    print("\n🔧 Setting up environment variable...")
    
    api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return False
    
    if len(api_key) < 20:
        print("❌ API key appears too short. Please check your key.")
        return False
    
    print(f"\n✅ API key received (length: {len(api_key)} characters)")
    print(f"Key preview: {api_key[:5]}...{api_key[-5:]}")
    
    # Set environment variable
    os.environ["ROBOFLOW_API_KEY"] = api_key
    
    print("\n🔧 Environment variable set for this session.")
    print("To make it permanent:")
    
    if sys.platform == "win32":
        print("Windows:")
        print(f'  setx ROBOFLOW_API_KEY "{api_key}"')
    else:
        print("Linux/Mac:")
        print(f'  export ROBOFLOW_API_KEY="{api_key}"')
        print("  Add to ~/.bashrc or ~/.zshrc for permanence")
    
    # Test the key
    return test_api_key(api_key)

def update_config_file():
    """Update config file directly"""
    print("\n🔧 Updating config file...")
    
    api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return False
    
    if len(api_key) < 20:
        print("❌ API key appears too short. Please check your key.")
        return False
    
    # Update config.py
    config_path = Path("config.py")
    if not config_path.exists():
        print("❌ config.py not found.")
        return False
    
    try:
        with open(config_path, "r") as f:
            content = f.read()
        
        # Replace the API key line
        import re
        new_content = re.sub(
            r'ROBOFLOW_API_KEY = os\.getenv\("ROBOFLOW_API_KEY", "[^"]*"\)',
            f'ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "{api_key}")',
            content
        )
        
        with open(config_path, "w") as f:
            f.write(new_content)
        
        print("✅ Config file updated successfully!")
        return test_api_key(api_key)
        
    except Exception as e:
        print(f"❌ Error updating config file: {e}")
        return False

def test_current_key():
    """Test the current API key"""
    current_key = os.getenv("ROBOFLOW_API_KEY", "RzOXFbriJONcee7MHKN8")
    return test_api_key(current_key)

def test_api_key(api_key):
    """Test if the API key is valid"""
    print(f"\n🧪 Testing API key...")
    
    try:
        import requests
        
        # Test with a simple request
        test_url = f"https://detect.roboflow.com/braille-to-text-0xo2p/1"
        response = requests.get(f"{test_url}?api_key={api_key}", timeout=10)
        
        print(f"📡 API test response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API key is valid!")
            return True
        elif response.status_code == 401:
            print("❌ API key is invalid or expired")
            return False
        elif response.status_code == 403:
            print("❌ API key doesn't have access to this model")
            return False
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def main():
    """Main function"""
    try:
        success = setup_api_key()
        
        if success:
            print("\n🎉 Setup completed successfully!")
            print("You can now run the detector:")
            print("  python test_detector.py")
        else:
            print("\n❌ Setup failed. Please check your API key and try again.")
            
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")

if __name__ == "__main__":
    main() 