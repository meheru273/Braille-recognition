"""
Configuration file for Braille Detection System
"""

import os

# Roboflow Configuration - Multiple Models
ROBOFLOW_API_KEY_1 = os.getenv("ROBOFLOW_API_KEY_1", "RzOXFbriJONcee7MHKN8")
ROBOFLOW_API_KEY_2 = os.getenv("ROBOFLOW_API_KEY_2", "NRmMU6uU07XILRg52e7n")

# Model Configurations
MODEL_1 = {
    "workspace": "braille-to-text-0xo2p",
    "workflow_id": "custom-workflow",
    "api_key": ROBOFLOW_API_KEY_1
}

MODEL_2 = {
    "workspace": "braille-image", 
    "workflow_id": "custom-workflow",
    "api_key": ROBOFLOW_API_KEY_2
}

# Default model to use
DEFAULT_MODEL = "MODEL_1"  # or "MODEL_2"

# API Endpoints
ROBOFLOW_API_URL = "https://serverless.roboflow.com"

# Detection Settings
DETECTION_CONFIDENCE_THRESHOLD = 0.1
DETECTION_OVERLAP_THRESHOLD = 0.5
MIN_CONFIDENCE_FOR_TEXT = 0.4

# AI Assistant Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Settings
GROQ_MODEL = "llama-3.1-8b-instant"
OPENAI_MODEL = "gpt-3.5-turbo"

# API Timeouts
REQUEST_TIMEOUT = 30
BATCH_TIMEOUT = 20

# Image Processing
MAX_IMAGE_SIZE = (1920, 1080)
IMAGE_QUALITY = 85
MAX_FILE_SIZE = 4 * 1024 * 1024  # 4MB

# Debug Settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Use inference_sdk (set to False if you want to use HTTP requests for Vercel)
USE_INFERENCE_SDK = True

def get_active_model():
    """Get the currently active model configuration"""
    if DEFAULT_MODEL == "MODEL_1":
        return MODEL_1
    else:
        return MODEL_2

def validate_config():
    """Validate configuration and return status"""
    issues = []
    
    # Check API keys
    if not ROBOFLOW_API_KEY_1 or len(ROBOFLOW_API_KEY_1) < 10:
        issues.append("Model 1 API key is missing or too short")
    
    if not ROBOFLOW_API_KEY_2 or len(ROBOFLOW_API_KEY_2) < 10:
        issues.append("Model 2 API key is missing or too short")
    
    # Check workspaces
    if not MODEL_1["workspace"]:
        issues.append("Model 1 workspace is not configured")
    
    if not MODEL_2["workspace"]:
        issues.append("Model 2 workspace is not configured")
    
    active_model = get_active_model()
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "model_1_configured": bool(ROBOFLOW_API_KEY_1 and len(ROBOFLOW_API_KEY_1) >= 10),
        "model_2_configured": bool(ROBOFLOW_API_KEY_2 and len(ROBOFLOW_API_KEY_2) >= 10),
        "ai_configured": bool(GROQ_API_KEY or OPENAI_API_KEY),
        "active_model": active_model["workspace"],
        "active_api_key_length": len(active_model["api_key"]) if active_model["api_key"] else 0,
        "use_inference_sdk": USE_INFERENCE_SDK
    }

def get_api_key_info(api_key):
    """Get masked API key info for logging"""
    if not api_key:
        return "Not configured"
    
    if len(api_key) < 10:
        return f"Too short ({len(api_key)} chars)"
    
    return f"{api_key[:5]}...{api_key[-5:]} ({len(api_key)} chars)"

if __name__ == "__main__":
    # Print configuration status
    print("🔧 Configuration Status")
    print("=" * 40)
    
    config_status = validate_config()
    
    print(f"✅ Valid: {config_status['valid']}")
    print(f"🔑 Model 1: {'✅ Configured' if config_status['model_1_configured'] else '❌ Not configured'}")
    print(f"🔑 Model 2: {'✅ Configured' if config_status['model_2_configured'] else '❌ Not configured'}")
    print(f"🤖 AI Assistant: {'✅ Configured' if config_status['ai_configured'] else '❌ Not configured'}")
    print(f"🎯 Active Model: {config_status['active_model']}")
    print(f"📏 Active API Key Length: {config_status['active_api_key_length']}")
    print(f"🔧 Using Inference SDK: {config_status['use_inference_sdk']}")
    
    print(f"\n📋 Model Details:")
    print(f"   • Model 1: {MODEL_1['workspace']} (Key: {get_api_key_info(MODEL_1['api_key'])})")
    print(f"   • Model 2: {MODEL_2['workspace']} (Key: {get_api_key_info(MODEL_2['api_key'])})")
    
    if config_status['issues']:
        print(f"\n❌ Issues Found:")
        for issue in config_status['issues']:
            print(f"   • {issue}")
    else:
        print(f"\n✅ Configuration looks good!")
    
    print(f"\n📋 Settings:")
    print(f"   • API URL: {ROBOFLOW_API_URL}")
    print(f"   • Confidence Threshold: {DETECTION_CONFIDENCE_THRESHOLD}")
    print(f"   • Request Timeout: {REQUEST_TIMEOUT}s")
    print(f"   • Debug Mode: {DEBUG_MODE}") 