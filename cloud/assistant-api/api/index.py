from http.server import BaseHTTPRequestHandler
import json
import os
import base64
import requests
from datetime import datetime
from typing import Dict, List
import io
from PIL import Image

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        
        # CORS headers
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        if self.path == '/api/health':
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'roboflow_configured': bool(os.getenv("ROBOFLOW_API_KEY")),
                'ai_configured': bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))
            }
            
            self.wfile.write(json.dumps(health_data).encode())
            
        else:
            # Serve HTML interface
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
    
    def do_POST(self):
        """Handle POST requests for image processing"""
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # For now, return a test response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Simple response to test the endpoint
            response_data = {
                'success': True,
                'detected_text': 'Test response - endpoint is working',
                'explanation': 'This is a test response to verify the API is accessible',
                'confidence': 0.9,
                'detection_count': 1,
                'total_time_ms': 100,
                'timestamp': datetime.now().isoformat()
            }
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                'success': False,
                'error': str(e),
                'detected_text': '',
                'explanation': f'Error: {str(e)}',
                'confidence': 0.0,
                'detection_count': 0,
                'total_time_ms': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

# HTML Template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Braille Recognition System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .upload-section {
            border: 2px dashed #ccc;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        .upload-btn:hover {
            transform: translateY(-2px);
        }
        .file-input { display: none; }
        .status {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
        .status.loading { background: #fff3cd; color: #856404; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }
        .detected-text {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 15px;
            font-family: monospace;
            font-size: 1.2em;
        }
        .explanation {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
        }
        .hidden { display: none; }
        .preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .stats {
            background: #f1f3f4;
            padding: 10px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔤 Braille Recognition</h1>
            <p>Upload an image containing braille text for recognition</p>
        </div>
        
        <div class="upload-section">
            <p>📁 Drop an image here or click to browse</p>
            <br>
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                Choose File
            </button>
            <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFile(event)">
        </div>
        
        <div id="status" class="status hidden"></div>
        <div id="preview" class="hidden">
            <img id="previewImage" class="preview" alt="Preview">
        </div>
        <div id="results" class="results hidden">
            <div>
                <h3>🔤 Detected Text</h3>
                <div id="detectedText" class="detected-text"></div>
            </div>
            <div>
                <h3>💡 Explanation</h3>
                <div id="explanation" class="explanation"></div>
            </div>
            <div id="stats" class="stats"></div>
        </div>
    </div>

    <script>
        function handleFile(event) {
            const file = event.target.files[0];
            if (file && file.type.startsWith('image/')) {
                processFile(file);
            } else {
                showStatus('Please select an image file', 'error');
            }
        }

        function processFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('previewImage').src = e.target.result;
                document.getElementById('preview').classList.remove('hidden');
            };
            reader.readAsDataURL(file);

            showStatus('Processing image...', 'loading');
            
            const formData = new FormData();
            formData.append('image', file);

            // Try the current path first, then fallback to /api
            fetch(window.location.pathname, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayResults(data);
                    showStatus('Processing complete!', 'success');
                } else {
                    showStatus('Error: ' + (data.error || 'Unknown error'), 'error');
                }
            })
            .catch(error => {
                showStatus('Network error: ' + error.message, 'error');
            });
        }

        function displayResults(data) {
            document.getElementById('detectedText').textContent = data.detected_text || 'No text detected';
            document.getElementById('explanation').textContent = data.explanation || 'No explanation available';
            
            const stats = `Detected ${data.detection_count} characters • Confidence: ${(data.confidence * 100).toFixed(1)}% • Processing time: ${data.total_time_ms}ms`;
            document.getElementById('stats').textContent = stats;
            
            document.getElementById('results').classList.remove('hidden');
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.classList.remove('hidden');
            
            if (type === 'success') {
                setTimeout(() => status.classList.add('hidden'), 3000);
            }
        }

        const uploadSection = document.querySelector('.upload-section');
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#667eea';
        });
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.style.borderColor = '#ccc';
        });
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#ccc';
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                processFile(files[0]);
            }
        });
    </script>
</body>
</html>'''