from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import os
from PIL import Image
import tempfile
from detector import BrailleDetector

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Serve the HTML test UI"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Braille Detection API Test</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .content {
            padding: 40px;
        }

        .upload-section {
            background: #f8f9ff;
            border: 3px dashed #4facfe;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .upload-section:hover {
            background: #f0f4ff;
            border-color: #2196f3;
            transform: translateY(-2px);
        }

        .upload-section.dragover {
            background: #e3f2fd;
            border-color: #1976d2;
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 4rem;
            color: #4facfe;
            margin-bottom: 20px;
        }

        .upload-text {
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 15px;
        }

        .file-input {
            display: none;
        }

        .upload-btn {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }

        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
        }

        .process-btn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            margin: 20px 0;
        }

        .process-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        }

        .process-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .preview-section {
            margin: 30px 0;
        }

        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }

        .results-section {
            margin-top: 30px;
        }

        .result-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #4facfe;
        }

        .result-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .detected-text {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #555;
            background: #f8f9ff;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            border-left: 4px solid #4facfe;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .stat-item {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
        }

        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #4facfe;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #ffebee;
            border-left-color: #f44336;
            color: #c62828;
        }

        .success {
            background: #e8f5e8;
            border-left-color: #4caf50;
            color: #2e7d32;
        }

        .images-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }

        .image-card {
            text-align: center;
        }

        .image-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }

        @media (max-width: 768px) {
            .images-container {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔤 Braille Detection API</h1>
            <p>Upload an image with Braille text to detect and convert it to readable text</p>
        </div>

        <div class="content">
            <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Click to select an image or drag and drop</div>
                <button class="upload-btn" type="button">Choose File</button>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
            </div>

            <div class="preview-section" id="previewSection" style="display: none;">
                <h3>Selected Image:</h3>
                <img id="previewImage" class="preview-image" alt="Preview">
                <div style="text-align: center;">
                    <button class="process-btn" onclick="processImage()">🔍 Detect Braille</button>
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing image... Please wait</p>
            </div>

            <div class="results-section" id="results"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        const apiEndpoint = '/api'; // Same endpoint for POST requests

        // File input handling
        document.getElementById('fileInput').addEventListener('change', handleFileSelect);

        // Drag and drop handling
        const uploadSection = document.querySelector('.upload-section');
        uploadSection.addEventListener('dragover', handleDragOver);
        uploadSection.addEventListener('dragleave', handleDragLeave);
        uploadSection.addEventListener('drop', handleDrop);

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file && file.type.startsWith('image/')) {
                selectedFile = file;
                showPreview(file);
            }
        }

        function handleDragOver(event) {
            event.preventDefault();
            uploadSection.classList.add('dragover');
        }

        function handleDragLeave(event) {
            event.preventDefault();
            uploadSection.classList.remove('dragover');
        }

        function handleDrop(event) {
            event.preventDefault();
            uploadSection.classList.remove('dragover');
            
            const files = event.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                selectedFile = files[0];
                showPreview(files[0]);
            }
        }

        function showPreview(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('previewImage').src = e.target.result;
                document.getElementById('previewSection').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        async function processImage() {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }

            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').innerHTML = '';

            try {
                // Convert file to base64
                const base64 = await fileToBase64(selectedFile);
                
                // Make API call
                const response = await fetch(apiEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: base64
                    })
                });

                const result = await response.json();
                
                // Hide loading
                document.getElementById('loading').classList.remove('show');

                if (result.success) {
                    showResults(result);
                } else {
                    showError(result.error || 'Unknown error occurred');
                }

            } catch (error) {
                document.getElementById('loading').classList.remove('show');
                showError('Network error: ' + error.message);
            }
        }

        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result);
                reader.onerror = error => reject(error);
            });
        }

        function showResults(result) {
            const resultsContainer = document.getElementById('results');
            
            let html = `
                <div class="result-card success">
                    <div class="result-title">✅ Detection Complete</div>
                    
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-number">${result.total_characters}</div>
                            <div class="stat-label">Characters Detected</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">${result.detected_text.length}</div>
                            <div class="stat-label">Text Rows</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">${result.predictions.length}</div>
                            <div class="stat-label">Total Predictions</div>
                        </div>
                    </div>
                </div>

                <div class="result-card">
                    <div class="result-title">📝 Detected Text</div>
                    <div class="detected-text">
                        ${result.full_text || 'No text detected'}
                    </div>
                </div>
            `;

            if (result.detected_text.length > 1) {
                html += `
                    <div class="result-card">
                        <div class="result-title">📄 Text by Rows</div>
                        <div class="detected-text">
                            ${result.detected_text.map((row, index) => 
                                `Row ${index + 1}: ${row}`
                            ).join('<br>')}
                        </div>
                    </div>
                `;
            }

            if (result.annotated_image) {
                html += `
                    <div class="result-card">
                        <div class="result-title">🎯 Detection Visualization</div>
                        <div class="images-container">
                            <div class="image-card">
                                <div class="image-title">Original Image</div>
                                <img src="${document.getElementById('previewImage').src}" class="preview-image" alt="Original">
                            </div>
                            <div class="image-card">
                                <div class="image-title">Detected Characters</div>
                                <img src="${result.annotated_image}" class="preview-image" alt="Annotated">
                            </div>
                        </div>
                    </div>
                `;
            }

            resultsContainer.innerHTML = html;
        }

        function showError(message) {
            const resultsContainer = document.getElementById('results');
            resultsContainer.innerHTML = `
                <div class="result-card error">
                    <div class="result-title">❌ Error</div>
                    <p>${message}</p>
                </div>
            `;
        }
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def do_POST(self):
        """Handle Braille detection API requests"""
        try:
            # Parse request
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Get base64 image
            if 'image' not in data:
                self.send_error(400, "No image provided")
                return
            
            # Decode base64 image
            image_data = base64.b64decode(data['image'].split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            
            # Save to temporary file for API call
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, format='PNG')
                temp_path = tmp_file.name
            
            try:
                # Initialize detector and process
                detector = BrailleDetector()
                result = detector.detect_braille(temp_path)
                predictions = detector.extract_predictions(result)
                rows = detector.organize_text_by_rows(predictions)
                annotated_image_b64 = detector.create_annotated_image_base64(image.copy(), predictions)
                
                # Prepare response
                response_data = {
                    'success': True,
                    'detected_text': rows,
                    'full_text': ' '.join(rows),
                    'predictions': predictions,
                    'annotated_image': f"data:image/png;base64,{annotated_image_b64}",
                    'total_characters': len([p for p in predictions if p['confidence'] >= 0.4])
                }
                
            finally:
                # Clean up temp file
                os.unlink(temp_path)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS, GET')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS, GET')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()