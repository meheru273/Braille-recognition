# api/index.py - Vercel serverless function handler
import json
import os
from urllib.parse import parse_qs
from assistant import BrailleAssistant

def handler(request):
    """Main Vercel handler function"""
    try:
        # Initialize assistant
        assistant = BrailleAssistant()
        
        # Get request method and path
        method = request.get('method', 'GET')
        path = request.get('path', '/')
        
        # Handle different routes
        if method == 'GET' and path == '/':
            return serve_html()
        
        elif method == 'POST' and path == '/api/chat':
            return handle_chat(request, assistant)
        
        elif method == 'POST' and path == '/api/process-braille':
            return handle_braille_processing(request, assistant)
        
        elif method == 'GET' and path == '/health':
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'status': 'healthy', 'timestamp': str(__import__('datetime').datetime.now())})
            }
        
        else:
            return {
                'statusCode': 404,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Not found'})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': f'Server error: {str(e)}'})
        }

def serve_html():
    """Serve the main HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Braille Recognition Assistant</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                padding: 30px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px; 
                color: #333;
            }
            .header h1 { 
                font-size: 2.5em; 
                margin-bottom: 10px; 
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .input-group { 
                margin-bottom: 20px; 
            }
            label { 
                display: block; 
                margin-bottom: 8px; 
                font-weight: 600; 
                color: #555;
            }
            input, textarea, select { 
                width: 100%; 
                padding: 12px; 
                border: 2px solid #e0e0e0; 
                border-radius: 8px; 
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input:focus, textarea:focus, select:focus { 
                outline: none; 
                border-color: #667eea; 
            }
            .btn { 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 16px; 
                font-weight: 600;
                transition: transform 0.2s;
                margin: 5px;
            }
            .btn:hover { 
                transform: translateY(-2px); 
            }
            .btn:disabled { 
                opacity: 0.6; 
                cursor: not-allowed; 
                transform: none;
            }
            .result { 
                margin-top: 20px; 
                padding: 20px; 
                background: #f8f9fa; 
                border-radius: 8px; 
                border-left: 4px solid #667eea;
            }
            .chat-container { 
                max-height: 400px; 
                overflow-y: auto; 
                border: 2px solid #e0e0e0; 
                border-radius: 8px; 
                padding: 15px; 
                margin-bottom: 15px;
                background: #fafafa;
            }
            .message { 
                margin-bottom: 15px; 
                padding: 10px; 
                border-radius: 8px; 
            }
            .user-message { 
                background: #667eea; 
                color: white; 
                margin-left: 20px; 
            }
            .assistant-message { 
                background: white; 
                border: 1px solid #e0e0e0; 
                margin-right: 20px;
            }
            .loading { 
                display: none; 
                text-align: center; 
                color: #667eea; 
                font-weight: 600;
            }
            .tabs { 
                display: flex; 
                margin-bottom: 20px; 
                border-bottom: 2px solid #e0e0e0;
            }
            .tab { 
                padding: 12px 24px; 
                cursor: pointer; 
                border: none; 
                background: none; 
                font-size: 16px; 
                color: #666;
                transition: all 0.3s;
            }
            .tab.active { 
                color: #667eea; 
                border-bottom: 2px solid #667eea; 
            }
            .tab-content { 
                display: none; 
            }
            .tab-content.active { 
                display: block; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔤 Braille Recognition Assistant</h1>
                <p>AI-powered braille text processing and intelligent assistance</p>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="switchTab('braille')">Braille Processing</button>
                <button class="tab" onclick="switchTab('chat')">AI Chat</button>
            </div>

            <!-- Braille Processing Tab -->
            <div id="braille-tab" class="tab-content active">
                <div class="input-group">
                    <label for="braille-input">Detected Braille Text (comma-separated):</label>
                    <textarea id="braille-input" rows="4" placeholder="Enter detected braille characters, e.g: ⠓⠑⠇⠇⠕, ⠺⠕⠗⠇⠙"></textarea>
                </div>

                <button class="btn" onclick="processBraille()">🔍 Process Braille</button>

                <div class="loading" id="braille-loading">Processing braille text...</div>

                <div id="braille-result" class="result" style="display: none;">
                    <h3>Results:</h3>
                    <div id="braille-output"></div>
                </div>
            </div>

            <!-- Chat Tab -->
            <div id="chat-tab" class="tab-content">
                <div id="chat-messages" class="chat-container">
                    <div class="message assistant-message">
                        <strong>Assistant:</strong> Hello! I'm your AI assistant. Ask me anything or request help with text processing.
                    </div>
                </div>

                <div class="input-group">
                    <input type="text" id="chat-input" placeholder="Type your message..." onkeypress="handleChatKeyPress(event)">
                </div>

                <button class="btn" onclick="sendMessage()">💬 Send Message</button>
                <button class="btn" onclick="clearChat()" style="background: #dc3545;">🗑️ Clear Chat</button>

                <div class="loading" id="chat-loading">Thinking...</div>
            </div>
        </div>

        <script>
            let chatHistory = [];

            function switchTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });

                // Show selected tab
                document.getElementById(tabName + '-tab').classList.add('active');
                event.target.classList.add('active');
            }

            async function processBraille() {
                const input = document.getElementById('braille-input').value.trim();
                const resultDiv = document.getElementById('braille-result');
                const outputDiv = document.getElementById('braille-output');
                const loading = document.getElementById('braille-loading');

                if (!input) {
                    alert('Please enter some braille text to process.');
                    return;
                }

                loading.style.display = 'block';
                resultDiv.style.display = 'none';

                try {
                    const response = await fetch('/api/process-braille', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            braille_strings: input.split(',').map(s => s.trim())
                        })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        outputDiv.innerHTML = `
                            <div style="margin-bottom: 15px;">
                                <strong>Processed Text:</strong> ${data.text}
                            </div>
                            <div style="margin-bottom: 15px;">
                                <strong>Explanation:</strong> ${data.explanation}
                            </div>
                            <div>
                                <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
                            </div>
                        `;
                        resultDiv.style.display = 'block';
                    } else {
                        outputDiv.innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
                        resultDiv.style.display = 'block';
                    }
                } catch (error) {
                    outputDiv.innerHTML = `<div style="color: red;">Network error: ${error.message}</div>`;
                    resultDiv.style.display = 'block';
                }

                loading.style.display = 'none';
            }

            async function sendMessage() {
                const input = document.getElementById('chat-input');
                const message = input.value.trim();
                const messagesDiv = document.getElementById('chat-messages');
                const loading = document.getElementById('chat-loading');

                if (!message) return;

                // Add user message
                const userMessageDiv = document.createElement('div');
                userMessageDiv.className = 'message user-message';
                userMessageDiv.innerHTML = `<strong>You:</strong> ${message}`;
                messagesDiv.appendChild(userMessageDiv);

                input.value = '';
                loading.style.display = 'block';

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message })
                    });

                    const data = await response.json();

                    const assistantMessageDiv = document.createElement('div');
                    assistantMessageDiv.className = 'message assistant-message';
                    
                    if (response.ok) {
                        assistantMessageDiv.innerHTML = `<strong>Assistant:</strong> ${data.response}`;
                    } else {
                        assistantMessageDiv.innerHTML = `<strong>Assistant:</strong> <span style="color: red;">Error: ${data.error}</span>`;
                    }
                    
                    messagesDiv.appendChild(assistantMessageDiv);
                } catch (error) {
                    const errorMessageDiv = document.createElement('div');
                    errorMessageDiv.className = 'message assistant-message';
                    errorMessageDiv.innerHTML = `<strong>Assistant:</strong> <span style="color: red;">Network error: ${error.message}</span>`;
                    messagesDiv.appendChild(errorMessageDiv);
                }

                loading.style.display = 'none';
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function handleChatKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            function clearChat() {
                const messagesDiv = document.getElementById('chat-messages');
                messagesDiv.innerHTML = `
                    <div class="message assistant-message">
                        <strong>Assistant:</strong> Hello! I'm your AI assistant. Ask me anything or request help with text processing.
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    """
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': html_content
    }

def handle_chat(request, assistant):
    """Handle chat requests"""
    try:
        # Get request body
        body = request.get('body', '{}')
        if isinstance(body, str):
            data = json.loads(body)
        else:
            data = body
        
        message = data.get('message', '').strip()
        if not message:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Message is required'})
            }
        
        # Process with assistant
        response = assistant.chat(message)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'response': response})
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': f'Chat processing error: {str(e)}'})
        }

def handle_braille_processing(request, assistant):
    """Handle braille processing requests"""
    try:
        # Get request body
        body = request.get('body', '{}')
        if isinstance(body, str):
            data = json.loads(body)
        else:
            data = body
        
        braille_strings = data.get('braille_strings', [])
        if not braille_strings:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Braille strings are required'})
            }
        
        # Process with assistant
        result = assistant.process_braille_strings(braille_strings)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'text': result.text,
                'explanation': result.explanation,
                'confidence': result.confidence
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': f'Braille processing error: {str(e)}'})
        }