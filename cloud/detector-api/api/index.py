from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

INFERENCE_API_URL = "https://serverless.roboflow.com"
API_KEY = "RzOXFbriJONcee7MHKN8"
WORKSPACE_NAME = "braille-to-text-0xo2p"
WORKFLOW_ID = "custom-workflow"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    # Prepare the API call to the inference service
    files = {'image': open(file_path, 'rb')}
    data = {
        'workspace_name': WORKSPACE_NAME,
        'workflow_id': WORKFLOW_ID,
        'use_cache': 'true',
    }
    headers = {'Authorization': f'Bearer {API_KEY}'}

    try:
        # Send image to inference API
        response = requests.post(
            f"{INFERENCE_API_URL}/workflows/{WORKFLOW_ID}/run",
            files=files,
            data=data,
            headers=headers
        )

        # Handle the response from the inference service
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Error processing image", "details": response.json()}), 500
    finally:
        files['image'].close()

if __name__ == '__main__':
    app.run(debug=True)
