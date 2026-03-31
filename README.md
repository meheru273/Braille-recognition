# Braille Recognition System

A complete end-to-end system for detecting braille characters in images, interpreting them with AI, and providing an interactive chat interface. Supports both local development and cloud deployment on Vercel with Firebase integration.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Local Development](#local-development)
  - [Cloud Deployment](#cloud-deployment)
- [API Reference](#api-reference)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [Known Issues](#known-issues)
- [Contributing](#contributing)

---

## Overview

The Braille Recognition System converts braille text in images to readable text using a computer vision model hosted on Roboflow. The detected characters are then interpreted by an LLM (Groq or OpenAI), which corrects OCR errors, generates confidence scores, and provides explanations. Users can follow up with a chat interface that retains conversation context and can search Wikipedia for additional information.

The system is split into three independently deployable microservices for the cloud, or a single local script for development.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        Client / User                       │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│                   Firebase API (FastAPI)                    │
│          cloud/firebase-api  — main orchestrator           │
│  • /detect-braille   • /chat   • /chat-threads             │
│  • /user-detections  • /batch-detect                       │
└──────────┬──────────────────────────────┬──────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌─────────────────────────────┐
│   Detector API      │      │       Assistant API          │
│  cloud/detector-api │      │   cloud/assistant-api        │
│  POST /upload       │      │  POST /api/process-braille   │
│                     │      │  POST /api/chat              │
└────────┬────────────┘      └──────────────┬──────────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────┐               ┌─────────────────────┐
│  Roboflow API   │               │  Groq / OpenAI API  │
│  (CV model)     │               │  + Wikipedia search  │
└─────────────────┘               └─────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Firebase Firestore   │
              │  Firebase Storage     │
              └───────────────────────┘
```

---

## Project Structure

```
Braille-recognition/
├── app/                          # Local development application
│   ├── main.py                   # Integration test / demo entry point
│   ├── detector.py               # Braille detection using Roboflow SDK
│   └── assistant.py              # AI assistant built with LangGraph
│
├── cloud/                        # Vercel serverless microservices
│   ├── README.md                 # Cloud API config & troubleshooting
│   │
│   ├── assistant-api/            # Combined braille + AI processing API
│   │   ├── api/index.py          # Vercel serverless handler (Flask)
│   │   ├── assistant.py          # Lightweight LLM client with fallback
│   │   ├── connector.py          # Controller orchestrating detector + assistant
│   │   ├── detector.py           # Cloud-optimized braille detector
│   │   ├── requirements.txt
│   │   └── vercel.json           # 30-second function timeout
│   │
│   ├── detector-api/             # Detection-only API
│   │   ├── api/index.py          # Flask image upload + Roboflow inference
│   │   ├── detector.py           # Braille detection module
│   │   ├── public/index.html     # Minimal HTML upload interface
│   │   └── requirements.txt
│   │
│   └── firebase-api/             # Main orchestration API with persistence
│       ├── api/index.py          # FastAPI app (682 lines) — primary API
│       ├── firebase-service.py   # Firestore + Storage integration
│       ├── requirements.txt
│       └── vercel.json
│
└── test/                         # Sample images for testing
    ├── before.jpg
    ├── before2.jpg
    ├── before4.jpg
    ├── before5.jpg
    └── annotated_result.png      # Example output with bounding boxes
```

---

## Features

- **Braille OCR** — Detects all 26 braille alphabet characters from an image using a custom Roboflow model. Returns bounding boxes, class labels, and confidence scores per character.
- **Row-aware text reconstruction** — Groups detected characters into horizontal rows using Y-coordinate clustering and sorts them left-to-right, producing readable text lines.
- **Annotated output images** — Generates a copy of the input image with colored bounding boxes and labels for each detected character (26 distinct colors).
- **AI interpretation** — Passes raw detected character strings to an LLM (Groq llama-3.1-8b-instant or OpenAI gpt-3.5-turbo) to correct errors, build readable sentences, and generate a confidence score.
- **Wikipedia-enhanced explanations** — The assistant searches Wikipedia for relevant context when explaining braille content.
- **Stateful chat** — Persistent conversation threads keyed by `thread_id`, allowing follow-up questions about previously detected text.
- **Fallback mode** — The cloud assistant works without an LLM API key using basic pattern matching, degrading gracefully rather than failing.
- **Firebase persistence** — Detection results, images, users, and chat threads are stored in Firestore and Firebase Storage.
- **Batch detection** — Supports uploading up to 5 images in a single request.
- **Inter-service health checks** — `/health` and `/service-status` endpoints report the state of each downstream service.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Computer Vision | Roboflow Inference SDK (`inference_sdk`) |
| LLM (primary) | Groq API — `llama-3.1-8b-instant` |
| LLM (fallback) | OpenAI API — `gpt-3.5-turbo` |
| Workflow orchestration | LangGraph + LangChain |
| Wikipedia tool | LangChain Wikipedia integration |
| Cloud API framework | FastAPI + Mangum (ASGI adapter) |
| Lightweight API | Flask |
| Image processing | Pillow (PIL) |
| HTTP client | httpx (async), requests |
| Database | Firebase Firestore |
| File storage | Firebase Storage |
| Deployment | Vercel (serverless functions) |
| Environment config | python-dotenv |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- A [Roboflow](https://roboflow.com) account with the `braille-to-text-0xo2p` workspace and a valid API key (32+ characters)
- A [Groq](https://console.groq.com) API key **or** an [OpenAI](https://platform.openai.com) API key
- (Cloud only) A Firebase project with Firestore and Storage enabled

---

### Environment Variables

Create a `.env` file in the project root (or in the relevant `cloud/*/` subdirectory):

```dotenv
# Required
ROBOFLOW_API_KEY=your_roboflow_api_key_here    # 32+ characters

# At least one LLM key is required for AI features
GROQ_API_KEY=gsk_your_groq_key_here
# OR
OPENAI_API_KEY=sk-your_openai_key_here

# Firebase (cloud deployment only)
FIREBASE_CREDENTIALS={"type":"service_account",...}   # JSON string
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_STORAGE_BUCKET=your-project.appspot.com

# Cloud inter-service URLs (firebase-api only)
DETECTOR_API_URL=https://your-detector.vercel.app
ASSISTANT_API_URL=https://your-assistant.vercel.app
```

> The Groq key takes precedence if both are set. Groq is recommended as it is faster and free-tier friendly.

---

### Local Development

Install dependencies:

```bash
pip install inference-sdk pillow python-dotenv langgraph langchain langchain-openai langchain-community wikipedia
```

Run the integration demo (detects braille in `test/before.jpg` and prints AI output):

```bash
cd app
python main.py
```

The script will:
1. Detect braille characters in `test/before.jpg`
2. Save an annotated image to `test/annotated_result.png`
3. Pass detected characters to the AI assistant for interpretation
4. Run a sample chat interaction

---

### Cloud Deployment

Each subdirectory under `cloud/` is an independent Vercel project.

#### 1. Deploy the Detector API

```bash
cd cloud/detector-api
vercel deploy
```

#### 2. Deploy the Assistant API

```bash
cd cloud/assistant-api
vercel deploy
```

#### 3. Deploy the Firebase (main) API

Set environment variables in the Vercel dashboard, then:

```bash
cd cloud/firebase-api
vercel deploy
```

Set `DETECTOR_API_URL` and `ASSISTANT_API_URL` to the URLs from steps 1 and 2.

---

## API Reference

All endpoints are served by the Firebase API (`cloud/firebase-api/api/index.py`).

### `POST /detect-braille`

Runs the full detection + AI interpretation pipeline on an uploaded image.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image file | Yes | JPEG or PNG braille image |
| `user_id` | string | No | User identifier for storing results |
| `session_id` | string | No | Session identifier |

**Response:**
```json
{
  "success": true,
  "detected_text": "hello world",
  "explanation": "The braille reads 'hello world', a common greeting.",
  "confidence": 0.87,
  "raw_detections": [...],
  "firebase_doc_id": "abc123"
}
```

---

### `POST /chat`

Send a chat message, optionally with braille context.

**Request:** `application/json`

```json
{
  "message": "What does this braille mean?",
  "context": "hello world",
  "user_id": "user123",
  "thread_id": "thread456"
}
```

**Response:**
```json
{
  "response": "The braille text 'hello world' is a greeting...",
  "thread_id": "thread456"
}
```

---

### `GET /chat-threads`

List a user's conversation threads.

**Query params:** `user_id` (required), `limit` (default 10)

---

### `GET /chat-threads/{thread_id}`

Get full message history for a thread.

**Query params:** `user_id` (required)

---

### `DELETE /chat-threads/{thread_id}`

Delete a conversation thread and all its messages.

---

### `GET /user-detections`

Get a user's detection history from Firestore.

**Query params:** `user_id` (required), `limit` (default 10)

---

### `POST /users`

Create or update a user profile.

```json
{
  "user_id": "user123",
  "email": "user@example.com",
  "display_name": "Alice"
}
```

---

### `POST /batch-detect`

Detect braille in up to 5 images at once.

**Request:** `multipart/form-data` — multiple `files[]` fields.

---

### `GET /health`

Returns service health and configuration status.

---

### `GET /service-status`

Returns detailed status of all downstream services (Roboflow, LLM, Firebase).

---

## Data Flow

### Detection & Interpretation

```
1. User uploads image
        ↓
2. BrailleDetector sends image to Roboflow Inference API
        ↓
3. Raw predictions returned: [{x, y, width, height, confidence, class}, ...]
        ↓
4. Characters clustered into rows by Y-coordinate
   → Rows sorted left-to-right by X-coordinate
        ↓
5. Text rows passed to BrailleAssistant (LLM)
   → LLM corrects errors, assembles sentences, scores confidence
   → Wikipedia search for additional context (optional)
        ↓
6. Result stored in Firestore, image in Firebase Storage
        ↓
7. Response returned to client
```

### Chat

```
1. User sends message (with optional braille context)
        ↓
2. BrailleAssistant determines if Wikipedia lookup would help
        ↓
3. LLM generates response using conversation history (MemorySaver)
        ↓
4. Message pair stored in Firestore chat thread
        ↓
5. Response returned to client
```

---

## Configuration

### Roboflow Model

| Setting | Value |
|---|---|
| API URL | `https://serverless.roboflow.com` |
| Workspace | `braille-to-text-0xo2p` |
| Workflow ID | `custom-workflow` |
| Classes | 26 (a–z) |
| Confidence threshold | 0.4 (configurable) |

### LLM Models

| Provider | Model | Trigger |
|---|---|---|
| Groq | `llama-3.1-8b-instant` | `GROQ_API_KEY` starts with `gsk_` |
| OpenAI | `gpt-3.5-turbo` | `OPENAI_API_KEY` present |
| Fallback | — | No key set; uses basic pattern matching |

### Firebase Collections

| Collection | Contents |
|---|---|
| `braille_detections` | Detection results with metadata |
| `users` | User profiles and activity |
| `chat_threads` | Conversation thread metadata |
| `chat_messages` | Individual messages per thread |

Storage path: `braille_images/{user_id}/{session_id}_{filename}`

---

## Known Issues

### 1. Incomplete Roboflow API Key
The default API key in the codebase (`RzOXFbriJONcee7MHKN8`) is only 16 characters. Valid Roboflow keys are 32+ characters. This causes `403 Access Denied` errors.

**Fix:** Log in to [app.roboflow.com](https://app.roboflow.com), go to **Settings → Roboflow API**, copy your full key, and set it as `ROBOFLOW_API_KEY` in your `.env` file.

### 2. Syntax Error in `cloud/detector-api/detector.py`
Line 12 is missing a proper assignment. The `InferenceHTTPClient` constructor call needs to be assigned to `self.client`.

### 3. Missing `import os` in `cloud/detector-api/api/index.py`
The file references `os.environ` without importing the `os` module.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests if applicable
4. Commit: `git commit -m "Add your feature"`
5. Push: `git push origin feature/your-feature`
6. Open a Pull Request

Please ensure your code handles the fallback/no-API-key case gracefully, as the system is designed to degrade without crashing when external services are unavailable.
