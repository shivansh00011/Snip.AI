# Snip.AI 🎬✂️

> **Transcribe, summarize, and chat with any YouTube video — powered by AI.**

Snip.AI is a cross-platform Flutter application that lets you paste a YouTube link and instantly get a full transcription, an AI-generated summary, and an interactive chat interface to ask questions about the video's content. Under the hood, it combines `faster-whisper` for speech recognition, Google Gemini for summarization and chat, and a vector database (Qdrant) for Retrieval-Augmented Generation (RAG).

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)

---

## Features

| Feature | Description |
|---|---|
| 🎙️ **Transcription** | Downloads audio via `yt-dlp` and transcribes it with `faster-whisper` (Tiny model, INT8 quantized for speed) |
| 📝 **AI Summarization** | Sends the transcript through Google Gemini 2.0 Flash to produce structured, detailed summaries with headings and bullet points |
| 💬 **RAG Chat** | Ask any question about the video; the system retrieves relevant transcript chunks from Qdrant and feeds them to Gemini for grounded answers |
| 📌 **Source Citations** | Chat responses include timestamped source segments from the original transcript so you can verify answers |
| 🖥️ **Cross-Platform UI** | Built with Flutter — runs on Android, iOS, macOS, Linux, and Windows |
| ⚡ **Async Processing** | Vector DB storage happens in background tasks so the UI is never blocked |

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              Flutter Frontend                │
│  ┌──────────┐          ┌───────────────────┐ │
│  │ Home Page│          │  Results Page     │ │
│  │ (URL     │─────────▶│  ┌─────────────┐ │ │
│  │  Input)  │          │  │   Summary   │ │ │
│  └──────────┘          │  ├─────────────┤ │ │
│                        │  │  RAG Chat   │ │ │
│                        │  └─────────────┘ │ │
│                        └───────────────────┘ │
└────────────────────┬─────────────────────────┘
                     │ HTTP (REST)
                     ▼
┌──────────────────────────────────────────────┐
│           FastAPI Backend (Python)           │
│                                              │
│  POST /transcribe          POST /chat        │
│      │                         │             │
│      ▼                         ▼             │
│  yt-dlp → MP3          Query Embedding       │
│      │                         │             │
│      ▼                         ▼             │
│  faster-whisper         Qdrant Vector DB     │
│  (Whisper Tiny)              │               │
│      │                         ▼             │
│      ▼                    Gemini API         │
│  Gemini API               (Answer)           │
│  (Summary)                                   │
│      │                                       │
│      ▼                                       │
│  Qdrant (Store chunks + summary)             │
└──────────────────────────────────────────────┘
```

### Data Flow

1. User pastes a YouTube URL → Flutter sends `POST /transcribe`
2. Backend downloads audio with `yt-dlp`, transcribes with `faster-whisper`
3. Transcript is chunked into ~30-second segments
4. Long transcripts are split and summarized in parallel chunks via Gemini
5. Chunks and summary are embedded with `sentence-transformers` and stored in Qdrant
6. Flutter renders the summary using `flutter_markdown`
7. User asks a question → Flutter sends `POST /chat`
8. Backend embeds the query, searches Qdrant for relevant chunks, and calls Gemini with the retrieved context
9. Gemini returns a grounded answer with source references

---

## Tech Stack

### Frontend
| Library | Version | Purpose |
|---|---|---|
| Flutter | SDK | Cross-platform UI framework |
| `google_fonts` | ^6.2.1 | Montserrat & Poppins typography |
| `flutter_markdown` | ^0.6.18 | Rendering AI-generated markdown summaries |
| `http` | ^1.4.0 | HTTP client for API calls |
| `cupertino_icons` | ^1.0.8 | iOS-style icons |

### Backend
| Library | Purpose |
|---|---|
| FastAPI | Async REST API framework |
| `faster-whisper` | Optimized Whisper transcription (INT8) |
| `yt-dlp` | YouTube audio/video downloading |
| `sentence-transformers` | Text embedding (`all-MiniLM-L6-v2`, 384-dim) |
| `qdrant-client` | Vector database client (in-memory mode) |
| `httpx` | Async HTTP client for Gemini API calls |
| Google Gemini 2.0 Flash | Summarization and RAG-powered chat |
| `numpy` | Fallback random embeddings |

---

## Prerequisites

- **Flutter SDK** ≥ 3.27.0 ([install guide](https://docs.flutter.dev/get-started/install))
- **Python** 3.8+ with `pip`
- **Node.js** (optional, for tooling)
- **yt-dlp** — installed separately or via pip:
  ```bash
  pip install yt-dlp
  ```
- **ffmpeg** — required by `yt-dlp` for audio conversion:
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu/Debian
  sudo apt install ffmpeg
  # Windows
  winget install ffmpeg
  ```
- A valid **Google Gemini API key** (free tier available at [Google AI Studio](https://aistudio.google.com/))

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shivansh00011/Snip.AI.git
cd Snip.AI
```

### 2. Install Flutter dependencies

```bash
flutter pub get
```

### 3. Install Python backend dependencies

```bash
pip install -r server/requirements.txt
```

> **Tip:** It is recommended to use a virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r server/requirements.txt
> ```

### 4. Configure your Gemini API key

Open `server/main.py` and replace the placeholder with your key:

```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

> **Security note:** For production use, set this as an environment variable and load it with `os.environ.get("GEMINI_API_KEY")` instead of hardcoding it.

---

## Running the App

### Start the backend server

```bash
uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

You should see startup logs confirming that the embedding model, Whisper model, and Qdrant are initialized:

```
INFO: Loading embedding model...
INFO: Embedding model loaded successfully
INFO: Loading Whisper model...
INFO: Whisper model loaded successfully
INFO: Connecting to Qdrant...
INFO: Qdrant initialized successfully
```

Visit `http://127.0.0.1:8000` to verify the server is running. The health endpoint returns service status.

### Start the Flutter app

```bash
flutter run
```

Select your target platform when prompted (Chrome, macOS, Android emulator, etc.).

For a specific platform:

```bash
flutter run -d macos       # macOS desktop
flutter run -d chrome      # Web browser
flutter run -d linux       # Linux desktop
flutter run -d windows     # Windows desktop
```

---

## API Reference

### `GET /`

Health check — returns service availability status.

**Response:**
```json
{
  "message": "Server is up and running 🔥",
  "services": {
    "embedding_model": "Available",
    "whisper_model": "Available",
    "vector_db": "Available"
  }
}
```

---

### `POST /transcribe`

Downloads audio from a YouTube URL, transcribes it, generates a summary, and stores everything in the vector DB.

**Request body:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=..."
}
```

**Response:**
```json
{
  "transcript": "Full transcript text...",
  "summary": "## AI Summary\n\n...",
  "session_id": "uuid-v4",
  "chunks_count": 24,
  "metadata": {
    "title": "Video Title",
    "duration": 3600.0,
    "video_url": "https://...",
    "transcript_id": "uuid-v4"
  }
}
```

> **Note:** The `session_id` / `transcript_id` must be saved on the client to enable chat for this video.

---

### `POST /chat`

Queries the transcribed content using RAG (Retrieval-Augmented Generation).

**Request body:**
```json
{
  "query": "What did the speaker say about climate change?",
  "session_id": "uuid-v4"
}
```

**Response:**
```json
{
  "answer": "According to the transcript, the speaker discussed...",
  "context": {
    "chunks": [
      {
        "text": "...relevant segment...",
        "start_time": 142.5,
        "end_time": 172.5,
        "score": 0.89,
        "chunk_index": 4,
        "type": "chunk"
      }
    ],
    "summary": {
      "text": "## Summary...",
      "type": "summary",
      "score": 1.0
    }
  }
}
```

---

### `GET /status/{session_id}`

Check whether transcript chunks have been stored in the vector DB for a given session.

**Response:**
```json
{
  "status": "completed",
  "message": "Transcript processing complete",
  "chunks_available": true,
  "chunks_count": 48
}
```

---

## Project Structure

```
Snip.AI/
├── lib/
│   ├── main.dart                          # App entry point & routing
│   └── pages and logic/
│       ├── home.dart                      # URL input & feature showcase UI
│       └── results.dart                   # Summary view + RAG chat UI
│
├── server/
│   └── main.py                            # FastAPI backend (all logic)
│
├── android/                               # Android platform files
├── ios/                                   # iOS platform files
├── macos/                                 # macOS platform files
├── linux/                                 # Linux platform files
├── windows/                               # Windows platform files
├── web/                                   # Web platform files
│
├── pubspec.yaml                           # Flutter dependencies
└── README.md
```

---

## How It Works

### Transcription Pipeline

1. `yt-dlp` downloads the YouTube video audio as MP3 (max 5-minute timeout)
2. `faster-whisper` (Tiny model with INT8 quantization) transcribes the audio into timed segments
3. Segments are chunked into ~30-second windows to preserve temporal context

### Summarization

- Transcripts under ~4,000 words are sent to Gemini in a single request
- Longer transcripts are split into 3,000-word chunks, each summarized independently, then a final synthesis pass combines them
- The system retries up to 3 times per chunk on failure, with a fallback using the first and last segments

### RAG Chat

- User queries are embedded using `sentence-transformers` (`all-MiniLM-L6-v2`)
- Qdrant performs cosine similarity search filtered by `session_id` to retrieve the top 5 relevant chunks
- The video summary (stored as a special point) is always prepended to the context
- The assembled context + user query is sent to Gemini with an instruction prompt to provide grounded, citation-backed answers

### Embedding Storage

- Each ~30-second transcript chunk becomes a Qdrant point with metadata: `session_id`, `start_time`, `end_time`, `chunk_index`
- The full summary is stored as a separate point with `type: "summary"` for fast retrieval
- Storage happens asynchronously via FastAPI `BackgroundTasks` so the API response is not delayed

---

## Configuration

| Setting | Location | Default | Description |
|---|---|---|---|
| Gemini API Key | `server/main.py` | `""` | Required — get from Google AI Studio |
| Whisper model size | `server/main.py` | `"tiny"` | Options: `tiny`, `base`, `small`, `medium`, `large` |
| Whisper compute type | `server/main.py` | `"int8"` | Options: `int8`, `float16`, `float32` |
| Chunk size (seconds) | `server/main.py` | `30.0` | Length of transcript segments for RAG |
| Embedding model | `server/main.py` | `all-MiniLM-L6-v2` | 384-dim sentence transformer |
| Embedding dimensions | `server/main.py` | `384` | Must match embedding model output |
| Qdrant mode | `server/main.py` | In-memory | Change to `QdrantClient(host="localhost", port=6333)` for persistence |
| Backend URL | `lib/pages and logic/home.dart` | `http://127.0.0.1:8000` | Update for production deployments |

---

## Known Limitations

- **In-memory vector DB:** Qdrant runs in-memory by default, meaning all stored transcripts are lost when the server restarts. Switch to a persistent Qdrant instance for production use.
- **Whisper Tiny accuracy:** The `tiny` model is fast but may struggle with heavy accents, technical jargon, or low-quality audio. Use `small` or `medium` for better accuracy at the cost of speed.
- **Gemini API rate limits:** Free-tier Gemini accounts have rate and quota limits that may cause failures on very long videos.
- **API key exposure:** The Gemini API key is currently hardcoded in `server/main.py`. Move it to an environment variable before deploying.
- **No authentication:** The backend has no user authentication. Any client with network access can submit URLs and use your Gemini quota.
- **Backend URL hardcoded:** The Flutter app points to `http://127.0.0.1:8000`. This needs to be configurable for multi-device or production use.
- **Private/restricted videos:** `yt-dlp` cannot download private, age-restricted, or DRM-protected content.

---

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and test them
4. Commit with a clear message: `git commit -m "Add: my feature description"`
5. Push to your fork: `git push origin feature/my-feature`
6. Open a Pull Request against `main`

### Ideas for contributions

- Persistent Qdrant storage with session management
- Support for non-YouTube video sources (direct MP4 URLs, Vimeo, etc.)
- Export transcript/summary as PDF or Markdown
- Dark/light theme toggle in the Flutter UI
- Environment variable support for backend configuration
- Docker Compose setup for one-command deployment
- User authentication and per-user transcript history

