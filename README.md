# Snip.AI

Snip.AI is a Flutter-based application designed to enhance your interaction with YouTube videos. With its AI-powered features, you can transcribe, summarize, and interact with videos as though they're supported by GPT-4. The project combines cutting-edge natural language processing with a user-friendly interface.

---

## Features

### 1. Transcription
Extract spoken content from any YouTube video effortlessly. Ideal for capturing and searching through spoken content.

### 2. Summarization
Break down lengthy videos into concise, structured summaries, saving hours of time while getting to the crux of the content.

### 3. Dynamic Interaction
Ask questions or interact with videos using advanced GPT-4-like capabilities—query, summarize, and explore in a conversational format.

---

## Key Technologies
- **Flutter**: Provides a seamless cross-platform UI for mobile, desktop, and web.
- **Python Backend**:
  - The backend, built with FastAPI, processes transcription, generates summaries, and supports interactive chat through its server endpoints.
  - Utilizes libraries like `faster-whisper` for transcription tasks and integrates with a vector database (e.g., Qdrant) for search capabilities.

---

## Architecture Overview

### Flutter Frontend
- The design ensures interactive and visually appealing interfaces for users.
- Widgets laid out in **Material Design** provide responsive and accessible UI components.
- Implements flutter-specific features like advanced button designs (e.g., ElevatedButton with spinners used).

---

### Backend Details
Found in the server directory (`server/main.py`):
- **Endpoints include**:
  - `/transcribe`: Processes YouTube videos with `yt-dlp`.
  - `/chat`: Handles interactive RAG (Retrieval-Augmented Generation).
- Modularized components like `search_vector_db` to fetch summaries using a vectorized database for faster results.

#### Error Handling
Implemented detailed error logging for issues like whisper-model loading failures or permission issues.

---

## Installation and Setup

### Prerequisites
- Flutter SDK
- Python with pip installed
- `yt-dlp` for downloading YouTube videos and whisper models using the `faster-whisper` library.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/shivansh00011/Snip.AI.git
   cd Snip.AI
   ```

2. Set up dependencies:
   ```bash
   flutter pub get
   pip install -r server/requirements.txt
   ```

3. Run the Flutter application:
   ```bash
   flutter run
   ```

4. Start the backend server:
   ```bash
   uvicorn server.main:app --reload
   ```

---

## Usage
1. Paste the YouTube link into the app.
2. Select an action (Transcribe, Summarize, or Ask Anything).
3. View results and interact with the output as required.

**Example Screenshot** _(Placeholder for screenshots)_: 
_Drop in relevant UI snapshots for better understanding._

---

## Contributing
Have suggestions or want to extend the capabilities? Contributions are welcome! Fork this repository and create a pull request following these steps:
1. Fork this repository.
2. Make your changes.
3. Open a pull request.

See `CONTRIBUTING.md` (if available) for more details.

---

## License
This project is released under the [MIT License](LICENSE).