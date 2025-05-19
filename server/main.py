from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import subprocess
import os
import uuid
import glob
import httpx
import math
import logging
import traceback
from typing import Dict, Any, Optional, List
import numpy as np

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and clients
embedding_model = None
qdrant_client = None
whisper_model = None
COLLECTION_NAME = "video_transcripts"
EMBEDDING_DIM = 384  # Dimensions for all-MiniLM-L6-v2

class VideoRequest(BaseModel):
    youtube_url: str

class ChatRequest(BaseModel):
    query: str
    session_id: str  # To identify which transcript to search

class ChunkData(BaseModel):
    text: str
    start_time: float
    end_time: float
    embedding: Optional[List[float]] = None

class TranscriptMetadata(BaseModel):
    video_url: str
    title: Optional[str] = None
    duration: Optional[float] = None
    transcript_id: str = Field(..., description="Unique ID for the transcript")

# Fixed: Use hardcoded API key since env var approach isn't working
GEMINI_API_KEY = "Your_Gemini_API_Key"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Global exception handler for better error reporting
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = {
        "error": str(exc),
        "type": type(exc).__name__,
        "path": request.url.path
    }
    
    # Log the full exception with traceback
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=error_detail
    )

# Initialize models and database connections
@app.on_event("startup")
async def startup_event():
    global embedding_model, qdrant_client, whisper_model
    
    # Initialize sentence transformer model for embeddings with better error handling
    try:
        logger.info("Loading embedding model...")
        # Import SentenceTransformer here to isolate any import errors
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
    except ImportError as e:
        logger.error(f"Missing dependency: SentenceTransformer. Error: {str(e)}")
        logger.error("Please install with: pip install sentence-transformers")
        embedding_model = None
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        # Continue with reduced functionality
        embedding_model = None
    
    # Initialize whisper model with better error handling
    try:
        logger.info("Loading Whisper model...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("tiny", compute_type="int8")
        logger.info("Whisper model loaded successfully")
    except ImportError as e:
        logger.error(f"Missing dependency: faster_whisper. Error: {str(e)}")
        logger.error("Please install with: pip install faster-whisper")
        whisper_model = None
    except Exception as e:
        logger.error(f"Failed to load whisper model: {str(e)}")
        whisper_model = None
    
    # Initialize Qdrant client (local instance)
    logger.info("Connecting to Qdrant...")
    try:
        # First check if qdrant_client is importable
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError as e:
            logger.error(f"Missing dependency: qdrant_client. Error: {str(e)}")
            logger.error("Please install with: pip install qdrant-client")
            qdrant_client = None
            return
            
        qdrant_client = QdrantClient(":memory:")  # In-memory for development
        # For production, use:
        # qdrant_client = QdrantClient(host="localhost", port=6333)
        
        # Create collection if it doesn't exist
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        logger.info("Qdrant initialized successfully")
    except NameError:
        # This will happen if the imports failed
        qdrant_client = None
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {str(e)}")
        qdrant_client = None
        # We'll still start the app, but RAG functionality will be limited

@app.get("/")
async def root():
    status = {
        "message": "Server is up and running 🔥",
        "services": {
            "embedding_model": "Available" if embedding_model else "Unavailable",
            "whisper_model": "Available" if whisper_model else "Unavailable", 
            "vector_db": "Available" if qdrant_client else "Unavailable"
        }
    }
    return status

# Check prerequisites before running critical functions
def check_dependencies(required_modules: Dict[str, Any], error_msg: str = "Required dependencies not available") -> None:
    """Checks if required modules are available and raises a clear exception if not"""
    missing_modules = [name for name, module in required_modules.items() if module is None]
    if missing_modules:
        missing_str = ", ".join(missing_modules)
        raise HTTPException(
            status_code=503, 
            detail=f"{error_msg}: {missing_str} not available. Please check server logs."
        )

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a list of text chunks"""
    if embedding_model is None:
        # Return random embeddings as fallback when model is not available
        logger.warning("Embedding model not loaded, using random embeddings")
        return [np.random.rand(EMBEDDING_DIM).tolist() for _ in texts]
    
    try:
        embeddings = embedding_model.encode(texts)
        # Convert numpy arrays to list of floats for JSON serialization
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        # Fallback to random embeddings
        return [np.random.rand(EMBEDDING_DIM).tolist() for _ in texts]

def chunk_transcript(transcript_segments, chunk_size_seconds=30.0):
    """
    Chunk transcript into segments of approximately chunk_size_seconds
    Returns list of chunks with text and timing information
    """
    chunks = []
    current_chunk = ""
    chunk_start_time = 0
    
    if not transcript_segments:
        logger.warning("No transcript segments provided to chunk_transcript")
        return []
    
    for segment in transcript_segments:
        if not current_chunk:  # If starting a new chunk
            chunk_start_time = segment.start
            
        current_chunk += segment.text + " "
        
        # If this segment ends beyond our chunk size, save the chunk
        if segment.end - chunk_start_time >= chunk_size_seconds:
            chunks.append({
                "text": current_chunk.strip(),
                "start_time": chunk_start_time,
                "end_time": segment.end
            })
            current_chunk = ""
    
    # Don't forget the last chunk if there's anything left
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "start_time": chunk_start_time,
            "end_time": transcript_segments[-1].end if transcript_segments else 0
        })
    
    return chunks

async def store_in_vector_db(session_id: str, chunks: List[Dict[str, Any]], metadata: Dict[str, Any], summary: str):
    """Store transcript chunks and summary in vector database with embeddings"""
    if not qdrant_client:
        logger.error("Qdrant client not initialized - skipping vector storage")
        return False
    
    try:
        # Import here to handle potential missing module
        from qdrant_client.http import models
        
        # Generate embeddings for all chunks and summary
        texts = [chunk["text"] for chunk in chunks] + [summary]
        embeddings = create_embeddings(texts)
        
        # Prepare points for Qdrant
        points = []
        
        # Store chunks
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings[:-1])):  # Exclude summary embedding
            point_id = str(uuid.uuid4())
            
            # Create point payload
            payload = {
                "session_id": session_id,
                "text": chunk["text"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "chunk_index": i,
                "type": "chunk",
                **metadata  # Include video metadata
            }
            
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Store summary as a separate point
        summary_point_id = str(uuid.uuid4())
        summary_payload = {
            "session_id": session_id,
            "text": summary,
            "type": "summary",
            **metadata
        }
        
        points.append(models.PointStruct(
            id=summary_point_id,
            vector=embeddings[-1],  # Use the last embedding (summary)
            payload=summary_payload
        ))
        
        # Upsert points to Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"Successfully stored {len(points)} points (chunks + summary) for session {session_id}")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency for vector storage: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error storing transcript in vector DB: {str(e)}")
        return False

async def search_vector_db(query: str, session_id: str, limit: int = 5) -> List[Dict]:
    """Search vector database for relevant chunks and summary based on query"""
    if not qdrant_client:
        logger.error("Vector search services not available")
        raise Exception("Vector search capabilities not available")
    
    try:
        # Import here to handle potential missing module
        from qdrant_client.http import models
        
        # Create embedding for query
        query_embedding = create_embeddings([query])[0]
        
        # First, get the summary for this session
        summary_results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    ),
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="summary")
                    )
                ]
            ),
            limit=1
        )
        
        # Then search for relevant chunks
        chunk_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    ),
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="chunk")
                    )
                ]
            ),
            limit=limit
        )
        
        # Format results
        results = []
        
        # Add summary if found
        if summary_results and summary_results[0]:
            summary = summary_results[0][0]  # Get the first (and should be only) summary
            results.append({
                "text": summary.payload.get("text"),
                "type": "summary",
                "score": 1.0  # Summary always gets highest relevance
            })
        
        # Add chunks
        for result in chunk_results:
            results.append({
                "text": result.payload.get("text"),
                "start_time": result.payload.get("start_time"),
                "end_time": result.payload.get("end_time"),
                "score": result.score,
                "chunk_index": result.payload.get("chunk_index"),
                "type": "chunk"
            })
        
        return results
    except ImportError as e:
        logger.error(f"Missing dependency for vector search: {str(e)}")
        raise Exception(f"Vector search requires qdrant-client: {str(e)}")
    except Exception as e:
        logger.error(f"Error searching vector DB: {str(e)}")
        raise Exception(f"Search failed: {str(e)}")

async def summarize_text(text: str, is_chunk: bool = False) -> str:
    """
    Summarize text using Gemini API with improved error handling and response parsing
    """
    request_headers = {
        "Content-Type": "application/json"
    }
    
    # More detailed prompts for full summary vs chunk summary
    if is_chunk:
        prompt = f"""
Extract detailed key points from this transcript chunk:

{text}

Please provide an in-depth analysis covering:
- Main topics and themes, with detailed explanations
- Important facts, including any data or statistics
- Different perspectives or arguments discussed
- Examples or anecdotes mentioned
- Implications or conclusions drawn by speakers

Format the response with clear bullet points or numbered sections.
"""
    else:
        prompt = f"""
Provide a thorough and detailed summary of the following transcript:

{text}

Your summary should include:
1. Comprehensive explanation of main topics and themes
2. Detailed insights and important information, including supporting data
3. Different viewpoints or perspectives, if any
4. Examples, anecdotes, or relevant stories
5. Clear explanation of conclusions and takeaways

Organize the summary with headings, bullet points, or numbered lists for clarity.
"""
    
    json_data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1500,
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout
            response = await client.post(GEMINI_API_URL, headers=request_headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            
            # Enhanced error logging and response validation
            if "error" in data:
                error_message = data.get("error", {}).get("message", "Unknown Gemini API error")
                logger.error(f"Gemini API returned error: {error_message}")
                raise Exception(f"Gemini API error: {error_message}")
            
            logger.info(f"Gemini API response received successfully")
            
            # Extract the generated summary text with improved parsing
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    content_parts = candidate["content"]["parts"]
                    if content_parts and "text" in content_parts[0]:
                        return content_parts[0]["text"].strip()
            
            # If we couldn't parse the response, log it and return an error
            logger.error(f"Unexpected Gemini API response structure: {data}")
            return "No summary could be generated due to API response format."
            
    except httpx.HTTPStatusError as e:
        # Specific handling for HTTP status errors
        logger.error(f"HTTP error {e.response.status_code} when calling Gemini API: {e.response.text}")
        raise Exception(f"Gemini API HTTP error {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        # Network-related errors (connection, timeout, etc)
        logger.error(f"Request error when calling Gemini API: {str(e)}")
        raise Exception(f"Network error when connecting to Gemini API: {str(e)}")
    except Exception as e:
        logger.error(f"Error in summarize_text: {str(e)}")
        raise Exception(f"Summary generation failed: {str(e)}")

async def process_long_transcript(transcript: str) -> str:
    """
    Process a transcript that may be too long for a single API call
    with improved error handling and fallback mechanisms
    """
    try:
        # Estimate token count (rough approximation)
        estimated_tokens = len(transcript.split())
        logger.info(f"Transcript length: {estimated_tokens} tokens")
        
        # If transcript is empty, return early
        if not transcript or estimated_tokens == 0:
            logger.warning("Empty transcript provided to process_long_transcript")
            return "No transcript content to summarize."
        
        # If transcript is short enough, process it directly
        if estimated_tokens < 4000:
            return await summarize_text(transcript)
        
        # Otherwise, chunk the transcript and summarize each chunk
        words = transcript.split()
        chunk_size = 3000  # Tokens per chunk
        chunk_count = math.ceil(len(words) / chunk_size)
        logger.info(f"Chunking transcript into {chunk_count} segments")
        
        chunk_summaries = []
        for i in range(chunk_count):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(words))
            chunk_text = " ".join(words[start_idx:end_idx])
            
            logger.info(f"Processing chunk {i+1}/{chunk_count}")
            # Retry mechanism for chunk summarization
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    chunk_summary = await summarize_text(chunk_text, is_chunk=True)
                    chunk_summaries.append(chunk_summary)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt+1} failed, retrying... Error: {str(e)}")
                    else:
                        logger.error(f"All attempts failed for chunk {i+1}")
                        # Add a placeholder instead of failing completely
                        chunk_summaries.append(f"[Summary for segment {i+1} unavailable due to processing error]")
        
        # If we couldn't get any chunk summaries, try a fallback approach
        if not any(summary for summary in chunk_summaries if not summary.startswith("[Summary for segment")):
            logger.warning("All chunk summaries failed. Attempting fallback with first and last chunks")
            # Try with just first and last segments as a fallback
            first_chunk = " ".join(words[:chunk_size])
            last_chunk = " ".join(words[-chunk_size:])
            combined_chunks = f"[Beginning of transcript] {first_chunk}\n\n[...]\n\n[End of transcript] {last_chunk}"
            return await summarize_text(combined_chunks)
        
        # Combine chunk summaries and create a final detailed summary
        combined_points = "\n\n".join(chunk_summaries)
        final_prompt = f"""
Create a cohesive, detailed final summary based on these key points extracted from different sections of a longer transcript:

{combined_points}

Organize the information into a clear, comprehensive summary that captures:
- Main themes and topics
- Detailed insights and important takeaways
- Supporting data and examples
- Different perspectives and conclusions

Format the summary with headings, bullet points, or numbered lists for clarity and readability.
"""

        json_data = {
            "contents": [{
                "parts": [{
                    "text": final_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1500,
            }
        }
        
        request_headers = {
            "Content-Type": "application/json"
        }
        
        logger.info("Generating final summary from chunks")
        async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout
            response = await client.post(GEMINI_API_URL, headers=request_headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                error_message = data.get("error", {}).get("message", "Unknown Gemini API error")
                logger.error(f"Gemini API returned error in final summary: {error_message}")
                # Return a basic summary from chunks instead of failing completely
                return "## Generated Summary\n\n" + "\n\n".join(chunk_summaries)
            
            # Extract the final summary with improved parsing
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    content_parts = candidate["content"]["parts"]
                    if content_parts and "text" in content_parts[0]:
                        return content_parts[0]["text"].strip()
            
            logger.error(f"Unexpected Gemini API response structure: {data}")
            # Return a basic summary from chunks instead of failing completely
            return "## Generated Summary\n\n" + "\n\n".join(chunk_summaries)
    except Exception as e:
        logger.error(f"Error in process_long_transcript: {str(e)}")
        raise Exception(f"Error processing transcript: {str(e)}")

async def get_video_metadata(url: str) -> Dict[str, Any]:
    """Extract metadata from YouTube video"""
    try:
        # Check if yt-dlp is installed
        try:
            # Try running yt-dlp --version to check if it's available
            subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
        except FileNotFoundError:
            logger.error("yt-dlp not found. Please install with: pip install yt-dlp")
            return {
                "title": "Unknown (yt-dlp not installed)",
                "duration": None,
                "video_url": url
            }
        except subprocess.CalledProcessError:
            logger.error("yt-dlp command failed")
            return {
                "title": "Unknown (yt-dlp error)",
                "duration": None,
                "video_url": url
            }
            
        # Use yt-dlp to get video metadata
        result = subprocess.run([
            "yt-dlp", "--print", "%(title)s|%(duration)s", "--no-download", url
        ], check=True, capture_output=True, text=True)
        
        # Parse the output
        output = result.stdout.strip()
        parts = output.split('|')
        
        metadata = {
            "title": parts[0] if len(parts) > 0 else "Unknown",
            "duration": float(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
            "video_url": url
        }
        
        return metadata
    except Exception as e:
        logger.warning(f"Failed to extract video metadata: {str(e)}")
        return {
            "title": "Unknown",
            "duration": None,
            "video_url": url
        }

def check_media_permissions():
    """Check if the server has permission to create media files in the current directory"""
    try:
        # Try to create a test file
        test_file = "permission_test.txt"
        with open(test_file, "w") as f:
            f.write("test")
        
        # Clean up
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"No permission to write files in current directory: {str(e)}")
        return False

@app.post("/transcribe")
async def transcribe_video(request: VideoRequest, background_tasks: BackgroundTasks):
    global whisper_model  # Add this line to declare whisper_model as global
    url = request.youtube_url
    session_id = str(uuid.uuid4())
    audio_basename = f"{session_id}"
    
    logger.info(f"Processing video URL: {url}")
    logger.info(f"Session ID: {session_id}")
    
    # First check if we have the required dependencies and permissions
    if not whisper_model:
        try:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("tiny", compute_type="int8")
            logger.info("Loaded whisper model on demand")
        except ImportError:
            logger.error("Missing dependency: faster_whisper")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Speech recognition service not available",
                    "details": "Missing dependency: faster_whisper. Please install with: pip install faster-whisper"
                }
            )
        except Exception as e:
            logger.error(f"Failed to load whisper model: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Speech recognition service not available",
                    "details": f"Error loading whisper model: {str(e)}"
                }
            )
    
    # Check file system permissions
    if not check_media_permissions():
        return JSONResponse(
            status_code=503,
            content={
                "error": "Server configuration error",
                "details": "No permission to write files in the current directory"
            }
        )
    
    try:
        # Get video metadata
        metadata = await get_video_metadata(url)
        if "error" in metadata:
            return JSONResponse(
                status_code=400,
                content={"error": metadata["error"]}
            )
            
        metadata["transcript_id"] = session_id
        
        # Download audio using yt-dlp with more detailed error handling
        try:
            logger.info("Starting yt-dlp download")
            result = subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "-o", f"{audio_basename}.%(ext)s", url
            ], check=True, capture_output=True, text=True, timeout=300)  # 5-minute timeout
            
            logger.info(f"yt-dlp stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"yt-dlp stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("yt-dlp process timed out")
            return JSONResponse(
                status_code=408,
                content={"error": "Download timed out. The video might be too large or the server is under heavy load."}
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
            logger.error(f"yt-dlp error: {error_msg}")
            
            # Try to give more specific error messages for common issues
            if "unavailable" in error_msg.lower() or "not available" in error_msg.lower():
                return JSONResponse(
                    status_code=400,
                    content={"error": "Video is unavailable. It might be private, removed, or region-restricted."}
                )
            elif "copyright" in error_msg.lower():
                return JSONResponse(
                    status_code=400,
                    content={"error": "Unable to access this video due to copyright restrictions."}
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Error downloading video: {error_msg}"}
                )
        
        # Find the downloaded audio file
        matching_files = glob.glob(f"{audio_basename}.*")
        logger.info(f"Found files: {matching_files}")
        
        if not matching_files:
            logger.error("No audio file found after download")
            return JSONResponse(
                status_code=404,
                content={"error": "Audio file not found after download. YouTube URL may be invalid or content is unavailable."}
            )
            
        audio_file = matching_files[0]
        logger.info(f"Using audio file: {audio_file}")
        
        # Check if file exists and is valid
        if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
            logger.error(f"Downloaded file {audio_file} is missing or empty")
            return JSONResponse(
                status_code=500,
                content={"error": "Downloaded audio file is missing or empty"}
            )
        
        # Transcribe with better error handling
        try:
            logger.info("Starting transcription")
            segments, info = whisper_model.transcribe(audio_file)
            segments_list = list(segments)  # Convert generator to list
            
            if not segments_list:
                logger.warning("No segments returned from transcription")
                # Clean up audio file first
                try:
                    os.remove(audio_file)
                    logger.info(f"Removed temporary file: {audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
                    
                return JSONResponse(
                    status_code=200,  # This is still a valid result, just empty
                    content={"warning": "No speech detected in the audio", 
                             "transcript": "", 
                             "summary": "No speech content to summarize",
                             "session_id": session_id,
                             "metadata": metadata}
                )
                
            full_transcript = " ".join([seg.text.strip() for seg in segments_list])
            logger.info(f"Transcription completed. Length: {len(full_transcript)} characters")
            
            # Create chunks for vector storage
            logger.info("Creating chunks for vector storage")
            transcript_chunks = chunk_transcript(segments_list)
            logger.info(f"Created {len(transcript_chunks)} chunks")
            
            # Process the transcript for summary (handles chunking if needed)
            try:
                logger.info("Generating summary")
                summary = await process_long_transcript(full_transcript)
                logger.info(f"Summary generated. Length: {len(summary)} characters")
                
                # Only store in vector DB if Qdrant is available
                if qdrant_client:
                    # Schedule background task to store in vector DB
                    background_tasks.add_task(
                        store_in_vector_db,
                        session_id=session_id,
                        chunks=transcript_chunks,
                        metadata=metadata,
                        summary=summary
                    )
                else:
                    logger.warning("Vector DB not available. Transcript will not be stored for search.")
                
                # Clean up audio file
                try:
                    os.remove(audio_file)
                    logger.info(f"Removed temporary file: {audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
                
                return {
                    "transcript": full_transcript,
                    "summary": summary,
                    "session_id": session_id,
                    "chunks_count": len(transcript_chunks),
                    "metadata": metadata
                }
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                # Return transcript but with error info for summary
                return {
                    "transcript": full_transcript,
                    "summary": f"Summary generation failed: {str(e)}",
                    "session_id": session_id,
                    "chunks_count": len(transcript_chunks),
                    "metadata": metadata
                }
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            # Clean up audio file in case of error
            try:
                os.remove(audio_file)
                logger.info(f"Removed temporary file: {audio_file}")
            except Exception as clean_error:
                logger.warning(f"Failed to remove temporary file: {str(clean_error)}")
                
            return JSONResponse(
                status_code=500,
                content={"error": f"Transcription failed: {str(e)}"}
            )
        
    except Exception as e:
        logger.error(f"Unhandled error in transcribe_video: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.post("/chat")
async def chat_with_transcript(request: ChatRequest):
    """Query the transcribed content using RAG"""
    try:
        check_dependencies(
            {"qdrant_client": qdrant_client, "embedding_model": embedding_model}, 
            "Vector search is not available"
        )
        
        query = request.query
        session_id = request.session_id
        
        logger.info(f"Processing chat query: '{query}' for session: {session_id}")
        
        # Get relevant chunks and summary from vector DB
        search_results = await search_vector_db(query, session_id, limit=5)
        
        if not search_results:
            logger.warning(f"No relevant chunks found for query: {query}")
            return {
                "answer": "I couldn't find relevant information in the transcript to answer your question.",
                "context": []
            }
        
        # Separate chunks and summary from search results
        chunks = [r for r in search_results if r.get("type") == "chunk"]
        summary = next((r for r in search_results if r.get("type") == "summary"), None)
        
        # Create context from the most relevant chunks and summary
        context_parts = []
        if summary:
            context_parts.append(f"Summary of the video:\n{summary['text']}\n")
        
        if chunks:
            context_parts.append("Relevant transcript segments:")
            context_parts.extend([f"Segment {i+1}: {chunk['text']}" for i, chunk in enumerate(chunks)])
        
        context_text = "\n\n".join(context_parts)
        
        # Prepare prompt for Gemini
        prompt = f"""
You are an AI assistant that helps users understand video content based on transcripts and summaries. 
Your task is to provide the most helpful answer possible using the available information from both the summary and transcript segments.

Query: {query}

Context:
{context_text}

Instructions:
1. If you can answer the query using the summary, do so first and then add relevant details from the transcript segments.
2. If the summary doesn't contain enough information, use the transcript segments to provide a detailed answer.
3. If you can partially answer the query, provide what you know and explain what information is missing.
4. Only say "I don't have enough information" if you truly cannot provide any relevant information from either the summary or transcript segments.

Please provide a comprehensive answer based on the available information.
"""
        
        json_data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000,
            }
        }
        
        request_headers = {
            "Content-Type": "application/json"
        }
        
        # Call Gemini API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_API_URL, headers=request_headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                error_message = data.get("error", {}).get("message", "Unknown Gemini API error")
                logger.error(f"Gemini API returned error: {error_message}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"AI response generation failed: {error_message}"}
                )
            
            # Extract the answer with improved parsing
            answer = "No response could be generated."
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    content_parts = candidate["content"]["parts"]
                    if content_parts and "text" in content_parts[0]:
                        answer = content_parts[0]["text"].strip()
        
        # Return the answer and the context
        return {
            "answer": answer,
            "context": {
                "chunks": chunks,
                "summary": summary
            }
        }
        
    except HTTPException as e:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Chat processing failed: {str(e)}"}
        )

@app.get("/status/{session_id}")
async def get_processing_status(session_id: str):
    """Check transcript processing status for a session"""
    try:
        if not qdrant_client:
            return {
                "status": "unknown",
                "message": "Vector database service unavailable",
                "chunks_available": False
            }
            
        # Check if any chunks exist for this session ID
        from qdrant_client.http import models
        
        search_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            ),
            limit=1
        )
        
        if search_result and search_result[0]:
            chunk_count = qdrant_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id)
                        )
                    ]
                )
            ).count
            
            return {
                "status": "completed",
                "message": "Transcript processing complete",
                "chunks_available": True,
                "chunks_count": chunk_count
            }
        else:
            return {
                "status": "pending",
                "message": "Transcript processing in progress or session not found",
                "chunks_available": False
            }
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return {
            "status": "error",
            "message": f"Error checking status: {str(e)}",
            "chunks_available": False
        }
