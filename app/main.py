from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.engine import TranslationEngine
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tongue - Human-like Translator")
engine = TranslationEngine()

@app.post("/translate-video")
async def translate_video(video: UploadFile = File(...), target_lang: str = Form("hi")):
    """
    Translates an uploaded video file to the target language, cloning the original speaker's emotion and voice.
    Supports standard video formats (mp4, mkv, mov, etc.).
    """
    # Use the original filename extension but save locally to process
    ext = os.path.splitext(video.filename)[1]
    if not ext:
        ext = ".mp4" # fallback
        
    temp_input = f"temp_upload{ext}"
    
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process the video
        translated_video_path = await engine.process_video(temp_input, target_lang)
        
        return FileResponse(
            translated_video_path, 
            media_type="video/mp4",
            filename=os.path.basename(translated_video_path)
        )
    except Exception as e:
        logger.error(f"Video translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded raw file
        if os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception:
                pass

@app.get("/languages")
async def get_languages():
    """
    Returns supported languages for XTTS v2.
    """
    return {
        "languages": [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "hi"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": engine.device}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
