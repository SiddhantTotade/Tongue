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

# main.py Snippet
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_lang = "hi"
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                import json
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        current_lang = data.get("lang", current_lang)
                        logger.info(f"Language switched to {current_lang}")
                except json.JSONDecodeError:
                    logger.error("Failed to parse WebSocket text message")
                    
            elif "bytes" in message:
                audio_bytes = message["bytes"]
                output_path = await engine.process_chunk(audio_bytes, current_lang)
                
                with open(output_path, "rb") as f:
                    await websocket.send_bytes(f.read())
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")

@app.post("/translate")
async def translate_audio(audio: UploadFile = File(...), target_lang: str = Form("hi")):
    """
    Translates uploaded audio to the target language with human-like speech.
    """
    temp_input = f"temp_{audio.filename}"
    
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        translated_audio_path = await engine.process(temp_input, target_lang)
        
        return FileResponse(
            translated_audio_path, 
            media_type="audio/wav",
            filename=os.path.basename(translated_audio_path)
        )
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup is handled inside engine.process, but just in case of failure:
        if os.path.exists(temp_input):
            os.remove(temp_input)

@app.post("/set-reference")
async def set_reference(audio: UploadFile = File(...)):
    """
    Upload a 6-10 second high-quality audio clip to use for voice cloning.
    """
    try:
        with open("reference.wav", "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        return {"message": "Reference voice updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save reference: {str(e)}")

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
