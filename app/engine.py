import os
import torch
import tempfile
import asyncio
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from TTS.api import TTS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # 1. Initialize STT (Whisper)
        # faster-whisper 0.10.1 compatible
        # float16 for CUDA, int8 for CPU
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.stt_model = WhisperModel(
            "large-v3",
            device=self.device,
            compute_type=compute_type,
            download_root="/app/models/whisper"
        )

        # 2. Initialize TTS (XTTS v2)
        # XTTS v2 is the gold standard for open-source human-like speech
        # We initialize it once to cache the model
        logger.info("Initializing XTTS v2 model (this may take a while on first run)...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
        # Ensure models directory exists
        os.makedirs("/app/models/xtts", exist_ok=True)
    
    # Add this to your TranslationEngine class in engine.py
    async def process_stream(self, audio_bytes, target_lang):
        # Save bytes to a temp file or use a buffer
        temp_input = "stream_input.wav"
        with open(temp_input, "wb") as f:
            f.write(audio_bytes)
        
        # Reuse your existing logic but return the bytes of the output
        output_path = await self.process(temp_input, target_lang)
    
        with open(output_path, "rb") as f:
            return f.read()
    
    # Add to your engine.py

    async def process_chunk(self, audio_bytes, target_lang):
        # Use a temporary file to store the incoming chunk
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_in:
            temp_in.write(audio_bytes)
            temp_in_path = temp_in.name

        try:
            # Run your existing translation/cloning logic
            output_path = await self.process(temp_in_path, target_lang)
            
            with open(output_path, "rb") as f:
                return f.read()
        finally:
            # Cleanup to prevent disk bloat
            if os.path.exists(temp_in_path):
                os.remove(temp_in_path)

    async def process(self, audio_path, target_lang):
        loop = asyncio.get_running_loop()
        
        try:
            # 1. Speech to Text (Non-blocking)
            def transcribe():
                logger.info("Starting transcription...")
                segments, info = self.stt_model.transcribe(audio_path, beam_size=5)
                # Iterate inside the executor to avoid blocking the event loop
                text = " ".join([segment.text for segment in segments])
                return text

            english_text = await loop.run_in_executor(None, transcribe)
            
            if not english_text.strip():
                logger.warning("No speech detected.")
                return audio_path

            logger.info(f"Transcribed: {english_text}")

            # 2. Translation
            # Wrapping sync call in executor
            translated_text = await loop.run_in_executor(
                None, 
                lambda: GoogleTranslator(source='auto', target=target_lang).translate(english_text)
            )
            logger.info(f"Translated ({target_lang}): {translated_text}")

            # 3. Human-like Text to Speech (XTTS v2)
            output_filename = f"translated_{os.path.basename(audio_path)}.wav"
            output_path = os.path.join(os.getcwd(), output_filename)
            
            # For XTTS, we need a reference speaker. 
            # NOTE: For best results, the user should provide 'reference.wav'
            # We'll use a placeholder speaker if reference.wav is missing, 
            # though XTTS usually requires a file.
           # This looks in the same folder as engine.py
            ref_path = os.path.join(os.path.dirname(__file__), "reference.wav")
            
            # Check if reference exists, if not, we might need a fallback or fail gracefully
            if not os.path.exists(ref_path):
                logger.warning("reference.wav not found! Please provide a 6-10s high-quality audio clip as 'reference.wav' for voice cloning.")
                # We will try to use the first available speaker if possible, 
                # but XTTS v2 API usually needs a wav for the 'speaker_wav' argument.
                # If no reference, this might error. I'll add an endpoint to upload it.

            def generate_tts():
                # XTTS v2 supports many languages including hi, mr, ta, etc.
                # If no reference.wav, we try to use a default speaker if possible,
                # but XTTS v2 usually requires a reference wav for cloning.
                # For now, we'll pass the speaker_wav only if it exists.
                kwargs = {
                    "text": translated_text,
                    "language": target_lang,
                    "file_path": output_path
                }
                
                if os.path.exists(ref_path):
                    kwargs["speaker_wav"] = ref_path
                else:
                    # Fallback to a built-in speaker if possible
                    # Note: xtts_v2 doesn't always have a 'default' speaker name, 
                    # it prefers reference wavs. We'll use a known one if we can.
                    logger.warning("No reference.wav found. Attempting to use default speaker 'Damien Sincere'.")
                    kwargs["speaker"] = "Damien Sincere" 

                self.tts.tts_to_file(**kwargs)
                return output_path

            await loop.run_in_executor(None, generate_tts)

            # Cleanup input
            if os.path.exists(audio_path):
                os.remove(audio_path)

            return output_path

        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            raise e