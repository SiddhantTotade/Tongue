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
            "base", # Changed from large-v3 for much faster real-time processing
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

    async def process_video(self, video_path, target_lang):
        loop = asyncio.get_running_loop()
        
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            # 1. Extract audio from video to use as dynamic emotion/voice reference
            def extract_audio():
                logger.info(f"Extracting audio from {video_path}...")
                video = VideoFileClip(video_path)
                audio_ext_path = f"temp_extracted_{os.path.basename(video_path)}.wav"
                video.audio.write_audiofile(audio_ext_path, logger=None)
                video.close()
                return audio_ext_path
            
            extracted_audio_path = await loop.run_in_executor(None, extract_audio)
            
            # 1.5 Isolate vocals and background
            def isolate_audio():
                logger.info(f"Isolating vocals and background from {extracted_audio_path}...")
                import subprocess
                output_dir = "separated_audio"
                
                # Demucs CLI: separates into vocals.wav and no_vocals.wav
                cmd = [
                    "python", "-m", "demucs.separate", 
                    "-n", "htdemucs", 
                    "--two-stems=vocals", 
                    "-o", output_dir, 
                    extracted_audio_path
                ]
                subprocess.run(cmd, check=True)
                
                base_name = os.path.splitext(os.path.basename(extracted_audio_path))[0]
                model_out_dir = os.path.join(output_dir, "htdemucs", base_name)
                
                v_path = os.path.join(model_out_dir, "vocals.wav")
                b_path = os.path.join(model_out_dir, "no_vocals.wav")
                
                if not os.path.exists(v_path) or not os.path.exists(b_path):
                    raise Exception(f"Demucs failed: Expected files not found in {model_out_dir}")
                    
                return v_path, b_path, output_dir
                
            vocals_path, bg_path, demucs_out_dir = await loop.run_in_executor(None, isolate_audio)
            
            # 2. Transcribe Isolated Vocals
            def transcribe():
                logger.info("Starting transcription on isolated vocals...")
                segments, info = self.stt_model.transcribe(
                    vocals_path, 
                    beam_size=1, 
                    language="en", 
                    vad_filter=True, 
                    condition_on_previous_text=False
                )
                text = " ".join([segment.text for segment in segments])
                return text

            english_text = await loop.run_in_executor(None, transcribe)
            
            if not english_text.strip():
                logger.warning("No speech detected.")
                return video_path
                
            logger.info(f"Transcribed: {english_text}")
            
            # 3. Translate
            translated_text = await loop.run_in_executor(
                None, 
                lambda: GoogleTranslator(source='en', target=target_lang).translate(english_text)
            )
            logger.info(f"Translated ({target_lang}): {translated_text}")
            
            # 4. Generate TTS using the isolated vocals as the clean reference
            def generate_tts():
                output_audio_path = f"translated_audio_{os.path.basename(video_path)}.wav"
                
                logger.info("Computing conditioning latents from isolated vocals for clean emotion cloning...")
                gpt_cond_latent, speaker_embedding = self.tts.synthesizer.tts_model.get_conditioning_latents(audio_path=[vocals_path])
                
                # Split text into chunks to avoid the XTTS 400 token limit
                import re
                import numpy as np
                
                # Split by sentence terminators including Hindi Purna Viram (।)
                sentences = re.split(r'(?<=[.!?।])\s+', translated_text)
                chunks = []
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 200:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        # If a single sentence is still too long, we forcefully split it
                        if len(sentence) >= 200:
                            words = sentence.split()
                            temp_chunk = ""
                            for word in words:
                                if len(temp_chunk) + len(word) < 200:
                                    temp_chunk += word + " "
                                else:
                                    chunks.append(temp_chunk.strip())
                                    temp_chunk = word + " "
                            current_chunk = temp_chunk
                        else:
                            current_chunk = sentence + " "
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    
                all_wavs = []
                for chunk in chunks:
                    if not chunk.strip(): continue
                    logger.info(f"Generating TTS for chunk: {chunk[:30]}...")
                    out = self.tts.synthesizer.tts_model.inference(
                        text=chunk,
                        language=target_lang,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=0.7,
                    )
                    all_wavs.append(np.array(out["wav"]))
                    
                if all_wavs:
                    wav = np.concatenate(all_wavs)
                else:
                    wav = np.zeros(24000)
                
                import torchaudio
                if isinstance(wav, list):
                    wav = torch.tensor(wav)
                elif not isinstance(wav, torch.Tensor):
                    wav = torch.from_numpy(np.array(wav))
                    
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                    
                torchaudio.save(output_audio_path, wav.cpu(), 24000)
                return output_audio_path

            tts_audio_path = await loop.run_in_executor(None, generate_tts)
            
            # 5. Mix Audio and Mux back to Video
            def mux_video():
                logger.info("Mixing translated audio with background track and muxing into video...")
                from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
                video = VideoFileClip(video_path)
                
                new_vocals = AudioFileClip(tts_audio_path)
                bg_audio = AudioFileClip(bg_path)
                
                # Combine vocals and background
                mixed_audio = CompositeAudioClip([bg_audio, new_vocals])
                
                final_video = video.set_audio(mixed_audio)
                
                # Output as .mp4 regardless of input format to ensure universal compatibility
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_video_path = f"translated_{base_name}.mp4"
                
                final_video.write_videofile(
                    output_video_path, 
                    codec="libx264", 
                    audio_codec="aac", 
                    logger=None
                )
                
                video.close()
                new_vocals.close()
                bg_audio.close()
                final_video.close()
                
                return output_video_path
                
            final_video_path = await loop.run_in_executor(None, mux_video)
            
            # Cleanup
            import shutil
            for p in [extracted_audio_path, tts_audio_path, video_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            if os.path.exists(demucs_out_dir):
                shutil.rmtree(demucs_out_dir, ignore_errors=True)
                        
            return final_video_path
            
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            raise e