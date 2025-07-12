from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import openai
from gtts import gTTS
import uuid
import os
import aiofiles
from dotenv import load_dotenv
from openai import OpenAI

import logging

from dotenv import load_dotenv
_ = load_dotenv(override=True)

logger = logging.getLogger("v-agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
app = FastAPI()

load_dotenv()

# Allow local frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your API keys from environment or directly here
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
openai.api_key = TOGETHER_API_KEY
openai.api_base = "https://api.together.xyz/v1"

@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}
@app.post("/pronounce/")
async def process_pronunciation(audio: UploadFile, target: str = Form(...)):
    logger.info("Received request to /pronounce/")

    # Save uploaded audio
    audio_path = f"uploads/{uuid.uuid4()}.wav"
    os.makedirs("uploads", exist_ok=True)

    logger.info(f"Saving uploaded audio to {audio_path}")
    async with aiofiles.open(audio_path, 'wb') as out_file:
        content = await audio.read()
        await out_file.write(content)

    # Step 1: Transcribe using Groq
    logger.info("Sending audio to Groq for transcription...")
    with open(audio_path, "rb") as f:
        transcript_response = groq_client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
            language="en"
        )

    transcript = transcript_response.text
    logger.info(f"Transcript: {transcript}")
    # Step 2: Generate feedback using Together API
    logger.info("Generating feedback with Together API...")

    prompt = f"""You are a helpful pronunciation coach.
                The correct sentence is: "{target}"
                The student said: "{transcript}"
                Give specific pronunciation corrections and tips.
                """
    client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1"
    )

    chat = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[{"role": "user", "content": prompt}]
    )
    feedback = chat.choices[0].message.content
    logger.info(f"Feedback: {feedback}")
    # Step 3: TTS (speak the correct sentence)
    logger.info("Generating TTS audio...")
    tts = gTTS(text=target)
    tts_path = f"tts/{uuid.uuid4()}.mp3"
    os.makedirs("tts", exist_ok=True)
    tts.save(tts_path)
    logger.info(f"TTS saved to {tts_path}")

    return {
        "transcript": transcript,
        "feedback": feedback,
        "tts_audio_url": f"/tts_audio/{os.path.basename(tts_path)}"
    }

@app.get("/tts_audio/{file_name}")
def get_audio(file_name: str):
    logger.info(f"Serving TTS audio file: {file_name}")
    return FileResponse(path=f"tts/{file_name}", media_type="audio/mpeg")

