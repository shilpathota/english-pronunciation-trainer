import streamlit as st
import uuid
import os
import openai
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv
from openai import OpenAI
import tempfile

# Load environment variables (for local dev)
if "GROQ_API_KEY" not in st.secrets:
    load_dotenv()

# Unified config
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY", os.getenv("TOGETHER_API_KEY"))

# Clients
groq_client = Groq(api_key=GROQ_API_KEY)
openai.api_key = TOGETHER_API_KEY
openai.api_base = "https://api.together.xyz/v1"

st.title("🎙️ English Pronunciation Coach")

# Target sentence
target = st.text_input("Target sentence:", value="I would like a cup of coffee.")

# Upload audio
audio_file = st.file_uploader("Upload your spoken WAV file:", type=["wav"])

if audio_file and st.button("Submit"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Transcription
    with open(tmp_path, "rb") as f:
        transcript_response = groq_client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
            language="en"
        )

    transcript = transcript_response.text
    st.markdown("📝 **Transcript:**")
    st.code(transcript)

    # Feedback prompt
    prompt = f"""You are a helpful pronunciation coach.
The correct sentence is: "{target}"
The student said: "{transcript}"
Give specific pronunciation corrections and tips."""

    client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
    chat = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[{"role": "user", "content": prompt}]
    )

    feedback = chat.choices[0].message.content
    st.markdown("📢 **Feedback:**")
    st.success(feedback)

    # TTS
    tts = gTTS(text=target)
    tts_path = f"{uuid.uuid4()}.mp3"
    tts.save(tts_path)

    st.markdown("🔊 **Correct Pronunciation (TTS):**")
    audio_bytes = open(tts_path, "rb").read()
    st.audio(audio_bytes, format="audio/mp3")
    os.remove(tts_path)
