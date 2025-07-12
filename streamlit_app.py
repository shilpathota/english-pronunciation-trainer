import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import asyncio
import av
import os
import uuid
import tempfile
from groq import Groq
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv
import openai as openai_lib
import numpy as np
import wave

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
openai_lib.api_key = TOGETHER_API_KEY
openai_lib.api_base = "https://api.together.xyz/v1"

st.title("🎙️ English Pronunciation Coach")

# Target sentence input
target_sentence = st.text_input("🎯 Target sentence:", "I would like a cup of coffee.")

# Placeholder for audio buffer
audio_buffer = []

class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        pcm = frame.to_ndarray()
        self.frames.append(pcm)
        return frame

# Use WebRTC for mic input
webrtc_ctx = webrtc_streamer(
    key="pronunciation-coach",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

processor = AudioProcessor()

if webrtc_ctx.audio_receiver:
    while True:
        try:
            audio_frame = webrtc_ctx.audio_receiver.get_frames(timeout=1)[0]
            processor.recv(audio_frame)
        except asyncio.TimeoutError:
            break

# Save recording
if st.button("🛑 Submit & Get Feedback"):
    if processor.frames:
        pcm_data = np.concatenate(processor.frames).astype(np.int16)

        # Save as WAV
        wav_path = f"{uuid.uuid4()}.wav"
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm_data.tobytes())

        st.audio(wav_path)

        # Step 1: Transcribe
        with open(wav_path, "rb") as f:
            transcript_response = groq_client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo",
                language="en"
            )
        transcript = transcript_response.text
        st.markdown("📝 **Transcript:**")
        st.info(transcript)

        # Step 2: Feedback via Together
        prompt = f"""You are a helpful pronunciation coach.
The correct sentence is: "{target_sentence}"
The student said: "{transcript}"
Give specific pronunciation corrections and tips.
"""
        client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
        chat = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=[{"role": "user", "content": prompt}]
        )
        feedback = chat.choices[0].message.content
        st.markdown("📢 **Feedback:**")
        st.success(feedback)

        # Step 3: TTS
        tts = gTTS(text=target_sentence)
        tts_path = f"{uuid.uuid4()}.mp3"
        tts.save(tts_path)

        st.markdown("🔊 **Correct Pronunciation (TTS):**")
        audio_bytes = open(tts_path, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")

        os.remove(wav_path)
        os.remove(tts_path)
    else:
        st.warning("No audio was recorded. Please speak into your mic first.")
