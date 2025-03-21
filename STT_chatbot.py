import streamlit as st
import torch
import numpy as np
import wave
import os
import pyttsx3  
import datetime
import pyaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
groq_llm = ChatGroq(
    model_name="mistral-saba-24b",
    groq_api_key=groq_api_key,
    temperature=0.1
)

# Initialize Streamlit UI
st.set_page_config(page_title="AI Voice Chatbot", layout="centered")
st.title("💬 AI Voice Chatbot")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS for Chat Styling
st.markdown(
    """
    <style>
        .user-msg {background-color: #0078FF; color: white; padding: 10px; border-radius: 10px; max-width: 70%; text-align: left;}
        .ai-msg {background-color: #F0F0F0; color: black; padding: 10px; border-radius: 10px; max-width: 70%; text-align: left;}
        .message-container {display: flex; align-items: center; margin: 5px 0;}
        .user-container {justify-content: flex-end;}
        .ai-container {justify-content: flex-start;}
    </style>
    """,
    unsafe_allow_html=True
)

# Load ASR Model
@st.cache_resource
def load_wav2vec2_model(model_path):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

# Load model
model_path = "D:/Speech_recognition/Model_Training/wav2vec2_finetuned"
processor, model, device = load_wav2vec2_model(model_path)

# Initialize TTS
tts_engine = pyttsx3.init()

# Recording Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Set recording time
AUDIO_FILE = "recorded_audio/recorded.wav"

def record_audio():
    """Records audio using PyAudio and saves it as a WAV file."""
    audio = pyaudio.PyAudio()

    # Open audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    st.success("🎙 Recording... Speak now!")
    frames = []

    # Record audio
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio
    os.makedirs("recorded_audio", exist_ok=True)
    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    st.success(f"✅ Audio recorded and saved as `{AUDIO_FILE}`")
    return AUDIO_FILE

# Function to transcribe audio
def transcribe_audio(audio_path):
    """Loads a WAV file and transcribes it using Wav2Vec2."""
    with wave.open(audio_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize

    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    input_values = input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

# Function to refine transcription
def batch_refine_transcriptions(asr_outputs):
    if not asr_outputs:
        return []

    prompt = f"""
    You are an advanced ASR correction assistant that fixes phonetic and contextual mistakes in transcriptions.
    - Correct phonetic errors 
    - Correct name recognition 
    - Ensure proper grammar while preserving original meaning.
    
    Here are multiple ASR outputs. Correct each one and return ONLY the corrected texts:
    {asr_outputs}
    """

    response = groq_llm.invoke(prompt)
    if not response or not hasattr(response, "content"):
        return ["❌ Error: No response from AI"]
    
    corrected_texts = response.content.strip().split("\n")
    return corrected_texts

# Function to interact with AI
def chat_with_ai(user_input):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(("user", timestamp, user_input))

    prompt = f"Chat history:\n" + "\n".join(
        [f"User: {msg[2]}" if msg[0] == "user" else f"AI: {msg[2]}" for msg in st.session_state.chat_history]
    ) + f"\n\nUser: {user_input}\nAI:"
    
    response = groq_llm.invoke(prompt)

    ai_response = "❌ Error: No response from AI" if not response or not hasattr(response, "content") else response.content.strip()
    st.session_state.chat_history.append(("ai", timestamp, ai_response))

st.subheader("🎤 Voice Input")

if st.button("🎙 Record Audio"):
    audio_file = record_audio()
    if audio_file:
        raw_transcription = transcribe_audio(audio_file)
        refined_transcription = batch_refine_transcriptions([raw_transcription])[0]
        st.write(f"📝 **Transcription:** {refined_transcription}")
        chat_with_ai(refined_transcription)

# Chat Input
st.subheader("💬 Chat with AI")
user_input = st.text_input("Type your message:")
if st.button("Send"):
    if user_input:
        chat_with_ai(user_input)

# Display Chat Messages
st.subheader("🗨️ Chat History")
for msg_type, timestamp, message in st.session_state.chat_history:
    if msg_type == "user":
        st.markdown(f'<div class="message-container user-container"><div class="user-msg">🧑‍💻 [{timestamp}] You: {message}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-container ai-container"><div class="ai-msg">🤖 [{timestamp}] AI: {message}</div></div>', unsafe_allow_html=True)

# Clear Chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
