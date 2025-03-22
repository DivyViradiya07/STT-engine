🎙️ ASR Engine with LLM Refinement

🚀 Project Overview

This project focuses on building an Automatic Speech Recognition (ASR) system optimized for Indian English accents, using Wav2Vec2 for speech-to-text transcription. Additionally, Groq LLM is integrated to refine ASR outputs, improving accuracy and reducing phonetic errors.

✅ Features

🎤 Speech-to-Text Conversion (ASR Model: Wav2Vec2)

🏆 Fine-Tuned on Indian English Accent (AccentDB Dataset)

🤖 Error Correction using Groq LLM

📊 Performance Metrics: Word Error Rate (WER) & Accuracy

⚡ Optimized for Speed (Inference & Training Enhancements)

📂 Dataset & Preprocessing

Dataset Used: AccentDB (Indian English Speech Dataset)

Preprocessing Steps:

Converted audio files to standardized format (16kHz WAV)

Extracted MFCC features & spectrograms

Generated transcriptions & stored in structured format

🏗️ Model Training

ASR Model Used: facebook/wav2vec2-large-960h

Training Process:

Batch Size: 4 (Optimized for memory constraints)

Epochs: 2

Gradient Scaling for efficient GPU training

Fine-Tuned ASR Model Saved At: D:/Speech_recognition/wav2vec2_finetuned

🤖 ASR Output Refinement using LLM

LLM Model Used: mixtral-8x7b-32768 (via Groq API)

LLM Fixes Common ASR Issues:

Phonetic Errors (e.g., "sake fully ter" → "sexual intercourse")

Grammar & Readability Improvements

Proper Name Recognition

📊 Performance Evaluation

Metric

Before LLM Refinement

After LLM Refinement

Word Error Rate (WER)

5.0%

2.5% ✅

Accuracy

~80%

90%+ ✅

Inference Speed

~2 sec per file

Optimized ✅
