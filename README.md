# STT-Engine: AI-Powered Speech Recognition and Chatbot Integration

## Overview

STT-Engine is an AI-driven Speech-to-Text (STT) system designed to transcribe speech accurately, even across diverse accents and dialects. By fine-tuning the **Facebook Wav2Vec2** model on the AccentDB dataset, the system enhances transcription accuracy for Indian dialects. Additionally, it integrates a chatbot feature, enabling interactive and intelligent user experiences. The **Groq API** is used for LLM-based transcription refinement, ensuring higher accuracy.

## Features

- **Accent Adaptation**: Fine-tuned **Facebook Wav2Vec2** model on the AccentDB dataset to improve transcription accuracy for various Indian dialects.
- **Transcription Refinement**: Utilizes **Groq API (LLM)** to correct phonetic and contextual errors in transcriptions, enhancing overall quality.
- **Real-Time Interaction**: Integrates a chatbot that processes user speech input and provides intelligent responses, leveraging the refined transcriptions.

## Repository Structure

- **Model_Training/**
  - `Model_training.ipynb`: Notebook detailing the process of fine-tuning the **Facebook Wav2Vec2** model on the AccentDB dataset.
  - `Model_performance.ipynb`: Notebook evaluating the performance of the fine-tuned model, including metrics like Word Error Rate (WER) and accuracy.

- **Pre_Processing/**
  - `Pre_processing.ipynb`: Notebook focusing on preprocessing steps such as data cleaning, augmentation, and preparation for model training.

- **Demo.ipynb**: Interactive notebook showcasing the functionalities of the STT system, including transcription and chatbot interactions.

- **STT_chatbot.py**: Streamlit-based application that provides a user-friendly interface for real-time speech transcription and chatbot interaction.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install librosa numpy pandas
pip install soundfile pydub
pip install openai streamlit
```

-Finetuned Model and Processed Spectrograms Available on:
    GDRIVE: https://drive.google.com/drive/folders/1vTC1Vpsd3YAsCvNKu01mZzRZxjgshhMt?usp=sharing

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DivyViradiya07/STT-engine.git
   cd STT-engine
   ```

2. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

### Usage

1. **Data Preprocessing**

   Navigate to the `Pre_Processing` directory and run the `Pre_processing.ipynb` notebook to preprocess the AccentDB dataset.

2. **Model Training**

   In the `Model_Training` directory, execute the `Model_training.ipynb` notebook to fine-tune the **Facebook Wav2Vec2** model on the preprocessed data.

3. **Model Evaluation**

   Use the `Model_performance.ipynb` notebook to assess the performance of the fine-tuned model, focusing on metrics like WER and accuracy.

4. **Demonstration**

   Run the `Demo.ipynb` notebook to interactively test the STT system's transcription and chatbot capabilities.

5. **Real-Time Chatbot Interaction**

   Launch the Streamlit application for real-time interaction:

   ```bash
   streamlit run STT_chatbot.py
   ```

   This will open a web interface where you can speak and receive transcriptions and chatbot responses in real-time.

## Performance Metrics

- **Word Error Rate (WER)**:
  - Without refinement: 5%
  - With **Groq API LLM refinement**: 2.5%

- **Transcription Accuracy**:
  - Without refinement: 70%
  - With **Groq API LLM refinement**: 90%

## Future Enhancements

- **Extended Language Support**: Incorporate additional languages and dialects to broaden the system's applicability.
- **Mobile Integration**: Develop mobile applications to facilitate on-the-go speech recognition and chatbot interactions.
- **Enhanced Chatbot Intelligence**: Integrate more advanced NLP models to improve the chatbot's contextual understanding and response generation.

## Acknowledgments

- **Facebook Wav2Vec2** for the ASR model and the Transformers library.
- The creators of the AccentDB dataset for providing valuable speech data.
- **Groq API** for the language model used in transcription refinement and chatbot responses.

