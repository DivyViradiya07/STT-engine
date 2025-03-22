# STT-Engine: Real-Time Speech-to-Text with Chatbot Integration

## Overview
STT-Engine is a real-time speech-to-text system that leverages the Wav2Vec2 model fine-tuned on Indian dialect audio from AccentDB. It refines transcriptions using an LLM and integrates a chatbot for response generation. The system supports both file-based and real-time speech input.

## Workflow

1. **Dataset Preparation**
   - Used the **AccentDB** dataset, which contains Indian dialect audio.
   - Audio files were preprocessed to enhance quality.

2. **Feature Extraction**
   - **Spectrograms** were generated from audio files for visualization and analysis.

3. **Transcription Generation**
   - Speech-to-text transcriptions were created and stored in **JSON format**.

4. **Batch Processing**
   - A total of **1484 audio files** were divided into **batches of 4** for efficient training.

5. **Model Fine-Tuning**
   - The **Wav2Vec2** model was fine-tuned using **200 batches over 2 epochs**.
   - Fine-tuning was done using transfer learning, where the pre-trained Wav2Vec2 model was adapted to the Indian dialect dataset.
   - The model was trained using **CTC (Connectionist Temporal Classification) loss**, which helps align speech with text without requiring pre-segmented data.
   - **AdamW optimizer** was used with a scheduled learning rate to improve convergence.
   - Training logs, loss values, and accuracy metrics were recorded to track model performance.
   - After training, the model was evaluated on validation data to check for overfitting and fine-tuned further if needed.

6. **Transcription Refinement**
   - Used an **LLM to refine the transcriptions** during testing, ensuring better accuracy and readability.

7. **Testing**
   - Tested the model on a demo **wav file named "harvard"**.
   - Performed real-time speech-to-text testing using recorded audio.

8. **Chatbot Integration**
   - Implemented a **real-time chatbot** that uses STT output as input and generates responses using an LLM.

## Dependencies
For DataSet:
refer to: https://www.kaggle.com/datasets/imsparsh/accentdb-core-extended

for PreTrained Model:
GDRIVE: https://drive.google.com/drive/folders/1vTC1Vpsd3YAsCvNKu01mZzRZxjgshhMt?usp=sharing

Ensure you have the following dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install librosa numpy pandas
pip install soundfile pydub
pip install openai streamlit
```

## How to Run the Application

1. **Clone the Repository**
   ```bash
   git clone https://github.com/DivyViradiya07/STT-engine.git
   cd STT-engine
   ```

2. **Prepare the Environment**
   - Install the required dependencies using the command above.

3. **Run the Training and Testing Scripts**
   - To Finetune Wav2vec2 model for Speech to Text transformation 
      refer to file named: Model_training
    
   - Run the testing script to evaluate performance: Model_performance.ipynb
   - for testing / Demo run notebook: Demo.py

4. **Real-Time Speech-to-Text & Chatbot**
   - Start the Streamlit chatbot interface:
     ```bash
     streamlit run STT-chatbot.py
     ```

This will launch a web-based UI where you can interact with the real-time STT and chatbot system.

