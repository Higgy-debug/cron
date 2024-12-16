import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from openai import OpenAI
import os
import tempfile

# Set up the Whisper model
@st.cache_resource
def load_whisper_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"task": "transcribe", "language": "english"}
    )
    return pipe

# Function for chatbot interaction
def chat_with_transcription(transcription, user_input):
    client = OpenAI(api_key='')
    response = client.chat.completions.create(
        model="gpt-4o",  # or your preferred model
        messages=[
            {"role": "system", "content": "You are an AI assistant helping to answer questions about a transcribed audio. Use the provided transcription to answer the user's questions accurately."},
            {"role": "user", "content": f"Transcription: {transcription}\n\nUser question: {user_input}"}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

# Streamlit app
def main():
    st.title("Reckitt Audio Transcription and Summarization")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an MP3 file", type="mp3")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            input_path = tmp_file.name
        
        st.audio(input_path, format='audio/mp3')
        
        # Initialize session state for transcription
        if 'transcription' not in st.session_state:
            st.session_state.transcription = None
        
        # Transcribe button
        if st.button("Transcribe"):
            pipe = load_whisper_model()
            result = pipe(input_path, return_timestamps=True)
            st.session_state.transcription = result["text"]
            st.success("Transcription completed!")
        
        # Display transcription in a dropdown if available
        if st.session_state.transcription:
            with st.expander("Show Transcription"):
                st.markdown(st.session_state.transcription)
        
        # Summarize button
        if st.button("Summarize"):
            if st.session_state.transcription:
                client = OpenAI(api_key='')
                response = client.chat.completions.create(
                    model="gpt-4o",  # or your preferred model
                    messages=[
                        {"role": "system", "content": "You are generating a transcript summary. Create a detailed and structured summary of the provided transcription. Focus on key points including personal background, product preferences, challenges faced, cleaning routines, and brand perceptions. The summary should include an overview of cleaning products discussed, reasons for preferences, and key insights into customer habits and suggestions. Keep the summary clear and concise, maintaining the user's main ideas and experiences."},
                        {"role": "user", "content": f"The audio transcription is: {st.session_state.transcription}"}
                    ],
                    temperature=0.0,
                )
                summary = response.choices[0].message.content
                st.subheader("Summary")
                st.markdown(summary)
            else:
                st.warning("Please transcribe the audio first.")
        
        # Chat interface
        st.subheader("Chat with the Transcription")
        user_input = st.text_input("Ask a question about the transcription:")
        if st.button("Send"):
            if st.session_state.transcription and user_input:
                response = chat_with_transcription(st.session_state.transcription, user_input)
                st.markdown(f"**AI:** {response}")
            elif not st.session_state.transcription:
                st.warning("Please transcribe the audio first.")
            else:
                st.warning("Please enter a question.")
        
        # Clean up temporary file
        os.unlink(input_path)

if __name__ == "__main__":
    main()