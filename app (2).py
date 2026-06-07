import streamlit as st
from openai import OpenAI
from pathlib import Path
import tempfile
import time

# Initialize OpenAI client
client = OpenAI()

def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily and return the path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio file using OpenAI Whisper"""
    try:
        with open(audio_path, "rb") as audio_file:
            # Add a small delay to ensure file is properly saved
            time.sleep(1)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            if not transcript:
                raise ValueError("Transcription returned empty result")
            return transcript
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def summarize_transcript(transcript):
    """Summarize transcript using GPT-4"""
    try:
        # Corrected model name to gpt-4-turbo-preview
        # You can also use "gpt-3.5-turbo" if you prefer
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # or "gpt-3.5-turbo"
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize the following meeting transcript into 5 key bullet points:\n\n{transcript}"
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        summary = response.choices[0].message.content
        if not summary:
            raise ValueError("Summary generation returned empty result")
        return summary
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return None

# Streamlit UI
st.title("Meeting Summarizer")
st.write("Upload an audio recording to get a summary")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])

if uploaded_file:
    # Show processing message
    processing_message = st.empty()
    processing_message.info("Processing your audio file... Please wait.")
    
    # Save uploaded file
    temp_path = save_uploaded_file(uploaded_file)
    
    if temp_path:
        try:
            # Get transcript
            processing_message.info("Transcribing audio...")
            transcript = transcribe_audio(temp_path)
            
            if transcript:
                st.subheader("Transcript")
                st.text_area("Full Transcript", transcript, height=200)
                
                # Get summary
                processing_message.info("Generating summary...")
                summary = summarize_transcript(transcript)
                
                if summary:
                    st.subheader("Summary")
                    st.write(summary)
                    processing_message.success("Processing complete!")
                else:
                    st.error("Failed to generate summary. Please try again.")
            else:
                st.error("Failed to transcribe audio. Please check the file and try again.")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            
        finally:
            # Clean up temporary file
            try:
                Path(temp_path).unlink()
            except Exception as e:
                st.warning(f"Warning: Could not delete temporary file: {str(e)}")
    else:
        st.error("Failed to process uploaded file. Please try again.")

# Add API key check
if not client.api_key:
    st.error("OpenAI API key not found. Please set your OPENAI_API_KEY environment variable.")
