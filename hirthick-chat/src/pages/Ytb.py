import os
import streamlit as st
import re
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import AnalyzeDocumentChain, LLMChain
from langchain.llms import LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI

st.set_page_config(layout="wide", page_icon="💬", page_title="Hirthick | Chat-Bot 🤖")

# Instantiate the main components
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

st.markdown(
    """
    <h1 style='text-align: center;'>Ask Hirthick to summarize YouTube video! 😁</h1>
    """,
    unsafe_allow_html=True,
)

# Load the LLM model
llm = LlamaCpp(
    streaming=True,
    n_gpu_layers=-1,
    model_path=,
    temperature=0.1,
    top_p=1,
    verbose=True,
    n_ctx=2048
    )

def get_youtube_id(url):
    """
    Extract the video ID from a YouTube URL.
    """
    video_id = None
    match = re.search(r"(?<=v=)[^&#]+", url)
    if match:
        video_id = match.group()
    else:
        match = re.search(r"(?<=youtu.be/)[^&#]+", url)
        if match:
            video_id = match.group()
    return video_id

# Get the YouTube video URL from the user
video_url = st.text_input(placeholder="Enter YouTube Video URL", label_visibility="hidden", label=" ")

# Process the video URL and transcript
if video_url:
    video_id = get_youtube_id(video_url)
    if video_id != " ":
    
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=('en', 'fr', 'es', 'zh-cn', 'hi', 'ar', 'bn', 'ru', 'pt', 'sw'))
        final_string = " "
        for item in transcript:
            text = item['text']
            final_string += text + " "
            

            # Split the transcript into chunks
        text_splitter = CharacterTextSplitter()
        chunks = text_splitter.split_text(final_string)

            # Analyze and summarize the transcript
        
        summary_chain = load_summarize_chain(llm=llm,chain_type="stuff", verbose=True)
        summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
        answer = summarize_document_chain.run(chunks)
        st.subheader(answer)