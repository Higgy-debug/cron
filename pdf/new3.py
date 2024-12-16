import os
import streamlit as st
import re
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import VectorStore
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(layout="wide", page_icon="💬", page_title="Robby | Chat-Bot 🤖")

# Instantiate the main components
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

st.markdown(
    f"""
    <h1 style='text-align: center;'> Ask Robby to summarize youtube video ! 😁</h1>
    """,
    unsafe_allow_html=True,
)

user_api_key = utils.load_api_key()

sidebar.about()

if not user_api_key:
    layout.show_api_key_missing()

else:
    # Load your LLM model
    def create_conversational_chain():
        llm = LlamaCpp(
            streaming=True,
            n_gpu_layers=-1,
            model_path='llm/mistral-7b-instruct-v0.2.Q8_0.gguf',
            temperature=0.1,
            top_p=1,
            verbose=True,
            n_ctx=2048
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff', memory=memory)
        return chain

    script_docs = []

    def get_youtube_id(url):
        video_id = None
        match = re.search(r"(?<=v=)[^&#]+", url)
        if match :
            video_id = match.group()
        else : 
            match = re.search(r"(?<=youtu.be/)[^&#]+", url)
            if match :
                video_id = match.group()
        return video_id

    video_url = st.text_input(placeholder="Enter Youtube Video URL", label_visibility="hidden", label =" ")
    if video_url :
        video_id = get_youtube_id(video_url)

        if video_id != "":
            t = YouTubeTranscriptApi.get_transcript(video_id, languages=('en','fr','es', 'zh-cn', 'hi', 'ar', 'bn', 'ru', 'pt', 'sw' ))
            finalString = ""
            for item in t:
                text = item['text']
                finalString += text + " "

            text_splitter = CharacterTextSplitter()
            chunks = text_splitter.split_text(finalString)

            # Create conversational chain
            conversational_chain = create_conversational_chain()

            # Get responses
            for chunk in chunks:
                response = conversational_chain.run(chunk)
                st.write(response)
