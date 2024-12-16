import pandas as pd
import pdfplumber
import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from modules.embedder import Embedder
from modules.chatbot import Chatbot
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp

llm = LlamaCpp(
    streaming=True,
    n_gpu_layers=-1,
    model_path='llm/mistral-7b-instruct-v0.2.Q6_K.gguf',
    temperature=0.1,
    top_p=1,
    verbose=True,
    n_ctx=2048
    )

class Utilities:

    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder()

        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            # Get the document embeddings for the uploaded file
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)

            # Create a Chatbot instance with your "gguf" model and temperature
            chatbot = Chatbot(model, temperature, vectors)

            # Initialize st.session_state["chatbot"]
            st.session_state["chatbot"] = chatbot

        st.session_state["ready"] = True

        return chatbot

class Chatbot:
    def __init__(self, model, temperature, vectors):
        self.model = model
        self.temperature = temperature
        self.vectors = vectors

    qa_template = """
        You are a helpful AI assistant named Hirthick. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.

        context: {context}
        =========
        question: {question}
        ======
        """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        retriever = self.vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
            max_tokens_limit=4097,
            combine_docs_chain_kwargs={'prompt': self.QA_PROMPT}
        )

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        # count_tokens_chain(chain, chain_input)
        return result["answer"]