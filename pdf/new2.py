import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
import torch
from collections import defaultdict

DEVICE = "cuda:0"
print(DEVICE)
torch.set_num_threads(1)

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about the PDF"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey User!"]

    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = []

def create_conversational_chain(vector_store):
    llm = LlamaCpp(
        streaming=True,
        n_gpu_layers=-1,
        model_path='/home/hirthick/poc/llm/mistral-7b-instruct-v0.2.Q8_0.gguf',
        temperature=0.1,
        top_p=1,
        verbose=True,
        n_ctx=2048
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

@st.cache(allow_output_mutation=True)
def process_uploaded_files(uploaded_files):
    print("Processing files")
    text = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)

        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(text)
    
    return text_chunks

@st.cache(allow_output_mutation=True)
def create_chain(text_chunks):
    print("Creating chains")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs={'device': DEVICE})

    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    chain = create_conversational_chain(vector_store)

    return chain

def calculate_source_page_accuracy(test_queries, chatbot_responses):
    """
    Calculate the Source Page Accuracy for the chatbot's responses.
    
    Args:
        test_queries (list): A list of tuples (query, correct_source_pages), where correct_source_pages is a list of page numbers.
        chatbot_responses (list): A list of tuples (query, answer, retrieved_pages), where retrieved_pages is a list of page numbers.
    
    Returns:
        dict: A dictionary containing the Source Page Accuracy metrics.
    """
    
    total_queries = len(test_queries)
    exact_match_count = 0
    partial_match_count = 0
    no_match_count = 0
    
    correct_pages_retrieved = defaultdict(int)
    incorrect_pages_retrieved = defaultdict(int)
    
    for (query, correct_pages), (_, _, retrieved_pages) in zip(test_queries, chatbot_responses):
        correct_pages = set(correct_pages)
        retrieved_pages = set(retrieved_pages)
        
        if retrieved_pages == correct_pages:
            exact_match_count += 1
        elif retrieved_pages.intersection(correct_pages):
            partial_match_count += 1
        else:
            no_match_count += 1
        
        for page in correct_pages.intersection(retrieved_pages):
            correct_pages_retrieved[page] += 1
        
        for page in retrieved_pages.difference(correct_pages):
            incorrect_pages_retrieved[page] += 1
    
    source_page_accuracy = {
        'total_queries': total_queries,
        'exact_match': exact_match_count,
        'partial_match': partial_match_count,
        'no_match': no_match_count,
        'source_page_accuracy': (exact_match_count + partial_match_count) / total_queries,
        'avg_correct_pages_retrieved': sum(correct_pages_retrieved.values()) / total_queries,
        'avg_incorrect_pages_retrieved': sum(incorrect_pages_retrieved.values()) / total_queries
    }
    
    return source_page_accuracy

def main():
    initialize_session_state()
    st.title("Multi-PDF Chatbot")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    test_queries = [
        ("What is the capital of France?", [10, 11]),
        ("When was the Eiffel Tower built?", [15, 16, 17]),
        ("Who was the first President of the United States?", [])
    ]
    chatbot_responses = []

    if uploaded_files:
        text_chunks = process_uploaded_files(uploaded_files)
        chain = create_chain(text_chunks)
        display_chat_history(chain)
        
        # Calculate and display source page accuracy
        source_page_accuracy = calculate_source_page_accuracy(test_queries, chatbot_responses)
        st.write("Source Page Accuracy Metrics:")
        st.write(source_page_accuracy)

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    retrieved_docs = None  # Initialize retrieved_docs
    chatbot_responses = []  # Initialize chatbot_responses list

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output, retrieved_pages = conversation_chat(user_input, chain, st.session_state['history'])
                chatbot_responses.append((user_input, output, retrieved_pages))  # Append response to list

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['feedback'].append(None)  # Placeholder for feedback

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

                # Display retrieved context for verification
                if retrieved_docs:
                    st.write("Retrieved Context:")
                    for doc in retrieved_docs:
                        st.write(doc[:500])  # Display first 500 characters for brevity

                # Feedback mechanism