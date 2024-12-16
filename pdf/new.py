import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
import torch
import fitz  # PyMuPDF for handling PDFs and extracting images

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

def create_conversational_chain(vector_store):
    llm = LlamaCpp(
        streaming=True,
        n_gpu_layers = -1,
        model_path='llm/mistral-7b-instruct-v0.2.Q8_0.gguf',
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
    images_metadata = []  # Store image metadata
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
            # Extract image metadata
            images_metadata.extend(extract_images_metadata(temp_file_path))

        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(text)
    
    return text_chunks, images_metadata

def extract_images_metadata(pdf_path):
    images_metadata = []
    doc = fitz.open(pdf_path)
    for page_number, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            images_metadata.append({
                "page_number": page_number,
                "image_index": img_index,
                "image_info": img_info[0]  # Extracting the first image info from the list
            })
    return images_metadata



@st.cache(allow_output_mutation=True)
def create_chain(text_chunks):
    print("Creating chains")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs={'device': DEVICE})

    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    chain = create_conversational_chain(vector_store)

    return chain

def main():
    initialize_session_state()
    st.title("Multi-PDF Chatbot")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
       text_chunks, images_metadata = process_uploaded_files(uploaded_files)
       chain = create_chain(text_chunks)
       display_chat_history(chain, images_metadata)

def conversation_chat(query, chain, history, images_metadata):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    
    image_urls = []
    if 'metadata' in result:
        relevant_pages = result['metadata'].get('relevant_pages', [])
        for metadata in images_metadata:
            if metadata['page_number'] in relevant_pages:
                image_urls.append(metadata['image_info'])

    return result["answer"], image_urls


def display_chat_history(chain, images_metadata):
    reply_container = st.container()
    container = st.container()
    image_urls = []  # Initialize image_urls

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output, image_urls = conversation_chat(user_input, chain, st.session_state['history'], images_metadata)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                # Display images if available in the response
                if image_urls and i < len(image_urls):
                    st.image(image_urls[i]['image_data'])  # Assuming 'image_data' contains image data

if __name__ == "__main__":
    main()



