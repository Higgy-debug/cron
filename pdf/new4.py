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
import PyPDF2

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
    extracted_images_folder = tempfile.mkdtemp()
    
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
            extract_and_save_images(temp_file_path, extracted_images_folder)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(text)
    
    return text_chunks, extracted_images_folder


def extract_and_save_images(pdf_file_path, output_folder):
    reader = PyPDF2.PdfReader(pdf_file_path)
    
    page_number = 0
    for page in range(len(reader.pages)):
        pdf_page = reader.pages[page]
        xObject = pdf_page['/Resources']['/XObject'].getObject()
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()
                mode = ''
                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    mode = 'RGB'
                else:
                    mode = 'P'

                with open(os.path.join(output_folder, f"page{page_number}_image{obj}.jpg"), "wb") as f:
                    f.write(data)
        page_number += 1

@st.cache(allow_output_mutation=True)
def create_chain(text_chunks, extracted_images_folder):
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
       text_chunks, extracted_images_folder = process_uploaded_files(uploaded_files)
       chain = create_chain(text_chunks, extracted_images_folder)
       display_chat_history(chain)

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

if __name__ == "__main__":
    main()
