import gradio as gr
import os
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Function to generate prompt
def generate_prompt(prompt: str, system_prompt: str) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

def process_pdf_and_chat(pdf_file, question):

    loader = PyPDFDirectoryLoader(r'Vitamin_and_Mineral_Requirements_in_Huma (1).pdf')
    docs = loader.load()

    if not docs:
        return "No documents loaded."

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
    )

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)

    if not texts:
        return "No texts split from documents."

    # Create vector store
    db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="pdfss")

    
    SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

    template = generate_prompt(
        """
    {context}

    Question: {question}
    """,
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Set up QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa_chain(question)

    return result 


model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    revision="main",
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    inject_fused_attention=False,
    device=DEVICE,
    quantize_config=None,
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

# Gradio interface
inputs = [gr.components.File(label="Upload PDF"),gr.components.Textbox(label="Ask a Question")]
outputs = gr.components.Textbox(label="Chat Result")

gr.Interface(
    fn=process_pdf_and_chat,
    inputs=inputs,
    outputs=outputs,
    title="PDF Chatbot",
    description="Upload a PDF file to chat with the model.",
    theme="compact",
).launch()