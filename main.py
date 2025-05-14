import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

st.set_page_config(page_title="🦙 Local AWS PDF Q&A Bot")

st.title("🦙 Local LLaMA 3 - AWS Cloud Practitioner Q&A Bot")
st.write("Ask any question based on your uploaded AWS notes PDF!")

uploaded_file = st.file_uploader("Upload your AWS PDF", type="pdf")

if uploaded_file is not None:
    with open("aws_notes.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded!")

    # Load and split PDF
    loader = PyPDFLoader("aws_notes.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Use HuggingFace embeddings (no API needed!)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Use Ollama + LLaMA 3 locally
    llm = Ollama(model="llama3")

    # RAG Chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = st.text_input("💬 Ask a question:")
    if question:
        with st.spinner("Thinking with local LLaMA... 🦙"):
            answer = qa.run(question)
            st.write("### 🤖 Answer:")
            st.write(answer)
