import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def load_vector_db():
    if "vectors" not in st.session_state:
        try:
            knowledge_base_file = './data/data.txt'
            if not os.path.exists(knowledge_base_file):
                st.error(f"File {knowledge_base_file} does not exist.")
                return

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            docs = TextLoader(knowledge_base_file).load()

            if not docs:
                st.error("No documents were loaded.")
                return

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)
            vector_store = FAISS.from_documents(split_docs, embeddings)
            st.session_state.vectors = vector_store

        except Exception as e:
            st.error(f"Error loading documents: {e}")
