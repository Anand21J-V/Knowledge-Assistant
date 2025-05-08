from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import create_retrieval_chain

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

def get_rag_chain(query: str, prompt):
    if "vectors" not in st.session_state:
        return "Knowledge base not loaded properly."

    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain.invoke({'input': query})['answer']
