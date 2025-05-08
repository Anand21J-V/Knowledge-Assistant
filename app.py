import os
import streamlit as st
import time
import re
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.tools import Tool

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load LLM (Llama 3 from Groq)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama-3.3-70b-versatile")

# Streamlit UI
st.set_page_config(page_title="Knowledge Assistant", layout="wide")
st.title("Knowledge Assistant with Agentic Workflow")

# Embedding logic 
def load_vector_db():
    if "vectors" not in st.session_state:
        try:
            # Absolute path to the specific file
            knowledge_base_file = './data/data.txt'

            if not os.path.exists(knowledge_base_file):
                st.error(f"File {knowledge_base_file} does not exist.")
                return  

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            loader = TextLoader(knowledge_base_file)  
            docs = loader.load()

            if not docs:
                st.error("No documents were loaded from the file.")
                return 

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)
            vector_store = FAISS.from_documents(split_docs, embeddings)
            st.session_state.vectors = vector_store  # Initialising the vector store in session state

        except Exception as e:
            st.error(f"Error loading documents: {e}")

# Tools
def simple_calculator(query: str) -> str:
    try:
        expression = re.findall(r"[-+*/().\d\s]+", query)
        if expression:
            return f"The result is: {eval(expression[0])}"
        return "No valid expression found."
    except:
        return "Could not evaluate the expression."

def simple_define_tool(word: str) -> str:
    dictionary = {
        "rag": "Retrieval-Augmented Generation, enhancing LLMs with external documents.",
        "llm": "Large Language Model trained on big text datasets.",
        "embedding": "Numerical vector representation of text.",
        "agent": "A logic-based system that routes tasks to tools or chains."
    }
    return dictionary.get(word.lower(), f"No definition found for: {word}")

# Prompt for RAG
prompt = ChatPromptTemplate.from_template("""
Answer the question using the following context. Be concise and accurate.
<context>
{context}
</context>

Question: {input}
""")

# Input
query = st.text_input("Ask your question here...")

# Inference block
if query:
    load_vector_db()  
    decision_log = ""

    # Route logic
    if re.search(r"\b(calculate|add|subtract|multiply|divide|eval|[-+*/=])\b", query, re.IGNORECASE):
        decision_log = "🛠 Used Calculator Tool"
        answer = simple_calculator(query)

    elif re.search(r"\bdefine\s+(\w+)", query, re.IGNORECASE):
        decision_log = "📘 Used Dictionary Tool"
        word = re.findall(r"\bdefine\s+(\w+)", query, re.IGNORECASE)[0]
        answer = simple_define_tool(word)

    else:
        decision_log = "RAG and LLM - Knowledge assistant chatbot for Internshala"
        if "vectors" in st.session_state:  # Ensure vectors exist before using them
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            output = rag_chain.invoke({'input': query})
            answer = output['answer']
        else:
            answer = "Knowledge base not loaded properly."

    # Display response
    st.markdown(f"Decision: {decision_log}")
    st.markdown(f"Answer:\n{answer}")