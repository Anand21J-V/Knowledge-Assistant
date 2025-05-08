# knowledge_assistant/app.py

import streamlit as st
from utils.vector_store import load_vector_db
from utils.tools import simple_calculator, simple_define_tool
from utils.prompts import get_rag_prompt
from services.rag_pipeline import get_rag_chain
import re
import time

st.set_page_config(page_title="Knowledge Assistant", layout="wide")
st.title("Knowledge Assistant with Agentic Workflow")

query = st.text_input("Ask your question here...")

if query:
    load_vector_db()
    decision_log = ""

    if re.search(r"\\b(calculate|add|subtract|multiply|divide|eval|[-+*/=])\\b", query, re.IGNORECASE):
        decision_log = "\U0001F6E0 Used Calculator Tool"
        answer = simple_calculator(query)

    elif re.search(r"\\bdefine\\s+(\\w+)", query, re.IGNORECASE):
        decision_log = "Used Dictionary Tool"
        word = re.findall(r"\\bdefine\\s+(\\w+)", query, re.IGNORECASE)[0]
        answer = simple_define_tool(word)

    else:
        decision_log = "RAG and LLM - Knowledge Assistant Chatbot for decision"
        answer = get_rag_chain(query, get_rag_prompt())

    st.markdown(f"Decision: {decision_log}")
    st.markdown(f"Answer:\n{answer}")
