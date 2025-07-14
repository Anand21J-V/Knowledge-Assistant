
import streamlit as st
from utils.vector_store import load_vector_db
from utils.tools import simple_calculator, simple_define_tool
from utils.prompts import get_rag_prompt
from services.rag_pipeline import get_rag_chain
import re

st.set_page_config(page_title="🧠 Knowledge Assistant", layout="wide")

# ===== Title Section =====
st.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        <h1 style='color: #4A90E2;'>🧠 Knowledge Assistant</h1>
        <h4 style='color: gray;'>An Agentic AI Assistant for Answers, Definitions, and Calculations</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ===== Input Area =====
with st.container():
    query = st.text_input("💬 Ask me anything...", placeholder="Try 'define quantum', or 'calculate 2+2*5'")

# ===== Query Handling =====
if query:
    load_vector_db()
    decision_log = ""
    
    # Decision logic
    if re.search(r"\b(calculate|add|subtract|multiply|divide|eval|[-+*/=])\b", query, re.IGNORECASE):
        decision_log = "🧮 **Used Calculator Tool**"
        answer = simple_calculator(query)

    elif re.search(r"\bdefine\s+(\w+)", query, re.IGNORECASE):
        decision_log = "📘 **Used Dictionary Tool**"
        word = re.findall(r"\bdefine\s+(\w+)", query, re.IGNORECASE)[0]
        answer = simple_define_tool(word)

    else:
        decision_log = "🤖 **RAG and LLM - Knowledge Assistant Chatbot**"
        answer = get_rag_chain(query, get_rag_prompt())

    # ===== Output Section =====
    with st.container():
        st.markdown("---")
        st.markdown(f"**🔍 Decision:** {decision_log}")
        st.markdown(f"**🧾 Answer:**\n\n{answer}")
