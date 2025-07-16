import streamlit as st
from services.rag_pipeline import build_graph
import time

# Page config
st.set_page_config(page_title="AI Assistant", layout="centered")

# ---- STYLES ----
st.markdown("""
    <style>
        .big-font {
            font-size:22px !important;
            font-weight: 600;
        }
        .small-font {
            font-size:15px;
            color: #555;
        }
        .stTextArea textarea {
            font-size: 16px;
            padding: 16px;
            border-radius: 12px;
        }
        .stButton button {
            font-size: 16px;
            padding: 10px 30px;
            border-radius: 8px;
            background-color: #3b82f6;
            color: white;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1 style='text-align: center;'>Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='small-font' style='text-align: center;'>Ask smart questions powered by LLM + tools + vector memory</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- INPUT ----
query = st.text_area("💬 Ask your question:", height=160, placeholder="e.g. Define entropy, What is the price of T1000, Solve sin(1), etc.")
submit = st.button("🚀 Get Answer")

# ---- PROCESS ----
if submit and query:
    with st.spinner("🧠 Thinking..."):
        graph = build_graph()
        start_time = time.time()
        output = graph.invoke({"query": query})
        end_time = time.time()

    st.success("✅ Answer Ready")

    # ---- OUTPUT ----
    st.markdown("### 📘 Result")
    st.markdown(f"<div class='big-font'>{output.get('result', 'No answer available.')}</div>", unsafe_allow_html=True)

    # ---- EXPANDABLE LOGS ----
    with st.expander("🧾 Detailed Logs", expanded=False):
        st.markdown("**Query Processed:**")
        st.code(query, language="text")

        st.markdown("**Tool Used:**")
        st.code(output.get("tool", "N/A"), language="text")

        if output.get("tool_output"):
            st.markdown("**Tool Output:**")
            st.code(output["tool_output"], language="text")

        st.markdown("**Final Answer:**")
        st.code(output.get("result", ""), language="markdown")

        st.markdown("**⏱️ Time Taken:**")
        st.code(f"{end_time - start_time:.2f} seconds", language="text")

elif submit and not query:
    st.warning("⚠️ Please enter a question to get an answer.")




