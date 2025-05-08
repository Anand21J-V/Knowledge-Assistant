from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt():
    return ChatPromptTemplate.from_template("""
Answer the question using the following context. Be concise and accurate.
<context>
{context}
</context>

Question: {input}"""
)
