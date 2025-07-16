import os
from dotenv import load_dotenv
from typing import TypedDict
from utils.prompts import prompt_template
from utils.vector_store import VectorStore

from groq import Groq
from langgraph.graph import StateGraph, END

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
vector_store = VectorStore("data/data.txt")

# Agent Format
class AgentState(TypedDict):
    query: str
    tool: str
    tool_output: str
    result: str


# LLM for decision making to which tool to use

def decide_tool(state):
    query = state["query"]

    docs = vector_store.retrieve(query)

    relevant = any(
        word.lower() in " ".join(docs).lower()
        for word in query.split() if len(word) > 2
    )

    if docs and relevant:
        return {"tool": "rag"}

    tool_selector_prompt = f"""
You're an intelligent agent. The user asked: \"{query}\".

Choose the best tool to answer it. Options:
- calculator (math expressions like sin, cos, 2+2)
- dictionary (for definitions like 'define entropy', 'food')
- logic_solver (for reasoning problems or functional equations)

Just respond with the correct tool name.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": tool_selector_prompt}]
    )

    tool = response.choices[0].message.content.strip().lower()
    if tool not in ["calculator", "dictionary", "logic_solver"]:
        tool = "dictionary"  

    return {"tool": tool}



# Run tool using LLM-powered logic

def run_tool(state):
    query = state["query"]
    tool = state["tool"]

    if tool == "calculator":
        prompt = f"""
You are a scientific calculator. Evaluate the following math expression and return only the result:

Expression: {query}
"""
    elif tool == "dictionary":
        prompt = f"""
You are a dictionary. Provide a clear, concise definition for the following word or phrase:

Define: {query}
"""
    elif tool == "logic_solver":
        prompt = f"""
You are a math reasoning expert. Analyze and solve the functional equation or logical query given below step by step. Use rigorous reasoning and clearly explain each step:

Problem: {query}
"""
    else:
        return {"tool_output": ""}

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"tool_output": response.choices[0].message.content.strip()}


# Format tool result as final output

def format_tool_result(state):
    query = state["query"]
    tool = state["tool"]
    tool_output = state["tool_output"]

    prompt = f"""
You are a helpful AI assistant.

The user asked: "{query}"
You used the tool: {tool}
The tool returned: {tool_output}

Now generate a clear and helpful response for the user using that result.
Avoid redundant explanations. Just be informative and natural.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"result": response.choices[0].message.content.strip()}


# RAG response Format

def run_rag(state):
    query = state["query"]
    docs = vector_store.retrieve(query)
    context = "\n".join(docs)

    prompt = prompt_template.format(context=context, query=query)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"result": response.choices[0].message.content.strip()}


# Build the main LangGraph by connecting Nodes and tools, start and End Points.

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("decide_tool", decide_tool)
    builder.add_node("run_tool", run_tool)
    builder.add_node("format_tool_result", format_tool_result)
    builder.add_node("run_rag", run_rag)

    builder.set_entry_point("decide_tool")

    builder.add_conditional_edges("decide_tool", lambda s: s["tool"], {
        "calculator": "run_tool",
        "dictionary": "run_tool",
        "logic_solver": "run_tool",
        "rag": "run_rag"
    })

    builder.add_edge("run_tool", "format_tool_result")
    builder.add_edge("format_tool_result", END)
    builder.add_edge("run_rag", END)

    return builder.compile()
