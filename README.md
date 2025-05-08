# Knowledge Assistant with Agentic Workflow - Internshala Assignment

This project is a smart knowledge assistant built using **Streamlit** that leverages **Retrieval-Augmented Generation (RAG)** and tool-based routing to answer user queries. It uses **Llama 3.3-70B** from **Groq** as the backend language model and supports dynamic query handling—ranging from simple calculations and definitions to context-aware retrieval from a knowledge base.

---

## What This App Does

This assistant provides three main functionalities:

1. **Math Solver:** Detects mathematical expressions in the user's query and evaluates them.
2. **Dictionary Tool:** Defines certain technical terms like "RAG", "LLM", etc.
3. **Knowledge-Based QA using RAG:** When the input doesn't match the above patterns, it uses vector-based document retrieval and LLM response generation to answer the query based on the given data.

---

## Project Structure

```
.
├── app.py                      # Main Streamlit app
├── .env                        # Contains GROQ_API_KEY
├── data/
│   └── data.txt                # Source file for knowledge base
├── requirements.txt            # Python dependencies
├── utils/
│   ├── __init__.py             # Initializes the utils module
│   ├── vector_store.py         # Manages the vector store
│   ├── tools.py                # Contains helper tools like calculator and dictionary
│   └── prompts.py              # Contains prompt templates
└── services/
    ├── __init__.py             # Initializes the services module
    └── rag_pipeline.py         # Handles the RAG pipeline logic

```

---

## Technologies Used

* **Streamlit** – Frontend interface
* **LangChain** – Toolchains, prompt templates, and document QA
* **FAISS** – Vector similarity search
* **HuggingFace Embeddings** – Text embedding generator
* **Groq + Llama 3.3 70B** – Backend LLM
* **Python `re` module** – Regex-based tool routing
* **dotenv** – Secure environment variable management

---

## How It Works

### 1. **Query Input**

User types a query in the input box. Based on regex-based classification, the system routes it to the right tool.

### 2. **Tool Routing Logic**

* If the query includes math operators or keywords like `calculate`, it's sent to the **calculator**.
* If it contains `define [word]`, it's sent to a **local dictionary tool**.
* Otherwise, it falls back to the **RAG-based document QA** system.

### 3. **Knowledge Retrieval**

* `data/data.txt` is used to load content via `TextLoader`.
* It is split into chunks using `RecursiveCharacterTextSplitter`.
* The chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in FAISS vector store.
* At inference time, the top 3 most relevant chunks are retrieved and passed to the LLM via a structured prompt.

### 4. **LLM Inference**

The structured prompt is passed to Llama 3.3 via the Groq API, and the response is displayed to the user.

---

## Sample Queries

* `calculate 42 * 19`
* `define LLM`
* `What is the main purpose of RAG in LLM pipelines?`

---

## Setup Instructions

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/knowledge-assistant.git
cd knowledge-assistant
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Add Environment Variables**

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. **Run the Application**

```bash
streamlit run app.py
```

---

## How to Add New Knowledge

To update the assistant’s knowledge base:

1. Add new content to `data/data.txt`.
2. The embedding and FAISS indexing are handled automatically when a query is made.

---

## Author
Anand Kumar Vishwakarma


