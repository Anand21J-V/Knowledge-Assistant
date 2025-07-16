import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

VECTOR_DIR = "vectorstore/faiss_index"

class VectorStore:
    def __init__(self, file_path):
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load from disk if exists
        if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
            self.db = FAISS.load_local(VECTOR_DIR, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            self.db = self._create_and_save_index(file_path)

    def _create_and_save_index(self, file_path):
        with open(file_path, "r") as f:
            text = f.read()

        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=chunk) for chunk in chunks]

        db = FAISS.from_documents(docs, self.embedding_model)
        db.save_local(VECTOR_DIR)
        return db

    def retrieve(self, query, k=3):
        results = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
