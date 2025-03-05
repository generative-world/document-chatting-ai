from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from app.config import VECTORSTORE_PATH

def create_retriever():
    vectorstore = Chroma(persist_directory=VECTORSTORE_PATH)
    return vectorstore.as_retriever()
