from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from app.config import EMBEDDING_MODEL_NAME, VECTORSTORE_PATH

def chunk_and_embed_documents(documents):
    # Load embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Initialize text splitter
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # Split documents into chunks
    chunks = []
    for document in documents:
        chunks.extend(text_splitter.split_text(document))

    # Create embeddings for the chunks
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]
    
    # Store chunks and embeddings into Chroma DB
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=VECTORSTORE_PATH)
    
    return vectorstore
