import os
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_community.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import TextLoader

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEEDING_MODEL_NAME = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEEDING_MODEL_NAME,
        model_kwargs = {"device": "cpu"}
        )

def split_transcript(transcript: str, chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(transcript)
 
 
def reset_vector_store():
    """Drop the old collection so rebuilding doesn't stack duplicate chunks."""
    try:
        load_vector_store().delete_collection()
    except Exception:
        pass

def build_vector_store(transcript: str)-> Chroma:
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i}) 
        for i, chunk in enumerate(chunks)  

    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
        persist_directory=CHROMA_DIR,
    )
    return vector_store

def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type= 'similarity'
        search_kwargs = {'k': k}
    )

def chunk_count() -> int:
    if not os.path.isdir(CHROMA_DIR):
        return 0
    try:
        return load_vector_store()._collection.count()
    except Exception:
        return 0

    