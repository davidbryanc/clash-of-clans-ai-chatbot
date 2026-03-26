# build_pinecone_db.py (VERSI 2.0 - dengan Hugging Face Embeddings)

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- PERUBAHAN IMPORT ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
# -------------------------

print("Loading environment variables...")
load_dotenv()

# Kita tidak lagi butuh Google API Key untuk script ini
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT are missing.")

def main():
    data_directory = "parsed_data"
    pinecone_index_name = "coc-wiki"

    # ... (Bagian loader dan text_splitter sama seperti sebelumnya) ...
    loader = DirectoryLoader(
        data_directory, 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    print(f"Total chunks to be uploaded: {len(docs)}")

    # --- PERUBAHAN UTAMA DI SINI ---
    print("Initializing HuggingFace embedding model (all-MiniLM-L6-v2)...")
    # Library ini akan otomatis mengunduh model saat pertama kali dijalankan
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # --------------------------------

    print(f"Uploading documents to Pinecone index '{pinecone_index_name}'...")
    PineconeVectorStore.from_documents(
        docs, 
        embeddings, 
        index_name=pinecone_index_name
    )

    print("\n✅ Success! Your knowledge base has been uploaded to Pinecone using HuggingFace embeddings.")

if __name__ == "__main__":
    main()