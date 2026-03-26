# app.py

import os
from dotenv import load_dotenv

# Import library untuk UI
import chainlit as cl

# Import library untuk logika RAG
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# =====================================================================
# SETUP AWAL
# Muat semua kunci rahasia dari file .env
# =====================================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Cek apakah semua kunci ada
if not all([PINECONE_API_KEY, GOOGLE_API_KEY]):
    raise ValueError("PINECONE_API_KEY or GOOGLE_API_KEY are missing from .env file.")

# Nama index di Pinecone dan nama model embedding
pinecone_index_name = "coc-wiki"
embedding_model_name = "all-MiniLM-L6-v2"

# =====================================================================
# FUNGSI-FUNGSI UTAMA UNTUK CHAINLIT
# =====================================================================

# Dekorator @cl.on_chat_start akan menjalankan fungsi ini setiap kali 
# pengguna baru memulai sesi chat.
@cl.on_chat_start
async def start_rag_chain():
    """
    Inisialisasi dan setup semua komponen RAG Chain saat chat dimulai.
    """
    # Kirim pesan ke UI untuk memberitahu user bahwa sistem sedang disiapkan
    await cl.Message(content="Memulai sesi... Menyiapkan basis data pengetahuan.").send()

    # 1. Menyiapkan model embedding (Hugging Face)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # 2. Menghubungkan ke database vektor Pinecone yang sudah ada
    vector_store = PineconeVectorStore(
        index_name=pinecone_index_name, 
        embedding=embeddings
    )
    
    # 3. Menyiapkan "Otak" AI (LLM) dari Google
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)

    # 4. Membuat Prompt Template yang cerdas
    prompt = ChatPromptTemplate.from_template(
        """
        Anda adalah seorang ahli strategi game Clash of Clans yang sangat membantu.
        Jawab pertanyaan berikut dengan sejelas dan seramah mungkin, HANYA berdasarkan konteks yang diberikan.
        Jika informasi tidak ada di dalam konteks, katakan dengan sopan, "Maaf, saya tidak memiliki informasi spesifik mengenai hal itu dalam basis data saya saat ini."
        Jangan mengarang jawaban.

        Konteks:
        {context}

        Pertanyaan: {input}
        
        Jawaban Ahli:
        """
    )

    # 5. Membuat RAG Chain yang utuh (menggunakan arsitektur modern)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 6. Menyimpan chain yang sudah jadi ke dalam "user session"
    # Ini agar kita bisa menggunakannya lagi saat user mengirim pesan
    cl.user_session.set("retrieval_chain", retrieval_chain)

    # Kirim pesan bahwa sistem sudah siap
    await cl.Message(content="Basis data siap! Silakan ajukan pertanyaan tentang Clash of Clans.").send()


# Dekorator @cl.on_message akan menjalankan fungsi ini setiap kali
# pengguna mengirimkan sebuah pesan.
@cl.on_message
async def main(message: cl.Message):
    """
    Fungsi utama yang dipanggil untuk setiap pesan dari pengguna.
    """
    # 1. Ambil RAG chain yang sudah kita simpan dari sesi
    retrieval_chain = cl.user_session.get("retrieval_chain")
    
    # 2. Buat "callback" untuk streaming jawaban kata per kata
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True
    )
    cb.answer_reached = True

    # 3. Jalankan chain dengan pertanyaan dari user
    #    Chainlit secara otomatis akan menampilkan loading indicator
    response = await retrieval_chain.ainvoke({"input": message.content}, callbacks=[cb])
    
    # 4. Ambil jawaban akhir dari response
    answer = response["answer"]

    # (Opsional) Jika Anda ingin menampilkan dokumen sumber yang digunakan
    # sources = response["context"]
    # if sources:
    #     source_elements = [cl.Text(content=doc.page_content, name=os.path.basename(doc.metadata.get("source", ""))) for doc in sources]
    #     await cl.Message(content="Dokumen sumber yang digunakan:", elements=source_elements).send()

    # Kirim jawaban akhir ke UI
    if not cb.has_streamed_final_answer:
        await cl.Message(content=answer).send()