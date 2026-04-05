# app.py - versi Gradio
import os
import gradio as gr
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Load RAG chain sekali saja
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name="coc-wiki", embedding=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# === PROMPT 1: Reformulate pertanyaan ===
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Diberikan riwayat percakapan dan pertanyaan terbaru dari pengguna, "
     "rumuskan ulang pertanyaan tersebut menjadi pertanyaan yang berdiri sendiri. "
     "Jangan jawab pertanyaannya, cukup rumuskan ulang jika perlu."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# === PROMPT 2: Answer dengan Guardrails ketat ===
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Anda adalah asisten ahli Clash of Clans. Anda HANYA boleh menjawab pertanyaan yang berkaitan dengan game Clash of Clans.

ATURAN KETAT yang harus dipatuhi:
1. Jawab HANYA berdasarkan konteks yang diberikan di bawah.
2. Jika konteks tidak mengandung informasi yang relevan dengan pertanyaan, jawab PERSIS dengan: "Maaf, saya tidak memiliki informasi mengenai hal tersebut dalam basis data saya."
3. Jika pertanyaan sama sekali tidak berkaitan dengan Clash of Clans (misal: politik, masakan, matematika, dll), jawab PERSIS dengan: "Saya hanya dapat menjawab pertanyaan seputar game Clash of Clans."
4. JANGAN mengarang, berasumsi, atau menggunakan pengetahuan di luar konteks yang diberikan.
5. JANGAN menyebut bahwa Anda memiliki "konteks" atau "dokumen" — cukup jawab secara natural.

Konteks:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# === BUILD CHAIN ===
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
document_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

print("✅ RAG Chain with Citation & Guardrails ready!")

# === FUNGSI CITATION ===
def format_sources(source_documents) -> str:
    """Ekstrak nama file unik dari metadata dokumen yang di-retrieve."""
    if not source_documents:
        return ""
    
    sources = set()
    for doc in source_documents:
        # Ambil metadata 'source' — biasanya berisi path file seperti 'Archer.txt'
        source = doc.metadata.get("source", "")
        if source:
            # Ambil nama file saja, buang path direktori
            filename = os.path.basename(source)
            sources.add(filename)
    
    if not sources:
        return ""
    
    source_list = ", ".join(sorted(sources))
    return f"\n\n📚 **Sumber:** {source_list}"

# === FUNGSI CHAT ===
def chat(message, history):
    chat_history = []
    for item in history:
        if isinstance(item, dict):
            if item["role"] == "user":
                chat_history.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                chat_history.append(AIMessage(content=item["content"]))
        else:
            human, ai = item
            chat_history.append(HumanMessage(content=human))
            chat_history.append(AIMessage(content=ai))

    response = rag_chain.invoke({
        "input": message,
        "chat_history": chat_history
    })

    answer = response["answer"]

    # ✅ Jangan tampilkan citation kalau AI menjawab "tidak tahu"
    NO_CITATION_PHRASES = [
        "saya tidak memiliki informasi",
        "saya hanya dapat menjawab",
    ]
    
    should_show_citation = not any(
        phrase in answer.lower() for phrase in NO_CITATION_PHRASES
    )

    citation = ""
    if should_show_citation:
        source_docs = response.get("context", [])
        citation = format_sources(source_docs)

    return answer + citation

# === GRADIO UI ===
demo = gr.ChatInterface(
    fn=chat,
    title="⚔️ Clash of Clans Expert Chatbot",
    description="Tanyakan apapun tentang Clash of Clans — troops, buildings, strategi, dan lainnya!"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)