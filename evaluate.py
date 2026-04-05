# evaluate.py
# Jalankan lokal: python evaluate.py
# JANGAN dijalankan di HF Spaces (tidak ada akses write ke disk)

import os
import csv
import time
from datetime import datetime
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# =====================================================================
# SETUP CHAIN (sama seperti app.py, tanpa history untuk evaluasi)
# =====================================================================
print("⏳ Initializing RAG chain for evaluation...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name="coc-wiki", embedding=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rumuskan ulang pertanyaan menjadi pertanyaan yang berdiri sendiri."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Anda adalah asisten ahli Clash of Clans.

ATURAN KETAT:
1. Jawab HANYA berdasarkan konteks yang diberikan.
2. Jika konteks tidak relevan, jawab: "Maaf, saya tidak memiliki informasi mengenai hal tersebut dalam basis data saya."
3. Jika pertanyaan tidak berkaitan dengan Clash of Clans, jawab: "Saya hanya dapat menjawab pertanyaan seputar game Clash of Clans."
4. JANGAN mengarang jawaban.

Konteks:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
document_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

print("✅ Chain ready. Starting evaluation...\n")

# =====================================================================
# FUNGSI EVALUASI SEDERHANA
# Teknik: Keyword matching — cek apakah kata kunci dari ideal answer
# ada di dalam jawaban AI. Cocok untuk portofolio tanpa butuh LLM judge.
# =====================================================================
def simple_score(ai_answer: str, ideal_answer: str) -> tuple[float, str]:
    """
    Score 0.0 - 1.0 berdasarkan keyword overlap.
    Return: (score, verdict)
    """
    ai_lower = ai_answer.lower()
    
    # Ekstrak kata penting dari ideal answer (lebih dari 4 huruf)
    keywords = [w.lower() for w in ideal_answer.split() if len(w) > 4]
    
    if not keywords:
        return 0.0, "NO_KEYWORDS"
    
    matched = sum(1 for kw in keywords if kw in ai_lower)
    score = matched / len(keywords)
    
    if score >= 0.5:
        verdict = "✅ PASS"
    elif score >= 0.3:
        verdict = "⚠️ PARTIAL"
    else:
        verdict = "❌ FAIL"
    
    return round(score, 2), verdict

# =====================================================================
# JALANKAN EVALUASI
# =====================================================================
input_file  = "eval_dataset.csv"
output_file = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

results = []
total_score = 0.0

with open(input_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"📋 Evaluating {len(rows)} questions...\n")
print("-" * 60)

for i, row in enumerate(rows, 1):
    question    = row["question"]
    ideal       = row["ideal_answer"]
    
    print(f"[{i}/{len(rows)}] Q: {question}")
    
    # Invoke RAG chain (tanpa history untuk evaluasi bersih)
    response    = rag_chain.invoke({"input": question, "chat_history": []})
    ai_answer   = response["answer"]
    
    # Ambil nama sumber dokumen
    source_docs = response.get("context", [])
    sources     = ", ".join(set(
        os.path.basename(doc.metadata.get("source", "unknown"))
        for doc in source_docs
    )) or "none"
    
    # Hitung score
    score, verdict = simple_score(ai_answer, ideal)
    total_score += score
    
    print(f"     Verdict : {verdict} (score: {score})")
    print(f"     Sources : {sources}")
    print()
    
    results.append({
        "question"       : question,
        "ideal_answer"   : ideal,
        "ai_answer"      : ai_answer,
        "sources"        : sources,
        "score"          : score,
        "verdict"        : verdict,
        "timestamp"      : datetime.now().isoformat(),
    })
    
    # Delay agar tidak kena rate limit Gemini
    time.sleep(1.5)

# =====================================================================
# SIMPAN LOG & RINGKASAN
# =====================================================================
fieldnames = ["question", "ideal_answer", "ai_answer", "sources", "score", "verdict", "timestamp"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

avg_score   = total_score / len(results)
pass_count  = sum(1 for r in results if "PASS"    in r["verdict"])
partial     = sum(1 for r in results if "PARTIAL" in r["verdict"])
fail_count  = sum(1 for r in results if "FAIL"    in r["verdict"])

print("=" * 60)
print("📊 EVALUATION SUMMARY")
print("=" * 60)
print(f"Total questions : {len(results)}")
print(f"✅ PASS         : {pass_count}")
print(f"⚠️  PARTIAL      : {partial}")
print(f"❌ FAIL         : {fail_count}")
print(f"📈 Avg Score    : {avg_score:.2f} / 1.00")
print(f"\n💾 Log saved to : {output_file}")
print("=" * 60)