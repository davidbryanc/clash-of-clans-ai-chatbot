# ⚔️ Clash of Clans Expert Chatbot

> Chatbot berbasis RAG (Retrieval-Augmented Generation) yang menjawab pertanyaan seputar game Clash of Clans secara akurat, dengan memori percakapan, citation sumber, dan pipeline evaluasi otomatis.

🔗 **Live Demo:** [davidbc17-coc-chatbot-expert.hf.space](https://davidbc17-coc-chatbot-expert.hf.space)

---

## 🧠 Arsitektur Sistem

```
User Query
    │
    ▼
[Gradio UI] ──────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
[Chat History] ──► [History-Aware Retriever] ◄── [Pinecone Vector DB]
                          │                       (coc-wiki index)
                          ▼
                   [Reformulated Query]
                          │
                          ▼
                   [Gemini 1.5 Flash] ◄── [Retrieved Context + Sources]
                          │
                          ▼
                   [Answer + Citation]
                          │
                          ▼
                      [Gradio UI]
```

---

## 🚀 Fitur Utama

### 1. RAG Pipeline (Retrieval-Augmented Generation)
Data dari Fandom Wiki CoC (format XML) diproses, di-chunk, di-embed, dan disimpan di Pinecone. Saat ada pertanyaan, sistem mencari chunk paling relevan sebelum menjawab — sehingga jawaban selalu berdasarkan fakta, bukan halusinasi.

### 2. Conversational Memory
Menggunakan `create_history_aware_retriever` dari LangChain. Pertanyaan ambigu seperti *"Berapa HP-nya?"* secara otomatis di-reformulate menjadi *"Berapa HP Barbarian King?"* berdasarkan riwayat percakapan sebelumnya.

### 3. Citation & Source Transparency
Setiap jawaban menampilkan sumber dokumen yang digunakan (contoh: `📚 Sumber: Archer.txt, Super Archer.txt`), sehingga pengguna bisa memverifikasi keakuratan jawaban.

### 4. Guardrails (Anti-Halusinasi)
Prompt engineering ketat memaksa model untuk:
- Menjawab *"Maaf, saya tidak memiliki informasi..."* jika konteks tidak relevan
- Menjawab *"Saya hanya dapat menjawab pertanyaan seputar CoC"* jika pertanyaan di luar topik
- Tidak mengarang jawaban di luar konteks yang diberikan

### 5. Evaluation Pipeline (MLOps)
Script `evaluate.py` mengukur kualitas sistem secara otomatis menggunakan dataset 15 pasang Q&A, menghasilkan log CSV dengan score per pertanyaan dan ringkasan performa keseluruhan.

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| UI | Gradio |
| Deployment | Hugging Face Spaces (Docker) |
| Vector Database | Pinecone |
| LLM | Google Gemini 1.5 Flash |
| Embedding Model | `all-MiniLM-L6-v2` (HuggingFace) |
| RAG Framework | LangChain |
| Data Source | Clash of Clans Fandom Wiki (XML Export) |

---

## 📁 Struktur Proyek

```
coc-chatbot/
├── app.py                  # Aplikasi utama Gradio + RAG chain
├── parse_xml.py            # Parser XML Wiki → file .txt per artikel
├── build_pinecone_db.py    # Ingest dokumen .txt → Pinecone vector DB
├── evaluate.py             # Script evaluasi otomatis RAG pipeline
├── eval_dataset.csv        # Dataset evaluasi (15 pasang Q&A)
├── Dockerfile              # Container config untuk HF Spaces
├── requirements.txt        # Python dependencies
├── chainlit.md             # (Legacy) Dokumentasi awal Chainlit
├── coc_export1.xml         # Raw data export dari Fandom Wiki
├── parsed_data/            # Hasil parsing: 100+ file .txt per artikel
│   ├── Archer.txt
│   ├── Barbarian King.txt
│   └── ... (100+ files)
└── README.md
```

---

## ⚙️ Cara Menjalankan Lokal

### 1. Clone & Setup Environment
```bash
git clone https://github.com/davidbc17/coc-chatbot-expert.git
cd coc-chatbot-expert
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Konfigurasi API Keys
Buat file `.env`:
```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 3. (Opsional) Rebuild Vector Database
```bash
# Parse XML menjadi file .txt
python parse_xml.py

# Upload ke Pinecone
python build_pinecone_db.py
```

### 4. Jalankan Aplikasi
```bash
python app.py
```

### 5. Jalankan Evaluasi
```bash
python evaluate.py
# Output: evaluation_log_YYYYMMDD_HHMMSS.csv
```

---

## 📊 Hasil Evaluasi

Evaluasi dijalankan terhadap 15 pertanyaan mencakup topik troops, spells, heroes, dan guardrails.

| Metrik | Hasil |
|---|---|
| ✅ PASS (score ≥ 0.5) | 9 / 15 |
| ⚠️ PARTIAL (score 0.3–0.5) | 3 / 15 |
| ❌ FAIL (score < 0.3) | 3 / 15 |
| 📈 Average Score | 0.59 / 1.00 |

**Catatan:** Score 0.59 menggunakan metrik keyword matching sederhana yang cukup ketat. Guardrails untuk pertanyaan di luar topik bekerja 100% (2/2 test case passed). Area perbaikan utama ada di ideal answer dataset dan ukuran chunk vector database.

---

## 🔮 Langkah Berikutnya (Next Steps)

- **LangSmith** — Platform tracing native LangChain untuk memvisualisasikan setiap langkah chain, mengidentifikasi bottleneck, dan menjalankan evaluasi berbasis LLM yang lebih akurat
- **Weights & Biases (W&B)** — Logging metrik evaluasi dari waktu ke waktu untuk tracking improvement secara eksperimental
- **Re-chunking Strategy** — Eksperimen dengan chunk size dan overlap yang berbeda untuk meningkatkan retrieval accuracy
- **Hybrid Search** — Kombinasi semantic search (vector) + keyword search (BM25) untuk hasil retrieval yang lebih baik
- **LLM-as-Judge** — Mengganti keyword matching dengan LLM untuk menilai kualitas jawaban secara semantik

---

## 👤 Author

**David Bryan Christiansen** — [Github](https://github.com/davidbryanc/clash-of-clans-ai-chatbot)
