#  Clash of Clans AI Expert ⚔️

Sebuah chatbot canggih berbasis AI yang berfungsi sebagai ahli strategi game Clash of Clans. Dibangun dengan arsitektur **Retrieval-Augmented Generation (RAG)** modern, aplikasi ini mampu menjawab pertanyaan spesifik tentang Troops, Buildings, Heroes, dan Spells dengan akurat dan relevan.

**[Link ke Aplikasi Live di Hugging Face]** <-- _(Tambahkan link ini setelah Anda deploy di Hari ke-4)_

![Screenshot Aplikasi Chainlit Anda] <-- _(Opsional: Tambahkan screenshot setelah aplikasi Anda berjalan)_

## 🔥 Fitur Utama

- **Basis Pengetahuan Komprehensif:** Menggunakan data yang diekstrak langsung dari export XML Fandom Wiki untuk memastikan akurasi.
- **Arsitektur RAG Profesional:** Menggabungkan kekuatan model embedding open-source dengan database vektor cloud untuk pencarian semantik yang cepat dan efisien.
- **Antarmuka Chat Modern:** Dibangun menggunakan **Chainlit** untuk pengalaman pengguna yang interaktif dan responsif.
- **Siap Produksi:** Menggunakan tumpukan teknologi yang relevan dengan industri, termasuk database terkelola (Pinecone) dan deployment di Hugging Face.

## 🛠️ Tumpukan Teknologi & Arsitektur

Proyek ini dibangun dengan pendekatan modular yang mencerminkan praktik terbaik dalam pengembangan aplikasi LLM.

1.  **Data Ingestion & Processing:**
    - **Sumber Data:** Export XML dari Fandom Wiki.
    - **Parsing:** Menggunakan `xml.etree.ElementTree` untuk streaming dan `wikitextparser` untuk membersihkan sintaks wikitext, termasuk konversi tabel statistik menjadi teks deskriptif.

2.  **Embedding & Indexing (The "Brain"):**
    - **Embedding Model:** `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`. Model open-source yang efisien dan berjalan secara lokal, mengurangi ketergantungan API dan biaya.
    - **Vector Database:** **Pinecone**. Database vektor cloud yang terkelola, memberikan skalabilitas dan performa tingkat produksi.

3.  **Retrieval & Generation:**
    - **Orchestration:** **LangChain**. Menggunakan arsitektur chain modern (LCEL) untuk alur kerja yang fleksibel dan kuat.
    - **LLM:** **Google Gemini API** (`gemini-1.5-flash`). Memanfaatkan kekuatan salah satu model bahasa terbesar untuk menghasilkan jawaban yang fasih dan kontekstual.
    - **Prompt Engineering:** Menerapkan template prompt kustom untuk mengontrol persona AI, membatasi jawaban hanya pada konteks yang diberikan, dan menangani kasus di mana informasi tidak ditemukan.

4.  **User Interface & Deployment:**
    - **UI Framework:** **Chainlit**. Kerangka kerja spesialis untuk membangun antarmuka chatbot dengan cepat.
    - **Hosting:** **Hugging Face Spaces**. Platform standar komunitas AI untuk hosting dan memamerkan proyek machine learning.

## 🚀 Rencana Pengembangan Selanjutnya (Improvement)

- **[ ] Implementasi Memori Percakapan:** Menggunakan `ConversationalRetrievalChain` untuk memungkinkan chatbot mengingat konteks dari obrolan sebelumnya.
- **[ ] Menampilkan Sumber (Citation):** Mengembalikan metadata dari dokumen yang diambil untuk menunjukkan kepada pengguna "bukti" dari mana jawaban berasal.
- **[ ] Sistem Evaluasi (MLOps):** Membangun script evaluasi untuk mengukur akurasi RAG chain terhadap dataset pertanyaan-jawaban yang sudah disiapkan.

## ⚙️ Menjalankan Secara Lokal

1.  **Clone repository ini.**
    ```bash
    git clone https://github.com/NamaAnda/coc-chatbot.git
    cd coc-chatbot
    ```
2.  **Install semua dependensi.**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup environment variables.**
    - Buat file `.env` dan isi dengan `GOOGLE_API_KEY` dan `PINECONE_API_KEY` & `PINECONE_ENVIRONMENT`.
4.  **Jalankan aplikasi Chainlit.**
    ```bash
    chainlit run app.py -w
    ```

---
