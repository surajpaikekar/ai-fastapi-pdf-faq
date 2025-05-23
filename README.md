# 🧠 FastAPI PDF FAQ AI Project

This is a Python-based **backend application** using **FastAPI** that allows users to:

- Upload a PDF document
- Generate top 10 FAQs from the PDF
- Ask any custom question from the PDF
- Ask any question using **live internet search** (via tools like Gemini)

---

## 🚀 Features

- ✨ Uses **Transformers**, **HuggingFace**, **LangChain**
- 🤖 Supports **LLAMA**, **Ollama**, **Gemini 2.2 Flash**, etc.
- 📄 Processes PDFs for intelligent Q&A
- 🔁 Multiple model pipelines: huggingface transformer, Ollama, Gemini
- 🧪 Testable directly via **Postman** (No frontend required)

---

## 🧪 How to Use (Test via Postman)

### 📤 1. Generate FAQs (POST)

**Endpoint:** `http://127.0.0.1:8000/generate-faqs`  
**Method:** POST  
**Body:** `form-data`

| Key  | Value            |
|------|------------------|
| file | Upload your PDF |

---

### 🧠 2. Ask Question (POST)

**Endpoint:** `http://127.0.0.1:8000/ask-question/`  
**Method:** POST  
**Body:** `form-data`

| Key       | Value                                        |
|-----------|----------------------------------------------|
| file      | Upload the same PDF                          |
| question  | Type your question (e.g., "What is chapter 1 about?") |
| user_id   | Required for Ollama & Gemini (e.g., `"user123"`) |

---

## 🧰 Technologies Used

| Component       | Tool/Framework              |
|-----------------|-----------------------------|
| Backend         | FastAPI                     |
| AI Models       | HuggingFace Transformers    |
| Embeddings      | Sentence Transformers (`MiniLM`) |
| Vector DB       | FAISS                       |
| Chain Logic     | LangChain                   |
| Internet Answer | Gemini 2.2 Flash (via API)  |
| Multi-Model     | LLAMA, Ollama, Transformers |

---

## 📦 Setup Instructions (Optional)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run FastAPI
uvicorn main:app --reload
