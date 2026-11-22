# Pathochat – AI-powered Medical Chat Assistant

A conversational AI tool designed to answer pathology related questions using vector databases, LLMs, and PDF-based knowledge sources.
Try it at : https://exemplary-heart-production.up.railway.app/ 

---

## Features

- Natural language medical Q&A  
- Custom vector database with FAISS  
- PDF-based medical content (e.g., Pathoma, Robbins & Cotran)  
- Streamlit-based frontend  
- Git LFS support for large assets  

---

## Requirements

- Python 3.8+
- [pipenv](https://pipenv.pypa.io/en/latest/)
- Git LFS (for handling large files)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Mehdi-Abidi/Pathochat.git
cd Pathochat
```

### 2. Install Dependencies Using Pipenv

```bash
pip install pipenv
pipenv install
```

Or if you're using `requirements.txt`:

```bash
pipenv install -r requirements.txt
```

### 3. Activate the Pipenv Shell

```bash
pipenv shell
```

### 4. Add Your Environment Variables

Create a `.env` file in the root of the project:

```env
HF_TOKEN="your_hugging_face_token"
LLM_API_KEY="your_groq_key"
```

With `LLM_API_KEY` set, the app answers through an OpenAI-compatible endpoint. The defaults are Groq and `qwen/qwen3.8-27b`, and you can change them with `LLM_BASE_URL` and `LLM_MODEL`. Without `LLM_API_KEY` the app falls back to Mistral-7B on the Hugging Face endpoint, which needs `HF_TOKEN`.

---

## Run the App

Launch the Streamlit app:

```bash
streamlit run Pathochat.py
```

Your browser should open automatically at:  
[http://localhost:8501](http://localhost:8501)

---

## Evaluation (November 2026)

PathoChat was scored on 100 held-out pathology questions from MedMCQA. The original setup (Mistral-7B with a context-only prompt) answered 59 correctly. The current setup (Qwen3.8-27B with a prompt that may fall back on standard pathology knowledge) answers 86. The same model without retrieval answers 91, so on exam questions the gain comes from the model and the prompt, and weak passages can mislead a strong model. The method, all results and the script are in [evaluation/](evaluation/README.md).

---

## 📁 Git LFS Setup (For Large Files)

### 1. Install Git LFS

```bash
git lfs install
```

### 2. Track large files (example: PDFs)

```bash
git lfs track "*.pdf"
```

### 3. Pull any LFS-tracked files

```bash
git lfs pull
```

---

##  Project Structure

```
Medbot/
├── data/                          # PDF textbooks (LFS tracked)
├── vector_store/                  # FAISS vector DB files
├── Pathochat.py                   # Main Streamlit app
├── middle_ware.py                 # Core logic / model handlers
├── llm_database.py                # Embedding + vector DB interface
├── requirements.txt               # Fallback dependency list
├── Pipfile / Pipfile.lock         # Pipenv environment
├── .env                           # API keys (excluded from Git)
└── README.md
```

---


