# Pathochat – AI-powered Medical Chat Assistant

A conversational AI tool designed to answer pathology related questions using vector databases, LLMs, and PDF-based knowledge sources.

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
HF_TOKEN="your_openai_key_here"
```

---

## Run the App

Launch the Streamlit app:

```bash
streamlit run Pathochat.py
```

Your browser should open automatically at:  
[http://localhost:8501](http://localhost:8501)

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


