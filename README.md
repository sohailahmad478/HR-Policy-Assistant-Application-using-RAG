HR Policy Assistant — RAG

A beginner-friendly but improved Retrieval-Augmented Generation (RAG) application for HR policy questions.

What the app does

User uploads a text-based HR policy PDF.

PyMuPDF extracts text page by page.

Text is cleaned and split into overlapping chunks.

Sentence Transformers (all-MiniLM-L6-v2) creates normalized embeddings.

FAISS searches the most relevant chunks using cosine similarity.

The retrieved chunks are sent to Groq openai/gpt-oss-20b.

The app returns a grounded answer and lets the user inspect the retrieved source text and page numbers.

Project structure

hr-policy-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml   # local only; never commit this file

Tech stack

Python

Streamlit

PyMuPDF

Sentence Transformers

FAISS

Groq API

GitHub

Streamlit Community Cloud

Run locally on Windows

Use Python 3.12 so your local environment matches Streamlit Community Cloud's current default.

1. Open the project in VS Code

Open the hr-policy-assistant folder in VS Code.

2. Create a virtual environment

PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1

If activation is blocked:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Add your Groq API key

Create:

.streamlit/secrets.toml

Put this inside:

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

Never commit secrets.toml to GitHub.

5. Start the app

streamlit run app.py

Open the local URL shown in the terminal, normally:

http://localhost:8501

GitHub deployment

From the project folder:

git init
git add .
git status
git commit -m "Build HR policy RAG assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hr-policy-assistant.git
git push -u origin main

Before pushing, check that your secret is ignored:

git check-ignore -v .streamlit/secrets.toml

Streamlit Community Cloud deployment

Open Streamlit Community Cloud.

Connect your GitHub account.

Create a new app.

Select your hr-policy-assistant repository.

Select branch main.

Select app.py as the main file.

Open Advanced settings.

Select Python 3.12.

In Secrets, paste:

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

Deploy the app.

Important notes

Scanned PDFs

The current app needs selectable text. Image-only/scanned PDFs require OCR before indexing.

Privacy

This is a learning/portfolio application. Do not upload confidential employee records to a public app without proper authentication, authorization, privacy controls, and organizational approval.

FAISS persistence

The current app builds a FAISS index for the uploaded document during the active Streamlit session. It does not use an external vector database.

Why this version is improved

Compared with the simplest RAG demo, this version adds:

page-aware document chunks

overlapping word-based chunking

normalized embeddings + cosine similarity through FAISS inner product

cached embedding model

document statistics

chat-style Q&A history

retrieved-source inspection

PDF page references in the prompt

safer fallback when information is not in the document

clearer API-key and deployment errors

duplicate PDF detection using a SHA-256 signature

a cleaner Streamlit interface

Suggested viva explanation

What is RAG?

RAG combines retrieval and generation. First, the system retrieves relevant information from the uploaded HR policy. Then the language model uses the retrieved information to generate an answer.

Why FAISS?

FAISS provides fast vector similarity search. It allows the application to retrieve the most relevant HR policy chunks without sending the complete PDF to the LLM.

Why Sentence Transformers?

Sentence Transformers converts text into numerical embeddings so semantically similar questions and policy passages can be compared.

Why PyMuPDF?

PyMuPDF extracts text and preserves page-level information from PDF files.

Why Groq?

Groq provides an API for fast inference with the selected openai/gpt-oss-20b model.
