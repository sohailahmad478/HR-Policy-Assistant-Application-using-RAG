# HR Policy Assistant

An HR Policy Assistant application using
Retrieval-Augmented Generation (RAG).

## Technologies

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Groq
- GitHub

## How it works

1. User uploads an HR policy PDF.
2. PyMuPDF extracts the text.
3. The text is divided into chunks.
4. Sentence Transformers creates embeddings.
5. FAISS stores and searches the embeddings.
6. Relevant chunks are retrieved.
7. Groq GPT-OSS 20B generates the answer.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
