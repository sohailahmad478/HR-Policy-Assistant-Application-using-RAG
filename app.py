import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄"
)

st.title("📄 HR Policy Assistant")
st.write("Ask questions about your HR policy PDF.")


# -----------------------------
# Load embedding model
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -----------------------------
# Read PDF
# -----------------------------

def read_pdf(file):

    document = fitz.open(
        stream=file.read(),
        filetype="pdf"
    )

    text = ""

    for page in document:
        text += page.get_text()

    return text


# -----------------------------
# Split text
# -----------------------------

def create_chunks(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# -----------------------------
# Create FAISS index
# -----------------------------

def create_index(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Search relevant chunks
# -----------------------------

def search_policy(question, chunks, index):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    distances, indices = index.search(
        question_embedding,
        3
    )

    results = []

    for i in indices[0]:
        if i < len(chunks):
            results.append(chunks[i])

    return results


# -----------------------------
# Groq
# -----------------------------

def generate_answer(question, context):

    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = f"""
You are an HR policy assistant.

Answer the user's question using ONLY
the HR policy information provided below.

If the answer is not available in the
policy, say:

"The uploaded HR policy does not
contain this information."

HR POLICY:

{context}

QUESTION:

{question}
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Upload PDF
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload HR Policy PDF",
    type=["pdf"]
)


if uploaded_file:

    text = read_pdf(uploaded_file)

    chunks = create_chunks(text)

    index, embeddings = create_index(
        chunks
    )

    st.success(
        "HR Policy PDF uploaded successfully!"
    )

    st.write(
        f"Created {len(chunks)} text chunks."
    )

    question = st.text_input(
        "Ask a question about the HR policy:"
    )

    if question:

        relevant_chunks = search_policy(
            question,
            chunks,
            index
        )

        context = "\n\n".join(
            relevant_chunks
        )

        answer = generate_answer(
            question,
            context
        )

        st.subheader("Answer")

        st.write(answer)
