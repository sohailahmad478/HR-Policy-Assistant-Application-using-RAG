import hashlib
from typing import List, Dict

import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

APP_TITLE = "HR Policy Assistant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

st.set_page_config(page_title=APP_TITLE, page_icon="👥", layout="wide")

st.title("👥 HR Policy Assistant")
st.caption("Ask questions from an uploaded HR policy PDF using Retrieval-Augmented Generation (RAG).")

@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    if "GROQ_API_KEY" not in st.secrets or not st.secrets["GROQ_API_KEY"]:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in Streamlit Secrets.")
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def extract_pages(pdf_bytes: bytes) -> List[Dict]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = " ".join(text.split())
            if text:
                pages.append({"page": page_no, "text": text})
    return pages


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > chunk_size:
            chunks.append(" ".join(current))
            overlap_words = []
            overlap_len = 0
            for old_word in reversed(current):
                added = len(old_word) + (1 if overlap_words else 0)
                if overlap_len + added > overlap:
                    break
                overlap_words.append(old_word)
                overlap_len += added
            current = list(reversed(overlap_words))
            current_len = overlap_len
        current.append(word)
        current_len += extra

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    chunks = []
    for page in pages:
        for chunk_no, text in enumerate(split_text(page["text"]), start=1):
            chunks.append({
                "text": text,
                "page": page["page"],
                "chunk": chunk_no,
            })
    return chunks


def build_index(chunks: List[Dict], model):
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(question: str, index, chunks: List[Dict], model, top_k: int = TOP_K):
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    k = min(top_k, len(chunks))
    scores, ids = index.search(query_vector, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)
    return results


def build_context(results: List[Dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source {i} | PDF page {r['page']} | similarity {r['score']:.3f}]\n{r['text']}"
        for i, r in enumerate(results, start=1)
    )


def answer_question(question: str, results: List[Dict]) -> str:
    client = get_groq_client()
    context = build_context(results)
    prompt = f"""
You are an HR policy assistant.

Use ONLY the HR policy context below. Do not invent facts, benefits, leave rules, dates,
approval requirements, exceptions, or procedures.

If the policy does not contain enough information to answer the question, say clearly:
"The uploaded HR policy does not contain enough information to answer this question."

Give a concise, professional answer in plain language. Include PDF page references such as
"(PDF page 4)" when relevant. Do not provide legal advice or legal conclusions.

HR POLICY CONTEXT:
{context}

QUESTION:
{question}
""".strip()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=700,
    )
    return (response.choices[0].message.content or "No answer was returned.").strip()


def signature(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clear_document():
    for key in ["doc_signature", "filename", "pages", "chunks", "index", "messages"]:
        st.session_state.pop(key, None)


with st.sidebar:
    st.header("📄 Policy document")
    uploaded = st.file_uploader("Upload an HR policy PDF", type=["pdf"])

    if st.button("🗑️ Clear document", use_container_width=True):
        clear_document()
        st.rerun()

    st.divider()
    st.subheader("RAG settings")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Top-K:** `{TOP_K}`")
    st.write(f"**Chunk size:** `{CHUNK_SIZE}` chars")
    st.write(f"**Overlap:** `{CHUNK_OVERLAP}` chars")
    st.divider()
    st.caption("For confidential HR data, add authentication and appropriate privacy controls before production use.")


if uploaded:
    pdf_bytes = uploaded.getvalue()
    sig = signature(pdf_bytes)

    if st.session_state.get("doc_signature") != sig:
        with st.spinner("Reading PDF and building the FAISS index..."):
            pages = extract_pages(pdf_bytes)
            if not pages:
                st.error("No selectable text was found. Scanned PDFs need OCR before they can be indexed.")
                st.stop()

            chunks = build_chunks(pages)
            if not chunks:
                st.error("No usable text chunks were created from the PDF.")
                st.stop()

            model = load_embedding_model()
            index = build_index(chunks, model)

            st.session_state["doc_signature"] = sig
            st.session_state["filename"] = uploaded.name
            st.session_state["pages"] = pages
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["messages"] = []

    st.success(f"Loaded **{st.session_state['filename']}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("PDF pages", len(st.session_state["pages"]))
    c2.metric("Text chunks", len(st.session_state["chunks"]))
    c3.metric("Retrieved per question", TOP_K)

    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View retrieved policy sections"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**PDF page {source['page']} · similarity {source['score']:.3f}**"
                        )
                        st.write(source["text"])

    question = st.chat_input("Ask a question about the HR policy...")

    if question:
        model = load_embedding_model()
        results = retrieve(question, st.session_state["index"], st.session_state["chunks"], model)

        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the policy and generating an answer..."):
                try:
                    answer = answer_question(question, results)
                    st.markdown(answer)
                except Exception as exc:
                    answer = f"I could not generate the answer. Please check your Groq API key and deployment settings.\n\nError: `{exc}`"
                    st.error(answer)

            with st.expander("View retrieved policy sections"):
                for source in results:
                    st.markdown(f"**PDF page {source['page']} · similarity {source['score']:.3f}**")
                    st.write(source["text"])

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": results,
        })
else:
    st.info("Upload an HR policy PDF from the sidebar to start.")
    st.markdown("### Example questions")
    st.markdown("- What is the annual leave policy?\n- How long is the probation period?\n- Who is eligible for maternity leave?\n- What is the remote-work policy?")
