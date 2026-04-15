import streamlit as st
import os
from rag_pipeline import (
    load_models,
    extract_text,
    chunk_documents,
    build_vectorstore,
    get_rag_chain
)

# ── Page config ────────────────────────────────
st.set_page_config(
    page_title="DeepCite",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DeepCite")
st.caption("Upload research papers and ask questions in plain English")

# ── Session state ──────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "papers_loaded" not in st.session_state:
    st.session_state.papers_loaded = []

# ── Load models once ───────────────────────────
@st.cache_resource
def get_models():
    return load_models()

embedding_model, llm = get_models()

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Research Papers")

    uploaded_files = st.file_uploader(
        "Drop your PDF files here",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.papers_loaded
        ]

        if new_files:
            with st.spinner(f"Processing {len(new_files)} paper(s)..."):
                all_chunks = []

                for pdf_file in new_files:
                    st.write(f"Reading: {pdf_file.name}")
                    pdf_bytes = pdf_file.read()
                    raw_docs, total_pages = extract_text(
                        pdf_bytes,
                        pdf_file.name
                    )
                    chunks = chunk_documents(raw_docs)
                    all_chunks.extend(chunks)
                    st.session_state.papers_loaded.append(pdf_file.name)
                    st.write(f"✅ {total_pages} pages, {len(chunks)} chunks")

                st.write("Building vector store...")
                st.session_state.vectorstore = build_vectorstore(
                    all_chunks,
                    embedding_model
                )
                st.success("Ready to answer questions! ✅")

    if st.session_state.papers_loaded:
        st.divider()
        st.subheader("📑 Loaded Papers")
        for paper in st.session_state.papers_loaded:
            st.write(f"• {paper}")

    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── Main chat ──────────────────────────────────
if not st.session_state.vectorstore:
    st.info("👈 Upload research papers from the sidebar to get started!")

else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about your papers..."):
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Searching papers..."):
                rag_chain = get_rag_chain(
                    st.session_state.vectorstore,
                    llm
                )
                answer = rag_chain.invoke(question)
                st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })