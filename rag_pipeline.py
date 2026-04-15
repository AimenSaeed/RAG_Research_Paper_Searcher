import fitz
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Models ─────────────────────────────────────────────────
def load_models():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    return embedding_model, llm

# ── PDF extraction ─────────────────────────────────────────
def load_pdfs(data_folder="data"):
    documents = []
    pdf_files = list(Path(data_folder).glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in /data folder!")
        return []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text().strip()
            if not text:
                continue
            documents.append({
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_num + 1
                }
            })

        print(f"Done — {total_pages} pages extracted from {pdf_path.name} ✅")
        doc.close()

    return documents

# ── PDF extraction from bytes (for Streamlit uploads) ──────
def extract_text(pdf_bytes, filename):
    raw_docs = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text().strip()
        if not text:
            continue
        raw_docs.append({
            "text": text,
            "metadata": {
                "source": filename,
                "page": page_num + 1
            }
        })

    doc.close()
    return raw_docs, total_pages

# ── Chunking ───────────────────────────────────────────────
def chunk_documents(documents, chunk_size=1000, overlap_pct=0.15):
    chunk_overlap = int(chunk_size * overlap_pct)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(Document(
                page_content=chunk,
                metadata=doc["metadata"]
            ))
    return all_chunks

# ── Vector store ───────────────────────────────────────────
def build_vectorstore(chunks, embedding_model):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

def load_vectorstore(embedding_model):
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

# ── Format docs ────────────────────────────────────────────
def format_docs(docs):
    formatted = []
    for doc in docs:
        formatted.append(
            f"Source: {doc.metadata['source']} | "
            f"Page: {doc.metadata['page']}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)

# ── RAG chain ──────────────────────────────────────────────
def get_rag_chain(vectorstore, llm):
    prompt = ChatPromptTemplate.from_template("""
You are an expert research assistant helping users find information
from academic research papers.

Use ONLY the following context to answer the question.
If the answer is not in the context, say "I could not find this
information in the uploaded papers."
Always mention which paper and page number your answer comes from.

Context:
{context}

Question:
{question}

Answer:
""")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    def build_prompt_input(question):
        docs = retriever.invoke(question)
        return {
            "context": format_docs(docs),
            "question": question
        }

    return (
        RunnableLambda(build_prompt_input)
        | prompt
        | llm
        | StrOutputParser()
    )