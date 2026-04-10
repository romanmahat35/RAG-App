import os

import requests
import streamlit as st
from dotenv import load_dotenv

from utils import (
    chunk_text,
    embed_query,
    embed_texts,
    extract_excel_text,
    extract_pdf_text,
    load_embedding_model,
    retrieve_top_k,
)

load_dotenv()

DEFAULT_OLLAMA_MODEL = "llama2"

def build_prompt(question: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])
    return (
        "You are an intelligent assistant. Use the following extracted content from the uploaded PDF/Excel documents to answer the question. "
        "If the answer is not contained in the provided context, say that the information is not available.\n\n" 
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def answer_question(
    question: str,
    retrieved_chunks: list[tuple[str, float]],
    ollama_url: str,
    ollama_model: str,
) -> str:
    prompt = build_prompt(question, retrieved_chunks)

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specializing in answering user questions from uploaded documents."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 450,
    }
    response = requests.post(f"{ollama_url}/v1/chat/completions", json=payload)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def main():
    st.set_page_config(page_title="RAG PDF + Excel Q&A", layout="wide")
    st.title("RAG Document Q&A")
    st.markdown(
        "Upload a PDF and/or Excel file, then ask a question based on the document content. "
        "The app extracts text, builds retrieval embeddings, and uses an LLM to answer from the uploaded data."
    )

    ollama_url = st.sidebar.text_input(
        "Ollama URL", value=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    )
    ollama_model = st.sidebar.text_input(
        "Ollama Model", value=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Supported formats: PDF, Excel (.xls, .xlsx). Upload one or both files.")

    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
    excel_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

    # if not openai_api_key:
    #     st.warning("Set your OpenAI API key in the sidebar or via OPENAI_API_KEY in a .env file.")

    if not pdf_file and not excel_file:
        st.info("Upload at least one PDF or Excel file to begin.")
        return

    model = load_embedding_model()
    contents = []
    if pdf_file:
        with st.spinner("Extracting PDF text..."):
            pdf_text = extract_pdf_text(pdf_file)
        contents.append((f"PDF: {pdf_file.name}", pdf_text))

    if excel_file:
        with st.spinner("Extracting Excel text..."):
            excel_text = extract_excel_text(excel_file)
        contents.append((f"Excel: {excel_file.name}", excel_text))

    chunks = []
    if contents:
        for title, text in contents:
            if not text.strip():
                continue
            chunks.extend([f"{title} | {chunk}" for chunk in chunk_text(text)])

    if not chunks:
        st.error("Could not extract any text from the uploaded files.")
        return

    with st.spinner("Embedding document chunks..."):
        doc_embeddings = embed_texts(model, chunks)

    question = st.text_input("Ask a question about the uploaded documents")
    if st.button("Get Answer"):
        if not question.strip():
            st.error("Enter a question to continue.")
            return

        with st.spinner("Retrieving relevant document context..."):
            query_vec = embed_query(model, question)
            top_chunks = retrieve_top_k(chunks, doc_embeddings, query_vec)

        st.subheader("Retrieved context")
        for idx, (chunk, score) in enumerate(top_chunks, start=1):
            st.markdown(f"**Chunk {idx}** — similarity {score:.4f}")
            st.write(chunk[:1000] + ("..." if len(chunk) > 1000 else ""))

        with st.spinner("Generating answer..."):
            answer = answer_question(
                question,
                top_chunks,
                ollama_url,
                ollama_model,
            )

        st.subheader("Answer")
        st.write(answer)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Notes")
    st.sidebar.markdown(
        "- Upload both documents if your question spans PDF and Excel data.\n"
        "- The app uses semantic retrieval and Ollama to answer from the extracted content.\n"
        "- For best results, ask specific questions about the uploaded documents."
    )


if __name__ == "__main__":
    main()
