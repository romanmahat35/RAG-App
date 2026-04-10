from io import BytesIO

import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_pdf_text(uploaded_file: BytesIO) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def extract_excel_text(uploaded_file: BytesIO) -> str:
    uploaded_file.seek(0)
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    blocks = []
    for sheet_name, df in sheets.items():
        df = df.fillna("")
        rows = df.apply(lambda row: " | ".join(map(str, row.tolist())), axis=1).tolist()
        blocks.append(f"Sheet: {sheet_name}\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return np.array(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]


def retrieve_top_k(chunks: list[str], embeddings: np.ndarray, query_vector: np.ndarray, k: int = 4) -> list[tuple[str, float]]:
    similarities = np.dot(embeddings, query_vector)
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [(chunks[i], float(similarities[i])) for i in top_indices]
