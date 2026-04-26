import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "index.faiss"
PKL_PATH = DATA_DIR / "index.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


class RAGEngine:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(str(INDEX_PATH))
        self.docs = self._load_docs()

    def _load_docs(self):
        with open(PKL_PATH, "rb") as f:
            data = pickle.load(f)

        # يدعم أكثر من شكل شائع للـ pkl
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if "documents" in data:
                return data["documents"]
            if "docs" in data:
                return data["docs"]
            if "texts" in data:
                return data["texts"]

        raise ValueError(
            "لم أستطع فهم شكل index.pkl. يجب أن يحتوي list أو dict فيه documents/docs/texts."
        )

    def _doc_to_text(self, doc):
        if isinstance(doc, str):
            return doc

        if isinstance(doc, dict):
            text = doc.get("text") or doc.get("page_content") or doc.get("content") or ""
            source = doc.get("source") or doc.get("file") or doc.get("metadata", {}).get("source", "")
            return f"{text}\n\nالمصدر: {source}" if source else text

        if hasattr(doc, "page_content"):
            metadata = getattr(doc, "metadata", {})
            source = metadata.get("source", "")
            return f"{doc.page_content}\n\nالمصدر: {source}" if source else doc.page_content

        return str(doc)

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            if idx < len(self.docs):
                results.append({
                    "text": self._doc_to_text(self.docs[idx]),
                    "score": float(score),
                    "index": int(idx),
                })

        return results

    def answer(self, query: str, top_k: int = 5):
        chunks = self.retrieve(query, top_k=top_k)

        context = "\n\n---\n\n".join(
            [f"[Chunk {i+1}]\n{c['text']}" for i, c in enumerate(chunks)]
        )

        system_prompt = """
أنت مساعد RAG عربي محترف.
أجب فقط اعتمادًا على السياق المرفق.
إذا لم تجد الإجابة في السياق، قل بوضوح: لا توجد معلومة كافية في الملفات.
اكتب إجابة منظمة، دقيقة، واذكر المصادر أو المقاطع عند توفرها.
"""

        user_prompt = f"""
السؤال:
{query}

السياق:
{context}
"""

        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.2,
            max_tokens=1400,
        )

        return completion.choices[0].message.content, chunks
