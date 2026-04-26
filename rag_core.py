import os
import re
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from config import AppConfig


# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("pdpl_streamlit")


# =========================================================
# Secrets
# =========================================================
def get_secret(key: str) -> str:
    """
    Priority:
    1. Environment variable
    2. Streamlit secrets, if running inside Streamlit
    """
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    raise ValueError(f"Missing secret: {key}")


# =========================================================
# Text cleanup — مطابق لكود Colab
# =========================================================
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s*\d+\b", "", text)
    return text.strip()


# =========================================================
# Retrieval helpers — مطابقة لكود Colab
# =========================================================
def deduplicate_documents(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("article"),
            d.page_content.strip(),
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    return unique_docs


def rank_and_limit_documents(
    docs: List[Document],
    max_sources: int = 4
) -> List[Document]:
    selected = []
    seen_groups = set()

    for d in docs:
        group_key = (
            d.metadata.get("source"),
            d.metadata.get("article"),
        )

        if group_key not in seen_groups:
            seen_groups.add(group_key)
            selected.append(d)

        if len(selected) >= max_sources:
            break

    return selected


def format_sources(docs: List[Document]) -> List[str]:
    sources = []

    for d in docs:
        meta = d.metadata
        article = meta.get("article", "N/A")

        if "Article" in str(article):
            sources.append(
                f"{meta.get('source')} - Page {meta.get('page')} - {article}"
            )
        else:
            sources.append(
                f"{meta.get('source')} - Page {meta.get('page')} - Section: {article}"
            )

    return list(dict.fromkeys(sources))


def docs_to_context(docs: List[Document]) -> str:
    blocks = []

    for d in docs:
        text = clean_text(d.page_content)
        blocks.append(text)

    return "\n\n".join(blocks)


# =========================================================
# Load embeddings + FAISS
# =========================================================
def get_embeddings(config: AppConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(config: AppConfig) -> FAISS:
    index_path = config.vectorstore_path

    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector store folder not found: {index_path}"
        )

    faiss_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"

    if not faiss_file.exists():
        raise FileNotFoundError(f"Missing file: {faiss_file}")

    if not pkl_file.exists():
        raise FileNotFoundError(f"Missing file: {pkl_file}")

    embeddings = get_embeddings(config)

    logger.info(f"Loading FAISS vector store from: {index_path}")

    # ملاحظة أمنية:
    # استخدم allow_dangerous_deserialization=True فقط مع index.pkl موثوق منك.
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# =========================================================
# Prompt — مطابق لكود Colab
# =========================================================
def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a legal assistant specialized in the Saudi Personal Data Protection Law (PDPL).

Your task:
- Content: Answer accurately and concisely using ONLY the provided context.
- Format: Use bullet points only.
- Accuracy: Do not invent article numbers or legal conclusions.
- Scope: Only include information that directly addresses the user's query.
- Constraints: Do not mention source names or page numbers inside the body of the answer.
- Silence: If the answer is not supported by the context, reply exactly: Not found in documents.
- Disclaimer: End with a short legal disclaimer advising consultation with a qualified legal professional.
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ])


# =========================================================
# Create RAG chain
# =========================================================
def create_rag_chain(config: AppConfig | None = None):
    config = config or AppConfig()

    vectorstore = load_vectorstore(config)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.retrieval_k},
    )

    groq_api_key = get_secret("GROQ_API_KEY")

    llm = ChatGroq(
        model=config.groq_model,
        api_key=groq_api_key,
        temperature=config.temperature,
    )

    prompt = get_prompt()

    def prepare_context(question: str) -> Dict[str, Any]:
        docs = retriever.invoke(question)
        docs = deduplicate_documents(docs)
        docs = rank_and_limit_documents(
            docs,
            max_sources=config.final_max_sources,
        )

        return {
            "question": question,
            "context": docs_to_context(docs),
            "docs": docs,
        }

    chain = (
        RunnableLambda(prepare_context)
        | RunnablePassthrough.assign(
            answer=(
                RunnableLambda(
                    lambda x: {
                        "context": x["context"],
                        "question": x["question"],
                    }
                )
                | prompt
                | llm
                | StrOutputParser()
            )
        )
        | RunnableLambda(
            lambda x: {
                "answer": x["answer"],
                "sources": []
                if "not found in documents" in x["answer"].lower()
                else format_sources(x["docs"]),
            }
        )
    )

    return chain


# =========================================================
# Public ask function
# =========================================================
def answer_question(query: str, chain=None) -> Dict[str, Any]:
    query = (query or "").strip()

    if not query:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    chain = chain or create_rag_chain()
    result = chain.invoke(query)

    return {
        "answer": result["answer"].strip(),
        "sources": result["sources"],
    }


def answer_question_text(query: str, chain=None) -> str:
    result = answer_question(query, chain=chain)

    answer = result["answer"]
    sources = result["sources"]

    if sources:
        return answer + "\n\nSources:\n" + "\n".join(
            f"- {s}" for s in sources
        )

    return answer
