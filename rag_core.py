import os
import re
import logging
from typing import List, Dict, Any, Optional

try:
    import streamlit as st
except Exception:
    st = None

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from config import AppConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("pdpl_rag")


def get_secret(key: str) -> str:
    """
    يقرأ المفتاح من Streamlit Secrets أو Environment Variables.
    لا تضع GROQ_API_KEY داخل GitHub.
    """
    value = os.getenv(key)
    if value:
        return value

    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    raise ValueError(f"Missing secret: {key}")


def clean_text(text: str) -> str:

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s*\d+\b", "", text)
    return text.strip()


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
    max_sources: int = 4,
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


def get_embeddings(config: AppConfig) -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(config: AppConfig) -> FAISS:

    index_path = config.vectorstore_path

    faiss_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Vectorstore folder not found: {index_path}")

    if not faiss_file.exists():
        raise FileNotFoundError(f"Missing file: {faiss_file}")

    if not pkl_file.exists():
        raise FileNotFoundError(f"Missing file: {pkl_file}")

    embeddings = get_embeddings(config)

    logger.info(f"Loading FAISS vectorstore from: {index_path}")

    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_prompt() -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a legal assistant specialized in the Saudi Personal Data Protection Law (PDPL).

CORE PRINCIPLE: Answer minimally and precisely. Brevity is preferred over completeness.

Your task:
- Content: Answer ONLY the exact question asked. Do not anticipate follow-up questions, related obligations, or natural extensions of the topic.
- Minimality: Use the FEWEST bullets needed to answer (typically 1-4). Do not add bullets that provide background, examples, or implementation details unless the question explicitly requests them.
- Faithfulness: Use ONLY facts explicitly stated in the provided context. If a fact is related to the topic but does NOT directly answer the specific question, OMIT IT — even if the fact is legally correct.
- No paraphrasing of qualifiers: Preserve binding qualifiers from the source verbatim (e.g., "without undue delay", "not exceeding [N] days", "in writing"). Do not soften, generalize, or omit them.
- No restating: Do not express the same point twice in different words.
- No volunteering: Do not add exceptions, edge cases, conditions, examples, or "additional considerations" unless explicitly asked.
- Format: Bullet points only. Each bullet must directly answer the question.
- Accuracy: Do not invent article numbers, definitions, or legal conclusions.
- Citations: Do not mention source names or page numbers inside the body of the answer.
- Abstention: If the answer is not supported by the context, reply exactly: Not found in documents.
- Disclaimer: End with a short legal disclaimer advising consultation with a qualified legal professional.

SELF-CHECK before responding (silent):
1. Is every bullet directly answering the user's exact question?
2. Did I add any fact not strictly required to answer?
3. Did I add any bullet beyond the minimum needed?
If yes to (2) or (3) — remove it."""
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


def create_rag_chain(config: Optional[AppConfig] = None):
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
