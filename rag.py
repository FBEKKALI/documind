"""Coeur du pipeline RAG : ingestion, index FAISS, récupération et génération citée."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir du contexte fourni. "
    "Cite les passages utilisés avec leur numéro entre crochets, par exemple [1]. "
    "Si le contexte ne contient pas la réponse, réponds exactement : "
    "\"Je ne trouve pas cette information dans les documents.\" "
    "Réponds dans la langue de la question, de façon concise."
)


# ---------- Fournisseurs de modèles ----------
def get_embeddings():
    if os.getenv("PROVIDER", "azure").lower() == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))


def get_llm():
    if os.getenv("PROVIDER", "azure").lower() == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            temperature=0,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)


# ---------- Ingestion ----------
def load_documents(docs_dir: Path = DOCS_DIR):
    docs = []
    for path in sorted(docs_dir.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif suffix in {".md", ".txt"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR):
    documents = load_documents(docs_dir)
    if not documents:
        raise SystemExit(f"Aucun document trouvé dans {docs_dir}/ (PDF, MD ou TXT).")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    store = FAISS.from_documents(chunks, get_embeddings())
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    return len(documents), len(chunks)


_store = None


def load_store():
    global _store
    if _store is None:
        _store = FAISS.load_local(
            str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
        )
    return _store


# ---------- Récupération et génération ----------
def retrieve(question: str, k: int = TOP_K):
    results = load_store().similarity_search_with_score(question, k=k)
    return [
        {
            "source": Path(doc.metadata.get("source", "?")).name,
            "page": doc.metadata.get("page"),
            "score": float(score),
            "text": doc.page_content,
        }
        for doc, score in results
    ]


def _format_context(passages):
    blocks = []
    for i, p in enumerate(passages, start=1):
        where = p["source"] + (f", p.{p['page'] + 1}" if p["page"] is not None else "")
        blocks.append(f"[{i}] ({where})\n{p['text']}")
    return "\n\n".join(blocks)


def answer(question: str, k: int = TOP_K):
    passages = retrieve(question, k)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "Contexte :\n{context}\n\nQuestion : {question}")]
    )
    chain = prompt | get_llm() | StrOutputParser()
    text = chain.invoke({"context": _format_context(passages), "question": question})
    return {
        "answer": text,
        "sources": [
            {"id": i, "source": p["source"], "page": p["page"], "excerpt": p["text"][:300]}
            for i, p in enumerate(passages, start=1)
        ],
    }
