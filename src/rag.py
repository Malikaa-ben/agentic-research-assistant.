"""Indexation et recherche de PDF avec FAISS + embeddings Ollama."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, INDEX_DIR, PDF_DIR

_store = None  # cache en mémoire


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)


def build_index(pdf_dir: Path = PDF_DIR, index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    """Indexe tous les PDF du dossier. Retourne (nb_pdf, nb_chunks)."""
    global _store
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {pdf_dir}")

    docs = []
    for pdf in pdfs:
        docs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    _store = FAISS.from_documents(chunks, _embeddings())
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    _store.save_local(str(index_dir))
    return len(pdfs), len(chunks)


def get_store():
    """Charge l'index FAISS s'il existe, sinon None."""
    global _store
    if _store is None and (Path(INDEX_DIR) / "index.faiss").exists():
        _store = FAISS.load_local(
            str(INDEX_DIR), _embeddings(), allow_dangerous_deserialization=True
        )
    return _store
