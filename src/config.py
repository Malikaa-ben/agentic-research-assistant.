from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
INDEX_DIR = ROOT / "data" / "index"

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
TEMPERATURE = 0.0

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
