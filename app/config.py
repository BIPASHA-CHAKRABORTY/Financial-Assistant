
"""Runtime configuration. Values are deliberately supplied through the environment."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENT_STORAGE_PATH = Path(os.getenv("DOCUMENT_STORAGE_PATH", DATA_DIR / "documents"))
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", DATA_DIR / "indexes"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", os.getenv("LLM_MODEL", "gemini-2.5-flash"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

for directory in (DOCUMENT_STORAGE_PATH, VECTOR_STORE_PATH):
    directory.mkdir(parents=True, exist_ok=True)

import os

ENABLE_TABLE_EXTRACTION = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
ENABLE_CHART_ANALYSIS = os.getenv("ENABLE_CHART_ANALYSIS", "true").lower() == "true"
MIN_IMAGE_PX = int(os.getenv("MIN_IMAGE_PX", "150"))                 # ignore logos/icons
MAX_CHARTS_PER_DOCUMENT = int(os.getenv("MAX_CHARTS_PER_DOCUMENT", "30"))   # vision cost guard
TABLE_MAX_ROWS = int(os.getenv("TABLE_MAX_ROWS", "25"))
# GEMINI_VISION_MODEL is read directly in app/vision.py (falls back to GEMINI_MODEL)
