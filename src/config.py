"""Global configuration for the Text-to-Network Engine."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# Auto-load .env from project root so any entry point (pipeline.py, chat.py,
# streamlit, pytest) sees the same environment without needing `source .env`.
# Existing process env vars take precedence; .env values are only filled in
# for keys that are not already set.
def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
CONCEPT_DICTIONARY_PATH = CONFIG_DIR / "concept_dictionary.yaml"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Data source paths ──────────────────────────────────────────────
POLICY_PDF_PATH = DATA_DIR / "national-climate-resilience-and-adaptation-strategy.pdf"
YELP_EXTRACTED_DIR = DATA_DIR / "Yelp JSON" / "yelp_dataset"

# ── spaCy model ────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── Preprocessing ──────────────────────────────────────────────────
CUSTOM_STOP_WORDS: set[str] = {
    "also", "however", "including", "well", "within", "across",
    "example", "e.g.", "i.e.", "etc", "page", "figure", "table",
    "this", "that", "these", "those", "they", "them", "which",
    "what", "some", "both", "each", "every", "other", "another",
    "everyone", "everything", "something", "nothing", "anything",
    "someone", "anyone", "nobody", "many", "much", "more", "most",
    "such", "very", "quite", "really", "actually", "just", "even",
    "still", "already", "often", "always", "never",
}
PRESERVE_STOP_WORDS: set[str] = {
    "shall", "must", "will", "should", "may",
}

# ── Concept extraction ─────────────────────────────────────────────
NER_LABELS = {"PERSON", "ORG", "GPE", "EVENT", "NORP", "FAC", "LOC", "PRODUCT", "LAW"}
MIN_CONCEPT_FREQ = 2
TFIDF_TOP_N = 100

# ── Relationship extraction ────────────────────────────────────────
CO_OCCURRENCE_WINDOW = 2          # sentences
CAUSAL_VERBS = {
    "cause", "lead", "result", "drive", "enable", "support",
    "contribute", "influence", "affect", "impact", "reduce",
    "increase", "improve", "strengthen", "weaken", "threaten",
    "promote", "facilitate", "hinder", "prevent",
}

# ── Network construction ───────────────────────────────────────────
MIN_EDGE_WEIGHT = 1

# ── Visualisation ──────────────────────────────────────────────────
TOP_N_DISPLAY = 30
GRAPH_LAYOUT = "spring"

# ── LLM / RAG chatbot (Stage 8) ────────────────────────────────────
# Provider is one of:
#   "openai" | "dashscope" | "huggingface_api" | "huggingface_local" | "echo"
# "echo" is a zero-dependency fallback that returns retrieved context verbatim —
# useful for local testing when no LLM keys are configured.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "echo")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# API keys are read from the environment by each provider; centralised here
# only so callers can check whether a provider is configured.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "") or os.getenv(
    "HF_TOKEN", ""
)
# DashScope (Aliyun ModelScope) — uses the OpenAI-compatible endpoint so we
# can reuse the `openai` Python client. Default model is qwen-turbo, override
# via LLM_MODEL (e.g. qwen-plus, qwen-max, qwen2.5-72b-instruct).
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ── Neo4j backend (optional) ───────────────────────────────────────
# When NEO4J_URI is reachable and NEO4J_PASSWORD is set, the pipeline
# can push the conceptual network into a local Neo4j instance and the
# Stage-8 chatbot can query it via Cypher + the native vector index.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_VECTOR_INDEX = os.getenv("NEO4J_VECTOR_INDEX", "concept_embedding")
NEO4J_EDGE_VECTOR_INDEX = os.getenv("NEO4J_EDGE_VECTOR_INDEX", "related_embedding")
NEO4J_EMBEDDING_DIM = int(os.getenv("NEO4J_EMBEDDING_DIM", "384"))
