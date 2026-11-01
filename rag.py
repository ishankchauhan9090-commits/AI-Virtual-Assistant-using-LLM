import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import faiss
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


load_dotenv()

DOCUMENTS_DIR = Path("documents")
VECTORSTORE_DIR = Path("vectorstore")
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
RECORDS_PATH = VECTORSTORE_DIR / "chunks.jsonl"
LEGACY_CHUNKS_PATH = VECTORSTORE_DIR / "chunks.txt"
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "300"))
RETRIEVAL_CACHE_PREFIX = "nova:retrieval:v1:"
logger = logging.getLogger(__name__)
VECTORSTORE_DIR.mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model only when indexing or searching is needed."""
    return SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))


@lru_cache(maxsize=1)
def get_retrieval_cache():
    """Return a Redis client when configured; retrieval still works without Redis."""
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis

        return redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
    except Exception as exc:
        logger.info("Redis retrieval cache is not configured: %s", exc)
        return None


def extract_page_records(pdf_path, source_name=None, document_id=None):
    """Extract page-aware records so answers can cite the originating PDF page."""
    is_bytes = isinstance(pdf_path, (bytes, bytearray))
    reader = PdfReader(BytesIO(pdf_path)) if is_bytes else PdfReader(str(pdf_path))
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        for chunk in split_text(page_text, chunk_size=420, overlap=60):
            record = {
                "source": source_name or ("Uploaded PDF" if is_bytes else Path(pdf_path).name),
                "page": page_number,
                "text": chunk,
            }
            if document_id is not None:
                record["document_id"] = int(document_id)
            records.append(record)
    return records


def extract_text_from_pdf(pdf_path):
    """Extract all selectable text from a PDF."""
    return "\n".join(
        page.extract_text() or ""
        for page in PdfReader(str(pdf_path)).pages
    )


def split_text(text, chunk_size=500, overlap=60):
    """Split text into overlapping word chunks to retain boundary context."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
    ]


def _write_vectorstore(index, records):
    """Persist the index and source metadata used to interpret FAISS row IDs."""
    index_temp = INDEX_PATH.with_suffix(".faiss.tmp")
    records_temp = RECORDS_PATH.with_suffix(".jsonl.tmp")
    faiss.write_index(index, str(index_temp))
    records_temp.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(records_temp, RECORDS_PATH)
    os.replace(index_temp, INDEX_PATH)
    load_search_assets.cache_clear()


def _encode(records):
    embeddings = get_embedding_model().encode(
        [record["text"] for record in records],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    faiss.normalize_L2(embeddings)
    return embeddings


def _saved_document_metadata():
    """Resolve opaque stored filenames to the names users uploaded."""
    try:
        from database import get_connection

        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT document_id, filename, file_path FROM documents WHERE file_path IS NOT NULL"
            ).fetchall()
            return {
                str(Path(row["file_path"]).resolve()): {
                    "filename": row["filename"],
                    "document_id": row["document_id"],
                }
                for row in rows
            }
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("Could not load uploaded PDF names from the database: %s", exc)
        return {}


def create_vectorstore(extra_documents=None):
    """Build a fresh page-aware FAISS index from all PDFs in documents/."""
    pdf_files = sorted(DOCUMENTS_DIR.rglob("*.pdf"))
    source_metadata = _saved_document_metadata()
    records = []
    for pdf_file in pdf_files:
        try:
            metadata = source_metadata.get(str(pdf_file.resolve()), {})
            records.extend(extract_page_records(
                pdf_file,
                metadata.get("filename", pdf_file.name),
                metadata.get("document_id"),
            ))
        except Exception as exc:
            logger.warning("Skipping unreadable PDF %s: %s", pdf_file.name, exc)

    for content, source_name, document_id in extra_documents or []:
        records.extend(extract_page_records(content, source_name, document_id))

    if not records:
        INDEX_PATH.unlink(missing_ok=True)
        RECORDS_PATH.unlink(missing_ok=True)
        load_search_assets.cache_clear()
        return 0

    embeddings = _encode(records)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _write_vectorstore(index, records)
    return len(records)


def add_documents_to_vectorstore(pdf_paths):
    """Incrementally index a batch of PDFs, rebuilding once only for legacy data."""
    documents = []
    for item in pdf_paths:
        if isinstance(item, tuple) and len(item) == 3:
            content, source_name, document_id = item
        elif isinstance(item, tuple):
            content, source_name = item
            document_id = None
        else:
            content, source_name = item, None
            document_id = None
        if isinstance(content, (bytes, bytearray)):
            documents.append((bytes(content), source_name or "Uploaded PDF", document_id))
        else:
            path = Path(content)
            documents.append((path, source_name or path.name, document_id))
    new_records = []
    for content, source_name, document_id in documents:
        new_records.extend(extract_page_records(content, source_name, document_id))
    if not new_records:
        return 0

    # The previous index stored plain text with no source metadata. Migrate it once.
    if not INDEX_PATH.exists() or not RECORDS_PATH.exists():
        extra_documents = [item for item in documents if isinstance(item[0], bytes)]
        count = create_vectorstore(extra_documents=extra_documents)
        return count

    index = faiss.read_index(str(INDEX_PATH))
    records = _read_records(RECORDS_PATH)
    embeddings = _encode(new_records)
    if embeddings.shape[1] != index.d:
        extra_documents = [item for item in documents if isinstance(item[0], bytes)]
        return create_vectorstore(extra_documents=extra_documents)
    index.add(embeddings)
    _write_vectorstore(index, records + new_records)
    return len(new_records)


def remove_document_from_vectorstore(document_id, source_name=None):
    """Remove one PDF's chunks and rebuild the compact index from remaining text."""
    if not INDEX_PATH.exists():
        return 0
    if not RECORDS_PATH.exists():
        # Upgrade the old source-less chunks file while local originals are available.
        create_vectorstore()
    if not RECORDS_PATH.exists():
        return 0

    records = _read_records(RECORDS_PATH)
    removed = [
        record for record in records
        if record.get("document_id") == int(document_id)
        or (
            record.get("document_id") is None
            and source_name
            and record.get("source") == source_name
        )
    ]
    if not removed:
        return 0

    remaining = [record for record in records if record not in removed]
    if not remaining:
        INDEX_PATH.unlink(missing_ok=True)
        RECORDS_PATH.unlink(missing_ok=True)
        load_search_assets.cache_clear()
        return len(removed)

    embeddings = _encode(remaining)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _write_vectorstore(index, remaining)
    return len(removed)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _read_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                if isinstance(record, dict) and record.get("text"):
                    records.append(record)
            except json.JSONDecodeError:
                continue
    return records


@lru_cache(maxsize=4)
def load_search_assets(index_path, index_mtime_ns, records_path, records_mtime_ns):
    """Keep the FAISS index and BM25 statistics warm until the corpus changes."""
    index = faiss.read_index(index_path)
    records = _read_records(Path(records_path))
    if not records and LEGACY_CHUNKS_PATH.exists():
        records = [
            {"source": "Unknown document", "page": None, "text": line.strip()}
            for line in LEGACY_CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    tokenized = tuple(tuple(tokenize(record["text"])) for record in records)
    term_frequencies = tuple(Counter(tokens) for tokens in tokenized)
    document_frequency = Counter(
        term for frequencies in term_frequencies for term in frequencies
    )
    average_length = (
        sum(map(len, tokenized)) / len(tokenized) if tokenized else 0
    )
    return index, tuple(records), tokenized, term_frequencies, document_frequency, average_length


def _retrieval_cache_key(query, top_k, index_stat, records_stat):
    digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    corpus_version = f"{index_stat.st_mtime_ns}:{records_stat.st_mtime_ns}"
    return f"{RETRIEVAL_CACHE_PREFIX}{corpus_version}:{top_k}:{digest}"


def _format_source(record):
    source = record.get("source", "Unknown document")
    page = record.get("page")
    page_text = f", page {page}" if page else ""
    return f"[Source: {source}{page_text}]\n{record['text']}"


def search_documents(query, top_k=3):
    """Combine BM25 and FAISS with RRF and cache repeated retrievals in Redis."""
    if not query or not INDEX_PATH.exists():
        return []
    top_k = max(1, min(int(top_k), 20))
    metadata_path = RECORDS_PATH if RECORDS_PATH.exists() else LEGACY_CHUNKS_PATH
    if not metadata_path.exists():
        return []

    index_stat = INDEX_PATH.stat()
    records_stat = metadata_path.stat()
    cache = get_retrieval_cache()
    cache_key = _retrieval_cache_key(query, top_k, index_stat, records_stat)
    if cache:
        try:
            cached = cache.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as exc:
            logger.debug("Redis retrieval cache read failed: %s", exc)

    index, records, tokenized, term_frequencies, document_frequency, average_length = (
        load_search_assets(
            str(INDEX_PATH), index_stat.st_mtime_ns,
            str(metadata_path), records_stat.st_mtime_ns,
        )
    )
    if not records:
        return []

    query_embedding = _encode([{"text": query}])
    candidate_count = min(len(records), max(top_k * 5, 20))
    _, semantic_indices = index.search(query_embedding, candidate_count)
    semantic_ranking = [int(i) for i in semantic_indices[0] if 0 <= i < len(records)]

    query_frequencies = Counter(tokenize(query))
    lexical_scores = []
    k1, b = 1.5, 0.75
    total_documents = len(tokenized)
    for doc_id, tokens in enumerate(tokenized):
        frequencies = term_frequencies[doc_id]
        length = len(tokens)
        score = 0.0
        for term, query_frequency in query_frequencies.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_count = document_frequency.get(term, 0)
            idf = math.log(1 + (total_documents - document_count + 0.5) / (document_count + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / max(average_length, 1))
            score += idf * frequency * (k1 + 1) / denominator * query_frequency
        if score > 0:
            lexical_scores.append((score, doc_id))
    lexical_ranking = [doc_id for _, doc_id in sorted(lexical_scores, reverse=True)]

    fusion_scores = {}
    for ranking in (semantic_ranking, lexical_ranking):
        for rank, doc_id in enumerate(ranking, start=1):
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0.0) + 1 / (60 + rank)
    ordered_ids = sorted(fusion_scores, key=fusion_scores.get, reverse=True)
    ranked_ids = []
    if top_k > 3:
        source_counts = Counter()
        for doc_id in ordered_ids:
            source = records[doc_id].get("source", "Unknown document")
            if source_counts[source] >= 2:
                continue
            ranked_ids.append(doc_id)
            source_counts[source] += 1
            if len(ranked_ids) == top_k:
                break
    if len(ranked_ids) < top_k:
        ranked_ids.extend(doc_id for doc_id in ordered_ids if doc_id not in ranked_ids)
        ranked_ids = ranked_ids[:top_k]
    results = [_format_source(records[doc_id]) for doc_id in ranked_ids]

    if cache:
        try:
            cache.setex(cache_key, RETRIEVAL_CACHE_TTL_SECONDS, json.dumps(results, ensure_ascii=False))
        except Exception as exc:
            logger.debug("Redis retrieval cache write failed: %s", exc)
    return results


if __name__ == "__main__":
    print(f"Indexed {create_vectorstore()} page-aware chunks.")
