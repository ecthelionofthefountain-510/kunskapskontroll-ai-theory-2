# Build DB from .txt files

from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import get_db


# --- Loading --------------------------------------------------------


def _read_text_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def load_txt_documents(raw_dir: Path) -> list[Document]:
    docs: list[Document] = []

    for path in sorted(raw_dir.rglob("*.txt")):
        text = _read_text_best_effort(path).strip()
        if not text:
            continue

        try:
            source = path.relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            source = path.as_posix()

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "title": path.stem.replace("_", " ").strip(),
                },
            )
        )

    return docs


# --- Chunk IDs ------------------------------------------------------


def assign_chunk_ids(chunks: list[Document]) -> list[str]:
    per_source_counter: dict[str, int] = {}
    ids: list[str] = []

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown_source"))
        idx = per_source_counter.get(source, 0)
        per_source_counter[source] = idx + 1

        chunk.metadata["chunk_index"] = idx
        ids.append(f"{source}::chunk_{idx}")

    return ids


# --- CLI ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build/update Chroma index from .txt files (Tolkien RAG-chat)"
    )
    p.add_argument(
        "--rebuild", action="store_true", help="Clean rebuild the index from scratch"
    )
    p.add_argument("--chunk-size", type=int, default=900, help="Chunk-size (digits)")
    p.add_argument("--chunk-overlap", type=int, default=150, help="Overlap (digits)")
    p.add_argument(
        "--collection",
        type=str,
        default=os.getenv(
            "CHROMA_COLLECTION", os.getenv("RAG_COLLECTION", "tolkien_lore")
        ),
        help="Chroma collection name",
    )
    p.add_argument(
        "--persist-dir",
        type=str,
        default=os.getenv(
            "CHROMA_PERSIST_DIR", os.getenv("RAG_PERSIST_DIR", "data/chroma")
        ),
        help="Directory to Chroma persist-mapp",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embeddings-modell",
    )
    return p.parse_args()


# --- Pipeline -------------------------------------------------------


def build_index(
    *,
    raw_dir: Path,
    persist_dir: Path,
    collection: str,
    embedding_model: str,
    rebuild: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int]:
    """Build/update the Chroma index.

    Returns: (num_docs, num_chunks)
    """

    # Disable Chroma telemetry noise early
    os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Cant find data/raw/ directory: {raw_dir}")

    if rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)

    docs = load_txt_documents(raw_dir)
    if not docs:
        raise ValueError(f"No .txt documents found in: {raw_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    ids = assign_chunk_ids(chunks)

    db = get_db(
        persist_dir=str(persist_dir),
        collection_name=collection,
        embedding_model=embedding_model,
    )
    db.add_documents(chunks, ids=ids)

    return len(docs), len(chunks)


def main() -> None:
    load_dotenv()
    args = parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    persist_dir = Path(args.persist_dir)
    if not persist_dir.is_absolute():
        persist_dir = PROJECT_ROOT / persist_dir

    num_docs, num_chunks = build_index(
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        collection=args.collection,
        embedding_model=args.embedding_model,
        rebuild=bool(args.rebuild),
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
    )

    print("Done!")
    print(f"Document: {num_docs}")
    print(f"Chunks:   {num_chunks}")
    print(f"Index saved in: {persist_dir}")


if __name__ == "__main__":
    main()
