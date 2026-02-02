# Terminal chat

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.rag import answer_question, get_db


# --- CLI ------------------------------------------------------------

# Build command line interface
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tolkien RAG-chat (Chroma + OpenAI)")
    p.add_argument("--k", type=int, default=4, help="Number of retrieved documents")
    p.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.35")),
        help="Minimum relevance threshold (0-1) for retrieved documents",
    )
    p.add_argument(
        "--chat-model",
        type=str,
        default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        help="OpenAI chat-modell",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embeddings-modell",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=os.getenv("RAG_COLLECTION", "tolkien_lore"),
        help="Chroma collection name",
    )
    p.add_argument(
        "--persist-dir",
        type=str,
        default=os.getenv("RAG_PERSIST_DIR", "data/chroma"),
        help="Chroma persist directory",
    )
    return p.parse_args()


# --- App ------------------------------------------------------------


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.path.isdir(args.persist_dir):
        raise FileNotFoundError("Chroma persist directory not found. Run ingest.py first.")

    db = get_db(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )

    print("Tolkien RAG-chat. Type 'exit' to close.\n")

    last_topic: str | None = None
    last_language: str | None = None

    while True:
        question = input("Du: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        # Send question to RAG pipeline
        result = answer_question(
            db=db,
            question=question,
            last_topic=last_topic,
            last_language=last_language,
            k=args.k,
            threshold=args.threshold,
            chat_model=args.chat_model,
            temperature=0.0,
        )

        last_topic = result.topic
        last_language = result.language

        print("\nBot:", result.answer)
        if result.sources:
            print("\nSources:")
            for s in result.sources:
                print(" -", s)
        print()


if __name__ == "__main__":
    main()