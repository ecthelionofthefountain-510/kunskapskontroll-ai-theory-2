# Streamlit UI for Tolkien RAG Chatbot

from __future__ import annotations

import os
import sys
import time
import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make imports work when running: streamlit run src/web.py
APP_ROOT = Path(__file__).resolve().parents[1]  # src -> tolkien-rag-chatbot
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from src.rag import answer_question, get_db, to_float
except ModuleNotFoundError:
    # Fallback if someone runs from inside src/ in a weird way
    from rag import answer_question, get_db, to_float


# --- Helpers --------------------------------------------------------


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def is_vector_db_present(persist_dir: Path) -> bool:
    if not persist_dir.is_dir():
        return False
    if (persist_dir / "chroma.sqlite3").exists():
        return True
    # Best-effort fallback for alternative Chroma layouts
    try:
        return any(persist_dir.iterdir())
    except Exception:
        return False


def inject_focus_css(focus_color: str) -> None:
    st.markdown(
        f"""
        <style>
            div[data-testid="stChatInput"] > div:focus-within {{
                box-shadow: 0 0 0 1px {focus_color} !important;
                border-color: {focus_color} !important;
            }}

            /* Chat messages: card feel */
            [data-testid="stChatMessage"] {{
                border-radius: 16px;
                padding: 12px 14px;
                margin-bottom: 10px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                background: rgba(255, 255, 255, 0.03);
                backdrop-filter: blur(6px);
            }}

            /* Subtle accent on assistant messages (best-effort selector) */
            [data-testid="stChatMessage"].stChatMessage--assistant {{
                border-color: color-mix(in srgb, {focus_color} 35%, rgba(255, 255, 255, 0.06));
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_fonts() -> None:
    font_path = APP_ROOT / "assets" / "fonts" / "Aniron.ttf"
    if not font_path.exists():
        return

    font_b64 = base64.b64encode(font_path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: "Aniron";
            src: url("data:font/ttf;base64,{font_b64}") format("truetype");
            font-weight: normal;
            font-style: normal;
        }}

        .tolkien-title {{
            font-family: "Aniron", serif !important;
            letter-spacing: 1.5px;
            font-size: 2.4rem;
            margin-bottom: 0.2em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def typewriter(text: str, speed_ms: int) -> None:
    box = st.empty()
    out = ""
    delay = max(0.001, speed_ms / 1000.0)

    for i in range(0, len(text), 3):
        out += text[i : i + 3]
        box.markdown(out + "▌")
        time.sleep(delay)

    box.markdown(out)


def display_source_label(source: str) -> str:
    try:
        return Path(source).name
    except Exception:
        return source


def render_message(role: str, content: str, sources: list[str], *, avatar: str) -> None:
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if sources:
            with st.expander("Sources", expanded=False):
                for s in sources:
                    st.write(f"- {display_source_label(s)}")


def render_used_docs(docs, *, max_chars: int = 900) -> None:
    if not docs:
        st.write("Inga chunks användes (inga träffar).")
        return

    for i, d in enumerate(docs, start=1):
        src = str(d.metadata.get("source", "unknown"))
        chunk_idx = d.metadata.get("chunk_index", "?")
        label = display_source_label(src)

        with st.expander(f"{i}. {label} (chunk {chunk_idx})", expanded=False):
            text = d.page_content or ""
            st.code(text[:max_chars] + ("..." if len(text) > max_chars else ""))


# --- App ------------------------------------------------------------


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="Tolkien RAG Chat")
    inject_fonts()

    st.markdown(
        '<h1 class="tolkien-title">Tolkien RAG Chatbot</h1>', unsafe_allow_html=True
    )

    # UI settings
    accent = env("UI_ACCENT_COLOR", "#22c55e")
    inject_focus_css(accent)

    user_avatar = env("UI_USER_AVATAR", "👤")
    bot_avatar = env("UI_ASSISTANT_AVATAR", "🧙")

    # RAG config (paths + models)
    persist_dir = Path(env("CHROMA_PERSIST_DIR", str(APP_ROOT / "data" / "chroma")))
    if not persist_dir.is_absolute():
        persist_dir = APP_ROOT / persist_dir
    collection_name = env("CHROMA_COLLECTION", "tolkien_lore")
    default_chat_model = env("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    default_embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    default_threshold = to_float(env("RAG_RELEVANCE_THRESHOLD", "0.35"), 0.35)

    # Sidebar

    with st.sidebar:
        st.image(str(APP_ROOT / "assets" / "gandalf.gif"), use_container_width=True)
        st.divider()

    with st.sidebar:
        st.header("Settings")

        mode = st.selectbox(
            "Mode",
            ["Balanced", "Strict (less hallucinations)", "Loose (more hits)"],
            index=0,
        )

        if mode == "Strict (less hallucinations)":
            k = 3
            threshold = 0.45
        elif mode == "Loose (more hits)":
            k = 6
            threshold = 0.25
        else:
            k = 4
            threshold = float(default_threshold)

        with st.expander("Advanced", expanded=False):
            k = st.slider("Number of chunks (k)", 1, 12, int(k), 1)
            threshold = st.slider(
                "Relevance threshold", 0.0, 1.0, float(threshold), 0.01
            )

        st.divider()
        show_sources = st.toggle("Show sources", value=True)
        show_context = st.toggle("Show context (top chunks)", value=False)

        st.caption("Models")
        st.code(
            f"Chat: {default_chat_model}\nEmbeddings: {default_embedding_model}",
            language=None,
        )

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["last_topic"] = None
            st.session_state["last_language"] = None
            st.rerun()

    # Session state
    st.session_state.setdefault("history", [])  # list[(role, content, sources)]
    st.session_state.setdefault("last_topic", None)
    st.session_state.setdefault("last_language", None)

    # Ensure vector DB exists (deployment-friendly)
    if not is_vector_db_present(persist_dir):
        st.warning("No vector DB found yet.")
        st.caption(
            "On deploy you usually need to build the index from data/raw/. "
            "This requires OPENAI_API_KEY (Streamlit secrets/env)."
        )

        auto_build = env("AUTO_BUILD_VECTOR_DB", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        st.session_state.setdefault("db_build_attempted", False)

        col1, col2 = st.columns(2)
        with col1:
            build_clicked = st.button("Build vector DB", use_container_width=True)
        with col2:
            rebuild_clicked = st.button("Rebuild vector DB", use_container_width=True)

        should_build = (
            build_clicked
            or rebuild_clicked
            or (auto_build and not st.session_state["db_build_attempted"])
        )
        if should_build:
            st.session_state["db_build_attempted"] = True
            try:
                persist_dir.mkdir(parents=True, exist_ok=True)
                with st.spinner("Building vector DB (this may take a few minutes)…"):
                    # Lazy import to avoid slowing down normal app startup.
                    try:
                        from src.ingest import build_index
                    except ModuleNotFoundError:
                        from ingest import build_index

                    raw_dir = APP_ROOT / "data" / "raw"
                    build_index(
                        raw_dir=raw_dir,
                        persist_dir=persist_dir,
                        collection=collection_name,
                        embedding_model=default_embedding_model,
                        rebuild=bool(rebuild_clicked),
                        chunk_size=900,
                        chunk_overlap=150,
                    )

                st.success("Vector DB built. Reloading…")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to build vector DB: {e}")
                st.stop()

        st.stop()

    # Cache DB
    @st.cache_resource(show_spinner=False)
    def _db(persist: str, collection: str, embed_model: str):
        return get_db(
            persist_dir=persist,
            collection_name=collection,
            embedding_model=embed_model,
        )

    db = _db(str(persist_dir), collection_name, default_embedding_model)

    # Render history
    for role, content, sources in st.session_state["history"]:
        avatar = bot_avatar if role == "assistant" else user_avatar
        render_message(role, content, sources if show_sources else [], avatar=avatar)

    # Input
    question = st.chat_input("What's on your mind …")
    if not question:
        return

    # Echo user
    st.session_state["history"].append(("user", question, []))
    render_message("user", question, [], avatar=user_avatar)

    # Answer
    with st.chat_message("assistant", avatar=bot_avatar):
        with st.spinner("Letar i källor..."):
            result = answer_question(
                db=db,
                question=question,
                last_topic=st.session_state["last_topic"],
                last_language=st.session_state["last_language"],
                k=int(k),
                threshold=float(threshold),
                chat_model=default_chat_model,
            )

        st.session_state["last_topic"] = result.topic
        st.session_state["last_language"] = result.language

        typewriter(result.answer, speed_ms=10)

        if show_sources and result.sources:
            with st.expander("Sources", expanded=False):
                for s in result.sources:
                    st.write(f"- {display_source_label(s)}")

        if show_context:
            st.markdown("**Retrieved context (top chunks)**")
            render_used_docs(getattr(result, "used_docs", []))

    st.session_state["history"].append(
        ("assistant", result.answer, result.sources if show_sources else [])
    )


if __name__ == "__main__":
    main()
