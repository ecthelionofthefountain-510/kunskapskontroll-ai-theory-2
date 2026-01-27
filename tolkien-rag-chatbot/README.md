# Tolkien RAG Chatbot

En enkel RAG-chattbot som indexerar `.txt`-filer i `data/raw/` till Chroma och svarar på svenska med källor.

## Setup

1. Skapa och aktivera en venv

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Installera dependencies

```bash
pip install -r requirements.txt
```

3. Skapa `.env`

Kopiera `.env.example` → `.env` och fyll i `OPENAI_API_KEY`.

## Bygg index (ingest)

```bash
python -m src.ingest --rebuild
```

Vanliga flaggor:

- `--chunk-size 900` och `--chunk-overlap 150` (fixed-length chunking med overlap)
- `--rebuild` för att rensa och bygga om Chroma-index

## Starta chatten

```bash
python -m src.chat
```

## Webbsida (Streamlit)

```bash
streamlit run src/web.py
```

Vanliga flaggor:

- `--k 4` antal chunks som hämtas
- `--threshold 0.35` relevans-tröskel (om för låg → boten säger att den inte hittar i källor)
- `--chat-model gpt-4o-mini`
- `--embedding-model text-embedding-3-small`

## Varför dessa val (kopplat till RAG)

- **Chunking + overlap**: bättre träffar vid retrieval, men kräver tuning (chunk_size/overlap).
- **Relevans-tröskel**: minskar hallucinationer genom att hellre säga “hittar inte i källor”.
- **Metadata + chunk-id**: gör källhänvisningar stabilare och förbättrar spårbarhet.

## Tips: enkel evaluering

För att utvärdera RAG: skapa en liten lista testfrågor + “ideal_answer” och kontrollera att boten:

- svarar korrekt när svaret finns i `data/raw/`
- svarar “hittar inte i mina källor” när det inte finns stöd i kontexten
