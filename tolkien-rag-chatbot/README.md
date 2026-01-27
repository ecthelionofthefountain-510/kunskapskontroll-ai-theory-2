# Tolkien RAG Chatbot

En enkel RAG-baserad chattbot som indexerar `.txt`-filer i `data/raw/` till en vektordatabas (Chroma) och besvarar frågor med stöd i källorna.

Modellen får inte svara fritt, utan endast utifrån innehållet i de indexerade texterna.

---

## Setup

### 1. Skapa och aktivera en venv

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Installera dependencies

```bash
pip install -r requirements.txt
```

### 3. Skapa `.env`

Kopiera `.env.example` → `.env` och fyll i:

```env
OPENAI_API_KEY=din_api_nyckel
```

---

## Bygg index (ingest)

Detta steg läser textfiler, delar upp dem i chunks, skapar embeddings och sparar allt i Chroma.

```bash
python -m src.ingest --rebuild
```

Vanliga flaggor:

- `--chunk-size 900` – storlek på varje textbit
- `--chunk-overlap 150` – överlapp mellan chunks
- `--rebuild` – rensar och bygger om Chroma-index

---

## Starta chatten (terminal)

Ett minimalt gränssnitt för att testa RAG-flödet utan UI-logik.

```bash
python -m src.chat
```

Vanliga flaggor:

- `--k 4` – antal chunks som hämtas
- `--threshold 0.35` – relevans-tröskel
- `--chat-model gpt-4o-mini`
- `--embedding-model text-embedding-3-small`

---

## Webbsida (Streamlit)

Streamlit-appen använder exakt samma RAG-logik som terminalchatten, men med ett grafiskt gränssnitt.

```bash
streamlit run src/web.py
```

---

## Projektstruktur (översikt)

- `src/ingest.py` – bygger vektordatabasen från textfiler  
- `src/rag.py` – RAG-logik (retrieval, prompt, svar)  
- `src/chat.py` – terminalbaserat gränssnitt  
- `src/web.py` – Streamlit-UI  
- `data/raw/` – källtexter  
- `data/chroma/` – genererad vektordatabas (ej versionshanterad)

---

## Varför dessa val (kopplat till RAG)

- **Chunking + overlap**  
  Ger bättre träffar vid retrieval men kräver viss tuning.

- **Relevans-tröskel**  
  Minskar hallucinationer genom att hellre säga  
  “hittar inte i källor” om inget relevant material hittas.

- **Metadata + chunk-ID**  
  Gör källhänvisningar stabila och spårbara.

---

## Tips: enkel evaluering

För att utvärdera RAG-flödet kan man testa att:

- ställa frågor där svaret finns i `data/raw/`
- ställa frågor där svaret inte finns i källorna

Boten ska då antingen svara korrekt med källor,
eller tydligt säga att den inte hittar stöd i sina källor.

---

## Notering

Projektet är en proof-of-concept med fokus på att visa och förstå hela RAG-kedjan:
data → embeddings → retrieval → svar.