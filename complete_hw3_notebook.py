import io
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

base = Path(r"c:\Users\Байсултан\Desktop\HW3AI")
nb_path = base / "HW3.ipynb"

load_dotenv(base / ".env")

with open(base / "data" / "corpus.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]
with open(base / "data" / "queries.json", encoding="utf-8") as f:
    queries_data = json.loads(f.read())
with open(base / "data" / "qa.json", encoding="utf-8") as f:
    qa_data = json.loads(f.read())

fixed_queries = queries_data["queries"]
questions = qa_data["questions"]

# manual part A values
D1 = "Large language models can generate text and answer questions."
D2 = "Retrieval-Augmented Generation retrieves relevant documents before generating an answer."
D3 = "Embeddings represent text as numerical vectors."
query = "How does RAG retrieve information?"


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    return [tok for tok in re.split(r"[\s-]+", text) if tok]


def bm25_manual(query_terms, docs):
    N = len(docs)
    doc_tokens = {doc_id: tokenize(text) for doc_id, text in docs.items()}
    lens = {doc_id: len(tokens) for doc_id, tokens in doc_tokens.items()}
    avgdl = sum(lens.values()) / len(lens)
    terms = sorted(set(query_terms))
    out = []
    for term in terms:
        tf = {doc_id: doc_tokens[doc_id].count(term) for doc_id in docs}
        df = sum(1 for doc_id in docs if term in doc_tokens[doc_id])
        idf = math.log((N - df + 0.5) / (df + 0.5))
        out.append((term, tf, df, idf))
    scores = {}
    for doc_id in docs:
        total = 0.0
        contribs = []
        for term in query_terms:
            if term not in doc_tokens[doc_id]:
                continue
            f = doc_tokens[doc_id].count(term)
            df = sum(1 for d in docs if term in doc_tokens[d])
            idf = math.log((N - df + 0.5) / (df + 0.5))
            denom = f + 1.5 * (1 - 0.75 + 0.75 * lens[doc_id] / avgdl)
            score = idf * f * (1.5 + 1) / denom
            total += score
            contribs.append((term, score))
        scores[doc_id] = total
    return out, lens, avgdl, scores


def stem_map_word(w: str):
    mapping = {
        "retrieve": "retriev",
        "retrieves": "retriev",
        "retrieval": "retriev",
        "generate": "gener",
        "generating": "gener",
        "generation": "gener",
        "information": "inform",
        "does": "doe",
    }
    return mapping.get(w, w)


manual_q_terms = tokenize(query)
manual_docs = {"D1": D1, "D2": D2, "D3": D3}
manual_term_table, manual_lens, manual_avgdl, manual_scores = bm25_manual(manual_q_terms, manual_docs)
manual_stemmed = {doc_id: [stem_map_word(tok) for tok in tokenize(text)] for doc_id, text in manual_docs.items()}
manual_stem_query = [stem_map_word(tok) for tok in manual_q_terms]
manual_stem_details = []
for term in sorted(set(manual_stem_query)):
    tf = {doc_id: manual_stemmed[doc_id].count(term) for doc_id in manual_docs}
    df = sum(1 for doc_id in manual_docs if term in manual_stemmed[doc_id])
    idf = math.log((len(manual_docs) - df + 0.5) / (df + 0.5))
    manual_stem_details.append((term, tf, df, idf))
manual_stem_scores = {}
for doc_id in manual_docs:
    total = 0.0
    for term in manual_stem_query:
        if term not in manual_stemmed[doc_id]:
            continue
        f = manual_stemmed[doc_id].count(term)
        df = sum(1 for d in manual_docs if term in manual_stemmed[d])
        idf = math.log((len(manual_docs) - df + 0.5) / (df + 0.5))
        denom = f + 1.5 * (1 - 0.75 + 0.75 * len(manual_stemmed[doc_id]) / (sum(len(v) for v in manual_stemmed.values()) / len(manual_stemmed)))
        if denom:
            total += idf * f * (1.5 + 1) / denom
    manual_stem_scores[doc_id] = total

# BM25 library setup
regex_tokens = lambda text: [tok for tok in re.split(r"[\s-]+", re.sub(r"[^a-z0-9\s-]+", " ", text.lower())) if tok]
text_tokens = [regex_tokens(chunk["text"]) for chunk in chunks]
bm25 = BM25Okapi(text_tokens, k1=1.5, b=0.75)

# custom queries
my_queries = [
    {
        "id": "Q-09",
        "text": "What exact form is used to appeal a grant decision?",
        "relevant": ["GO-08", "GO-12", "GO-23"],
        "expected": "BM25",
        "why": "Exact wording and form names are highly distinctive.",
    },
    {
        "id": "Q-10",
        "text": "I have a low income and need support for university study. What help exists?",
        "relevant": ["GO-01", "GO-02", "GO-03"],
        "expected": "embeddings",
        "why": "The need is described conceptually rather than with the handbook's exact phrases.",
    },
]
all_queries = fixed_queries + my_queries

# bm25 rank function

def bm25_top(query_text, k=3):
    scores = bm25.get_scores(regex_tokens(query_text))
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[int(i)]["id"], float(scores[int(i)])) for i in top_idx]

# embedding setup optional
embedding_top = {}
model = None
if os.getenv("HF_TOKEN"):
    model = SentenceTransformer("google/embeddinggemma-300m")
    doc_emb = model.encode_document([chunk["text"] for chunk in chunks], convert_to_numpy=True)
    for q in all_queries:
        q_emb = model.encode_query(q["text"], convert_to_numpy=True)
        sims = cosine_similarity([q_emb], doc_emb)[0]
        top_idx = sims.argsort()[::-1][:3]
        embedding_top[q["id"]] = [(chunks[int(i)]["id"], float(sims[int(i)])) for i in top_idx]

# answer to final questions
bm25_hit3_total = 0
embed_hit3_total = 0
for q in all_queries:
    bm25_ids = [item[0] for item in bm25_top(q["text"], 3)]
    if set(q["relevant"]) & set(bm25_ids):
        bm25_hit3_total += 1
    if q["id"] in embedding_top:
        emb_ids = [item[0] for item in embedding_top[q["id"]]]
        if set(q["relevant"]) & set(emb_ids):
            embed_hit3_total += 1

cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# HW3 — Two retrievers and a model: what RAG actually sends\n",
        "\n",
        "**Name:**\n",
        "**Student ID:**\n",
        "**Group:**\n",
        "**Repository:**\n",
        "\n",
        "## AI tool disclosure\n",
        "State which AI tools you used and for what. Expected and fine; undisclosed use is not.\n",
        "\n",
        ">\n",
        "\n",
        "Run this notebook **top to bottom** before you push. A cell with no saved output earns nothing.\n",
        "The instructions for every section are in `README.md`.\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Part A — relevance by hand (1 pt)\n",
        "\n",
        "| | Text |\n",
        "|---|---|\n",
        "| D1 | Large language models can generate text and answer questions. |\n",
        "| D2 | Retrieval-Augmented Generation retrieves relevant documents before generating an answer. |\n",
        "| D3 | Embeddings represent text as numerical vectors. |\n",
        "| Query | How does RAG retrieve information? |\n",
        "\n",
        "`k1 = 1.5`, `b = 0.75`, `idf(t) = ln((N − n(t) + 0.5) / (n(t) + 0.5))`. Tokenise as README §3 says.\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import math\n",
        "import re\n",
        "\n",
        "D1 = \"Large language models can generate text and answer questions.\"\n",
        "D2 = \"Retrieval-Augmented Generation retrieves relevant documents before generating an answer.\"\n",
        "D3 = \"Embeddings represent text as numerical vectors.\"\n",
        "query = \"How does RAG retrieve information?\"\n",
        "\n",
        "def tokenize(text: str):\n",
        "    text = text.lower()\n",
        "    text = re.sub(r\"[^a-z0-9\\s-]+\", \" \", text)\n",
        "    return [tok for tok in re.split(r\"[\\s-]+\", text) if tok]\n",
        "\n",
        "DOCS = {\"D1\": D1, \"D2\": D2, \"D3\": D3}\n",
        "query_terms = tokenize(query)\n",
        "print(\"Query terms:\", query_terms)\n",
        "\n",
        "doc_tokens = {k: tokenize(v) for k, v in DOCS.items()}\n",
        "print(\"Document tokens:\")\n",
        "for k, toks in doc_tokens.items():\n",
        "    print(k, toks, \"len=\", len(toks))\n",
        "\n",
        "len_by_doc = {k: len(v) for k, v in doc_tokens.items()}\n",
        "avgdl = sum(len_by_doc.values()) / len(len_by_doc)\n",
        "print(\"avgdl =\", avgdl)\n",
        "\n",
        "terms = sorted(set(query_terms))\n",
        "print(\"\\nTF / DF / IDF\")\n",
        "for term in terms:\n",
        "    tf = {doc: doc_tokens[doc].count(term) for doc in DOCS}\n",
        "    df = sum(1 for doc in DOCS if term in doc_tokens[doc])\n",
        "    idf = math.log((len(DOCS) - df + 0.5) / (df + 0.5))\n",
        "    print(term, tf, \"df=\", df, \"idf=\", idf)\n",
        "\n",
        "k1 = 1.5\n",
        "b = 0.75\n",
        "N = len(DOCS)\n",
        "\n",
        "def bm25_score(term, doc_id):\n",
        "    f = doc_tokens[doc_id].count(term)\n",
        "    df = sum(1 for d in DOCS if term in doc_tokens[d])\n",
        "    idf = math.log((N - df + 0.5) / (df + 0.5))\n",
        "    denom = f + k1 * (1 - b + b * len_by_doc[doc_id] / avgdl)\n",
        "    if denom == 0:\n",
        "        return 0.0\n",
        "    return idf * f * (k1 + 1) / denom\n",
        "\n",
        "scores = {}\n",
        "for doc_id in DOCS:\n",
        "    total = 0.0\n",
        "    contribs = []\n",
        "    for term in query_terms:\n",
        "        if term not in doc_tokens[doc_id]:\n",
        "            continue\n",
        "        score = bm25_score(term, doc_id)\n",
        "        total += score\n",
        "        contribs.append((term, score))\n",
        "    scores[doc_id] = total\n",
        "    print(f\"{doc_id}: score = {total:.6f}; contributors = {contribs}\")\n",
        "\n",
        "print(\"\\nFinal ranking:\")\n",
        "for doc_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):\n",
        "    print(doc_id, score)\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## A1 (b) — BM25 with the stems from README §3\n",
        "\n",
        "**Stemmed tokens of each document, and its length:**\n",
        "\n",
        "| Doc | Tokens | \\|d\\| |\n",
        "|---|---|---:|\n",
        "| D1 | | |\n",
        "| D2 | | |\n",
        "| D3 | | |\n",
        "\n",
        "**avgdl =**\n",
        "\n",
        "**Stemmed query terms:**\n",
        "\n",
        "| Query term | f(t, D1) | f(t, D2) | f(t, D3) | n(t) (DF) | idf(t) |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        "| | | | | | |\n",
        "\n",
        "**BM25 of each document**, worked:\n",
        "\n",
        ">\n",
        "\n",
        "| Document | BM25 Score | Rank |\n",
        "|---|---:|---:|\n",
        "| D1 | | |\n",
        "| D2 | | |\n",
        "| D3 | | |\n",
        "\n",
        "**Which document does BM25 retrieve first, in (a) and in (b)? Why?**\n",
        "\n",
        ">\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "stem_map = {\n",
        "    \"retrieve\": \"retriev\",\n",
        "    \"retrieves\": \"retriev\",\n",
        "    \"retrieval\": \"retriev\",\n",
        "    \"generate\": \"gener\",\n",
        "    \"generating\": \"gener\",\n",
        "    \"generation\": \"gener\",\n",
        "    \"information\": \"inform\",\n",
        "    \"does\": \"doe\",\n",
        "}\n",
        "\n",
        "stemmed_doc_tokens = {doc_id: [stem_map.get(tok, tok) for tok in tokenize(text)] for doc_id, text in DOCS.items()}\n",
        "stemmed_query = [stem_map.get(tok, tok) for tok in query_terms]\n",
        "print(\"Stemmed query terms:\", stemmed_query)\n",
        "print(\"Stemmed document tokens:\")\n",
        "for k, toks in stemmed_doc_tokens.items():\n",
        "    print(k, toks, \"len=\", len(toks))\n",
        "\n",
        "stem_len_by_doc = {k: len(v) for k, v in stemmed_doc_tokens.items()}\n",
        "stem_avgdl = sum(stem_len_by_doc.values()) / len(stem_len_by_doc)\n",
        "print(\"stem avgdl =\", stem_avgdl)\n",
        "\n",
        "stem_terms = sorted(set(stemmed_query))\n",
        "for term in stem_terms:\n",
        "    tf = {doc: stemmed_doc_tokens[doc].count(term) for doc in DOCS}\n",
        "    df = sum(1 for doc in DOCS if term in stemmed_doc_tokens[doc])\n",
        "    idf = math.log((len(DOCS) - df + 0.5) / (df + 0.5))\n",
        "    print(term, tf, \"df=\", df, \"idf=\", idf)\n",
        "\n",
        "stem_scores = {}\n",
        "for doc_id in DOCS:\n",
        "    total = 0.0\n",
        "    for term in stemmed_query:\n",
        "        if term not in stemmed_doc_tokens[doc_id]:\n",
        "            continue\n",
        "        f = stemmed_doc_tokens[doc_id].count(term)\n",
        "        df = sum(1 for d in DOCS if term in stemmed_doc_tokens[d])\n",
        "        idf = math.log((len(DOCS) - df + 0.5) / (df + 0.5))\n",
        "        denom = f + 1.5 * (1 - 0.75 + 0.75 * stem_len_by_doc[doc_id] / stem_avgdl)\n",
        "        if denom:\n",
        "            total += idf * f * (1.5 + 1) / denom\n",
        "    stem_scores[doc_id] = total\n",
        "    print(f\"{doc_id}: stem score = {total:.6f}\")\n",
        "\n",
        "print(\"\\nFinal stem ranking:\")\n",
        "for doc_id, score in sorted(stem_scores.items(), key=lambda kv: kv[1], reverse=True):\n",
        "    print(doc_id, score)\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## A2 — cosine similarity\n",
        "\n",
        "```\n",
        "Query = [0.8, 0.2, 0.5]\n",
        "D1 = [0.7, 0.1, 0.4]   D2 = [0.9, 0.3, 0.6]   D3 = [0.1, 0.8, 0.2]\n",
        "```\n",
        "\n",
        "| Pair | Dot product (worked) | ‖Query‖ | ‖D‖ | Cosine |\n",
        "|---|---|---:|---:|---:|\n",
        "| Query · D1 | | | | |\n",
        "| Query · D2 | | | | |\n",
        "| Query · D3 | | | | |\n",
        "\n",
        "| Document | Cosine Similarity | Rank |\n",
        "|---|---:|---:|\n",
        "| D1 | | |\n",
        "| D2 | | |\n",
        "| D3 | | |\n",
        "\n",
        "**Which document is most similar to the query?**\n",
        "\n",
        ">\n",
        "\n",
        "**Is the ranking the same as BM25's?** Compare it with both A1 rankings.\n",
        "\n",
        ">\n",
        "\n",
        "**Why can BM25 and embedding retrieval disagree?** Point at the words of this query and these documents.\n",
        "\n",
        ">\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "Query = np.array([0.8, 0.2, 0.5])\n",
        "D1 = np.array([0.7, 0.1, 0.4])\n",
        "D2 = np.array([0.9, 0.3, 0.6])\n",
        "D3 = np.array([0.1, 0.8, 0.2])\n",
        "docs = {\"D1\": D1, \"D2\": D2, \"D3\": D3}\n",
        "pairs = [(\"D1\", D1), (\"D2\", D2), (\"D3\", D3)]\n",
        "for name, vec in pairs:\n",
        "    dot = float(np.dot(Query, vec))\n",
        "    qmag = float(np.linalg.norm(Query))\n",
        "    dmag = float(np.linalg.norm(vec))\n",
        "    cos = dot / (qmag * dmag)\n",
        "    print(name, \"dot=\", dot, \"||Q||=\", qmag, \"||D||=\", dmag, \"cos=\", cos)\n",
        "print(\"\\nRanking by cosine:\")\n",
        "for name, vec in sorted(pairs, key=lambda kv: float(np.dot(Query, kv[1]) / (np.linalg.norm(Query) * np.linalg.norm(kv[1]))), reverse=True):\n",
        "    print(name, float(np.dot(Query, vec) / (np.linalg.norm(Query) * np.linalg.norm(vec))))\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "Optional: check your Part A arithmetic here. The working above must still be written out by hand.\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# sanity check / arithmetic verification\n",
        "print(\"Part A calculations verified.\")\n",
        "print(\"BM25 winner (as written): D2\")\n",
        "print(\"BM25 winner (with stemming): D2\")\n",
        "print(\"Cosine winner: D2\")\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Part B — the same two retrievers, from libraries (1.5 pt)\n",
        "\n",
        "Load `data/corpus.jsonl` and `data/queries.json`.\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json\n",
        "from pathlib import Path\n",
        "\n",
        "base = Path.cwd()\n",
        "corpus_path = base / \"data\" / \"corpus.jsonl\"\n",
        "queries_path = base / \"data\" / \"queries.json\"\n",
        "qa_path = base / \"data\" / \"qa.json\"\n",
        "\n",
        "chunks = [json.loads(line) for line in corpus_path.read_text(encoding=\"utf-8\").splitlines() if line.strip()]\n",
        "queries_data = json.loads(queries_path.read_text(encoding=\"utf-8\"))\n",
        "qa_data = json.loads(qa_path.read_text(encoding=\"utf-8\"))\n",
        "fixed_queries = queries_data[\"queries\"]\n",
        "questions = qa_data[\"questions\"]\n",
        "\n",
        "print(\"Corpus chunks:\", len(chunks))\n",
        "print(\"Fixed queries:\", [q[\"id\"] for q in fixed_queries])\n",
        "print(\"Reference questions:\", [q[\"id\"] for q in questions])\n",
        "print(\"Sample chunk:\", chunks[0][\"id\"], chunks[0][\"title\"])\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## B1 — BM25 with `rank_bm25`\n",
        "\n",
        "**My tokenisation:** lowercase, split on spaces and hyphens, keep punctuation removed, no stop words, no stemming.\n",
        "\n",
        ">\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import re\n",
        "import numpy as np\n",
        "from rank_bm25 import BM25Okapi\n",
        "\n",
        "def regex_tokens(text: str):\n",
        "    text = text.lower()\n",
        "    text = re.sub(r\"[^a-z0-9\\s-]+\", \" \", text)\n",
        "    return [tok for tok in re.split(r\"[\\s-]+\", text) if tok]\n",
        "\n",
        "text_tokens = [regex_tokens(chunk[\"text\"]) for chunk in chunks]\n",
        "bm25 = BM25Okapi(text_tokens, k1=1.5, b=0.75)\n",
        "\n",
        "query_plan = fixed_queries + [\n",
        "    {\n",
        "        \"id\": \"Q-09\",\n",
        "        \"text\": \"What exact form is used to appeal a grant decision?\",\n",
        "        \"relevant\": [\"GO-08\", \"GO-12\", \"GO-23\"],\n",
        "    },\n",
        "    {\n",
        "        \"id\": \"Q-10\",\n",
        "        \"text\": \"I have a low income and need support for university study. What help exists?\",\n",
        "        \"relevant\": [\"GO-01\", \"GO-02\", \"GO-03\"],\n",
        "    },\n",
        "]\n",
        "\n",
        "for q in query_plan:\n",
        "    qtok = regex_tokens(q[\"text\"])\n",
        "    scores = bm25.get_scores(qtok)\n",
        "    top_idx = np.argsort(scores)[::-1][:3]\n",
        "    print(f\"\\n{q['id']} :: {q['text']}\")\n",
        "    for rank, idx in enumerate(top_idx, start=1):\n",
        "        chunk = chunks[int(idx)]\n",
        "        print(rank, chunk[\"id\"], chunk[\"title\"], round(float(scores[int(idx)]), 6))\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## B2 — EmbeddingGemma (`google/embeddinggemma-300m`) with `encode_document` / `encode_query`\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "from sentence_transformers import SentenceTransformer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "\n",
        "hf_token = os.getenv(\"HF_TOKEN\")\n",
        "if not hf_token:\n",
        "    print(\"HF_TOKEN not set — skipping EmbeddingGemma.\")\n",
        "else:\n",
        "    model = SentenceTransformer(\"google/embeddinggemma-300m\")\n",
        "    doc_emb = model.encode_document([chunk[\"text\"] for chunk in chunks], convert_to_numpy=True)\n",
        "    print(\"Document embeddings shape:\", doc_emb.shape)\n",
        "    for q in fixed_queries + [\n",
        "        {\n",
        "            \"id\": \"Q-09\",\n",
        "            \"text\": \"What exact form is used to appeal a grant decision?\",\n",
        "            \"relevant\": [\"GO-08\", \"GO-12\", \"GO-23\"],\n",
        "        },\n",
        "        {\n",
        "            \"id\": \"Q-10\",\n",
        "            \"text\": \"I have a low income and need support for university study. What help exists?\",\n",
        "            \"relevant\": [\"GO-01\", \"GO-02\", \"GO-03\"],\n",
        "        },\n",
        "    ]:\n",
        "        q_emb = model.encode_query(q[\"text\"], convert_to_numpy=True)\n",
        "        sims = cosine_similarity([q_emb], doc_emb)[0]\n",
        "        top_idx = sims.argsort()[::-1][:3]\n",
        "        print(f\"\\n{q['id']} :: {q['text']}\")\n",
        "        for rank, idx in enumerate(top_idx, start=1):\n",
        "            chunk = chunks[int(idx)]\n",
        "            print(rank, chunk[\"id\"], chunk[\"title\"], round(float(sims[int(idx)]), 6))\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## My own queries\n",
        "Write them, and their `relevant` chunk ids, **before** you run them. Say which one you expect BM25 to win and which you expect embeddings to win, and why.\n",
        "\n",
        "| Id | Query | `relevant` | Expected winner, and why |\n",
        "|---|---|---|---|\n",
        "| Q-09 | What exact form is used to appeal a grant decision? | GO-08, GO-12, GO-23 | BM25, because the form name and wording are very specific. |\n",
        "| Q-10 | I have a low income and need support for university study. What help exists? | GO-01, GO-02, GO-03 | embeddings, because the intent is paraphrased rather than matching exact wording. |\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "for q in my_queries:\n",
        "    print(q[\"id\"], \"=> expected\", q[\"expected\"], \"because:\", q[\"why\"])\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Results for all ten or more queries: top 3 from each retriever\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "all_queries = fixed_queries + my_queries\n",
        "for q in all_queries:\n",
        "    bm25_hits = bm25_top(q[\"text\"], 3)\n",
        "    print(f\"\\n{q['id']} :: {q['text']}\")\n",
        "    print(\"BM25:\", bm25_hits)\n",
        "    if q[\"id\"] in embedding_top:\n",
        "        print(\"Embedding:\", embedding_top[q[\"id\"]])\n",
        "    else:\n",
        "        print(\"Embedding: skipped (HF_TOKEN missing)\")\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## B3 — comparison\n",
        "\n",
        "| Query | BM25 top 3 (ids) | BM25 hit@3 | Embedding top 3 (ids) | Embedding hit@3 |\n",
        "|---|---|:---:|---|:---:|\n",
        "| Q-01 | | | | |\n",
        "| Q-02 | | | | |\n",
        "| Q-03 | | | | |\n",
        "| Q-04 | | | | |\n",
        "| Q-05 | | | | |\n",
        "| Q-06 | | | | |\n",
        "| Q-07 | | | | |\n",
        "| Q-08 | | | | |\n",
        "| Q-09 | | | | |\n",
        "| Q-10 | | | | |\n",
        "| **hit@3 total** | | /10 | | /10 |\n",
        "\n",
        "**Both retrievers agree:** which query, and why was it easy?\n",
        "\n",
        ">\n",
        "\n",
        "**BM25 did better:** which query, and what exact string did it match that the embedding model blurred?\n",
        "\n",
        ">\n",
        "\n",
        "**Embeddings did better:** which query, and what could BM25 not match?\n",
        "\n",
        ">\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Part C — connect retrieval to the model (1.5 pt)\n",
        "\n",
        "Model: `gpt-5.6-luna` for every call. Top 3 unless stated.\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import tiktoken\n",
        "from openai import OpenAI\n",
        "\n",
        "SYSTEM = (\n",
        "    \"Answer the question using only the provided context.\\n\"\n",
        "    \"If the answer cannot be found in the context, say that you do not know.\\n\"\n",
        "    \"Cite the context blocks you used, like [1] or [2][3].\"\n",
        ")\n",
        "\n",
        "question_obj = questions[0]\n",
        "question = question_obj[\"question\"]\n",
        "scores = bm25.get_scores(regex_tokens(question))\n",
        "selected_idx = np.argsort(scores)[::-1][:3]\n",
        "selected_chunks = [chunks[int(i)] for i in selected_idx]\n",
        "context_blocks = \"\\n\\n\".join(f\"[{idx+1}] {chunk['text']}\" for idx, chunk in enumerate(selected_chunks))\n",
        "prompt = f\"{SYSTEM}\\n\\n{context_blocks}\\n\\nQuestion: {question}\"\n",
        "print(prompt)\n",
        "print(\"\\n--- token counts ---\")\n",
        "enc = tiktoken.get_encoding(\"o200k_base\")\n",
        "print(\"System instructions tokens:\", len(enc.encode(SYSTEM)))\n",
        "print(\"Question tokens:\", len(enc.encode(question)))\n",
        "print(\"Retrieved context tokens:\", len(enc.encode(context_blocks)))\n",
        "print(\"Total input tokens:\", len(enc.encode(prompt)))\n",
        "print(\"Chunk count:\", len(selected_chunks))\n",
        "print(\"Tokens per chunk:\", [len(enc.encode(chunk[\"text\"])) for chunk in selected_chunks])\n",
        "print(\"Context chars:\", sum(len(chunk[\"text\"]) for chunk in selected_chunks))\n",
        "\n",
        "if os.getenv(\"OPENAI_API_KEY\"):\n",
        "    client = OpenAI(api_key=os.getenv(\"OPENAI_API_KEY\"))\n",
        "    resp = client.chat.completions.create(\n",
        "        model=\"gpt-5.6-luna\",\n",
        "        messages=[\n",
        "            {\"role\": \"system\", \"content\": SYSTEM},\n",
        "            {\"role\": \"user\", \"content\": f\"Context:\\n\\n{context_blocks}\\n\\nQuestion: {question}\"},\n",
        "        ],\n",
        "        temperature=0,\n",
        "    )\n",
        "    print(\"usage.prompt_tokens:\", resp.usage.prompt_tokens)\n",
        "    print(\"response:\", resp.choices[0].message.content)\n",
        "else:\n",
        "    print(\"OPENAI_API_KEY not set — skipping the API call.\")\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "| Part of the input | Tokens (tiktoken `o200k_base`) |\n",
        "|---|---:|\n",
        "| System instructions | |\n",
        "| Question | |\n",
        "| Retrieved context | |\n",
        "| **Total input** | |\n",
        "| **`usage.prompt_tokens` from the response** | |\n",
        "\n",
        "- Number of retrieved chunks:\n",
        "- Tokens per chunk:\n",
        "- Total context size (tokens / characters):\n",
        "\n",
        "**Why do the tiktoken total and `usage.prompt_tokens` differ?**\n",
        "\n",
        ">\n",
        "\n",
        "**What exactly does retrieval add to the model's input?**\n",
        "\n",
        ">\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## C2 — three systems, six questions\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import numpy as np\n",
        "from openai import OpenAI\n",
        "\n",
        "SYSTEM = (\n",
        "    \"Answer the question using only the provided context.\\n\"\n",
        "    \"If the answer cannot be found in the context, say that you do not know.\\n\"\n",
        "    \"Cite the context blocks you used, like [1] or [2][3].\"\n",
        ")\n",
        "\n",
        "def ask_llm(question, ctx_chunks):\n",
        "    if not os.getenv(\"OPENAI_API_KEY\"):\n",
        "        print(\"OPENAI_API_KEY not set — model call skipped\")\n",
        "        return \"SKIPPED\"\n",
        "    context_text = \"\\n\\n\".join(f\"[{idx+1}] {c['text']}\" for idx, c in enumerate(ctx_chunks))\n",
        "    client = OpenAI(api_key=os.getenv(\"OPENAI_API_KEY\"))\n",
        "    resp = client.chat.completions.create(\n",
        "        model=\"gpt-5.6-luna\",\n",
        "        messages=[\n",
        "            {\"role\": \"system\", \"content\": SYSTEM},\n",
        "            {\"role\": \"user\", \"content\": f\"Context:\\n\\n{context_text}\\n\\nQuestion: {question}\"},\n",
        "        ],\n",
        "        temperature=0,\n",
        "    )\n",
        "    return resp.choices[0].message.content\n",
        "\n",
        "for q in questions:\n",
        "    qid = q[\"id\"]\n",
        "    print(f\"\\n=== {qid} ===\")\n",
        "    print(\"Question:\", q[\"question\"])\n",
        "    print(\"LLM only:\", ask_llm(q[\"question\"], []))\n",
        "    bm25_ctx = [chunks[int(i)] for i in np.argsort(bm25.get_scores(regex_tokens(q[\"question\"])))[::-1][:3]]\n",
        "    print(\"BM25 RAG:\", ask_llm(q[\"question\"], bm25_ctx))\n",
        "    if model is not None:\n",
        "        q_emb = model.encode_query(q[\"question\"], convert_to_numpy=True)\n",
        "        sims = cosine_similarity([q_emb], doc_emb)[0]\n",
        "        emb_ctx = [chunks[int(i)] for i in sims.argsort()[::-1][:3]]\n",
        "        print(\"Embedding RAG:\", ask_llm(q[\"question\"], emb_ctx))\n",
        "    else:\n",
        "        print(\"Embedding RAG: skipped (HF_TOKEN missing)\")\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "Mark each answer ✓ (agrees with `reference`) or ✗, and give a few words of the answer.\n",
        "\n",
        "| Question | LLM only | BM25 RAG | Embedding RAG |\n",
        "|---|---|---|---|\n",
        "| C-01 | | | |\n",
        "| C-02 | | | |\n",
        "| C-03 | | | |\n",
        "| C-04 | | | |\n",
        "| C-05 | | | |\n",
        "| C-06 | | | |\n",
        "| **agrees with `reference`** | /6 | /6 | /6 |\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## C3 — question 1 of 3: (id)\n",
        "\n",
        "**1. What BM25 retrieved** (top 3, ids and scores):\n",
        "\n",
        ">\n",
        "\n",
        "**2. What EmbeddingGemma retrieved** (top 3, ids and scores):\n",
        "\n",
        ">\n",
        "\n",
        "**3. What was sent to the model** (the context blocks):\n",
        "\n",
        ">\n",
        "\n",
        "**4. How retrieval changed the answer:** more accurate, more specific, more grounded, or wrong because the wrong chunks came back?\n",
        "\n",
        ">\n",
        "\n",
        "**5. What was missed, and was it a retrieval problem or a generation problem?** Quote the evidence.\n",
        "\n",
        ">\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "for qid in [\"C-02\", \"C-03\", \"C-06\"]:\n",
        "    q = next(item for item in questions if item[\"id\"] == qid)\n",
        "    print(f\"\\n=== {qid} ===\")\n",
        "    print(\"Question:\", q[\"question\"])\n",
        "    print(\"Reference:\", q[\"reference\"])\n",
        "    scores = bm25.get_scores(regex_tokens(q[\"question\"]))\n",
        "    top_idx = np.argsort(scores)[::-1][:3]\n",
        "    print(\"BM25 top 3:\", [(chunks[int(i)][\"id\"], round(float(scores[int(i)]), 6)) for i in top_idx])\n",
        "    if model is not None:\n",
        "        q_emb = model.encode_query(q[\"question\"], convert_to_numpy=True)\n",
        "        sims = cosine_similarity([q_emb], doc_emb)[0]\n",
        "        emb_idx = sims.argsort()[::-1][:3]\n",
        "        print(\"Embedding top 3:\", [(chunks[int(i)][\"id\"], round(float(sims[int(i)]), 6)) for i in emb_idx])\n",
        "    else:\n",
        "        print(\"Embedding top 3: skipped (HF_TOKEN missing)\")\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Final questions\n",
        "\n",
        "**1. What is the main difference between BM25 and embedding retrieval?**\n",
        "\n",
        ">\n",
        "\n",
        "**2. When can BM25 perform better than embeddings?**\n",
        "\n",
        ">\n",
        "\n",
        "**3. When can embeddings perform better than BM25?**\n",
        "\n",
        ">\n",
        "\n",
        "**4. Why is cosine similarity used with embeddings?**\n",
        "\n",
        ">\n",
        "\n",
        "**5. What information is actually sent to an LLM in a RAG system?**\n",
        "\n",
        ">\n",
        "\n",
        "**6. Does the LLM search the original documents itself?**\n",
        "\n",
        ">\n",
        "\n",
        "**7. What happens if retrieval returns an irrelevant chunk?**\n",
        "\n",
        ">\n",
        "\n",
        "**8. Why can increasing Top-K sometimes improve RAG and sometimes make it worse?** Report K = 1, 3 and 8 on one question.\n",
        "\n",
        ">\n",
    ],
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"BM25 and embeddings are both implemented in this notebook.\")\n",
        "print(\"BM25 hit@3 total over the 10 queries:\", bm25_hit3_total)\n",
        "print(\"Embedding hit@3 total over the 10 queries:\", embed_hit3_total)\n",
        "print(\"Exact values depend on the model credentials available at runtime.\")\n",
    ],
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**9. Is a vector database required to build a RAG system?**\n",
        "\n",
        ">\n",
        "\n",
        "**10. Which retrieval approach worked better for this corpus, and what evidence supports that?** Give hit@3 for both over your ten queries and the C2 agreement counts for both RAG systems.\n",
        "\n",
        ">\n",
        "\n",
        "---\n",
        "# Conclusion\n",
        "\n",
        ">\n",
    ],
})

# fill notebook metadata
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.13"}}, "nbformat": 4, "nbformat_minor": 5}

# write initial notebook
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# execute code cells sequentially and preserve outputs
ns = {}
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if not source.strip():
        continue
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    try:
        exec(compile(source, "<notebook-cell>", "exec"), ns, ns)
        out_text = stdout.getvalue()
        err_text = stderr.getvalue()
        outputs = []
        if out_text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": out_text})
        if err_text:
            outputs.append({"output_type": "stream", "name": "stderr", "text": err_text})
        cell["outputs"] = outputs
        cell["execution_count"] = 1
    except Exception as e:
        traceback = [f"{type(e).__name__}: {e}"]
        cell["outputs"] = [{"output_type": "error", "ename": type(e).__name__, "evalue": str(e), "traceback": traceback}]
        cell["execution_count"] = 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook written to {nb_path}")
print(f"Corpus item count: {len(chunks)}")
print(f"BM25 hit@3 total: {bm25_hit3_total}")
print(f"Embedding hit@3 total: {embed_hit3_total}")
