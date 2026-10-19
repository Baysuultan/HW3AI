import json
from pathlib import Path

nb_path = Path(r"c:\Users\Байсултан\Desktop\HW3AI\HW3.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

cells = nb["cells"]

parts_a = '''
import math
import re
from collections import Counter

# Part A — exact BM25 calculations by hand

D1 = "Large language models can generate text and answer questions."
D2 = "Retrieval-Augmented Generation retrieves relevant documents before generating an answer."
D3 = "Embeddings represent text as numerical vectors."
query = "How does RAG retrieve information?"

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    return [tok for tok in re.split(r"[\s-]+", text) if tok]

DOCS = {"D1": D1, "D2": D2, "D3": D3}
query_terms = tokenize(query)
print("Query terms:", query_terms)

doc_tokens = {k: tokenize(v) for k, v in DOCS.items()}
print("Document tokens:")
for k, toks in doc_tokens.items():
    print(k, toks, "len=", len(toks))

len_by_doc = {k: len(v) for k, v in doc_tokens.items()}
avgdl = sum(len_by_doc.values()) / len(len_by_doc)
print("avgdl =", avgdl)

# TF, DF and IDF
terms = sorted(set(query_terms))
print("\nTF / DF / IDF")
for term in terms:
    tf = {doc: doc_tokens[doc].count(term) for doc in DOCS}
    df = sum(1 for doc in DOCS if term in doc_tokens[doc])
    idf = math.log((len(DOCS) - df + 0.5) / (df + 0.5))
    print(term, tf, "df=", df, "idf=", idf)

# BM25 with k1=1.5, b=0.75
k1 = 1.5
b = 0.75
N = len(DOCS)

def bm25_score(term, doc_id):
    f = doc_tokens[doc_id].count(term)
    df = sum(1 for d in DOCS if term in doc_tokens[d])
    idf = math.log((N - df + 0.5) / (df + 0.5))
    denom = f + k1 * (1 - b + b * len_by_doc[doc_id] / avgdl)
    if denom == 0:
        return 0.0
    return idf * f * (k1 + 1) / denom

scores = {}
for doc_id, text in DOCS.items():
    total = 0.0
    parts = []
    for term in query_terms:
        if term in doc_tokens[doc_id]:
            term_score = bm25_score(term, doc_id)
            total += term_score
            parts.append((term, term_score))
    scores[doc_id] = total
    print(f"{doc_id}: score = {total:.6f}; contributors = {parts}")

print("\nFinal ranking:")
for doc_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
    print(doc_id, score)

# A1(b) with stemming
stem_map = {
    "retrieve": "retriev",
    "retrieves": "retriev",
    "retrieval": "retriev",
    "generate": "gener",
    "generating": "gener",
    "generation": "gener",
    "information": "inform",
    "does": "doe",
}

stemmed_doc_tokens = {
    doc_id: [stem_map.get(tok, tok) for tok in toks]
    for doc_id, toks in doc_tokens.items()
}
query_stems = [stem_map.get(tok, tok) for tok in query_terms]
print("\nStemmed query terms:", query_stems)
print("Stemmed document tokens:")
for k, toks in stemmed_doc_tokens.items():
    print(k, toks, "len=", len(toks))

stem_len_by_doc = {k: len(v) for k, v in stemmed_doc_tokens.items()}
stem_avgdl = sum(stem_len_by_doc.values()) / len(stem_len_by_doc)
print("stem avgdl =", stem_avgdl)

stem_terms = sorted(set(query_stems))
for term in stem_terms:
    tf = {doc: stemmed_doc_tokens[doc].count(term) for doc in DOCS}
    df = sum(1 for doc in DOCS if term in stemmed_doc_tokens[doc])
    idf = math.log((len(DOCS) - df + 0.5) / (df + 0.5))
    print(term, tf, "df=", df, "idf=", idf)

stem_scores = {}
for doc_id in DOCS:
    total = 0.0
    contribs = []
    for term in query_stems:
        if term in stemmed_doc_tokens[doc_id]:
            f = stemmed_doc_tokens[doc_id].count(term)
            df = sum(1 for d in DOCS if term in stemmed_doc_tokens[d])
            idf = math.log((N - df + 0.5) / (df + 0.5))
            denom = f + k1 * (1 - b + b * stem_len_by_doc[doc_id] / stem_avgdl)
            score = idf * f * (k1 + 1) / denom
            total += score
            contribs.append((term, score))
    stem_scores[doc_id] = total
    print(f"{doc_id}: stem score = {total:.6f}; contributors = {contribs}")

print("\nFinal stem ranking:")
for doc_id, score in sorted(stem_scores.items(), key=lambda kv: kv[1], reverse=True):
    print(doc_id, score)
'''
cells[6]["source"] = parts_a.splitlines(keepends=True)

# Part B dataset loading
load_data = '''
import json
from pathlib import Path

base = Path.cwd()
corpus_path = base / "data" / "corpus.jsonl"
queries_path = base / "data" / "queries.json"
qa_path = base / "data" / "qa.json"

chunks = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
queries_data = json.loads(queries_path.read_text(encoding="utf-8"))
qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
fixed_queries = queries_data["queries"]
questions = qa_data["questions"]

print("Corpus chunks:", len(chunks))
print("Fixed queries:", [q["id"] for q in fixed_queries])
print("Reference questions:", [q["id"] for q in questions])
print("Sample chunk:", chunks[0]["id"], chunks[0]["title"])
'''
cells[8]["source"] = load_data.splitlines(keepends=True)

# Part B1 BM25 library
b1 = '''
import re
import numpy as np
from rank_bm25 import BM25Okapi


def regex_tokens(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    return [tok for tok in re.split(r"[\s-]+", text) if tok]

# Tokenization choice for this notebook
# lowercase + regex on alphanumeric groups and hyphen splits, no stemming, no stop-word removal.
text_tokens = [regex_tokens(chunk["text"]) for chunk in chunks]

bm25 = BM25Okapi(text_tokens, k1=1.5, b=0.75)

query_plan = fixed_queries + [
    {
        "id": "Q-09",
        "text": "What exact form is used to appeal a grant decision?",
        "relevant": ["GO-08", "GO-12", "GO-23"],
    },
    {
        "id": "Q-10",
        "text": "I have a low income and need support for university study. What help exists?",
        "relevant": ["GO-01", "GO-02", "GO-03"],
    },
]

for q in query_plan:
    q_tokens = regex_tokens(q["text"])
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:3]
    print(f"\n{q['id']} :: {q['text']}")
    for rank, idx in enumerate(top_idx, start=1):
        chunk = chunks[int(idx)]
        print(rank, chunk["id"], chunk["title"], round(float(scores[int(idx)]), 6))
'''
cells[10]["source"] = b1.splitlines(keepends=True)

# Part B2 EmbeddingGemma
b2 = '''
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("HF_TOKEN not set — skipping EmbeddingGemma.")
else:
    model = SentenceTransformer("google/embeddinggemma-300m")
    doc_emb = model.encode_document([chunk["text"] for chunk in chunks], convert_to_numpy=True)
    print("Document embeddings shape:", doc_emb.shape)
    for q in fixed_queries + [
        {
            "id": "Q-09",
            "text": "What exact form is used to appeal a grant decision?",
            "relevant": ["GO-08", "GO-12", "GO-23"],
        },
        {
            "id": "Q-10",
            "text": "I have a low income and need support for university study. What help exists?",
            "relevant": ["GO-01", "GO-02", "GO-03"],
        },
    ]:
        q_emb = model.encode_query(q["text"], convert_to_numpy=True)
        sims = cosine_similarity([q_emb], doc_emb)[0]
        top_idx = sims.argsort()[::-1][:3]
        print(f"\n{q['id']} :: {q['text']}")
        for rank, idx in enumerate(top_idx, start=1):
            chunk = chunks[int(idx)]
            print(rank, chunk["id"], chunk["title"], round(float(sims[int(idx)]), 6))
'''
cells[12]["source"] = b2.splitlines(keepends=True)

# custom queries markdown cell already in notebook; code to print them
custom_queries = '''
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

for q in my_queries:
    print(q["id"], "=> expected", q["expected"], "because:", q["why"])
'''
cells[14]["source"] = custom_queries.splitlines(keepends=True)

# results for all queries; if embedding model not available, it will skip gracefully.
results = '''
import os
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

all_queries = fixed_queries + my_queries

def q_tokens(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    return [tok for tok in re.split(r"[\s-]+", text) if tok]

# BM25
bm25 = BM25Okapi([q_tokens(chunk["text"]) for chunk in chunks], k1=1.5, b=0.75)

def bm25_top(query_text, k=3):
    scores = bm25.get_scores(q_tokens(query_text))
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[int(i)]["id"], float(scores[int(i)])) for i in top_idx]

# EmbeddingGemma (optional)
embedding_top = {}
if os.getenv("HF_TOKEN"):
    model = SentenceTransformer("google/embeddinggemma-300m")
    doc_emb = model.encode_document([chunk["text"] for chunk in chunks], convert_to_numpy=True)
    for q in all_queries:
        q_emb = model.encode_query(q["text"], convert_to_numpy=True)
        sims = cosine_similarity([q_emb], doc_emb)[0]
        top_idx = sims.argsort()[::-1][:3]
        embedding_top[q["id"]] = [(chunks[int(i)]["id"], float(sims[int(i)])) for i in top_idx]
else:
    print("HF_TOKEN not set — embedding results were not computed.")

for q in all_queries:
    bm25_hits = bm25_top(q["text"], 3)
    print(f"\n{q['id']} :: {q['text']}")
    print("BM25:", bm25_hits)
    if q["id"] in embedding_top:
        print("Embedding:", embedding_top[q["id"]])
    else:
        print("Embedding: skipped")
'''
cells[16]["source"] = results.splitlines(keepends=True)

# C1 prompt/build helper
c1 = '''
import os
import json
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM = (
    "Answer the question using only the provided context.\n"
    "If the answer cannot be found in the context, say that you do not know.\n"
    "Cite the context blocks you used, like [1] or [2][3]."
)

# pick a question from qa.json
question_obj = questions[0]
question = question_obj["question"]

# choose the top-3 BM25 chunks for this question
import re
import numpy as np
from rank_bm25 import BM25Okapi


def q_tokens(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    return [tok for tok in re.split(r"[\s-]+", text) if tok]

bm25 = BM25Okapi([q_tokens(chunk["text"]) for chunk in chunks], k1=1.5, b=0.75)
scores = bm25.get_scores(q_tokens(question))
selected_idx = np.argsort(scores)[::-1][:3]
selected_chunks = [chunks[int(i)] for i in selected_idx]

context_blocks = "\n\n".join(f"[{idx+1}] {chunk['text']}" for idx, chunk in enumerate(selected_chunks))
prompt = f"{SYSTEM}\n\n{context_blocks}\n\nQuestion: {question}"
print(prompt)
print("\n--- token counts ---")
enc = tiktoken.get_encoding("o200k_base")
print("System instructions tokens:", len(enc.encode(SYSTEM)))
print("Question tokens:", len(enc.encode(question)))
print("Retrieved context tokens:", len(enc.encode(context_blocks)))
print("Total input tokens:", len(enc.encode(prompt)))
print("Chunk count:", len(selected_chunks))
print("Tokens per chunk:", [len(enc.encode(chunk["text"])) for chunk in selected_chunks])
print("Context chars:", sum(len(chunk["text"]) for chunk in selected_chunks))

if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n\n{context_blocks}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    print("usage.prompt_tokens:", resp.usage.prompt_tokens)
    print("response:", resp.choices[0].message.content)
else:
    print("OPENAI_API_KEY not set — skipping the API call.")
'''
cells[21]["source"] = c1.splitlines(keepends=True)

# C2 model comparison on the six questions
c2 = '''
import os
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def q_tokens(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]+", " ", text)
    return [tok for tok in re.split(r"[\s-]+", text) if tok]

SYSTEM = (
    "Answer the question using only the provided context.\n"
    "If the answer cannot be found in the context, say that you do not know.\n"
    "Cite the context blocks you used, like [1] or [2][3]."
)

bm25 = BM25Okapi([q_tokens(chunk["text"]) for chunk in chunks], k1=1.5, b=0.75)
model = None
if os.getenv("HF_TOKEN"):
    model = SentenceTransformer("google/embeddinggemma-300m")
    doc_emb = model.encode_document([chunk["text"] for chunk in chunks], convert_to_numpy=True)


def top_bm25(question_text, k=3):
    scores = bm25.get_scores(q_tokens(question_text))
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[int(i)] for i in top_idx]


def top_embedding(question_text, k=3):
    if model is None:
        return []
    q_emb = model.encode_query(question_text, convert_to_numpy=True)
    sims = cosine_similarity([q_emb], doc_emb)[0]
    top_idx = sims.argsort()[::-1][:k]
    return [chunks[int(i)] for i in top_idx]


def ask_llm(question, ctx_chunks):
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — model call skipped")
        return "SKIPPED"
    context_text = "\n\n".join(f"[{idx+1}] {c['text']}" for idx, c in enumerate(ctx_chunks))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n\n{context_text}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content

for q in questions:
    qid = q["id"]
    print(f"\n=== {qid} ===")
    print("Question:", q["question"])
    llm_only = ask_llm(q["question"], [])
    print("LLM only:", llm_only)
    bm25_ctx = top_bm25(q["question"], 3)
    bm25_answer = ask_llm(q["question"], bm25_ctx)
    print("BM25 RAG:", bm25_answer)
    emb_ctx = top_embedding(q["question"], 3)
    emb_answer = ask_llm(q["question"], emb_ctx)
    print("Embedding RAG:", emb_answer)
'''
cells[24]["source"] = c2.splitlines(keepends=True)

# C3 retrieval analysis for three questions
c3 = '''
# Compare retrieval outputs for three questions without making extra model calls.
for qid in ["C-02", "C-03", "C-06"]:
    q = next(item for item in questions if item["id"] == qid)
    print(f"\n=== {qid} ===")
    print("Question:", q["question"])
    print("Reference:", q["reference"])
    # BM25 top 3
    scores = bm25.get_scores(q_tokens(q["question"]))
    top_idx = np.argsort(scores)[::-1][:3]
    print("BM25 top 3:", [(chunks[int(i)]["id"], round(float(scores[int(i)]), 6)) for i in top_idx])
    # Embeddings top 3 if available
    if os.getenv("HF_TOKEN"):
        q_emb = model.encode_query(q["question"], convert_to_numpy=True)
        sims = cosine_similarity([q_emb], doc_emb)[0]
        emb_idx = sims.argsort()[::-1][:3]
        print("Embedding top 3:", [(chunks[int(i)]["id"], round(float(sims[int(i)]), 6)) for i in emb_idx])
    else:
        print("Embedding top 3: skipped (HF_TOKEN missing)")
'''
cells[27]["source"] = c3.splitlines(keepends=True)
cells[29]["source"] = c3.splitlines(keepends=True)
cells[31]["source"] = c3.splitlines(keepends=True)

# Final questions summary
final_q = '''
# Summary for the final answers
print("BM25 and embeddings are both implemented in this notebook.")
print("To get the exact model outputs, set OPENAI_API_KEY and HF_TOKEN before running Part B2 and Part C.")
print("This notebook is designed to run gracefully even when the keys are missing.")
'''
cells[33]["source"] = final_q.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook updated:", nb_path)
