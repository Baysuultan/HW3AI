# HW3 — Two retrievers and a model: what RAG actually sends

**Course:** CSS-4007 · Artificial Intelligence · Narxoz University
**Assigned:** Week 4 — *Context Engineering*
**Due:** see the LMS
**Points:** 4 pts total → **4%** of your final grade

A RAG system is a search engine bolted to a language model. The model never
opens a document. Your code searches, picks a few chunks, pastes them into the
prompt, and the model continues that prompt the way it continues any other. So
everything RAG adds to an answer comes in with the text you retrieved, and
everything it gets wrong starts either in the search or in what the model did
with the result.

You will compute two kinds of relevance by hand, run the same two retrievers
from libraries over 74 real chunks, and then wire them to a model and compare
three systems on the same questions. The point is not the pipeline, which is
about forty lines. The point is that **you can tell a retrieval failure from a
generation failure by looking at what was sent**, and back your answer with
your own numbers rather than a guess.

## 0. How this works

You work in **your own fork** of this repository. Do not open pull requests
against it.

1. **Fork it** (the **Fork** button, top right). Keep your fork **public**, so
   the instructor can read it without being added to anything.
2. Clone *your fork* and work there, locally or in Colab.
3. Work through `HW3.ipynb`. **It is the submission.** Every section below has a
   heading in it, and every table you are asked for has a blank copy in it.
4. **Run the notebook top to bottom before you push**, so that every output is
   saved in the file. A cell with no output earns nothing. Neither does a table
   typed in by hand that no cell printed.
5. Push to your fork, and submit your fork's link on the LMS before the deadline.

> There is no autograder. A human reads your notebook. That cuts both ways:
> nothing checks your work as you go, so rerun things and paste the numbers you
> actually got.

This repository contains **data, this README and a notebook of empty
sections. It contains no code.** You write it. Part B tells you which libraries
to use. How you put them together is up to you.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put both keys in .env
jupyter lab HW3.ipynb
```

In Colab, `git clone` your fork, `pip install -r requirements.txt`, and put the
two keys in Colab's *Secrets* panel rather than in a cell.

You need two accounts:

| Provider | Env var | Used for | Cost |
|---|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.6-luna`, the model in Part C | cents |
| Hugging Face | `HF_TOKEN` | downloading `google/embeddinggemma-300m` | free |

**EmbeddingGemma is gated.** Before your first download, sign in on Hugging
Face, open the model page for `google/embeddinggemma-300m` and accept the Gemma
licence. Then create a *read* token under Settings → Access Tokens. Without both
steps the download fails with a 401 or 403 error, and the error does not tell
you why. The model is 300M parameters (about 1.2 GB) and runs on a laptop CPU.
Embedding all 74 chunks takes well under a minute. Keep it at the default
`float32`, because EmbeddingGemma does not support `float16`.

**`.env` is in `.gitignore`. Never commit a key.** Your fork is public, so a
committed key is readable by anyone, and bots scan GitHub for them within
minutes. That goes for a key pasted into a notebook cell too, and for a key
printed in a cell's output, since the notebook is what you push. If a key does get
pushed, revoke it immediately and say so in the notebook. Disclosure costs you
nothing; a silent leaked key is an integrity issue.

**Estimated total spend for the whole assignment: under $0.05.** About twenty
short calls to one cheap model. Retrieval costs nothing, because BM25 and
EmbeddingGemma both run on your machine.

## 2. What is in `data/`

| File | What it is |
|---|---|
| `corpus.jsonl` | 74 chunks, one JSON object per line: `id`, `source`, `title`, `lang`, `url`, `text` |
| `queries.json` | Part B's eight fixed queries, each listing the chunks that answer it |
| `qa.json` | Part C's six questions, each with a reference answer and the chunks that hold it |
| `SOURCES.md` | Where every chunk came from, and the licence it carries |

The corpus has two halves, and they were chosen to behave differently:

- **KZ-01 … KZ-51 are real.** They are the opening sections of 29 English
  Wikipedia articles about Kazakhstan: Astana, the Aral Sea, Baikonur, Abai,
  Al-Farabi, beshbarmak, the tenge, Narxoz University and more. A large model
  has read all of this in training. That makes it a fair test of whether
  retrieval adds anything the model did not already have.
- **GO-01 … GO-23 are fictional.** They are the applicant handbook of the same
  grant office as HW2: the same rule, the same amounts, now with forms,
  deadlines, appeals and exceptions. No model has seen this text, so every
  correct detail in an answer about it has to come from retrieval. Twenty
  chunks are in English, one is in Kazakh and two are in Russian.

**Read the handbook before you write any code.** Some of it is deliberate. One
chunk is an *archived* rule that no longer applies, and it uses the exact words
a student would type. Three forms have near-identical names. Some facts are
stated only in Russian. Your written answers will be about exactly those
places.

You may bring your own collection instead of, or in addition to, this one:
university rules, course descriptions, news, documentation. It needs at least
30 chunks. If you do, the eight fixed queries and six questions must still be
run on *this* corpus, so that every submission can be compared. Your own
collection is for your extra queries.

## 3. Part A — relevance by hand (Sublab Easy, 1 pt)

The notebook section *Part A*. No libraries, no model. Show every step,
because the steps are what is marked. You may check your arithmetic in a code
cell afterwards, but the working must be written out.

**The documents and the query:**

| | Text |
|---|---|
| D1 | `Large language models can generate text and answer questions.` |
| D2 | `Retrieval-Augmented Generation retrieves relevant documents before generating an answer.` |
| D3 | `Embeddings represent text as numerical vectors.` |
| Query | `How does RAG retrieve information?` |

### A1 — BM25

Use the BM25 formula from the Week 4 slides, with `k1 = 1.5` and `b = 0.75`:

```text
idf(t)      = ln( (N − n(t) + 0.5) / (n(t) + 0.5) )
score(q, d) = Σ over query terms t of
              idf(t) · f(t,d) · (k1 + 1) / ( f(t,d) + k1 · (1 − b + b · |d| / avgdl) )
```

**Tokenise like this, and do not improvise.** Lowercase everything. Split at
spaces and at hyphens, so `Retrieval-Augmented` becomes two tokens. Drop
punctuation. Remove no stop words. `|d|` is the number of tokens in the
document.

1. **(a) As written.** Build the tables the notebook asks for: term frequency of
   every query term in every document, document frequency, IDF, each
   document's length, the average length, and the BM25 score of each document.
   Rank them.
2. **(b) With stemming.** Real BM25 systems usually stem. Repeat the
   calculation with these stems substituted wherever the word appears:

   | Word | Stem |
   |---|---|
   | retrieve, retrieves, retrieval | `retriev` |
   | generate, generating, generation | `gener` |
   | information | `inform` |
   | does | `doe` |

   Every other word stays as it is.

Answer: **which document does BM25 retrieve first, in (a) and in (b), and
why?** If you get a result in (a) that looks like a mistake, check it again. If
it survives the check, it is not a mistake, and explaining it is the answer.

### A2 — cosine similarity

Assume an embedding model produced these vectors:

```text
Query = [0.8, 0.2, 0.5]
D1    = [0.7, 0.1, 0.4]
D2    = [0.9, 0.3, 0.6]
D3    = [0.1, 0.8, 0.2]
```

For each document, show the dot product with the query, the magnitude of the
query, the magnitude of the document and the cosine similarity. Rank them.
Answer:

- **Which document is most similar to the query?**
- **Is the ranking the same as BM25's?** Compare it with both of your A1 rankings.
- **Why can BM25 and embedding retrieval disagree?** Point at the words of this
  query and these documents, not at retrieval in general.

**Grading:** A1 (a) and (b) with every intermediate table, and A2 with every
intermediate value (0.6) + the written answers (0.4).

## 4. Part B — the same two retrievers, from libraries (Sublab Medium, 1.5 pt)

The notebook section *Part B*. Load `data/corpus.jsonl`. Each line is one
chunk, so there is no chunking to do.

### B1 — BM25 with `rank_bm25`

Build a `BM25Okapi` index over the chunk texts, with `k1=1.5, b=0.75`. **Do not
implement BM25 yourself here.** Part A was for that. Tokenisation is your
choice, but write down what you chose (lowercase? `\w+`? stemming? stop
words?). That choice changes the rankings, and you may need it when you explain
one.

For every query, print the top 3: rank, chunk id, the first line of the chunk,
and the score.

### B2 — EmbeddingGemma with `sentence-transformers`

Load `google/embeddinggemma-300m` with `SentenceTransformer`. Embed every chunk
once with `model.encode_document(...)`, and each query with
`model.encode_query(...)`. EmbeddingGemma was trained with a different
instruction prefix for each side, and those two methods add them for you.
Compute the cosine similarity between the query and every chunk (with
`model.similarity`, or `sklearn`'s `cosine_similarity`; you do not have to write
it yourself here) and print the top 3 in the same format as B1.

### The queries

Run **the eight queries in `data/queries.json`** and **at least two of your
own**, ten or more in all. Do not edit the eight: they are what makes every
submission comparable. Make your own two worth running. One should be a query
you expect BM25 to win, and one a query you expect embeddings to win. Say which
is which *before* you run them.

For every query, print hit@3 for both retrievers: whether any chunk in
`relevant` made the top 3. For your own queries, decide `relevant` yourself
before you look at the results.

### B3 — compare them

The notebook has a table for all ten queries side by side. Then find, from your
own runs:

- **one query where both retrievers agree**, and say why it was easy;
- **one where BM25 did better**, and name the exact string it matched that the
  embedding model blurred;
- **one where embeddings did better**, and name what BM25 could not match:
  a synonym, a paraphrase, another language.

Think about Q-02, Q-03, Q-04 and Q-07 before you pick. Q-04 is interesting
whichever retriever wins it: look at what came first, then read that chunk's
title.

**Grading:** both retrievers running over ten or more queries, with the top-3
tables and hit@3 for each (0.75) + the B3 comparison, with the three examples
explained in terms of what each retriever actually matched (0.75).

## 5. Part C — connect retrieval to the model (Sublab Hard, 1.5 pt)

The notebook section *Part C*. The pipeline is:

```text
question → retrieve (BM25 or EmbeddingGemma) → top-K chunks → build the prompt → gpt-5.6-luna → answer
```

Use `gpt-5.6-luna` for every call, so that the only thing that changes between
the three systems is what your code sends. Retrieve top 3 unless a question
says otherwise.

### C1 — look at what is sent

Take one question from `data/qa.json`. Retrieve its top 3 with EmbeddingGemma,
build the complete prompt, and **print it in full before you send it**, exactly
as the model will receive it: system message, the numbered context blocks, the
question. Start from this system message and change it if you have a reason:

```text
Answer the question using only the provided context.
If the answer cannot be found in the context, say that you do not know.
Cite the context blocks you used, like [1] or [2][3].
```

Then account for the input:

| Part of the input | Tokens |
|---|---|
| System instructions | |
| Question | |
| Retrieved context | |
| **Total input** | |

Count each part with `tiktoken`'s `o200k_base` encoding. Then send the prompt
and **read `usage.prompt_tokens` off the response**. The two totals will not
match exactly, and you have to say why. HW1 is where you saw that gap before.
Also report the number of chunks, the tokens per chunk and the total size of
the context in tokens and in characters.

Answer: **what exactly does retrieval add to the model's input?** Be literal.

### C2 — three systems, six questions

Run all six questions in `data/qa.json` through:

1. **LLM only**: the question, with no context.
2. **BM25 RAG**: the top 3 from B1, in the C1 prompt.
3. **Embedding RAG**: the top 3 from B2, in the C1 prompt.

Fill in the notebook's table: the answer from each system, and whether it
agrees with `reference`. Mark it yourself, the way you marked Kazakh in HW1. A
correct answer in different words is correct. An answer with the right number
and an invented form code is not.

### C3 — three questions, taken apart

Choose three questions from C2. At least one should be a question where a RAG
system gave a worse answer than you expected. For each, show:

1. what BM25 retrieved (top 3, ids and scores);
2. what EmbeddingGemma retrieved (top 3, ids and scores);
3. what was sent to the model (the context blocks);
4. how retrieval changed the answer: more accurate, more specific, more
   grounded, or wrong because the wrong chunks came back;
5. what was missed, and **whether it was a retrieval problem or a generation
   problem**. The test is simple. If the fact was in the context and the answer
   still missed it, generation failed. If the fact was not in the context,
   retrieval failed. Say which, and quote the evidence.

C-03 and C-06 are the two most likely to teach you something. C-06 has no
answer in the corpus. Only one response to it is correct.

**Grading:** C1 with the full prompt printed and both token counts (0.4) + the
C2 table for all six questions and three systems (0.4) + the C3 analysis of three
questions with retrieval and generation failures told apart (0.4) + the final
questions (0.3).

## 6. Final questions

At the end of the notebook, answer briefly. Two to four sentences each is
enough. Where a question can be answered from your own runs, answer it from
them.

1. What is the main difference between BM25 and embedding retrieval?
2. When can BM25 perform better than embeddings?
3. When can embeddings perform better than BM25?
4. Why is cosine similarity used with embeddings?
5. What information is actually sent to an LLM in a RAG system?
6. Does the LLM search the original documents itself?
7. What happens if retrieval returns an irrelevant chunk?
8. Why can increasing Top-K sometimes improve RAG and sometimes make it worse?
   (Try K = 1, 3 and 8 on one question and report what changed.)
9. Is a vector database required to build a RAG system? (Look at what your own
   notebook used to store the vectors.)
10. Which retrieval approach worked better for this corpus, and what evidence
    supports that? Give hit@3 for both over your ten queries and the C2
    agreement counts for both RAG systems.

Then write a short conclusion: a paragraph on what you would build next, and
why.

## 7. Submission checklist

- [ ] Part A: both BM25 calculations and the cosine calculation, every step shown
- [ ] Part B: BM25 and EmbeddingGemma run on the eight fixed queries and at least two of your own, with hit@3 for each
- [ ] Part B: the three B3 examples, each explained
- [ ] Part C: the full C1 prompt printed, with the tiktoken count and the `usage` count
- [ ] Part C: six questions × three systems, each answer marked against `reference`
- [ ] Part C: three questions taken apart, each failure named as retrieval or generation
- [ ] Final questions and conclusion answered
- [ ] Notebook run top to bottom, outputs saved
- [ ] `.env` **not** committed (`git log --all -- .env` returns nothing), and no key in any cell
- [ ] Your fork is public, and its link is submitted on the LMS

#8. Academic integrity
Discuss concepts with classmates freely; the code and the written answers must be your own. Disclose AI tool assistance in the notebook's first cell. This is expected and fine, and undisclosed use is not.

The obvious point, again: this homework is about telling what a model knew from what it was given. A model that writes your C3 analysis did not see your retrieval results, so it can only guess what they were. Your answers have to quote your chunk ids, your scores and your token counts. A submission whose analysis does not match its own outputs is easy to spot.