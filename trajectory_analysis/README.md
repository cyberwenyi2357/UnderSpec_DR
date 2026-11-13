# Trajectory metrics

Search-behaviour metrics for the deep research agents' trajectories (`gpt-5.4`, `dr-tulu`, `tongyi`),
extracted from the trajectory analysis working directory. Each module reads one tool-calls file (a JSON
list of question objects, one per research session) and writes the metric fields back into it. The
module docstrings give the exact definitions.

| Module | Metric | Main output fields | Needs |
|---|---|---|---|
| `query_drift` | Search query drift: `1 - cos(first search query, last search query)` | `query_drift`, `first_last_similarity` | OpenAI embeddings (`text-embedding-3-small`) |
| `transition_category` | Search query transition category per consecutive search pair | `consecutive_query_pairs[*].search_transition_category`, `search_transition_category_ratios` | OpenAI chat (`gpt-5.4-mini`) |
| `query_provenance` | Search query provenance: user question / prior queries / prior retrieved content / untraced | `avg_tokens_in_*_ratio`, `avg_untraced_tokens_ratio` | nltk (+ jieba) |
| `document_novelty` | Document novelty: share of returned documents not seen earlier in the session | `new_documents_ratio`, `avg_new_documents_ratio_excl_first` | - |
| `context_adoption` | Context-driven term adoption rate: share of a query's new words found in earlier retrieved content | `context_adoption_rate`, `avg_context_adoption_rate_excl_first` | nltk (+ jieba) |

```bash
pip install numpy nltk jieba openai python-dotenv
export OPENAI_API_KEY=...   # only for query_drift and transition_category

python -m trajectory_analysis.document_novelty   --model gpt-5.4 tool_calls.json out.json
python -m trajectory_analysis.context_adoption   --model tongyi  tool_calls.json out.json --tokenizer jieba
python -m trajectory_analysis.query_provenance   --model dr-tulu tool_calls.json out.json
python -m trajectory_analysis.query_drift        --model tongyi  tool_calls.json out.json --cache emb.jsonl
python -m trajectory_analysis.transition_category --model gpt-5.4 tool_calls.json out.json
```

## Input schemas

- `gpt-5.4`, `dr-tulu`: `{"id", "question", "tool_calls": [{"tool_name", "query", "documents": [{"title", "url", "snippet"}], "output"}]}`.
  Search tools: `google_search`, `snippet_search`; `browse_webpage` is not a search.
- `tongyi`: `{"id", "question", "tool_calls": [{"name", "arguments": {"query": [...], "tool_response", "urls", "visit_tool_calls": [...]}}]}`.
  One call issues a batch of queries. Search tools: `search`, `google_scholar`. `visit_tool_calls` holds the
  text of the `visit` calls made after that search and before the next one. Results go into `arguments`.

## Conventions kept from the original analysis

- Sessions with fewer than 2 searches have no drift. The `_excl_first` averages drop the first call (novelty
  is always 1 and adoption always 0 there), and they are the values the query-vs-variant comparisons used.
- DeepResearch Bench (half Chinese) uses `--tokenizer jieba` for provenance and adoption. With jieba,
  gpt-5.4 / dr-tulu stem English tokens while tongyi does not, and tongyi's retrieved context leaves out
  `visit` pages. The English tokenizer includes them.
