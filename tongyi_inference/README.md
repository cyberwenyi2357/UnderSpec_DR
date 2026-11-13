# Tongyi DeepResearch inference

Runs [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
(`Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`) on the queries in `../query/` and their underspecified variants.
The agent code is the ReAct inference code from the Tongyi DeepResearch repository.

1. Install: `pip install -r requirements.txt`.
2. Configure: `cp .env.example .env` and fill in
   - `MODEL_PATH`: `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` or a local download of it.
   - `SERPER_KEY_ID` (search and Google Scholar) and `JINA_API_KEYS` (page reading).
   - `API_KEY`, `API_BASE`, `SUMMARY_MODEL_NAME`: an OpenAI-compatible model that summarizes visited pages.
3. Build the per-variant datasets: `python make_eval_data.py` writes
   `eval_data/{benchmark}[_{subset}]_{variant}.jsonl` (one `{"question", "answer"}` object per line).
4. Start the vLLM servers (GPUs 1-3, ports 6011-6013) and run one dataset: set `DATASET` in `.env` to an
   `eval_data/*.jsonl` file and run `bash run_react_infer.sh`.
5. With the servers running, run all variants of a benchmark:

   | Script | Datasets |
   |---|---|
   | `queue_runs.sh` | ResearchQA |
   | `queue_runs_deepresearch_bench.sh` | DeepResearch Bench |
   | `queue_runs_openscholar.sh` | ScholarQA-Bench, `multi` subset |
   | `queue_runs_openscholar_cs.sh` | ScholarQA-Bench, `cs` subset |
   | `queue_runs_openscholar_cs_2.sh` | ScholarQA-Bench, `cs_2` subset |

Results go to `outputs/`, logs to `log/`.
