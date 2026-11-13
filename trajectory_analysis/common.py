"""Shared pieces for the trajectory metrics: the two tool-call schemas, tokenizers and file IO.

Tool-call files are JSON lists of question objects, one per research session:

    {"id": ..., "question": "<the query the agent was given>", "tool_calls": [...], ...}

Two schemas, one per agent family:

* ``gpt-5.4`` and ``dr-tulu`` (single-query schema): one query per call,
      {"tool_name": "google_search" | "snippet_search" | "browse_webpage",
       "query": "<string>", "documents": [{"title", "url", "snippet"}, ...], "output": "<browse text>"}
* ``tongyi`` (batch schema): one call issues a list of queries at once,
      {"name": "search" | "google_scholar" | "visit" | ...,
       "arguments": {"query": ["q1", "q2", ...], "tool_response": "<search result text>",
                     "urls": [...], "visit_tool_calls": ["<visit page text>", ...]}}
  ``visit_tool_calls`` holds the text of every ``visit`` call issued after this search and before the next
  one. Rarely the agent put the query list directly in ``arguments``; that is handled as the query.
"""

import json
import re
from pathlib import Path

MODELS = ["gpt-5.4", "dr-tulu", "tongyi"]
SEARCH_TOOLS = {
    "gpt-5.4": {"google_search", "snippet_search"},
    "dr-tulu": {"google_search", "snippet_search"},
    "tongyi": {"search", "google_scholar"},
}
ALL_SEARCH_TOOLS = set().union(*SEARCH_TOOLS.values())


def is_batch_schema(model):
    if model not in SEARCH_TOOLS:
        raise ValueError(f"unknown model {model!r}, expected one of {MODELS}")
    return model == "tongyi"


# ------------------------------------------------------------------ tool-call accessors

def tool_name(tc):
    return tc.get("tool_name") or tc.get("name")


def tool_query(tc):
    """The raw query of a call: a string (single-query schema) or a list of strings (batch schema)."""
    if "tool_name" in tc:
        return tc.get("query")
    args = tc.get("arguments")
    if isinstance(args, dict):
        return args.get("query")
    if isinstance(args, list):
        return args
    return None


def query_list(q):
    """Query (string or list) -> list of non-empty strings."""
    if isinstance(q, list):
        return [x for x in q if isinstance(x, str) and x]
    return [q] if isinstance(q, str) and q else []


def has_query(q):
    if isinstance(q, list):
        return any(isinstance(x, str) and x.strip() for x in q)
    return isinstance(q, str) and bool(q.strip())


def is_search(tc):
    return tool_name(tc) in ALL_SEARCH_TOOLS


def search_calls(obj):
    return [tc for tc in obj.get("tool_calls", []) if is_search(tc)]


def question(obj):
    return obj.get("question") or obj.get("user_query")


def tongyi_args(tc):
    """Batch-schema arguments dict, created in place when missing (results are written into it)."""
    if not isinstance(tc.get("arguments"), dict):
        tc["arguments"] = {"query": tc["arguments"]} if isinstance(tc.get("arguments"), list) else {}
    return tc["arguments"]


def tongyi_search_steps(obj):
    """[(search call arguments, [visit page texts])] for a tongyi session, in order.

    The visit pages of a search are the visit calls made after it and before the next search. They are either
    already attached as ``arguments.visit_tool_calls`` (strings or {"tool_response": ...}) or still separate
    ``visit`` tool calls in ``tool_calls`` (``arguments.tool_response``); both are collected."""
    steps = []
    for tc in obj.get("tool_calls", []):
        if is_search(tc):
            args = tongyi_args(tc)
            texts = []
            for v in args.get("visit_tool_calls") or []:
                texts.append(v.get("tool_response") if isinstance(v, dict) else v)
            steps.append((args, texts))
        elif tool_name(tc) == "visit" and steps and isinstance(tc.get("arguments"), dict):
            steps[-1][1].append(tc["arguments"].get("tool_response"))
    return [(args, [t for t in texts if isinstance(t, str)]) for args, texts in steps]


# ------------------------------------------------------------------ tokenizers

_STOP = None
_STEMMER = None
ALPHA_RE = re.compile(r"[a-zA-Z]+")
ALNUM_RE = re.compile(r"[a-zA-Z0-9]+")
LOWER_ALNUM_RE = re.compile(r"[a-z0-9]+")
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
ASCII_ALPHA_RE = re.compile(r"^[a-z]+$")
CN_STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "有", "与", "及", "等", "对", "为", "中", "上", "下",
    "以", "或", "也", "就", "都", "而", "并", "并且", "这", "那", "关于", "如何", "什么",
    "哪些", "怎么", "可以", "能否", "是否", "请", "帮", "一下", "目前", "进行", "以及",
    "个", "之", "其", "使", "被", "让", "把", "到", "又", "但", "只", "还", "已", "不",
    "你", "我们", "他们", "她们", "它们", "一个", "一些", "这些", "那些", "这个", "那个",
}


def stopwords_en():
    global _STOP
    if _STOP is None:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        _STOP = set(stopwords.words("english"))
    return _STOP


def stem(word):
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer
        _STEMMER = PorterStemmer()
    return _STEMMER.stem(word)


def tokens_raw(text):
    """[a-z0-9]+ on lowercased text, English stopwords removed, no stemming."""
    if not isinstance(text, str) or not text:
        return set()
    stop = stopwords_en()
    return {w for w in LOWER_ALNUM_RE.findall(text.lower()) if w not in stop}


def tokens_stemmed(text, regex=ALPHA_RE):
    """regex on lowercased text, English stopwords removed, Porter-stemmed."""
    if not isinstance(text, str) or not text:
        return set()
    stop = stopwords_en()
    return {stem(w) for w in regex.findall(text.lower()) if w not in stop}


def tokens_jieba(text, stem_english):
    """Mixed Chinese/English tokenization with jieba, used for DeepResearch Bench (half of its tasks are
    Chinese, which the [a-z]+ tokenizers would drop entirely). Chinese and English stopwords are removed;
    pure-ASCII-letter tokens are Porter-stemmed when stem_english is set."""
    if not isinstance(text, str) or not text:
        return set()
    import jieba
    stop = stopwords_en()
    out = set()
    for w in jieba.cut(text.lower()):
        w = w.strip()
        if not w or PUNCT_RE.match(w) or w in stop or w in CN_STOPWORDS:
            continue
        if stem_english and ASCII_ALPHA_RE.match(w):
            w = stem(w)
        out.add(w)
    return out


# ------------------------------------------------------------------ IO and small helpers

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def openai_client(env_file=None):
    """OpenAI client from OPENAI_API_KEY (environment, or a .env file when python-dotenv is installed)."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        pass
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def add_io_args(parser):
    parser.add_argument("input", help="tool-calls JSON file (list of question objects)")
    parser.add_argument("output", nargs="?", help="where to write the result (default: overwrite input)")
    parser.add_argument("--model", required=True, choices=MODELS, help="agent that produced the trajectories")
    return parser
