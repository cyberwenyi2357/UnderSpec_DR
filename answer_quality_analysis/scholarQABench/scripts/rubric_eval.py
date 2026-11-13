import argparse
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

import litellm
import numpy as np
import scipy.stats
from scholarqacs_rubric import PendingJudgeCall, RubricCorpusQaGenericMetric
from litellm.caching import Cache
from pydantic.v1 import BaseModel, Field
from tqdm import tqdm
import glob

LOGGER = logging.getLogger(__name__)


class TestCase(BaseModel):
    case_id: str = Field(description="The ID of the case.")
    annotator: str = Field(description="Annotator id for the question.")
    agreement: bool = Field(
        description="Indicator whether the question is annotated by multiple annotators."
    )

    initial_prompt: str = Field(
        description="The initial query from the user to the system."
    )
    metric_config: Dict[str, Any] = Field(
        description="The metric to use to score the response."
    )
    response: str = Field(description="The response from the system.")

    def run(self) -> Dict[str, Any]:
        metric = RubricCorpusQaGenericMetric(self.metric_config["config"])
        resp = dict()
        resp["scores"] = metric.score_output(self.response)
        resp["case_id"] = self.case_id
        resp["annotator"] = self.annotator
        resp["agreement"] = self.agreement
        resp["question"] = self.initial_prompt
        return resp


class LlmEval:
    def __init__(self, test_config: List[Dict[str, Any]], responses: Dict[str, str], use_rubrics: bool,
                 use_snippets: bool):
        self.test_configs = test_config
        self.responses = responses
        self.use_snippets = use_snippets
        self.use_rubrics = use_rubrics

    def make_test_cases(
            self, skip_duplicate_annotations=True
    ) -> List[TestCase]:

        test_cases = []
        seen_agreements = set()
        for conf in self.test_configs:
            if conf["case_id"] not in self.responses:
                continue
            response = self.responses[conf["case_id"]]
            if response is None or response == "":
                print(
                    f"Skipping case_id={conf['case_id']}: missing answer_text"
                )
                continue
            if not self.use_rubrics:
                for prop in conf["metric_config"]["config"]["other_properties"]:
                    prop["criterion"] = ""
            if not self.use_snippets:
                for prop in conf["metric_config"]["config"]["other_properties"]:
                    prop["evidence"] = []
            conf["response"] = response
            if (
                    conf["initial_prompt"] in seen_agreements
                    and skip_duplicate_annotations
            ):
                continue
            seen_agreements.add(conf["initial_prompt"])
            test_cases.append(TestCase(**conf))
        return test_cases


def run_batch_scoring(
    cases: List[tuple],
    poll_interval: int,
    chunk_size: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Score a flat list of (src, test_case) pairs using the Gemini Batch API instead of one
    synchronous LLM call per judge criterion. Returns {src: [resp, ...]} in the same shape
    ``TestCase.run()`` produces, keyed by src.
    """
    from gemini_batch import run_gemini_batch

    metrics = []
    case_local_keys = []
    all_calls: List[PendingJudgeCall] = []
    model_names = set()

    for idx, (src, test_case) in enumerate(cases):
        metric = RubricCorpusQaGenericMetric(test_case.metric_config["config"])
        metrics.append(metric)
        model_names.add(metric.config.model_name)
        local_calls = metric.build_judge_calls(test_case.response)
        case_local_keys.append(list(local_calls.keys()))
        for local_key, call in local_calls.items():
            all_calls.append(call.copy(update={"key": f"{idx}::{local_key}"}))

    if len(model_names) > 1:
        raise ValueError(
            f"Batch mode requires all test cases to use the same judge model, found: {model_names}"
        )
    model_name = next(iter(model_names))

    print(f"Submitting {len(all_calls)} judge calls for {len(cases)} test cases to the Gemini Batch API...")
    results = run_gemini_batch(
        model=model_name, calls=all_calls, poll_interval=poll_interval, chunk_size=chunk_size
    )

    results_by_src: Dict[str, List[Dict[str, Any]]] = dict()
    for idx, (src, test_case) in enumerate(cases):
        case_results = {
            local_key: results.get(f"{idx}::{local_key}") for local_key in case_local_keys[idx]
        }
        resp = dict()
        resp["scores"] = metrics[idx].score_from_responses(test_case.response, case_results)
        resp["case_id"] = test_case.case_id
        resp["annotator"] = test_case.annotator
        resp["agreement"] = test_case.agreement
        resp["question"] = test_case.initial_prompt
        results_by_src.setdefault(src, []).append(resp)

    return results_by_src


def calculate_icc(scores1, scores2):
    """Calculate the intraclass correlation"""
    score_pairs = list(zip(scores1, scores2))
    n = len(score_pairs)
    grand_mean = np.mean(scores1 + scores2)

    sum1 = sum((x - grand_mean) ** 2 for x in scores1)
    sum2 = sum((x - grand_mean) ** 2 for x in scores2)
    s2 = (sum1 + sum2) / (2 * n - 1)

    icc = sum((x - grand_mean) * (y - grand_mean) for x, y in score_pairs) / (
            (n - 1) * s2
    )
    return icc


def load_sys_responses(qa_files):
    sys_responses = dict()
    for qa_file in qa_files:
        curr_responses = dict()
        print(f"Reading responses for source {qa_file}...")
        with open(qa_file) as f:
            for line in f:
                qa = json.loads(line)
                curr_responses[qa["case_id"]] = qa["answer_text"]
        print(f"{len(curr_responses)} responses obtained for eval...")
        sys_responses[qa_file] = curr_responses
    return sys_responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-dir",
        type=str,
        required=True,
        help="Directory containing *.jsonl files with queries and responses from system(s) to be evaluated. "
             "(All the files will be picked, to filter to specific files provide src-names param)",
    )
    parser.add_argument(
        "--test-config",
        type=str,
        required=True,
        help="Json file containing rubrics for all the questions to be evaluated",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"./data/results_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.json",
        help="output file to store the results of the evaluation",
    )
    parser.add_argument(
        "--agreement",
        action="store_true",
        help="Calculate agreement between annotators",
        default=False,
    )
    parser.add_argument(
        "--snippets",
        action="store_true",
        help="Score system for answer snippets",
        default=False,
    )
    parser.add_argument(
        "--rubrics",
        action="store_true",
        help="Score system for answer rubrics",
        default=False,
    )
    parser.add_argument(
        "--src-names",
        type=str,
        help="names of the source files to evaluate (comma separated with .jsonl extension)",
        default=None,
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Score using the Gemini Batch API instead of synchronous per-call scoring. "
             "Requires all test cases' judge model to be a gemini/ model.",
        default=False,
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=int,
        default=20,
        help="Seconds between polls of the Gemini Batch API job status (only used with --batch).",
    )
    parser.add_argument(
        "--batch-chunk-size",
        type=int,
        default=500,
        help="Max number of judge calls per submitted Gemini batch job (only used with --batch).",
    )

    args = parser.parse_args()

    litellm.cache = Cache(type="disk", disk_cache_dir="./data/litellm_cache/")

    if args.src_names:
        srcs = [s.strip() for s in args.src_names.split(",")]
        qa_files = [f"{args.qa_dir}/{src}.jsonl" for src in srcs]
    else:
        qa_files = glob.glob(f"{args.qa_dir}/*.jsonl")
    qa_files.sort()
    print(f"{len(qa_files)} src files found: {qa_files}")

    sys_responses = load_sys_responses(qa_files)
    test_config = json.load(open(args.test_config, "r"))

    qn_by_case = dict()
    test_cases_by_src = dict()
    for src, responses in sys_responses.items():
        llm_eval = LlmEval(test_config, responses, args.rubrics, args.snippets)
        print()
        print(f"Creating test cases for src: {src}...")
        test_cases = llm_eval.make_test_cases(
            skip_duplicate_annotations=(not args.agreement)
        )
        print(f"Created {len(test_cases)} tests for src: {src}...")
        for test_case in test_cases:
            qn_by_case[test_case.case_id] = test_case.initial_prompt
        test_cases_by_src[src] = test_cases

    if args.batch:
        all_cases = [
            (src, test_case) for src, test_cases in test_cases_by_src.items() for test_case in test_cases
        ]
        results_by_src = run_batch_scoring(
            all_cases, poll_interval=args.batch_poll_interval, chunk_size=args.batch_chunk_size
        )
        for src in test_cases_by_src:
            results_by_src.setdefault(src, [])
    else:
        results_by_src = dict()
        for src, test_cases in test_cases_by_src.items():
            results_by_src[src] = []
            print(f"Running test cases for src: {src}...")
            with ThreadPoolExecutor(max_workers=32) as executor:
                future_to_test_case = {
                    executor.submit(test_case.run): test_case for test_case in test_cases
                }
                for future in tqdm(
                        as_completed(future_to_test_case), total=len(test_cases)
                ):
                    results_by_src[src].append(future.result())

    for src in results_by_src:
        results_by_src[src].sort(key=lambda x: (x["annotator"], x["case_id"]))

    with open(args.output, "w") as f:
        json.dump(results_by_src, f)
        print(f"Results written to {args.output}\n")

    for src, results in results_by_src.items():
        print(
            f'Avg score for src={src}: {round(statistics.mean([res["scores"]["score"] for res in results]), 3)}'
        )

    # Note: "agreement" here is not actually agreement in the annotators'
    # labels, but rather the agreement in the scores their rubrics assign to
    # the same response.
    if args.agreement:
        qn_results = dict()
        print("\nCalculating agreement...")
        for src, results in results_by_src.items():
            for res in results:
                if res["agreement"]:
                    if res["question"] not in qn_results:
                        qn_results[res["question"]] = [[], []]
                    if res["annotator"] == "Annotator 1 Assignments":
                        qn_results[res["question"]][0].append(-res["scores"]["ann_score"])
                    else:
                        qn_results[res["question"]][1].append(-res["scores"]["ann_score"])
        ktaus, pcorr = [], []
        for qn, scores in qn_results.items():
            ann1, ann2 = [int(x) for x in np.argsort(scores[0]) + 1], [int(x) for x in np.argsort(scores[1]) + 1]
            ktaus.append(np.abs(scipy.stats.kendalltau(ann1, ann2)[0]))
            pcorr.append(np.abs(scipy.stats.pearsonr(ann1, ann2)[0]))


        print(f"\nAgreement metrics across {len(ktaus)} cases:")
        print(f"   Pearson corr: {round(np.mean(pcorr), 3)}")
        print(f"    Kendall tau: {round(np.mean(ktaus), 3)}")


if __name__ == "__main__":
    main()
