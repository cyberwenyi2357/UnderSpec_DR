import argparse
import copy
import glob
import json
import logging
import os
import re
import numpy as np

from fastchat.conversation import get_conv_template
from transformers import  LlamaForCausalLM
from vllm import LLM, SamplingParams
import torch
from collections import Counter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

prompt_template = (
"An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given."
"1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general."
"2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric."
"3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\""
"4. Please do not generate any other opening, closing, and explanations."
"###The instruction to evaluate:\n{instruction}\n"
"###Response to evaluate:\n{response}\n"
"###Reference Answer (Score 5):\n{reference_answer}"
"###Score Rubrics:\n"
"\n\n[{criteria_description}]"
"\nScore 1: {score1_description}"
"\nScore 2: {score2_description}"
"\nScore 3: {score3_description}"
"\nScore 4: {score4_description}"
"\nScore 5: {score5_description}"
)

prompt_template_wo_header = (
"An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given."
"1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general."
"2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric."
"3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\""
"4. Please do not generate any other opening, closing, and explanations."
"\n{instruction}"
"\n{response}"
"\n{reference_answer}"
"\n\n[{criteria_description}]"
"\nScore 1: {score1_description}"
"\nScore 2: {score2_description}"
"\nScore 3: {score3_description}"
"\nScore 4: {score4_description}"
"\nScore 5: {score5_description}"
)


prompt_template_no_reference = (
"An instruction (might include an Input inside it), a response to evaluate and a score rubric representing a evaluation criteria are given."
"1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general."
"2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric."
"3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\""
"4. Please do not generate any other opening, closing, and explanations."
"\n{instruction}"
"\n{response}"
"\n\n[{criteria_description}]"
"\nScore 1: {score1_description}"
"\nScore 2: {score2_description}"
"\nScore 3: {score3_description}"
"\nScore 4: {score4_description}"
"\nScore 5: {score5_description}"
)


def read_txt_file(file_path):
    """
    Read a text file to string
    """
    with open(file_path, 'r') as file:
        return file.read()


def read_json(file_path):
    """
    Read a json file to dict
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def preprocess_text(text):
    """
    Clean up text: remove reference section, URLS, non-ascii chars
    """
    # clean up empty line
    paragraphs = text.split("\n")
    paragraphs = [i for i in paragraphs if len(i) > 0]
    # clean up section title and remove reference section
    cleaned_pargraphs = []
    for i in paragraphs:
        if i == "# References":
            break
        if i.startswith("#"):
            i = "section: " + i.replace("#", "").strip()
        cleaned_pargraphs.append(i)
    text = "\n".join(cleaned_pargraphs)
    # remove URLS
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove non-ascii char
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # remove citation bracket (e.g. [10])
    text = re.sub(r'\[\d+\]', '', text)
    # remove non alphanumeric char
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_conversation_prompt(filled_prompt):
    """
    From filled prompt, convert it into llama-2 conversation prompt
    """
    conv = get_conv_template("llama-2")
    conv.set_system_message("You are a fair evaluator language model.")
    conv.append_message(conv.roles[0], filled_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def format_prompt(instruction, response, rubric, no_reference=False, top_n=8):
    """
    Fill prompt_template with rubric and response
    """
    output = response["output"]
    if "final_passages" in response:
        text = "References:\n" + response["final_passages"]
        output += text
    else:
        if "docs" not in response and "ctxs" in response:
            response["docs"] = response["ctxs"][:top_n]

        if rubric["use_title"] is True and "docs" in response and response["docs"] is False:
            titles = "References:\n"
            for idx, doc in enumerate(response["docs"]):
                if "[{0}]".format(idx+1) in output:
                    titles += "[{0}] {1}\n".format(idx + 1, doc["title"])

        if rubric["use_title"] is True and "docs" in response and response["docs"] is True:
            text = "References:\n"
            for idx, doc in enumerate(response["docs"]):
                if "[{0}]".format(idx+1) in output:
                    titles += "[{0}] {1}\n".format(idx + 1, doc["title"])
                    text += "[{0}] {1}: {2}\n".format(idx + 1, doc["title"], doc["text"])
            output += text
    
    if no_reference is False:
        entry = {"instruction": instruction + " " + response["input"], "response": output, "reference_answer": response["answer"]}
    else:
        entry = {"instruction": instruction + " " + response["input"], "response": output}
    
    entry.update(rubric)
    if no_reference is False:  
        if args.without_header is True:
            filled_prompt = prompt_template_wo_header.format(**entry)
        else:
            filled_prompt = prompt_template.format(**entry)
    else:
        filled_prompt = prompt_template_no_reference.format(**entry)
            
    return get_conversation_prompt(filled_prompt)


def get_grading_dict(responses,
                    instruction,
                    model,
                    rubric_path="rubrics/prometheus_rubrics_v8.json",
                    disable_sample=False,
                    temperature=0.01,
                    top_p=0.95,
                    max_new_tokens=512,
                    repetition_penalty=1.03,
                    logger=None,
                    sampling_params=None, 
                    no_reference=False,
                    top_n=8,
                    aspects=None):
    grading = {}
    rubrics = read_json(rubric_path)

    # Read all files in the given directory
    for rubric_idx, rubric in enumerate(rubrics):
        if aspects is not None and rubric["aspect"] not in aspects:
            continue
        grading[rubric["criteria_description"]] = {}
        
        prompts = []
        for response_idx, response in enumerate(responses):
            # generate evaluation prompt and tokenize
            if logger is not None:
                logger.info(
                    f"processing for rubric {rubric_idx + 1}/{len(rubrics)}, response {response_idx + 1}/{len(responses)}, response length: {len(response)}")

            prompt = format_prompt(instruction=instruction, response=response, rubric=rubric, no_reference=no_reference, top_n=top_n)
            prompts.append(prompt)

        outputs = model.generate(prompts, sampling_params)
        decoded_outputs = [output.outputs[0].text for output in outputs]
        
        for response_idx, (response, decoded_output) in enumerate(zip(responses, decoded_outputs)):
            score = decoded_output[decoded_output.find("[RESULT] ") + len("[RESULT] "):].split("\n")[0]
            print(score)
            feedback = decoded_output[decoded_output.find("Feedback: ") + len("Feedback: "):] if "Feedback: " in decoded_output else decoded_output
            try:
                int(score)
            except Exception as e:
                pattern = r"the overall score is (\d+)"
                match = re.search(pattern, feedback)
                if match:
                    score = match.group(1)
            print("final score")
            print(score)
            grading[rubric["criteria_description"]][response_idx] = {"feedback": feedback, "score": score}
    return grading

def extract_score(result):
    result_dics = {}
    for k in result:
        scores = []
        for i in list(result[k].values()):
            if i["score"] in ["1", "2", "3", "4", "5"]:
                scores.append(int(i["score"]))
            elif i["score"].split("\n\n")[0] in ["1", "2", "3", "4", "5"]:
                scores.append(int(i["score"].split("\n\n")[0]))
        result_dics[k] = np.mean(scores)
        # result_dics[k + "_std"] = np.std(scores)
    result_dics["average"] = np.mean(list(result_dics.values()))
    return result_dics

def align_responses_with_gold(responses, gold_items):
    """Attach gold answers by the response ID's numeric suffix (1-based gold index)."""
    aligned_responses = []
    skipped_ids = []
    for response in responses:
        response_id = str(response.get("id", ""))
        match = re.search(r"(\d+)$", response_id)
        if match is None:
            skipped_ids.append(response_id or "<missing id>")
            continue

        gold_index = int(match.group(1)) - 1
        if gold_index < 0 or gold_index >= len(gold_items):
            skipped_ids.append(response_id)
            continue

        response["answer"] = gold_items[gold_index]["output"]
        aligned_responses.append(response)

    logger.info(
        "Aligned %d/%d responses with %d gold answers by ID suffix.",
        len(aligned_responses),
        len(responses),
        len(gold_items),
    )
    if skipped_ids:
        logger.warning(
            "Skipped %d responses that could not be matched: %s",
            len(skipped_ids),
            ", ".join(skipped_ids),
        )
    if not aligned_responses:
        raise ValueError(
            "No responses could be matched to gold answers. Response IDs must "
            "end in a 1-based source index, for example "
            "'query_fully_underspecified_26'."
        )
    return aligned_responses

def main(args):
    # Load and align data before loading the large evaluator model.
    results = json.load(open(args.batch_process_dir))
    responses = results["data"] if "data" in results else results

    if args.gold_answer_file is not None:
        gold_items = json.load(open(args.gold_answer_file))
        responses = align_responses_with_gold(responses, gold_items)

    # Loading evaluator LM
    if args.load_vllm is True:
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)
        model = LLM(args.model, 
            download_dir=args.download_dir,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            disable_custom_all_reduce=True)
    else:
        model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto")

    if args.self_consistency is True:
        grading_dict = {}
        for iter_i in range(3):
            print("start grading: {0}".format(iter_i))
            grading = get_grading_dict(responses=responses,
                                    instruction=args.instruction,
                                    model=model,
                                    rubric_path=args.rubric_path,
                                    disable_sample=args.disable_sample,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    max_new_tokens=args.max_new_tokens,
                                    repetition_penalty=args.repetition_penalty,
                                    logger=logger,
                                    sampling_params=sampling_params if args.load_vllm else None,
                                    no_reference=args.no_reference,
                                    top_n=args.top_n,)
            grading_dict[iter_i] = grading
        final_grading = {}
        for aspect in grading_dict[iter_i].keys():
            final_grading[aspect] = {}
            for instance in grading_dict[iter_i][aspect]:
                if args.most_common is True:
                    c = Counter([grading_dict[i][aspect][instance]["score"] for i in range(3)])
                    final_decision = c.most_common()[0][0]
                    final_grading[aspect][instance] = {"score": final_decision}
                else:
                    valid_choices = []
                    for i in range(3): 
                        pred = grading_dict[i][aspect][instance]["score"]
                        try:
                            pred = int(pred)
                            valid_choices.append(pred)
                        except: 
                            print("conversion error")
                        
                    final_decision = np.mean(valid_choices)
                    final_grading[aspect][instance] = {"score": final_decision}
        grading = final_grading

    else:
        grading = get_grading_dict(responses=responses,
                                instruction=args.instruction,
                                model=model,
                                rubric_path=args.rubric_path,
                                disable_sample=args.disable_sample,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_new_tokens=args.max_new_tokens,
                                repetition_penalty=args.repetition_penalty,
                                logger=logger,
                                sampling_params=sampling_params if args.load_vllm else None,
                                no_reference=args.no_reference,
                                top_n=args.top_n,
                                aspects=args.aspects)

    summary_result = extract_score(grading)
    grading["summary"] = summary_result

    print(summary_result)

    # Save grading dictionary to output path
    with open(os.path.join(args.output_path, "results.json"), 'w') as outfile:
        json.dump(grading, outfile, indent=2)
        logger.info("Grading complete. Output saved to: %s", args.output_path)


if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('-b', '--batch_process_dir', required=True, help='Directory of files to process')
    parser.add_argument('-f', '--gold_answer_file',  help='Gold answer')
    parser.add_argument('-o', '--output_path', required=True, help='Path to save the output JSON file')
    parser.add_argument('-i', "--instruction", default="Given a paper abstract, generate the Related Work section summarizing relevant papers.", help="Topic of the script your going to analyze")

    parser.add_argument("--rubric_path", default="eval_rubric_5.json", help='path to rubric json file')

    parser.add_argument('--tokenizer', default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--model',
                        default="kaist-ai/prometheus-13b-v1.0",
                        help="Model to use; options are 'kaist-ai/prometheus-13b-v1.0' or 'kaist-ai/prometheus-7b-v1.0'")
    parser.add_argument('--disable_sample', action='store_true', help='Whether to disable sampling; default is False')
    parser.add_argument('--load_vllm', action='store_true', help='Load checkpoints via vllm for inference efficiency.')
    parser.add_argument('--temperature', type=float, default=0.01, help='Temperature for generation; default is 0.01')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top P for generation; default is 0.95')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum new tokens to generate; default is 512')
    parser.add_argument('--repetition_penalty', type=float, default=1.03, help='Repetition penalty; default is 1.03')
    parser.add_argument('--no_reference', action='store_true', help="whether to use reference or not.")
    parser.add_argument('--top_n', type=int, default=8,)
    parser.add_argument('--aspects', type=str, nargs="+")
    parser.add_argument('--self_consistency',  action='store_true')
    parser.add_argument('--most_common',  action='store_true')
    parser.add_argument('--download_dir', type=str, default="~/.cache/huggingface")
    parser.add_argument('--without_header',  action='store_true')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    assert os.path.exists(args.batch_process_dir), f"batch_process_dir: {args.batch_process_dir} not exists"
    output_directory = args.output_path
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        logger.info("Created directory: %s", output_directory)

    main(args)
