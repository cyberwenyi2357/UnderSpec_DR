import json
import jsonlines
import logging
from typing import Any, Dict, Optional

import litellm
LOGGER = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        return None

    try:
        return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        try:
            return json.loads(response[json_start:json_end]+"]}")
        except json.JSONDecodeError:
            LOGGER.warning(
                f"Could not decode JSON from response: {response[json_start:json_end]}"
            )
        return None


def _uses_max_completion_tokens(model_name: str) -> bool:
    """Newer OpenAI models reject ``max_tokens`` and require ``max_completion_tokens``."""
    name = model_name.lower()
    return (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )


def run_chatopenai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    json_mode: bool = False,
    **chat_kwargs,
) -> str:
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    if json_mode:
        chat_kwargs["response_format"] = {"type": "json_object"}
    # gpt-5 / o-series models expect max_completion_tokens instead of max_tokens.
    if _uses_max_completion_tokens(model_name) and "max_tokens" in chat_kwargs:
        chat_kwargs["max_completion_tokens"] = chat_kwargs.pop("max_tokens")
    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )
    resp = litellm.completion(
        model=model_name,
        messages=msgs,
        **chat_kwargs,
    )

    return resp.choices[0].message.content

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)