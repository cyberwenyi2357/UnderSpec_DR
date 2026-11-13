import asyncio
import copy
import json
import re
import random
import os
import importlib

_search_env = None
for _module_name in ("dr_agent.search_env", "envs.search_env"):
    try:
        _search_env = importlib.import_module(_module_name)
        break
    except Exception:
        _search_env = None

if _search_env is not None:
    extract_fn_call = _search_env.extract_fn_call
    keep_first_n_words = _search_env.keep_first_n_words
else:
    def keep_first_n_words(text: str, n: int = 1000) -> str:
        words = (text or "").split()
        return " ".join(words[:n])

    def extract_fn_call(text):
        return []

# Default system prompt when wiring :class:`UserSimulator` to deep-research ``ask_user``
# without full :class:`UserEnv` / RL task metadata.
DEFAULT_ASK_USER_SIMULATOR_PROMPT = (
    "You are a human user chatting with an agent. "
    "Answer the agent's clarifying questions briefly and naturally, in plain language. "
    "If you truly cannot answer, say so in one short sentence. "
    "Do not use markdown or bullet lists. Never ask the assistant a question back."
)


def _ask_user_log_full_prompt() -> bool:
    v = os.environ.get("ASK_USER_LOG_FULL_PROMPT", "").strip().lower()
    return v in ("1", "true", "yes")


class UserSimulator:
    def __init__(self, prompt: str, model: str = None):
        self.prompt = prompt
        self.conversation = []
        self.model = model

    async def call_openai(self, messages: list, model: str = 'gpt-5-nano', max_retries: int = 3) -> str:
        from openai import AsyncOpenAI

        model = self.model if self.model is not None else model
        if model != 'gpt-5.4-mini':
            print(f'[UserSimulator] Using model: {model}')

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if _ask_user_log_full_prompt():
            print(
                "[UserSimulator] Full prompt (chat.messages) for this API call:\n"
                + json.dumps(messages, ensure_ascii=False, indent=2),
                flush=True,
            )

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[UserSimulator] Error after {max_retries} attempts: {str(e)}")
                    return f"Error after {max_retries} attempts: {str(e)}"
                await asyncio.sleep(1 * (attempt + 1))
        return "I cannot answer your question."

    async def execute(self, params: dict | str) -> str:
        if len(self.conversation) == 0:
            self.conversation.append({'role': 'system', 'content': self.prompt})

        query = params if isinstance(params, str) else params.get("query", "")
        query = (query or "").strip()
        # ``user`` turn = only the research assistant's question text (use f"…{query}…" if you add a wrapper)
        self.conversation.append({"role": "user", "content": query})
        if len(self.conversation) >= 10:
            return "I have already answered 10 questions. I cannot answer more. Do it yourself."
        response = await self.call_openai(self.conversation)
        self.conversation.append({'role': 'assistant', 'content': response})
        return response


class UserEnv:
    def __init__(self, base_env, env_type='auto'):
        self.base_env = base_env
        self.user_simulator = None
        self.preference = None
        self.preference_name = None
        if env_type == 'auto':
            base_class_names = [c.__name__ for c in type(base_env).__mro__]
            if 'LocalSearch' in base_class_names:
                self.env_type = 'search'
            elif 'GymEnv' in base_class_names:
                self.env_type = 'gym'
            else:
                self.env_type = 'gym'  # default
        else:
            self.env_type = env_type
        print(f'[UserEnv] Initialized with env_type={self.env_type}')

    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def _init_user_stats(self):
        self.base_env.stats['is_finish'] = 0
        self.base_env.stats['ask_turn'] = 0
        self.base_env.stats['if_ask'] = 0
        for level in range(1, 6):
            self.base_env.stats[f'level_{level}'] = 0
        self.base_env.stats['level_sum'] = 0

    def _extract_preference_info(self, extra_info: dict) -> tuple:
        if 'preference' not in extra_info:
            return None, None, None

        preference = extra_info['preference']
        if 'preference_name' in preference:
            preference_name = preference['preference_name']
        else:
            preference_name = None
            for key, value in preference.items():
                if value is not None and key != 'preference_name':
                    preference_name = key
                    break
        if preference_name is None:
            return preference, None, None
        preference_info = preference[preference_name]
        preference_str = preference_info.get('preference', '')

        return preference, preference_name, preference_str

    async def _handle_ask_question(self, arguments: dict) -> dict:
        ask_response = await self.user_simulator.execute(arguments)

        print(f'[UserEnv-{self.env_type}] Ask: {arguments}')
        print(f'[UserEnv-{self.env_type}] Response: {ask_response}')
        for level in range(1, 6):
            if f'[Cost {level}]' in ask_response:
                self.base_env.stats[f'level_{level}'] += 1
                self.base_env.stats['level_sum'] += level
                break

        self.base_env.stats['ask_turn'] += 1
        self.base_env.stats['if_ask'] = 1

        return {'observation': ask_response}

    def _build_prompt(self, preference_str, task_info):
        prompt = (
            f'You are playing the role of a human user responding to an agent\'s questions. '
            f'Your behavior is dictated by a specific user preference. You will assign a cost '
            f'based on the sensitivity of the information you share.\n\n---\n\n'
            f'## User Preference\n\nYour personality and response style are determined by the '
            f'user preference. You must strictly adhere to this role.\n\n'
            f'* **Your Preference**: {preference_str}\n\n'
            f'---\n\n## Information & Cost\n\n'
            f'You have access to the following categories of information, listed from least to '
            f'most sensitive. You will reveal information reluctantly, starting with the least '
            f'sensitive category.\n\n'
        )
        if self.env_type == 'gym':
            prompt += '* **Full Question**: The detailed problem description.\n* **Hint Information**: Additional helpful context or clues.\n* **Edit File Path**: The specific file(s) that need to be modified.\n* **Edit Function Name**: The specific function(s) or class(es) to be changed.\n\nYou must assign a cost from 1–5 based on the **most sensitive** information used in your answer.\n\n* **Cost 1**: You used and only used **Full Question** information.\n* **Cost 2**: You used **Hint Information**.\n* **Cost 3**: Information not provided. You refused to answer, said "I don\'t know".\n* **Cost 4**: You used the **Edit File Path**.\n* **Cost 5**: You used the **Edit Function Name**.\n'
        else:  # search
            prompt += '* **Full Search Problem**: Detailed problem description\n* **Ground Truth Document Snapshot**: The correct page to the search problem with the first 100 words.\n\nYou must assign a cost from 1–3 based on the **most sensitive** information used in your answer.\n\n* **Level 1**: Used only Full Search Problem\n* **Level 2**: Refused, said "I don\'t know", or gave general answers\n* **Level 3**: Used Ground Truth Document Snapshot\n'
        prompt += (
            '\n---\n\n## Answering Rules\n\n'
            '* **Adhere to your preference** above all else.\n'
            '* **Default Style**: Unless the preference dictates otherwise, keep replies short, casual, and natural. Use contractions (e.g., "it\'s," "I don\'t"). Talk casually, like a real person. Use short sentences and contractions. Always write in a single paragraph.\n'
            '* **Escalate Carefully**: Only reveal more sensitive information if the agent is stuck and the question requires it.\n'
            '* **Stay in Scope**: If a question is about something not covered by the provided information, respond with "I don\'t know" or a similar refusal.\n'
            '* **Be Human**: Never explain your role, the rules, or the cost. Do not use markdown, bullet points, or code blocks. Never ask the agent a question. Be very concise. It\'s okay to sound uncertain or reflective sometimes.\n\n'
            '---\n\n## Output Format\n\nYour response must contain exactly these parts in this order:\n\n'
            '1.  A single-paragraph, human-style, concise reply.\n'
            '2.  The cost tag on a new line (e.g., `[Cost 3]`).\n'
        )

        if self.env_type == 'gym':
            prompt += f'\n\n--------------------- TASK INFORMATION ---------------------\n\n**Full Question:**\n{task_info["problem_statement"]}\n\n**Hint Information:**\n{task_info.get("hint", "None")}\n\n**Edit File Path**:\n{task_info.get("edit_file", "")}\n\n**Edit Function Name**:\n{task_info.get("edit_function", "")}\n\n'
        else:  # search
            prompt += f'\n\n--------------------- TASK INFORMATION ---------------------\n\n**Full Search Problem:**\n{task_info["full_query"]}\n\n**Ground Truth Document Snapshot:**\n{task_info.get("gold_doc_content", "")}\n\n'

        prompt += (
            f'Again, your preference is: {preference_str}\n\n'
            'Now answer the following agent\'s question based on these instruction information, '
            'your response should be very concise, in one sentence with few words. Make sure the '
            'cost prediction is accurate and fully follow the system prompt instructions.\n\n'
        )
        return prompt

    async def init_env(self, item):
        await self.base_env.init_env(item)
        self._init_user_stats()

        extra_info = item.non_tensor_batch['extra_info'][0]
        preference, preference_name, preference_str = self._extract_preference_info(extra_info)

        self.preference = preference
        self.preference_name = preference_name

        task_info = {}

        # Task specific user prompt info
        if self.env_type == 'gym':
            def get_diff_file(diff_text: str):
                _FILE = re.compile(r'^\+\+\+\s+b/(.+)$')
                _HUNK = re.compile(r'^@@.*@@\s*(.*)$')
                _SYMS = re.compile(r'\b(?:def|class)\s+([A-Za-z_]\w*)')
                _CHANGE = re.compile(r'^[+-]\s*(?:def|class)\s+([A-Za-z_]\w*)')
                lines = diff_text.splitlines()
                files = [m[1] for l in lines if (m := _FILE.match(l))]
                syms = []
                for l in lines:
                    if (m := _HUNK.match(l)):
                        syms += _SYMS.findall(m[1])
                    elif (m := _CHANGE.match(l)):
                        syms.append(m[1])
                return '\n'.join(list(dict.fromkeys(files))), '\n'.join(list(dict.fromkeys(syms)))

            instance_info = self.base_env.instance_info
            edit_file, edit_function = get_diff_file(instance_info['patch'])
            if not getattr(self.base_env.config.plugin, 'use_gold', True):
                edit_file, edit_function = "", ""
            if preference_name == 'amateur':
                edit_file, edit_function = "", ""
            task_info = {
                'problem_statement': instance_info['problem_statement'],
                'hint': instance_info.get('hint', 'None'),
                'edit_file': edit_file,
                'edit_function': edit_function
            }

        elif self.env_type == 'search':
            max_local_search_gt_file = getattr(self.base_env.config.plugin, 'max_local_search_gt_file', None)
            self.base_env.instance_info = copy.deepcopy(extra_info)

            if max_local_search_gt_file:
                gold_docs = self.base_env.instance_info['evidence_docs']
                selected_docs = random.sample(list(gold_docs), min(len(gold_docs), max_local_search_gt_file))

                gold_doc_content = ""
                for doc in selected_docs:
                    doc_id = doc['docid']
                    open_pages = await self.base_env.client.open(None, doc_id)
                    for page in open_pages:
                        page['text'] = keep_first_n_words(page['text'], 50)
                        gold_doc_content += f"[DOC {doc_id}] \n{page['text']}\n\n"
            else:
                gold_doc_content = "No documents available."

            task_info = {
                'full_query': self.base_env.instance_info['full_query'],
                'gold_doc_content': gold_doc_content
            }

        if preference_name:
            user_prompt = self._build_prompt(preference_str, task_info)
        else:
            # Default prompt without preference - simplified version
            user_prompt = self._build_prompt('No specific preference', task_info)

        print(f'[UserEnv-{self.env_type}] Preference: {self.preference_name}')

        # Create user simulator
        self.user_simulator = UserSimulator(
            user_prompt,
            model=getattr(self.base_env.config.plugin, 'user_model', None)
        )

        self.base_env.NO_FNCALL_PROMPT = (
            "Please continue working on the task.\n"
            "If you want to ask user question, use ask_question tool.\n"
            "If you think you have solved the task, please first send your answer to user "
            "through message and then finish the interaction.\n"
            "If you want to give up, use the \"finish\" tool to finish the interaction."
        )

    async def run_action(self, response):
        fn_call = extract_fn_call(response)
        if fn_call is None or len(fn_call) == 0:
            return {'observation': 'No function call was detected in the model response.'}

        action = fn_call[0]
        arguments = json.loads(action['arguments']) if isinstance(action['arguments'], str) else action['arguments']
        name = action['function']

        if name == 'ask_question':
            return await self._handle_ask_question(arguments)
        elif name == 'finish':
            self.base_env.stats['is_finish'] = 1
        return await self.base_env.run_action(response)
