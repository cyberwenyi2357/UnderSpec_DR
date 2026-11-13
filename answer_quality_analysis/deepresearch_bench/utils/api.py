import os
from typing import Optional, Dict, Any
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

logging.getLogger('httpx').setLevel(logging.WARNING)

# Read API keys from environment variables
READ_API_KEY = os.environ.get("JINA_API_KEY", "")
FACT_Model = "gemini-3.8-flash"
Model = "gemini-pro-latest"


def _make_gemini_client(api_key: str, model: str):
    from google import genai
    from google.genai import types
    import logging
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('google.genai').setLevel(logging.WARNING)

    client = genai.Client(api_key=api_key, http_options={'timeout': 600000})

    def generate(user_prompt, system_prompt="", override_model=None):
        from google.genai import types as _types
        model_to_use = override_model or model
        contents = []
        if system_prompt:
            contents.append({"role": "system", "parts": [{"text": system_prompt}]})
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})
        try:
            response = client.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=_types.GenerateContentConfig(
                    thinking_config=None,
                    temperature=0
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}")

    return generate


def _make_openai_client(api_key: str, model: str):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=600)

    def generate(user_prompt, system_prompt="", override_model=None):
        model_to_use = override_model or model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            response = client.chat.completions.create(model=model_to_use, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}")

    return generate


class AIClient:

    def __init__(self, api_key=None, model=None, backend="gemini"):
        """
        backend: "gemini" (default) or "gpt"
        """
        self.backend = backend
        if backend == "gpt":
            _key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not _key:
                raise ValueError("GPT backend requires OPENAI_API_KEY environment variable.")
            self.model = model or "gpt-4.1"
            self._generate = _make_openai_client(_key, self.model)
        else:
            _key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not _key:
                raise ValueError("Gemini API key not provided! Please set GEMINI_API_KEY environment variable.")
            self.model = model or Model
            self._generate = _make_gemini_client(_key, self.model)

    def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        return self._generate(user_prompt, system_prompt, model)

class WebScrapingJinaTool:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not provided! Please set JINA_API_KEY environment variable.")

    def __call__(self, url: str) -> Dict[str, Any]:
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                "Accept": "application/json",
                'Authorization': self.api_key,
                'X-Timeout': "60000",
                "X-With-Generated-Alt": "true",
            }
            response = requests.get(jina_url, headers=headers)

            if response.status_code != 200:
                raise Exception(f"Jina AI Reader Failed for {url}: {response.status_code}")

            response_dict = response.json()

            return {
                'url': response_dict['data']['url'],
                'title': response_dict['data']['title'],
                'description': response_dict['data']['description'],
                'content': response_dict['data']['content'],
                'publish_time': response_dict['data'].get('publishedTime', 'unknown')
            }

        except Exception as e:
            logger.error(str(e))
            return {
                'url': url,
                'content': '',
                'error': str(e)
            }
        
_jina_tool = None


def scrape_url(url: str) -> Dict[str, Any]:
    # Created on first use so that importing utils (e.g. for RACE) doesn't require JINA_API_KEY.
    global _jina_tool
    if _jina_tool is None:
        _jina_tool = WebScrapingJinaTool()
    return _jina_tool(url)
    
def call_model(user_prompt: str) -> str:
    client = AIClient(model=FACT_Model)
    return client.generate(user_prompt)

if __name__ == "__main__":
    url = ""
    result = scrape_url(url)
    print(result)