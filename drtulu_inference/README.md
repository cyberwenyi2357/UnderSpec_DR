# `dr-agent-lib`

`dr-agent-lib` is an agent library for training and developing deep research agents. It supports:
- **MCP-Based Tool Backend**: Unified interface for web search and browsing tools
- **High Concurrency**: Global caching and async request management for RL training at scale
- **Flexible Prompting Interface**: Easy composition of search workflows with fine-grained control

## Quick start 

Run all commands from the `drtulu_inference/` directory. The agent backbone is either the released
[`rl-research/DR-Tulu-8B`](https://huggingface.co/rl-research/DR-Tulu-8B) (served locally with vLLM) or
GPT-5.4 (OpenAI API):

| Backbone | Workflow config |
|---|---|
| `rl-research/DR-Tulu-8B` | `workflows/auto_search_sft.yaml` (default) |
| GPT-5.4 | `workflows/auto_search_sft-gpt5.4.yaml` |

1. Installation 
    ```bash
    # [Optional] create an environment 
    # conda create -n dr_agent python=3.10 -y && conda activate dr_agent

    uv pip install -e ".[ui]"    
    # uv pip install vllm   # only needed for DR-Tulu-8B
    ```

2. Set up the environment variables for the backend service: `cp .env.example .env` and fill in

    ```bash
    SERPER_API_KEY=xxx # [Must have] Needed for serper search 
    S2_API_KEY=xxx     # Needed for snippet search 
    JINA_API_KEY=xxx   # Needed for jina-based browsing tool 
    OPENAI_API_KEY=xxx # Needed for GPT-5.4 and for the ask_user simulator (gpt-5.4-mini)
    ```
    You can get these API keys from the respective services.
    - `SERPER_API_KEY`: https://serper.dev/
    - `S2_API_KEY`: https://api.semanticscholar.org/
    - `JINA_API_KEY`: https://jina.ai/reader/

3. Start the MCP tool backend (port must match `mcp_port` in the workflow config):
    ```bash
    python -m dr_agent.mcp_backend.main --port 8001
    ```

4. For DR-Tulu-8B, serve the model with vLLM (1 GPU; port must match `search_agent_base_url`):
    ```bash
    CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --port 30001 --dtype auto --max-model-len 40960
    ```
    GPT-5.4 needs no GPU, only `OPENAI_API_KEY`.

5. Run the agent on a query file, e.g. one of `../query/*.jsonl`. Each line is one question; every
   non-empty variant field (`query`, `fully_underspecified`, `without_scope`, `without_term`,
   `without_preference`) is run and written to the output JSONL with its full trajectory.
    ```bash
    # DR-Tulu-8B
    python scripts/eval_query_template_batch.py --config workflows/auto_search_sft.yaml \
        --query-file ../query/researchqa.jsonl --output researchqa_dr-tulu.jsonl

    # GPT-5.4
    python scripts/eval_query_template_batch.py --config workflows/auto_search_sft-gpt5.4.yaml \
        --query-file ../query/researchqa.jsonl --output researchqa_gpt-5.4.jsonl
    ```
    The system prompt is chosen by `prompt_version` in the workflow config; the mapping from version to
    prompt file is in `dr_agent/shared_prompts/unified_tool_calling.py` (`v20250907` ->
    `original_prompt.yaml`, `v20250907_oai_safe` -> `original_prompt_oai_safe.yaml`, used for GPT-5.4).

6. Optionally, chat with the agent in a web UI:
    ```bash
    python workflows/auto_search_sft.py serve --port 8080                                              # DR-Tulu-8B
    python workflows/auto_search_sft.py serve --port 8080 --config workflows/auto_search_sft-gpt5.4.yaml # GPT-5.4
    ```

The other `workflows/*-local.yaml` files are configs for locally fine-tuned checkpoints that are not
released; they are not needed to reproduce the results.

> [!NOTE]
> If you run crawl4ai locally, you will need to install playwright and its dependencies.
## Deep Dive into the `serve` command: 

The `serve` command can turn any workflow into a fastapi service with the following endpoints 

- **`/chat`**: Simple request-response endpoint, returns the complete response as JSON
- **`/chat/stream`**: SSE streaming endpoint, streams thinking, tool calls, and answers in real-time

```bash 
# Launch the server (assumes MCP and model servers are running)
python workflows/auto_search_sft.py serve --port 8080

# Example: simple chat endpoint
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "What is the capital of France?"}'

# Example: streaming endpoint (SSE)
curl -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "What is the capital of France?"}'
```

### Protect the UI with the passwords

```bash
# You can generate a random password with 
# openssl rand -base64 32
python workflows/auto_search_sft.py serve --port 8080 --password "your-secure-password"
```

### Develop the chat frontend

```bash 
cd ../app && npm install && npm run dev 

# Change the ui url accordingly 
python workflows/auto_search_sft.py serve --port 8080 --ui-mode proxy --dev-url http://localhost:3000 
```

## Running evaluation on the datasets

We also provide a command to generate the results for examples in a given dataset: 

```bash
python workflows/auto_search_sft.py generate-dataset healthbench \
    --num-examples 5 \
    --max-concurrent 5 
    --output healthbench.jsonl
```

We include evaluation scripts for multiple benchmarks, including:
- **Long-form**: SQA-CS-V2, Deep Research Bench, ResearchQA, HealthBench, Genetic Diseases  
- **Short-form**: BrowseComp, SimpleQA, 2Wiki, etc.

For detailed evaluation instructions, benchmark descriptions, and usage examples, see [`evaluation/README.md`](evaluation/README.md). 

## [experimental] Interactive chat with CLI 

We provide an interactive cli demo for the auto_search workflow.
Requires 1-2 GPUs. We recommend running with `uv`, which should install everything you need and then launch the tool, but set your API keys first:

```bash
export SERPER_API_KEY="XXXX"
export S2_API_KEY="XXXX"
export JINA_API_KEY="XXXX"

uv run --extra vllm  python scripts/launch_chat.py --model rl-research/DR-Tulu-8B
```

Note for this cli demo, we use a slightly different prompt than the one used for evaluation in our paper, for demo purposes. The prompt is in the file `dr_agent/shared_prompts/unified_tool_calling_cli.yaml`.


We provide additional flags for the chat script, for e.g. showing full tool output:
```bash
usage: launch_chat.py [-h] [--config CONFIG] [--dataset-name DATASET_NAME]
                      [--model MODEL] [--config-overrides CONFIG_OVERRIDES]
                      [--verbose] [--show-full-tool-output] [--skip-checks]
                      [--mcp-port MCP_PORT] [--gpu-id GPU_ID]
                      [--no-auto-launch]

Self-contained launcher for interactive chat

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Config file path (default:
                        workflows/auto_search_sft.yaml)
  --dataset-name DATASET_NAME, -d DATASET_NAME
                        Dataset name for dataset-specific instructions
  --model MODEL, -m MODEL
                        Main model name (for search agent). If not provided,
                        uses config defaults.
  --config-overrides CONFIG_OVERRIDES
                        Config overrides (e.g., 'param1=value1,param2=value2')
  --verbose, -v         Enable verbose output
  --show-full-tool-output
                        Show full tool output instead of truncating to 500
                        characters
  --skip-checks         Skip checking/launching services
  --mcp-port MCP_PORT   MCP server port (default: 8000)
  --gpu-id GPU_ID       GPU ID for search agent vLLM server (default: 0,
                        browse agent uses GPU 1)
  --no-auto-launch      Don't automatically launch vLLM servers (check only)

Examples:
  # Basic usage (auto-launches MCP server and vLLM servers if needed)
  python scripts/launch_chat.py

  # With specific model (auto-launches both vLLM servers on GPUs 0 and 1)
  python scripts/launch_chat.py --model rl-research/DR-Tulu-8B

  # Skip service checks (if services are already running)
  python scripts/launch_chat.py --skip-checks

  # Don't auto-launch vLLM servers (just check)
  python scripts/launch_chat.py --no-auto-launch

  # Custom config file
  python scripts/launch_chat.py --config workflows/auto_search_sft.yaml
```