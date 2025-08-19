# LLM Shopping Experiment Framework

This repository provides a framework to test different Large Language Models (LLMs) in a **budget‑constrained shopping scenario**, simulating a Walmart retail API. It supports both **simulated strategies** and **real API calls** to OpenAI, Google Gemini, and Anthropic Claude when API keys are provided.

---

## Features

- **ShopAPI Simulation**
  - Provides a catalog of products with `id`, `name`, `price`, and `category` visible to LLMs.
  - Internal attributes (`sodium`, `alcohol`, `healthy_score`) are hidden from the models but available for analysis.
  - Supports searching, listing, retrieving product info, and purchasing baskets.

- **LLM Abstraction Layer**
  - `LLMBase`: Abstract base class for LLM interactions.
  - Implementations:
    - `ChatGPT5LLM` (OpenAI GPT‑5)
    - `Gemini25LLM` (Google Gemini 2.5)
    - `Claude5LLM` (Anthropic Claude 5)
  - Each model can:
    - Use **real API calls** (if `use_real_api=True` and API keys are set).
    - Fall back to a **simulated selection strategy** when offline.

- **ExperimentRunner**
  - Runs repeated trials across models and budgets.
  - Records results (JSON basket, spend, budget adherence, etc.) into a CSV.
  - Supports configurable temperature for controlled randomness.

- **CLI Tool**
  - Generate dummy catalogs or load from CSV.
  - Run experiments across multiple budgets.
  - Save experiment results to CSV.
  - Optional integration with real LLM provider APIs.

---

## Installation

```bash
# Clone repo
git clone https://github.com/your-org/llm-shopping-experiment.git
cd llm-shopping-experiment

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- pandas, numpy
- Provider SDKs (optional, only if using real APIs):
  - `openai`
  - `google-generativeai`
  - `anthropic`

---

## Usage

### 1. Run with Dummy Catalog (Simulation Only)
```bash
python llm_shopping_experiment.py --runs 3
```
This generates a random 50‑item catalog and runs 3 trials per model at budgets $27, $54, and $108.

### 2. Use a Custom Catalog
```bash
python llm_shopping_experiment.py --catalog_csv ./catalog.csv --runs 5
```

### 3. Enable Real API Calls
Make sure you have API keys as environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Then run:
```bash
python llm_shopping_experiment.py --use_real_api --runs 2 \
    --chatgpt5_model gpt-5 \
    --gemini_model gemini-2.5 \
    --claude_model claude-5
```

### 4. Command‑Line Options
| Option | Description |
|--------|-------------|
| `--catalog_csv PATH` | Path to product catalog CSV (else dummy generated) |
| `--output_csv PATH` | Path to save experiment results CSV |
| `--interventions_file PATH` | File with intervention prompts (one per line) |
| `--runs INT` | Runs per model |
| `--temperature FLOAT` | Sampling temperature (default: 0.7) |
| `--use_real_api` | If set, will call provider APIs when keys/SDKs available |
| `--openai_key` | Override OpenAI key (else env var used) |
| `--gemini_key` | Override Gemini key (else env var used) |
| `--anthropic_key` | Override Anthropic key (else env var used) |
| `--chatgpt5_model` | Model ID override for OpenAI (default: gpt-5) |
| `--gemini_model` | Model ID override for Gemini (default: gemini-2.5) |
| `--claude_model` | Model ID override for Claude (default: claude-5) |

---

## Output
The results CSV includes:
- Model name (`chatgpt5`, `gemini-2.5`, `claude-5`)
- Budget
- Run index
- Assistant JSON basket
- Purchased total
- Spend percentage of budget
- Whether budget was exceeded
- Metadata: system prompt length, number of interventions, temperature, API usage flag, provider model ID

Example row:
```csv
model,budget,run_idx,assistant_json,purchased_total,spend_pct,budget_exceeded,temperature,used_real_api,provider_model_id
chatgpt5,54.0,0,"{...}",53.2,0.9852,False,0.7,True,gpt-5
```

---

## Development Notes
- Offline simulation ensures experiments can run without API keys.
- When `use_real_api=True`, the framework calls:
  - OpenAI Chat Completions API (`gpt-5`)
  - Google Gemini (`gemini-2.5`)
  - Anthropic Claude (`claude-5`)
- Results can vary depending on model availability, API quotas, and randomness (temperature).

---

## License
MIT License © 2025