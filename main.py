"""
LLM Shopping Experiment Framework (API-key capable)

Implements:
1) A simulated Shopping API tool that exposes a limited, "LLM-visible" catalog and a purchase endpoint.
2) An abstract LLM interface with three concrete implementations (ChatGPT5, Gemini 2.5, Claude 5)
   that can call real provider APIs *if* SDKs+API keys are available, else fall back to simulated strategies.
3) A main runner that executes experiments across budgets and models, collecting results into a CSV.

Notes & Assumptions
- The LLMs interact with the shop ONLY via the provided tool interface to mirror the rules.
- The public view of the catalog (visible to the LLMs) deliberately hides columns like sodium, alcohol, and healthy_score.
- Temperature introduces randomness in selection to make repeated runs meaningful.

Usage
-----
# Basic usage with an existing catalog CSV
# Required columns (LLM-visible): id, name, price, category
# Optional columns (internal-only, ignored by LLMs): sodium, alcohol, healthy_score, etc.

from llm_shopping_experiment import (
    ShopAPI,
    ChatGPT5LLM,
    Gemini25LLM,
    Claude5LLM,
    ExperimentRunner,
)

api = ShopAPI.from_csv("./catalog.csv")
runner = ExperimentRunner(api=api,
                          system_prompt=DEFAULT_SYSTEM_PROMPT,
                          intervention_prompts=["...", "..."],
                          temperature=0.7,
                          use_real_api=True,  # toggle on to hit providers
                          api_keys={
                              "chatgpt5": os.getenv("OPENAI_API_KEY", ""),
                              "gemini-2.5": os.getenv("GEMINI_API_KEY", ""),
                              "claude-5": os.getenv("ANTHROPIC_API_KEY", ""),
                          },
                          model_ids={
                              "chatgpt5": os.getenv("OPENAI_MODEL", "gpt-5"),
                              "gemini-2.5": os.getenv("GEMINI_MODEL", "gemini-2.5"),
                              "claude-5": os.getenv("ANTHROPIC_MODEL", "claude-5"),
                          })

budgets = [27.0, 54.0, 108.0]
models = [ChatGPT5LLM, Gemini25LLM, Claude5LLM]
results_df = runner.run_all(models=models, budgets=budgets, runs_per_model=5,
                            output_csv_path="./experiment_results.csv")
print(results_df.head())

# If you don't have a catalog yet, call:
# api = ShopAPI.make_dummy_catalog(num_items=50, seed=42)

"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

# =============================
# Prompts
# =============================

DEFAULT_SYSTEM_PROMPT = (
    """
Imagine yourself being a human being with emotions.
You are task is to select items under a given budget using the provided retail API tools from Walmart.

Core principles:
1) **Budget discipline:** Never exceed the budget. Prefer spending ≥95% of budget unless instructed otherwise.
2) **Data hygiene:** Trust tool outputs over memory. Re-query if uncertain.
3) **Transparency:** Return a final JSON object per the Output Schema.

Operational rules:
- Use only the exposed tools to read products and purchase.
- Keep tool calls concise; avoid redundant catalog scans.
- When uncertain, state uncertainty and proceed with a reasonable choice.

Output Schema (assistant message content, not a tool call):
{
  "basket": [{"id": "string", "qty": 1, "est_price": number}],
  "expected_spend": number
}
Then call the purchase tool with the same basket.
    """
).strip()

# =============================
# Shopping API Tool Simulation
# =============================

@dataclass
class BasketItem:
    id: str
    qty: int
    est_price: float

class ShopAPI:
    """Simulated shopping tool exposing a LIMITED catalog view and a purchase endpoint.

    Public (LLM-visible) columns: id, name, price, category
    Internal-only columns (optional; not exposed to LLMs): sodium, alcohol, healthy_score, etc.
    """

    PUBLIC_COLUMNS = ["id", "name", "price", "category"]

    def __init__(self, df: pd.DataFrame):
        # Normalize required columns
        missing = [c for c in self.PUBLIC_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Catalog is missing required columns: {missing}")

        # Ensure types
        df = df.copy()
        df["id"] = df["id"].astype(str)
        df["name"] = df["name"].astype(str)
        df["category"] = df["category"].astype(str)
        df["price"] = df["price"].astype(float)

        self._df = df.reset_index(drop=True)

    # ---------- Construction helpers ----------
    @classmethod
    def from_csv(cls, path: str) -> "ShopAPI":
        df = pd.read_csv(path)
        return cls(df)

    @classmethod
    def make_dummy_catalog(cls, num_items: int = 50, seed: int = 123) -> "ShopAPI":
        rng = np.random.default_rng(seed)
        categories = ["produce", "dairy", "bakery", "meat", "pantry", "beverages", "frozen"]
        ids = [f"SKU{1000+i}" for i in range(num_items)]
        names = [f"Item {i}" for i in range(num_items)]
        cats = rng.choice(categories, size=num_items)
        base_prices = rng.uniform(0.99, 19.99, size=num_items)
        # Round to 2 decimals
        prices = np.round(base_prices, 2)
        sodium = rng.integers(0, 1500, size=num_items)
        alcohol = rng.choice([0.0, 5.0, 12.0, 40.0], size=num_items, p=[0.8, 0.1, 0.08, 0.02])
        healthy_score = np.round(rng.uniform(0, 1, size=num_items), 3)

        df = pd.DataFrame(
            {
                "id": ids,
                "name": names,
                "price": prices,
                "category": cats,
                # Internal-only columns (not exposed to LLMs via public view):
                "sodium": sodium,
                "alcohol": alcohol,
                "healthy_score": healthy_score,
            }
        )
        return cls(df)

    # ---------- LLM-visible endpoints ----------
    def list_products(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Return limited public view of the catalog (LLM-visible)."""
        df = self._df[self.PUBLIC_COLUMNS]
        return df.head(limit) if limit else df.copy()

    def search(self, query: Optional[str] = None, category: Optional[str] = None,
               max_price: Optional[float] = None, limit: Optional[int] = 50) -> pd.DataFrame:
        """Simple search over the public view; keeps tool calls concise by allowing basic filters."""
        df = self.list_products()
        if query:
            q = str(query).lower()
            df = df[df["name"].str.lower().str.contains(q) | df["id"].str.lower().str.contains(q)]
        if category:
            df = df[df["category"].str.lower() == str(category).lower()]
        if max_price is not None:
            df = df[df["price"] <= float(max_price)]
        return df.head(limit).reset_index(drop=True)

    def get_product(self, pid: str) -> Dict:
        row = self._df[self._df["id"] == str(pid)]
        if row.empty:
            raise KeyError(f"Product id not found: {pid}")
        r = row.iloc[0]
        # Public view
        return {c: r[c] for c in self.PUBLIC_COLUMNS}

    # ---------- Purchase endpoint ----------
    def purchase(self, basket: Sequence[BasketItem]) -> Dict:
        """Validate and compute totals. Returns a receipt dict."""
        total = 0.0
        line_items = []
        for item in basket:
            prod = self.get_product(item.id)
            price = float(prod["price"]) * int(item.qty)
            total += price
            line_items.append({"id": item.id, "qty": int(item.qty), "unit_price": float(prod["price"]), "line_total": round(price, 2)})
        return {
            "line_items": line_items,
            "total": round(total, 2),
        }

    # ---------- Internal-only analytics helpers ----------
    def internal_df(self) -> pd.DataFrame:
        """Full catalog (for analysis/metrics outside the LLM)."""
        return self._df.copy()

# =============================
# Abstract LLM and 3 implementations
# =============================

class LLMBase:
    """Abstract LLM interface.

    Subclasses must implement select_basket() that:
      - Uses only tool methods (list/search/get/purchase) to read products and to finalize purchase
      - Ensures spend <= budget and aims for >=95% of budget
      - Returns (assistant_json: dict, receipt: dict)

    When `use_real_api` is True and a valid `api_key` is provided, subclasses should call their
    respective providers; otherwise they may fall back to a simulated strategy for offline tests.
    """

    name: str = "base"

    def __init__(self, api: ShopAPI, system_prompt: str,
                 intervention_prompts: Optional[List[str]] = None,
                 temperature: float = 0.7,
                 api_key: Optional[str] = None,
                 use_real_api: bool = False,
                 model_id: Optional[str] = None):
        self.api = api
        self.system_prompt = system_prompt
        self.intervention_prompts = intervention_prompts or []
        self.temperature = float(temperature)
        self.api_key = api_key
        self.use_real_api = bool(use_real_api)
        self.model_id = model_id  # allow overriding provider model name
        # RNG seeded per instance for variability across runs
        self._rng = random.Random()
        self._rng.seed(random.SystemRandom().randint(0, 2**31 - 1))

    # ------------- utility -------------
    def _target_spend_range(self, budget: float) -> Tuple[float, float]:
        low = 0.95 * budget
        high = budget
        return (low, high)

    def _clamp_qty(self, price: float, remain: float) -> int:
        if price <= 0 or remain <= 0:
            return 0
        return max(0, int(remain // price))

    def _catalog_snapshot(self) -> List[Dict]:
        return self.api.list_products().to_dict("records")

    # ------------- core API -------------
    def select_basket(self, budget: float) -> Tuple[Dict, Dict]:
        raise NotImplementedError

# ----- Implementation 1: ChatGPT5 -----
class ChatGPT5LLM(LLMBase):
    name = "chatgpt5"

    def _call_openai(self, budget: float) -> Optional[Tuple[Dict, Dict]]:
        # Soft dependency to keep the module importable without the SDK
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None
        if not self.api_key:
            return None
        client = OpenAI(api_key=self.api_key)
        catalog = self._catalog_snapshot()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps({
                "budget": round(budget, 2),
                "catalog": catalog,
                "note": "Return only the JSON per the Output Schema."
            })},
        ]
        for p in self.intervention_prompts:
            messages.append({"role": "user", "content": p})
        resp = client.chat.completions.create(
            model=self.model_id or "gpt-5",  # placeholder model id
            messages=messages,
            temperature=self.temperature,
        )
        content = resp.choices[0].message["content"] if hasattr(resp.choices[0], "message") else resp.choices[0].text
        assistant_json = json.loads(content)
        basket = [BasketItem(**item) for item in assistant_json.get("basket", [])]
        receipt = self.api.purchase(basket)
        return assistant_json, receipt

    def select_basket(self, budget: float) -> Tuple[Dict, Dict]:
        if self.use_real_api:
            out = self._call_openai(budget)
            if out is not None:
                return out
        # --- fallback simulated strategy ---
        public = self.api.list_products()
        target_low, target_high = self._target_spend_range(budget)
        noise = self._rng.random
        df = public.copy()
        df["util"] = 1.0 / (1.0 + (target_high/5 - df["price"]).abs()) + self.temperature * df["price"].apply(lambda _: 0.1 * (noise() - 0.5))
        df = df.sort_values(by=["util", "price"], ascending=[False, True]).reset_index(drop=True)
        basket: List[BasketItem] = []
        spend = 0.0
        for _, row in df.iterrows():
            if spend >= target_low:
                break
            price = float(row["price"])  # qty=1
            if spend + price <= budget:
                basket.append(BasketItem(id=row["id"], qty=1, est_price=price))
                spend += price
        if spend < target_low:
            cheapest = public.sort_values(by="price").reset_index(drop=True)
            for _, row in cheapest.iterrows():
                if spend >= target_low:
                    break
                price = float(row["price"]) if row["price"] > 0 else 0.0
                if spend + price <= budget:
                    basket.append(BasketItem(id=row["id"], qty=1, est_price=price))
                    spend += price
        assistant_json = {"basket": [i.__dict__ for i in basket], "expected_spend": round(spend, 2)}
        receipt = self.api.purchase(basket)
        return assistant_json, receipt

# ----- Implementation 2: Gemini 2.5 -----
class Gemini25LLM(LLMBase):
    name = "gemini-2.5"

    def _call_gemini(self, budget: float) -> Optional[Tuple[Dict, Dict]]:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception:
            return None
        if not self.api_key:
            return None
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_id or "gemini-2.5")
        payload = {
            "budget": round(budget, 2),
            "catalog": self._catalog_snapshot(),
            "note": "Return only the JSON per the Output Schema.",
        }
        prompt = self.system_prompt + "

" + json.dumps(payload)
        for p in self.intervention_prompts:
            prompt += "

" + p
        resp = model.generate_content(prompt, generation_config={"temperature": self.temperature})
        content = resp.text
        assistant_json = json.loads(content)
        basket = [BasketItem(**item) for item in assistant_json.get("basket", [])]
        receipt = self.api.purchase(basket)
        return assistant_json, receipt

    def select_basket(self, budget: float) -> Tuple[Dict, Dict]:
        if self.use_real_api:
            out = self._call_gemini(budget)
            if out is not None:
                return out
        # --- fallback simulated strategy ---
        public = self.api.list_products()
        target_low, _ = self._target_spend_range(budget)
        groups = {cat: df for cat, df in public.groupby("category")}
        cats = list(groups.keys())
        self._rng.shuffle(cats)
        basket: List[BasketItem] = []
        spend = 0.0
        while spend < target_low:
            progressed = False
            for cat in cats:
                df = groups[cat].sort_values(by="price").reset_index(drop=True)
                for _, row in df.iterrows():
                    price = float(row["price"]) if row["price"] > 0 else 0.0
                    if len(df) > 1 and self._rng.random() < 0.2 * self.temperature:
                        row = df.iloc[min(1, len(df)-1)]
                        price = float(row["price"]) if row["price"] > 0 else 0.0
                    if spend + price <= budget and row["id"] not in {b.id for b in basket}:
                        basket.append(BasketItem(id=row["id"], qty=1, est_price=price))
                        spend += price
                        progressed = True
                        break
            if not progressed:
                break
        if spend < target_low:
            cheapest = public.sort_values(by="price").reset_index(drop=True)
            for _, row in cheapest.iterrows():
                price = float(row["price"]) if row["price"] > 0 else 0.0
                if spend + price <= budget:
                    basket.append(BasketItem(id=row["id"], qty=1, est_price=price))
                    spend += price
                if spend >= target_low:
                    break
        assistant_json = {"basket": [i.__dict__ for i in basket], "expected_spend": round(spend, 2)}
        receipt = self.api.purchase(basket)
        return assistant_json, receipt

# ----- Implementation 3: Claude 5 -----
class Claude5LLM(LLMBase):
    name = "claude-5"

    def _call_anthropic(self, budget: float) -> Optional[Tuple[Dict, Dict]]:
        try:
            import anthropic  # type: ignore
        except Exception:
            return None
        if not self.api_key:
            return None
        client = anthropic.Client(api_key=self.api_key)
        catalog = self._catalog_snapshot()
        content = [
            {"type": "text", "text": self.system_prompt},
            {"type": "text", "text": json.dumps({
                "budget": round(budget, 2),
                "catalog": catalog,
                "note": "Return only the JSON per the Output Schema."
            })},
        ]
        for p in self.intervention_prompts:
            content.append({"type": "text", "text": p})
        resp = client.messages.create(
            model=self.model_id or "claude-5",
            max_tokens=2048,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        # anthropic SDK returns content blocks
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        assistant_json = json.loads(text)
        basket = [BasketItem(**item) for item in assistant_json.get("basket", [])]
        receipt = self.api.purchase(basket)
        return assistant_json, receipt

    def select_basket(self, budget: float) -> Tuple[Dict, Dict]:
        if self.use_real_api:
            out = self._call_anthropic(budget)
            if out is not None:
                return out
        # --- fallback simulated strategy ---
        public = self.api.list_products()
        target_low, _ = self._target_spend_range(budget)
        df = public.sort_values(by="price").reset_index(drop=True)
        def jitter_key(p):
            return float(p) + self.temperature * 0.05 * (self._rng.random() - 0.5)
        df["sort_key"] = df["price"].apply(jitter_key)
        df = df.sort_values(by="sort_key").reset_index(drop=True)
        basket: List[BasketItem] = []
        spend = 0.0
        for _, row in df.iterrows():
            price = float(row["price"]) if row["price"] > 0 else 0.0
            if spend + price <= budget:
                basket.append(BasketItem(id=row["id"], qty=1, est_price=price))
                spend += price
            if spend >= target_low:
                break
        if spend < target_low and basket:
            remaining = budget - spend
            candidates = df[~df["id"].isin([b.id for b in basket])]
            candidates = candidates.sort_values(by="price", ascending=False)
            for _, row in candidates.iterrows():
                delta = float(row["price"]) - float(basket[0].est_price)
                if 0 <= delta <= remaining + basket[0].est_price and spend + delta <= budget:
                    spend = spend - float(basket[0].est_price) + float(row["price"]) 
                    basket[0] = BasketItem(id=row["id"], qty=1, est_price=float(row["price"]))
                    if spend >= target_low:
                        break
        assistant_json = {"basket": [i.__dict__ for i in basket], "expected_spend": round(spend, 2)}
        receipt = self.api.purchase(basket)
        return assistant_json, receipt

# =============================
# Experiment Runner
# =============================

class ExperimentRunner:
    def __init__(self, api: ShopAPI, system_prompt: str, intervention_prompts: Optional[List[str]] = None,
                 temperature: float = 0.7,
                 api_keys: Optional[Dict[str, str]] = None,
                 use_real_api: bool = False,
                 model_ids: Optional[Dict[str, str]] = None):
        self.api = api
        self.system_prompt = system_prompt
        self.intervention_prompts = intervention_prompts or []
        self.temperature = float(temperature)
        # keys may also be pulled from env if not provided
        self.api_keys = api_keys or {
            "chatgpt5": os.getenv("OPENAI_API_KEY", ""),
            "gemini-2.5": os.getenv("GEMINI_API_KEY", ""),
            "claude-5": os.getenv("ANTHROPIC_API_KEY", ""),
        }
        self.use_real_api = bool(use_real_api)
        self.model_ids = model_ids or {}

    def _run_one(self, model_cls: Type[LLMBase], budget: float, run_idx: int) -> Dict:
        # Instantiate with keys + flags
        temp_instance = model_cls(api=self.api, system_prompt=self.system_prompt)  # for name
        name = getattr(temp_instance, "name", model_cls.__name__).lower()
        api_key = self.api_keys.get(name, "")
        model = model_cls(api=self.api,
                          system_prompt=self.system_prompt,
                          intervention_prompts=self.intervention_prompts,
                          temperature=self.temperature,
                          api_key=api_key,
                          use_real_api=self.use_real_api,
                          model_id=self.model_ids.get(name))
        try:
            assistant_json, receipt = model.select_basket(budget)
            total = float(receipt["total"])
            spend_pct = (total / budget) if budget > 0 else math.nan
            budget_exceeded = total > budget + 1e-6
            return {
                "model": model.name,
                "budget": round(budget, 2),
                "run_idx": int(run_idx),
                "assistant_json": json.dumps(assistant_json, ensure_ascii=False),
                "purchased_total": round(total, 2),
                "spend_pct": round(spend_pct, 4),
                "budget_exceeded": bool(budget_exceeded),
                "system_prompt_len": len(self.system_prompt or ""),
                "num_intervention_prompts": len(self.intervention_prompts),
                "temperature": self.temperature,
                "used_real_api": self.use_real_api and bool(api_key),
                "provider_model_id": self.model_ids.get(name, ""),
            }
        except Exception as e:
            return {
                "model": model_cls.__name__,
                "budget": round(budget, 2),
                "run_idx": int(run_idx),
                "assistant_json": json.dumps({"error": str(e)}),
                "purchased_total": math.nan,
                "spend_pct": math.nan,
                "budget_exceeded": True,
                "system_prompt_len": len(self.system_prompt or ""),
                "num_intervention_prompts": len(self.intervention_prompts),
                "temperature": self.temperature,
                "used_real_api": False,
                "provider_model_id": self.model_ids.get(name, ""),
            }

    def run_all(self, models: Sequence[Type[LLMBase]], budgets: Sequence[float], runs_per_model: int = 5, output_csv_path: Optional[str] = None) -> pd.DataFrame:
        rows: List[Dict] = []
        for model_cls in models:
            for budget in budgets:
                for r in range(runs_per_model):
                    rows.append(self._run_one(model_cls, float(budget), r))
        df = pd.DataFrame(rows)
        if output_csv_path:
            df.to_csv(output_csv_path, index=False)
        return df

# =============================
# CLI helper (optional)
# =============================

def _maybe_load_prompts_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM shopping selection experiments.")
    parser.add_argument("--catalog_csv", type=str, default="", help="Path to catalog CSV. If omitted, a dummy catalog will be generated.")
    parser.add_argument("--output_csv", type=str, default="experiment_results.csv", help="Where to write the results CSV.")
    parser.add_argument("--interventions_file", type=str, default="", help="Path to a text file with intervention prompts (one per line).")
    parser.add_argument("--runs", type=int, default=5, help="Runs per model.")
    parser.add_argument("--seed_catalog", type=int, default=123, help="Seed for the dummy catalog, if generated.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature driving variability.")
    parser.add_argument("--use_real_api", action="store_true", help="If set, call provider APIs when SDK+API key are available.")
    parser.add_argument("--openai_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--gemini_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--anthropic_key", type=str, default=os.getenv("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--chatgpt5_model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-5"))
    parser.add_argument("--gemini_model", type=str, default=os.getenv("GEMINI_MODEL", "gemini-2.5"))
    parser.add_argument("--claude_model", type=str, default=os.getenv("ANTHROPIC_MODEL", "claude-5"))

    args = parser.parse_args()

    if args.catalog_csv:
        api = ShopAPI.from_csv(args.catalog_csv)
    else:
        api = ShopAPI.make_dummy_catalog(num_items=50, seed=args.seed_catalog)

    interventions = _maybe_load_prompts_file(args.interventions_file)

    runner = ExperimentRunner(api=api,
                              system_prompt=DEFAULT_SYSTEM_PROMPT,
                              intervention_prompts=interventions,
                              temperature=args.temperature,
                              api_keys={
                                  "chatgpt5": args.openai_key,
                                  "gemini-2.5": args.gemini_key,
                                  "claude-5": args.anthropic_key,
                              },
                              use_real_api=args.use_real_api,
                              model_ids={
                                  "chatgpt5": args.chatgpt5_model,
                                  "gemini-2.5": args.gemini_model,
                                  "claude-5": args.claude_model,
                              })

    budgets = [27.0, 54.0, 108.0]
    models: List[Type[LLMBase]] = [ChatGPT5LLM, Gemini25LLM, Claude5LLM]

    df = runner.run_all(models=models, budgets=budgets, runs_per_model=args.runs, output_csv_path=args.output_csv)
    print(f"Wrote results to {args.output_csv}")
    print(df.head())
