"""Accès à l'API Claude : routage multi-modèles, cache de prompt, suivi des coûts, garde budgétaire.

Le routage associe à chaque niveau de difficulté un modèle et un niveau d'effort :
  - "leger"   : Haiku 5.5  (extraction, résumé, mise en forme)
  - "code"    : Sonnet 5.5 (génération et correction de code)
  - "jugement": Opus 5.5   (méthodologie, orchestration, verdicts)
Le mode « tout_opus » sert de témoin de coût : tous les agents passent sur Opus.

La clé est lue dans la variable d'environnement ANTHROPIC_API_KEY (jamais dans le code).
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

# Prix en $ par million de jetons (grille publique Anthropic, novembre 2026)
PRICES = {
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_w": 0.125, "cache_r": 0.01},
    "claude-sonnet-5-5": {"in": 2.0, "out": 10.0, "cache_w": 2.50, "cache_r": 0.10},
    "claude-opus-5-5": {"in": 4.0, "out": 20.0, "cache_w": 5.0, "cache_r": 0.20},
}
HAIKU_LONG = {"in": 0.50, "out": 2.50, "cache_w": 0.625, "cache_r": 0.05}  # prompts > 100 k jetons

# Réflexion interne : désactivée pour les tâches légères (coût), adaptative sinon.
THINKING = {"leger": {"type": "disabled"}, "code": {"type": "adaptive"}, "jugement": {"type": "adaptive"}}

ROUTING = {
    "route": {
        "leger": ("claude-haiku-5-5", "low"),
        "code": ("claude-sonnet-5-5", "medium"),
        "jugement": ("claude-opus-5-5", "high"),
    },
    "tout_opus": {
        "leger": ("claude-opus-5-5", "high"),
        "code": ("claude-opus-5-5", "high"),
        "jugement": ("claude-opus-5-5", "high"),
    },
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    """Registre de tous les appels : jetons, coût, latence, par agent et par modèle."""
    path: Path
    budget_usd: float
    spent: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():  # reprise : on recompte ce qui a déjà été dépensé
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.spent += json.loads(line)["cost_usd"]

    def record(self, rec: dict):
        with self.lock:
            self.spent += rec["cost_usd"]
            with self.path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def check(self, margin: float = 0.0):
        if self.spent + margin > self.budget_usd:
            raise BudgetExceeded(f"Budget atteint : {self.spent:.2f} $ / {self.budget_usd:.2f} $")


def cost_of(model: str, usage) -> float:
    total_in = usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
    p = PRICES[model]
    if model == "claude-haiku-5-5" and total_in > 100_000:
        p = HAIKU_LONG
    return (usage.input_tokens * p["in"] + usage.output_tokens * p["out"]
            + (usage.cache_creation_input_tokens or 0) * p["cache_w"]
            + (usage.cache_read_input_tokens or 0) * p["cache_r"]) / 1e6


class LLM:
    def __init__(self, ledger: Ledger, mode: str = "route", run_id: str = ""):
        self.client = anthropic.Anthropic(max_retries=6)
        self.ledger = ledger
        self.mode = mode
        self.run_id = run_id

    def call(self, *, agent: str, tier: str, system: str, messages: list, max_tokens: int = 8000,
             cached_context: str | None = None, json_schema: dict | None = None) -> tuple[str, dict]:
        """Un appel. `cached_context` (ex. grille ASOP) est placé en tête avec cache_control."""
        self.ledger.check(margin=0.05)
        model, effort = ROUTING[self.mode][tier]
        sys_blocks = []
        if cached_context:
            sys_blocks.append({"type": "text", "text": cached_context, "cache_control": {"type": "ephemeral"}})
        sys_blocks.append({"type": "text", "text": system})
        output_config = {"effort": effort}
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        t0 = time.time()
        thinking = THINKING["jugement"] if self.mode == "tout_opus" else THINKING[tier]
        # Flux (streaming) obligatoire pour les réponses longues ; on récupère le message final.
        with self.client.messages.stream(model=model, max_tokens=max_tokens, system=sys_blocks,
                                         messages=messages, output_config=output_config,
                                         thinking=thinking) as stream:
            resp = stream.get_final_message()
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        rec = {
            "run_id": self.run_id, "mode": self.mode, "agent": agent, "tier": tier, "model": model,
            "effort": effort, "in": u.input_tokens, "out": u.output_tokens,
            "thinking": getattr(getattr(u, "output_tokens_details", None), "thinking_tokens", 0) or 0,
            "cache_w": u.cache_creation_input_tokens or 0, "cache_r": u.cache_read_input_tokens or 0,
            "cost_usd": cost_of(model, u), "latency_s": round(time.time() - t0, 1),
            "stop": resp.stop_reason, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.ledger.record(rec)
        return text, rec


def default_ledger(budget: float = 19.0) -> Ledger:
    root = Path(__file__).resolve().parents[1]
    return Ledger(root / "results" / "llm_calls.jsonl", budget_usd=budget)


def require_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY absente de l'environnement.")
