"""Client LLM avec garde budgétaire et journal de chaque appel.

La clé est lue dans la variable d'environnement ANTHROPIC_API_KEY (jamais dans le dépôt).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import anthropic

PRIX = {  # USD par million de jetons (entrée, sortie)
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


class BudgetDepasse(RuntimeError):
    pass


class Client:
    def __init__(self, journal: Path, budget_usd: float = 5.0):
        self.c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                     base_url="https://api.anthropic.com", max_retries=4)
        self.journal = journal
        self.budget = budget_usd
        self.cout = 0.0
        if journal.exists():
            for line in journal.read_text().splitlines():
                self.cout += json.loads(line)["cout_usd"]

    def call(self, tache: str, model: str, system: str, user: str, max_tokens: int) -> tuple[str, str]:
        if self.cout >= self.budget:
            raise BudgetDepasse(f"budget atteint : {self.cout:.2f} USD")
        t0 = time.time()
        r = self.c.messages.create(model=model, max_tokens=max_tokens, system=system,
                                   messages=[{"role": "user", "content": user}])
        pin, pout = PRIX[model]
        cout = (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
        self.cout += cout
        rec = dict(tache=tache, modele=model, jetons_entree=r.usage.input_tokens,
                   jetons_sortie=r.usage.output_tokens, cout_usd=round(cout, 6),
                   stop_reason=r.stop_reason, duree_s=round(time.time() - t0, 2))
        with self.journal.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        text = "".join(b.text for b in r.content if b.type == "text")
        if r.stop_reason == "max_tokens":
            print(f"ALERTE max_tokens sur {tache}")
        return text, r.stop_reason


def parse_json(text: str):
    """Extrait le premier bloc JSON (objet ou liste) d'une réponse."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    start = min([i for i in (s.find("{"), s.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("pas de JSON")
    return json.JSONDecoder().raw_decode(s[start:])[0]
