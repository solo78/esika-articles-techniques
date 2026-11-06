"""Mini-cadre multi-agents (équivalent fonctionnel de CrewAI, sans dépendance).

Correspondance avec les composants décrits dans le papier :
  - Agent           : rôle, objectif, histoire (persona) + niveau de modèle (routage)
  - Task            : description, résultat attendu, contexte (tâches amont), porte de qualité
  - MemoryStream    : flux de mémoire partagé (entrées, code, sorties, rapports, horodatage)
  - outils          : exécuteur de code Python, outil EDA, outil CAG (grille ASOP en cache)
La différence majeure avec le papier : aucune intervention humaine. Les erreurs
d'exécution et les échecs de portes sont renvoyés automatiquement à l'agent.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .llm import LLM


@dataclass
class Agent:
    name: str
    role: str
    goal: str
    backstory: str
    tier: str  # "leger" | "code" | "jugement"

    def system_prompt(self, extra: str = "") -> str:
        return (f"Tu es {self.role}.\nObjectif : {self.goal}\nProfil : {self.backstory}\n"
                "Tu travailles dans une équipe d'agents sans superviseur humain : sois rigoureux, "
                "vérifie toi-même tes résultats et signale honnêtement toute limite.\n" + extra)


@dataclass
class Task:
    id: str
    agent: Agent
    description: str
    expected_output: str
    kind: str = "code"  # "code" | "text" | "json"
    context: list[str] = field(default_factory=list)  # ids de tâches amont dont on reçoit le rapport
    gate: Callable[[Path], tuple[bool, str]] | None = None
    max_attempts: int = 3
    timeout_s: int = 1500
    json_schema: dict | None = None
    cached_context: str | None = None  # ex. grille ASOP (CAG)
    max_tokens: int = 24000


class MemoryStream:
    """Flux de mémoire : chaque entrée est horodatée et lisible par tous les agents."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []

    def add(self, **entry):
        entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.entries.append(entry)
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=1))

    def last(self, task_id: str) -> dict | None:
        for e in reversed(self.entries):
            if e.get("task") == task_id and e.get("type") == "result":
                return e
        return None

    def reports(self, task_ids: list[str]) -> str:
        parts = []
        for t in task_ids:
            e = self.last(t)
            if e:
                parts.append(f"### Rapport de la tâche « {t} » ({e['agent']})\n{e['report']}")
        return "\n\n".join(parts)


CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    blocks = CODE_RE.findall(text)
    return max(blocks, key=len) if blocks else None


def extract_report(text: str) -> str:
    m = re.search(r"#+\s*Rapport\s*\n(.*)", text, re.S | re.I)
    rep = m.group(1) if m else CODE_RE.sub("", text)
    return rep.strip()[:4000]


def run_python(code: str, workdir: Path, script_name: str, timeout_s: int) -> dict:
    """Outil « exécuteur de code » : lance le script dans le dossier de l'exécution."""
    scripts = workdir / "scripts"
    scripts.mkdir(exist_ok=True)
    path = scripts / script_name
    path.write_text(code)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", str(path)], cwd=workdir, capture_output=True,
                           text=True, timeout=timeout_s)
        ok, out, err = p.returncode == 0, p.stdout, p.stderr
        if p.returncode < 0:  # processus tué par un signal (souvent la mémoire)
            err += (f"\nProcessus interrompu par le signal {-p.returncode} "
                    "(9 = arrêt forcé, en général mémoire insuffisante : 7 Go disponibles).")
    except subprocess.TimeoutExpired as e:
        ok, out, err = False, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), \
            f"TIMEOUT après {timeout_s} s"
    return {"ok": ok, "stdout": out[-3500:], "stderr": err[-3500:], "seconds": round(time.time() - t0, 1)}


class Runner:
    """Exécute les tâches, avec boucle d'auto-correction (erreur -> agent) et portes de qualité."""

    def __init__(self, llm: LLM, workdir: Path, env_note: str, use_gates: bool = True):
        self.llm = llm
        self.workdir = workdir
        self.env_note = env_note
        self.use_gates = use_gates
        self.memory = MemoryStream(workdir / "memory.json")

    def _user_prompt(self, task: Task, extra_instruction: str = "") -> str:
        ctx = self.memory.reports(task.context)
        s = f"## Tâche « {task.id} »\n{task.description}\n\n## Résultat attendu\n{task.expected_output}\n"
        if ctx:
            s += f"\n## Contexte transmis par les autres agents\n{ctx}\n"
        if extra_instruction:
            s += f"\n## Consigne supplémentaire (orchestrateur ou juge)\n{extra_instruction}\n"
        if task.kind == "code":
            s += ("\n## Format de réponse\nUn seul bloc ```python``` complet et exécutable tel quel "
                  "(il sera lancé depuis le dossier de travail), puis une section « ## Rapport » "
                  "de 5 à 15 lignes décrivant ce qui a été fait, les choix et les chiffres clés. "
                  "Script concis : 250 lignes au maximum.")
        return s

    def run(self, task: Task, extra_instruction: str = "") -> dict:
        a = task.agent
        system = a.system_prompt(self.env_note if task.kind == "code" else "")
        messages = [{"role": "user", "content": self._user_prompt(task, extra_instruction)}]
        history = []
        for attempt in range(1, task.max_attempts + 1):
            text, rec = self.llm.call(agent=a.name, tier=a.tier, system=system, messages=messages,
                                      cached_context=task.cached_context, json_schema=task.json_schema,
                                      max_tokens=task.max_tokens)
            if task.kind != "code":
                out = {"ok": True, "task": task.id, "agent": a.name, "attempts": attempt,
                       "report": text, "history": history}
                if task.kind == "json":
                    try:
                        out["data"] = json.loads(text)
                    except json.JSONDecodeError:
                        out["ok"] = False
                        if attempt < task.max_attempts:  # réponse tronquée : on redemande plus court
                            history.append({"attempt": attempt, "error": "JSON invalide ou tronqué"})
                            messages = messages[:1] + [{"role": "user", "content":
                                "Ta réponse précédente était tronquée (JSON invalide). Recommence en étant "
                                "nettement plus concis : une ou deux phrases par champ au maximum."}]
                            messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
                            continue
                self.memory.add(type="result", task=task.id, agent=a.name, report=text[:6000],
                                attempts=attempt, ok=out["ok"])
                return out
            code = extract_code(text)
            if rec["stop"] == "max_tokens":
                code = None
                feedback = ("Ta réponse a été tronquée (limite de longueur atteinte). Renvoie un script plus "
                            "court et direct (moins de 250 lignes, sans commentaires superflus).")
                history.append({"attempt": attempt, "error": "réponse tronquée"})
            elif code is None:
                feedback = "Ta réponse ne contient pas de bloc ```python```. Recommence au bon format."
                history.append({"attempt": attempt, "error": "pas de code"})
            else:
                res = run_python(code, self.workdir, f"{task.id}_{attempt}.py", task.timeout_s)
                history.append({"attempt": attempt, "exec_ok": res["ok"], "seconds": res["seconds"],
                                "stderr_tail": res["stderr"][-600:]})
                if not res["ok"]:
                    feedback = (f"L'exécution a échoué.\nSTDOUT (fin) :\n{res['stdout'][-1500:]}\n"
                                f"STDERR (fin) :\n{res['stderr'][-2500:]}\nCorrige et renvoie le script complet.")
                else:
                    gate_ok, gate_msg = (True, "porte non appliquée")
                    if self.use_gates and task.gate is not None:
                        gate_ok, gate_msg = task.gate(self.workdir)
                    history[-1]["gate_ok"], history[-1]["gate_msg"] = gate_ok, gate_msg[:800]
                    if gate_ok:
                        report = extract_report(text) + f"\n\nSortie console (fin) :\n{res['stdout'][-1500:]}"
                        self.memory.add(type="result", task=task.id, agent=a.name, report=report,
                                        code=code, attempts=attempt, ok=True, gate=gate_msg)
                        return {"ok": True, "task": task.id, "agent": a.name, "attempts": attempt,
                                "report": report, "code": code, "history": history}
                    feedback = (f"Le script s'exécute, mais la porte de qualité automatique a refusé le "
                                f"résultat :\n{gate_msg}\nCorrige et renvoie le script complet.")
            self.memory.add(type="attempt", task=task.id, agent=a.name, attempt=attempt, feedback=feedback[:3000])
            shown = "[réponse tronquée, omise]" if rec["stop"] == "max_tokens" else text
            messages += [{"role": "assistant", "content": shown}, {"role": "user", "content": feedback}]
        self.memory.add(type="result", task=task.id, agent=a.name, report="ÉCHEC après tentatives",
                        attempts=task.max_attempts, ok=False)
        return {"ok": False, "task": task.id, "agent": a.name, "attempts": task.max_attempts,
                "report": feedback, "history": history}
