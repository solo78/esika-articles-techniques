"""Transforme le livrable d'une crew de modélisation en dossier de revue pour la crew MRM."""
import json
import shutil
import sys
from pathlib import Path


def build(run_dir: Path, out: Path):
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(run_dir / "artifacts", out, ignore=shutil.ignore_patterns("_mu_test.npy", "__pycache__"))
    mem = json.loads((run_dir / "memory.json").read_text())
    train = [e for e in mem if e.get("task") == "training" and e.get("type") == "result" and e.get("code")]
    (out / "train_script.py").write_text(train[-1]["code"] if train else "# script d'entraînement non disponible\n")
    for f in ("model_doc.md", "eval.json", "tuning.json"):
        if not (out / f).exists():
            (out / f).write_text("{}" if f.endswith(".json") else "Documentation absente.")
    data = out.parent / (out.name + "_data")
    if data.exists():
        shutil.rmtree(data)
    data.mkdir()
    for f in ("train.parquet", "test.parquet"):
        shutil.copy(run_dir / "data" / f, data / f)


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
