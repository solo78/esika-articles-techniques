"""Données freMTPL2freq : chargement, découpage de référence et empreinte du jeu de test.

Le découpage 80/20 est imposé (graine 2026) pour que toutes les exécutions,
autonomes ou non, soient évaluées sur le même jeu de test (comparaisons appariées).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "freMTPL2freq.parquet"
SPLIT_SEED = 2026
TEST_SIZE = 0.2
SAMPLE_FRAC = 0.4  # sous-échantillon pour le réglage, comme dans le papier (40 %)


def load_raw() -> pd.DataFrame:
    return pd.read_parquet(RAW)


def reference_split(df: pd.DataFrame | None = None):
    """Découpage de référence : train_test_split(test_size=0.2, random_state=2026)."""
    df = load_raw() if df is None else df
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def ids_hash(ids) -> str:
    """Empreinte SHA-256 de l'ensemble trié des IDpol (sert à vérifier le découpage)."""
    arr = np.sort(np.asarray(ids, dtype="int64"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def reference_test_hash() -> str:
    _, test = reference_split()
    return ids_hash(test["IDpol"])
