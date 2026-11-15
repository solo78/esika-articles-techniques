# Une archive de sinistres ne se lit pas, elle s'interroge

Démonstration associée à l'article Esika sur le knowledge management des sinistres graves, à partir de :
Loffredo et al. (2026), *EKA, Enterprise Knowledge Assistant: Collaborative Multi-Agent AI for Large Claims Handling*, Applied System Innovation 9(3), 62, https://doi.org/10.3390/asi9030062

Toutes les données sont **synthétiques** (générateur `src/corpus.py`) : aucune donnée réelle d'assureur.

## Contenu

```
src/corpus.py       générateur de synthèses de sinistres (vérité latente connue)
src/extraction.py   induction de schéma et extraction simulée avec erreurs (tirages communs)
src/retrieval.py    BM25 et LSA (proxy d'embeddings denses)
src/evaluate.py     une simulation : 4 types de questions Level-2, 5 dispositifs + oracle
src/figures.py      figures des résultats
src/llm.py          client Claude avec garde budgétaire et journal des appels
run_all.py          Monte Carlo (300 simulations, environ 3 min sur 2 cœurs)
run_llm.py          volet LLM réel (Haiku 4.5, Sonnet 4.6), réponses en cache dans results/llm/
results/            résultats CSV / JSON, figures, journal des appels LLM
article/            article.md et images
```

## Reproduire

```bash
pip install -r requirements.txt
python run_all.py --sims 300 --workers 2
export ANTHROPIC_API_KEY=...   # uniquement pour run_llm.py, jamais dans le dépôt
python run_llm.py              # relit le cache results/llm/ s'il existe (aucun appel refait)
```

## Résultats principaux

- `results/resultats_principaux.csv` : taux de succès et intervalles de Clopper-Pearson (réglages de référence et scénario calibré).
- `results/tests_apparies.csv` : tests de Wilcoxon appariés par simulation.
- `results/sensibilites.csv` : toutes les combinaisons (précision d'extraction, k du top-k, rappel du balayage).
- `results/llm/synthese_llm.json` : induction de schéma (E1), extraction réelle (E2), questions Level-2 réelles (E3), coût.
- `results/simulations_detail.csv.gz` : une ligne par question, dispositif et paramètre.

## Chronologie des critères

Les critères ont été fixés avant toute exécution. Le scénario « calibré E1/E2 » a été ajouté **après** le volet LLM, qui a montré que l'hypothèse initiale de couverture du schéma était trop pessimiste et que les faux positifs réels étaient plus fréquents que supposé. Dans les CSV, le paramètre 0 correspond à 2,6 % de faux positifs sur les facteurs et le paramètre -1 à 0,46 %. Les résultats du protocole initial sont conservés tels quels.

Correction après relecture indépendante : la première version de `run_llm.py` comptait un indicateur laissé vide (null) comme un faux positif (4,9 % annoncés au lieu de 2,6 %). Le calcul traite désormais null comme « non mentionné », comme la requête SQL de E3, et le scénario calibré a été relancé.

## Limites connues du code

- La LSA remplace multilingual-e5-large (modèle non téléchargeable dans l'environnement d'exécution).
- Le lecteur des RAG est supposé parfait sur les documents récupérés (hypothèse favorable au témoin).
- Le routage de la double vue est déterministe (planificateur parfait).
- Le modèle d'erreur du Monte Carlo suppose des erreurs indépendantes et ne simule pas la fragmentation du schéma observée en E1.
- Pour les précédents, la vue structurée et l'oracle classent les dossiers avec la fonction de notation elle-même (`true_sim`) appliquée aux champs extraits.

*Esika : Financial Engineering & Process Optimization*
