# Crews d'agents autonomes pour la tarification non vie et la revue de modèles

Code et résultats de l'article Esika « Retirer l'humain de la boucle : crews d'agents IA pour modéliser et valider un tarif auto », transposition du papier d'Okpala, Golgoon et Ravi Kannan (arXiv 2502.05439, 2025) à la fréquence de sinistres RC automobile.

## Ce que fait le code

- `src/framework.py` : mini-cadre multi-agents (Agent, Task, flux de mémoire, exécuteur de code, boucle d'auto-correction, portes de qualité).
- `src/llm.py` : appels à l'API Claude, routage multi-modèles (Haiku 5.5, Sonnet 5.5, Opus 5.5), cache de prompt, registre des coûts et garde budgétaire.
- `src/crews/modeling.py` : crew de modélisation autonome (variante « autonome » avec méthodologiste, portes, orchestrateur et juge ; variante « naïve » sans ces mécanismes).
- `src/crews/mrm.py` : crew de revue MRM autonome guidée par la grille ASOP (`src/asop_grid.md`).
- `src/gates.py`, `src/package_check.py` : portes de qualité et audit déterministe d'un dossier de modèle.
- `src/baseline.py` : témoins faits main (GLM de Poisson actuariel, LightGBM Poisson).
- `src/sabotage.py` : 7 dossiers de modèle dont 6 avec un défaut injecté, pour tester la crew MRM.
- `analyse.py` : recalcul indépendant de toutes les métriques, intervalles de confiance (bootstrap apparié, Clopper-Pearson), tableaux et figures.

## Reproduire

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # jamais dans le code
python run_all.py --prepare             # données, témoins, dossiers sabotés (sans LLM)
./run_mrm_batch.sh                      # crew MRM sur les 7 dossiers, 2 répétitions
./run_crew_batch.sh                     # crews de modélisation
python analyse.py                       # tableaux et figures dans results/
```

Données : `freMTPL2freq` (bibliothèque R CASdatasets, aussi publiée sur Kaggle), téléchargée depuis le dépôt GitHub de CASdatasets par `run_all.py --prepare`. Données publiques, aucune donnée client.

Le fichier de données n'est pas versionné : `run_all.py --prepare` le télécharge et le convertit en `data/freMTPL2freq.parquet` (version CASdatasets, 677 991 lignes).

Les dossiers de travail des exécutions (modèles entraînés, scripts générés par les agents) sont écrits hors du dépôt, dans `../runs/`. Les journaux d'appels (`results/llm_calls.jsonl`), les journaux de crews (`results/crew_runs/`, `results/mrm_runs/`) et les résultats agrégés sont versionnés.

## Limites connues du code

Le code est publié tel qu'il a produit les résultats de l'article. La relecture a relevé les points suivants, signalés dans l'article :

- Dans `src/crews/modeling.py`, un refus de la porte documentaire relance le rédacteur sans être inscrit dans l'historique de la tâche, et la version réécrite n'est pas recontrôlée. `analyse.py` reconstitue ces refus depuis `results/llm_calls.jsonl` (6 refus) et recontrôle la documentation finale de chaque exécution.
- Les juges lisent des extraits tronqués : 6 000 caractères de documentation pour le juge de modélisation, 5 000 pour le juge MRM, des extraits des JSON et rapports des agents.
- Chaque agent rédige son rapport dans le même message que son script, donc avant de voir le résultat de l'exécution. Les JSON et la sortie console font foi.
- Les reprises écrasent les scripts de la passe précédente (nom `{tâche}_{tentative}.py`).
- Pas de bac à sable : les agents exécutent leur code dans le même environnement que l'expérience, et les dossiers de travail de la revue MRM portent le nom du défaut injecté. Aucun agent n'a consulté ces noms ni le fichier de vérité des sabotages (vérifié dans les journaux), mais une version de production doit isoler les agents.

Esika : Financial Engineering & Process Optimization
