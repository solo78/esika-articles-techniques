# Imputer pour mieux contrôler ? Contrôle qualité d'un recensement d'affiliés

Code de l'article Esika inspiré de Saini et al. (BlackRock), *A Unified AI System For Data Quality Control and DataOps Management in Regulated Environments*, arXiv:2512.05559 (2025).

**Question testée** : dans le recensement d'une caisse de retraite, où une partie des salaires, cotisations et dates d'embauche manque, que gagne-t-on à ajouter, au-dessus de règles métier et de statistiques robustes :
- un détecteur non supervisé (Isolation Forest) sur données complétées naïvement (médiane) ;
- le même détecteur sur données imputées (imputation multiple) ;
- un contrôle « predict-then-check » : prédire chaque champ à partir des autres et signaler les écarts trop grands au regard de l'incertitude (modèle et imputation).

Toutes les données sont **synthétiques**.

## Structure

```
config.yaml          tous les paramètres (population, erreurs, non-réponse, seuils, scénarios)
run_all.py           lance les simulations, l'analyse et les figures
src/simulation.py    recensement synthétique, erreurs injectées, non-réponse MCAR / MAR / MNAR
src/controles.py     couches C0 à C4
src/evaluation.py    mesures de détection, engagement actuariel, correction des dossiers revus
src/experience.py    une simulation complète
src/analyse.py       agrégation, tests appariés, figures
results/             CSV, JSON et figures produits
article/             article et images
```

## Lancer

```bash
pip install -r requirements.txt   # versions figées utilisées pour les résultats publiés
python run_all.py            # 300 simulations x 9 scénarios (environ 70 min sur 2 cœurs)
python run_all.py --rapide   # 20 simulations par scénario
python run_all.py --analyse  # refait l'analyse à partir de results/simulations.csv
```

## Protocole en bref

- 4 000 affiliés par simulation, 3 % de lignes portant une erreur (5 types), 10 % de non-réponse par champ (MAR) dans le scénario de base.
- Mêmes graines pour tous les traitements et tous les scénarios : les comparaisons sont appariées (tests de Student et de Wilcoxon sur les différences).
- Aucun modèle ne voit la vérité. Les modèles sont appris sur les lignes non signalées par les règles et les statistiques.
- Imputation multiple (MICE) avec traitement passif de l'identité cotisation = taux x salaire x mois.
- Mêmes tirages d'imputation pour tous les traitements dans le calcul de l'engagement (nombres aléatoires communs).
- Oracle : mêmes erreurs, sans aucune valeur manquante.
- Impact métier : écart sur la valeur actuelle des droits acquis après correction des dossiers signalés.

Esika : Financial Engineering & Process Optimization
