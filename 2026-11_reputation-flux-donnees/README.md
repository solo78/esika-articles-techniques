# Réputation des flux de données partenaires et provisions techniques

Code de l'article Esika *« Faire confiance aux données de ses partenaires : ce qu'un système de réputation change, et ne change pas, pour les provisions »*.

Le projet transpose à l'assurance l'architecture proposée par Wu & Konstantinidis (2025), *Trust and Reputation in Data Sharing: A Survey* (arXiv:2508.14028). Il l'applique aux bordereaux mensuels de sinistres transmis par des partenaires : courtiers, bancassureurs, gestionnaires délégués, coassureurs.

Toutes les données sont **synthétiques**.

## Contenu

| Fichier | Rôle |
|---|---|
| `src/simulation.py` | 8 partenaires, sinistres et paiements exacts, bordereaux dégradés selon un profil de qualité, flux comptable mensuel |
| `src/quality.py` | Couche 2 : note de chaque bordereau sur 7 dimensions, alertes critiques, explication lisible |
| `src/reputation.py` | Couche 3 : réputation bayésienne Beta avec oubli, statut vert, orange ou rouge, facteur de confiance Z |
| `src/reserving.py` | Triangles trimestriels, Chain-Ladder, Bornhuetter-Ferguson, Benktander pondéré par la confiance |
| `src/experiment.py` | Comparaison des traitements : Oracle, Brut, Contrôle, Redressement, Benktander, Réputation, plus le test de sensibilité |
| `run_all.py` | Lance les simulations et produit tableaux et figures dans `results/` |
| `article/` | Texte de l'article et images |

## Lancer

```bash
pip install -r requirements.txt
python run_all.py --runs 300     # environ 1 minute
```

## Principaux résultats (300 simulations)

Erreur sur la provision du portefeuille, en % de la provision réelle :

| Traitement | Biais moyen | Erreur quadratique |
|---|---|---|
| Oracle | 0,0 % | 5,8 % |
| Brut | -18,3 % | 20,0 % |
| Contrôle ponctuel | -31,8 % | 32,8 % |
| Redressement comptable | -1,3 % | 5,9 % |
| Benktander | -1,6 % | 5,3 % |
| Benktander pondéré par la réputation | -0,7 % | 5,7 % |

Détection des dérives :
- la dégradation progressive (début au mois 19) est signalée en médiane au mois 23 ;
- la survenance mal codée (début au mois 24) est signalée en médiane au mois 27 ;
- le contrôle ponctuel ne la voit jamais ;
- fausses alertes sur les partenaires sains : moins de 4 % des mois.

## Paramètres illustratifs

Tolérances, poids, facteur d'oubli (0,85), seuils (0,85 et 0,70 sur le quantile à 5 %) et profils de défauts sont des choix d'illustration. Ils doivent être calibrés sur l'historique réel d'un portefeuille avant tout usage.
