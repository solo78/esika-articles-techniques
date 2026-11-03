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
| `src/experiment.py` | Comparaison des traitements (Oracle, Brut, Contrôle, Redressement, Benktander, Réputation), témoins de détection et tests de sensibilité |
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
| Oracle | +0,3 % | 6,0 % |
| Brut | -17,8 % | 19,5 % |
| Contrôle ponctuel | -31,5 % | 32,4 % |
| Redressement comptable | -1,0 % | 6,0 % |
| Benktander | -1,5 % | 5,4 % |
| Benktander pondéré par la réputation | -1,5 % | 5,8 % |

- Le contrôle ponctuel aggrave l'erreur : il supprime des doublons qui compensaient des lignes manquantes.
- Le rapprochement comptable fait l'essentiel (flux idéal ; avec un flux bruité de 5 % : biais +1,2 %, erreur 7,2 %).
- Pondérer Benktander par la réputation dégrade légèrement la précision (écart significatif), et nettement si l'a priori est biaisé.
- Détection : un seuil sur la note du bordereau alerte plus tôt que la réputation (P3 : mois 21 contre 23 ; P8 : mois 24 contre 27). La réputation donne un signal plus stable.

Détails : `results/erreur_portefeuille.csv`, `results/detection.json`, `results/tests_apparies.json`.

## Paramètres illustratifs

Tolérances, poids, facteur d'oubli (0,85), seuils (0,85 et 0,70 sur le quantile à 5 %) et profils de défauts sont des choix d'illustration. Ils doivent être calibrés sur l'historique réel d'un portefeuille avant tout usage.
