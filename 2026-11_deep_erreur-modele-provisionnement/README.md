# Comprendre l'erreur de modèle en provisionnement (analyse pédagogique)

Analyse pédagogique et reproduction du papier : Greg Taylor et Gráinne McGuire, *Model Error (or Ambiguity) and Its Estimation, with Particular Application to Loss Reserving*, Risks 2023, 11(11), 185, doi 10.3390/risks11110185.

L'article (`article/article.md`) explique la méthode pas à pas : famille de modèles LASSO, a priori de Laplace, portes d'inclusion, moyenne bayésienne de modèles, bootstrap. Il la reproduit sur un petit triangle calculable à la main (fil rouge) et sur les jeux 1 et 2 du papier, la critique et décrit un cas d'usage en assurance non vie (marge de risque IFRS 17).

**Toutes les données sont fictives**, générées par le code publié par les auteurs (blog MLR de l'Institute and Faculty of Actuaries). Aucune donnée client.

## Structure

```
src/
  data.py               générateur des triangles (jeux 1 et 2 du papier, fil rouge 10 x 10)
  basis.py              base de fonctions (rampes, interactions de Heaviside), section 6.1.2
  lasso_glmnet.R        LASSO Poisson par glmnet, validation croisée, dispersion phi (comme les auteurs)
  lasso.py              interface Python vers le script R
  bma.py                vraisemblance Gamma, a priori de Laplace, portes, moyenne bayésienne (éq. 10 à 27)
  primary.py            analyse du modèle principal
  bootstrap.py          bootstrap et matrice bootstrap (section 7, éq. 28 à 33)
  witnesses.py          témoins : Mack, GLM Gamma de structure connue
  fil_rouge.py          exemple 10 x 10 et vote réduit à 5 modèles
  criteria.py           évaluation scriptée des critères fixés avant exécution
  figures.py            figures de résultats
  diag_extrapolation.R  diagnostic : calendrier prolongé ou figé
run_all.py              exécute tout (environ 1 h 30 sur 2 cœurs ; cache dans work/)
diagnostic_extrapolation.py, diagnostic_biais_survenance.py, diagnostic_complements.py   diagnostics du biais et compléments (ajoutés après le contrôle de réalisme et la relecture)
notebook/parcours.ipynb suit l'article pas à pas (relit results/, recalcule le fil rouge)
results/                CSV de résultats et figures
article/                article.md et images
```

## Exécution

```bash
pip install -r requirements.txt     # et R >= 4.3 avec glmnet et MASS
python run_all.py --B 300 --n-real 20
python diagnostic_extrapolation.py
python diagnostic_biais_survenance.py
python diagnostic_complements.py
jupyter nbconvert --to notebook --execute --inplace notebook/parcours.ipynb
```

## Résultats clés (a priori 1se, portes x 1,1)

| Jeu | IMSE | Paramètre | Processus | Total | Papier (total) |
|---|---|---|---|---|---|
| 1 | 1,5 % | 7,1 % | 0,6 % | 7,3 % | 8,1 % |
| 2 | 1,8 % | 6,7 % | 0,5 % | 7,0 % | 9,9 % |

Contrôle de réalisme (20 triangles indépendants par jeu) : la provision du vote surestime les paiements réalisés de +23,9 % (jeu 1) et +34,7 % (jeu 2) en moyenne (t = 7,0 et 10,9) ; Mack : -0,4 % et +2,6 %, GLM de structure connue : +1,6 % et +1,1 % (écarts non distinguables de zéro).

## Écarts au protocole du papier (détaillés dans l'article, section 5.4)

1. **Résidus du bootstrap** réduits par la racine de la moyenne (bootstrap ODP) et non par la variance Gamma : la vérification d'homogénéité exigée par le papier échoue avec la variance Gamma sur ces données. Le vote garde la vraisemblance Gamma.
2. **Portes élargies d'un facteur 1,1** comme configuration de travail : avec les portes de la Table 2, 3 réplications sur 300 passent sur le jeu 1 (282 sur 300 sur le jeu 2). Décision prise après un pilote de 20 réplications sur le jeu 1, avant de voir le jeu 2 (chronologie complète : annexe A de l'article).
3. **Recentrage bootstrap** appliqué aux prévisions seulement (la vraisemblance reste celle de l'ajustement à chaque pseudo-triangle) : la description du papier est ambiguë sur ce point.
4. **Simplifications** : mise à l'échelle des fonctions de base par l'écart type sur le passé (non publiée par les auteurs) ; 300 réplications au lieu de 500 ; jeux 1 et 2 seulement ; graine numpy (tirage différent de celui du papier, espérances identiques).

## Limites connues du code

- L'erreur de processus utilise la même fonction de variance que le bootstrap (proportionnelle à la moyenne), et non la loi Gamma du papier.
- Sur 2 triangles sur 40 du contrôle de réalisme, le GLM Gamma servant à estimer phi diverge : repli sur l'estimateur de Pearson (colonne `phi_methode` de `results/realisme.csv`).
- Le contrôle de réalisme n'applique pas le bootstrap à chaque triangle (temps de calcul) : l'erreur annoncée de référence est celle du triangle principal.
- 1,1 % (jeu 1) et 0,85 % (jeu 2) des valeurs simulées par le bootstrap, négatives, sont ramenées à 0,1 % de la moyenne.
- Le calage de l'échelle de l'a priori n'atteint pas exactement le 1se sur le jeu 2 (mode sur le modèle 109 au lieu de 110) : point de grille le plus proche retenu.
- Les dossiers `work/` (cache, plusieurs Go) ne sont pas versionnés : `run_all.py` les recrée.

*Esika : Financial Engineering & Process Optimization*
