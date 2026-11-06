# Grille de revue fondée sur les ASOP (Actuarial Standards Board)

Grille de contrôle rédigée par Esika à partir des normes de pratique actuarielle américaines,
paraphrasées (le texte officiel n'est pas reproduit). Chaque point porte un identifiant stable,
la norme et la section d'origine. Usage : référentiel de la crew de revue (MRM) et guide de
rédaction de la documentation de modèle.

Normes utilisées : ASOP 56 Modélisation (doc. 195, en vigueur au 1er octobre 2020),
ASOP 23 Qualité des données (doc. 185, en vigueur au 30 avril 2017),
ASOP 12 Classification des risques (doc. 132, en vigueur au 1er mai 2006),
ASOP 41 Communications actuarielles (exigences générales de rapport et de divulgation).

## A. Objet et adéquation du modèle (ASOP 56)
- A1 (56 §3.1) L'objet du modèle et son usage prévu sont énoncés explicitement.
- A2 (56 §3.1.1 et 3.1.2) Les capacités du modèle (niveau de détail, dépendances, volatilité des sorties) sont adaptées à cet objet.
- A3 (56 §3.1.4) La structure du modèle est justifiée : risques et interactions importants pris en compte, forme du modèle, niveau de détail, risque de surapprentissage traité.
- A4 (56 §3.1.6 a et e) Les hypothèses reposent sur l'expérience et des données fiables ; le caractère raisonnable du résultat agrégé est vérifié, pas seulement celui de chaque hypothèse.
- A5 (56 §3.2) Les opérations clés, sensibilités, faiblesses connues et limites du modèle sont comprises et décrites.

## B. Données (ASOP 23, renvoyé par ASOP 56 §3.1.5)
- B1 (23 §3.2) Le choix des données est justifié : pertinence, actualité, cohérence interne et externe, exhaustivité, limites connues.
- B2 (23 §3.3) Une revue des données est réalisée : définition de chaque variable, recherche de valeurs douteuses et de relations incohérentes (par exemple exposition supérieure à un an, nombres de sinistres extrêmes, identifiants).
- B3 (23 §3.4) Les traitements appliqués aux données (plafonds, exclusions, corrections) sont décrits et justifiés ; leur effet possible sur les résultats est apprécié.
- B4 (23 §3.5) La confiance accordée aux données fournies par des tiers est déclarée.
- B5 (23 §4) La documentation déclare : sources, revue faite ou non, problèmes non résolus, ajustements significatifs, biais ou incertitudes possibles liés aux données.

## C. Classification des risques (ASOP 12)
- C1 (12 §3.2.1) Chaque variable tarifaire a un lien raisonnable avec la sinistralité attendue ; les interdépendances importantes sont prises en compte.
- C2 (12 §3.2.3) Les variables sont objectivement mesurables et difficiles à manipuler ; aucune variable n'est une conséquence du sinistre lui-même (pas de fuite de la cible) ni un simple identifiant administratif.
- C3 (12 §3.3.2 b) Les classes sont assez peuplées pour être crédibles (compromis homogénéité / crédibilité).
- C4 (12 §3.3.4) Les relativités obtenues sont raisonnables et cohérentes (sens de variation attendu, absence de ruptures inexpliquées).
- C5 (12 §3.2.5 et 4.1 a) Les contraintes légales sur les variables utilisables sont examinées (en Europe : pas de genre, attention aux variables sensibles).

## D. Contrôle du risque de modèle et validation (ASOP 56 §3.6)
- D1 (56 §3.6.1) Tests suffisants : rapprochement des entrées, contrôle de la logique, tests de sensibilité.
- D2 (56 §3.6.2) Validation hors échantillon : jeu de test indépendant jamais utilisé pour l'apprentissage ni le réglage, tests statistiques, comparaison à un modèle alternatif.
- D3 (56 §3.6.2) Le modèle est calibré : nombre de sinistres attendus cohérent avec l'observé, globalement et par segment.
- D4 (56 §3.6.3) Une revue par un autre professionnel qualifié est envisagée au regard de la complexité.
- D5 (56 §3.6.4) Gouvernance et contrôles : traçabilité des versions, reproductibilité (graines, paramètres).
- D6 (spécifique fréquence) L'exposition est traitée correctement : nombre attendu proportionnel à l'exposition (offset log ou poids), unité de temps explicite.

## E. Communication et documentation (ASOP 41, ASOP 56 §3.7 et §4.1)
- E1 (41 ; 56 §3.7) La documentation permet à un autre actuaire qualifié de juger du caractère raisonnable du travail (méthodes, hypothèses, données, paramètres).
- E2 (56 §4.1) Sont déclarés : l'objet du modèle, les limites et faiblesses connues, les incohérences d'hypothèses importantes.
- E3 (56 §4.1 ; 41) La dépendance envers des modèles, outils ou experts tiers est déclarée (y compris les outils d'IA ayant produit le code).
- E4 (41) Les chiffres annoncés sont exacts et reproductibles ; toute différence avec une communication antérieure est expliquée.
- E5 (41) Les limites d'usage et le public visé sont précisés pour prévenir une mauvaise utilisation.
