# Faire confiance aux données de ses partenaires : ce qu'un système de réputation change, et ne change pas, pour les provisions

*Transposer un travail de recherche récent sur la confiance dans le partage de données à un problème très concret de l'assurance : les bordereaux de courtiers, de bancassureurs et de gestionnaires délégués qui alimentent les provisions techniques. Avec une implémentation complète et des résultats chiffrés.*

![Des erreurs qui se compensent](images/fig0_erreurs_compensees.jpg)

---

## 1. Le problème : des provisions calculées sur des données que l'on ne produit pas

Une part importante des sinistres d'un assureur n'est pas gérée par l'assureur lui-même. Les courtiers, les bancassureurs, les gestionnaires délégués en santé et les coassureurs apériteurs transmettent chaque mois un **bordereau** : la liste des sinistres payés, avec leur date de survenance, leur montant et leur identifiant.

L'actuaire construit ensuite ses triangles de développement à partir de ces lignes, puis estime les provisions. La qualité de la provision dépend donc directement de la qualité de données produites par des tiers.

Le cadre prudentiel est clair sur ce point. La directive Solvabilité II (article 82) demande aux assureurs de disposer de processus garantissant le caractère approprié, l'exhaustivité et l'exactitude des données utilisées pour calculer les provisions techniques.

En pratique, le dispositif le plus répandu reste le **contrôle ponctuel à réception** : on vérifie le bordereau du mois, on supprime les doublons évidents, on rejette les lignes incohérentes. Puis on oublie. Chaque bordereau est jugé seul, sans mémoire de ce que le partenaire a transmis les mois précédents.

Ce type de contrôle a deux angles morts :
- **les dérives lentes**, où chaque bordereau reste « acceptable » pris isolément ;
- **les défauts invisibles ligne à ligne**, comme une date de survenance systématiquement mal codée.

Cet article explore une piste issue de la recherche en informatique : traiter la confiance dans un partenaire comme une **réputation**, construite dans le temps à partir de preuves mesurables.

## 2. Ce que propose la recherche

Le point de départ est une revue de littérature publiée en août 2025 par Wenbo Wu et George Konstantinidis (Université de Southampton) : *Trust and Reputation in Data Sharing: A Survey* (arXiv 2508.14028).

Les auteurs étudient les **systèmes de gestion de la confiance et de la réputation** (en anglais *Trust and Reputation Management Systems*, TRMS). Ce sont les mécanismes qui permettent à des acteurs qui échangent des données de décider à qui se fier. On les connaît surtout par les notes des plateformes de commerce en ligne, mais ils existent aussi dans les réseaux d'objets connectés, les véhicules communicants ou les systèmes multi-agents.

> **Encadré : trois mots à ne pas confondre**
>
> Le survey distingue trois notions.
> - La **confiance** est la croyance subjective d'un acteur dans le fait qu'un autre se comportera comme prévu.
> - La **fiabilité** (*trustworthiness*) est la qualité objective qui rend un acteur digne de confiance. C'est ce que l'on cherche à mesurer.
> - La **réputation** est la perception collective de cette fiabilité, construite à partir de l'historique.
>
> Dans cet article, on mesure la fiabilité de chaque bordereau, puis on en déduit une réputation qui sert à décider.

### Une architecture en quatre couches

Les auteurs proposent une architecture en quatre couches :
1. **l'écosystème** : fournisseurs de données, consommateurs, intermédiaires et tiers de confiance ;
2. **l'évaluation atomique** : chaque échange reçoit une note, à partir de signaux explicites (une plainte, une note) ou implicites (des contrôles automatiques de qualité, de conformité, de délai) ;
3. **l'inférence de réputation** : les notes successives sont agrégées dans le temps, avec un oubli progressif, une détection d'anomalies et un modèle de calcul (statistique, bayésien, à base de graphes ou d'apprentissage) ;
4. **les services** : des décisions, des tableaux de bord et des explications lisibles des scores.

### Le constat central

Sur une trentaine de systèmes analysés, l'évaluation porte presque toujours sur le **comportement des acteurs**, et presque jamais sur la **qualité de la donnée elle-même**. Les auteurs appellent donc à un score composite de confiance dans la donnée, construit sur des dimensions mesurables : exactitude, complétude, cohérence, fraîcheur, traçabilité, unicité. Ses poids doivent dépendre de l'usage qui sera fait de la donnée.

C'est une proposition, pas un résultat : le survey ne l'implémente pas et ne la chiffre pas. Ses exemples portent sur l'Internet des objets, les véhicules et la santé, jamais sur l'assurance. C'est précisément ce que nous avons voulu tester.

## 3. La transposition à l'assurance

![Architecture transposée](images/fig1_architecture.jpg)

La correspondance des rôles est directe :
- **fournisseurs de données** : les partenaires (courtiers, bancassureurs, gestionnaires délégués, coassureurs) ;
- **consommateur** : l'actuariat, qui transforme les bordereaux en triangles puis en provisions ;
- **intermédiaire** : le système d'information qui reçoit, stocke et transforme les fichiers ;
- **tiers de confiance** : la fonction actuarielle et le contrôle interne, garants de la qualité des données utilisées.

L'assurance dispose en outre d'un signal que la plupart des écosystèmes étudiés par le survey n'ont pas : le **flux comptable**. Les montants payés par un gestionnaire délégué ou réglés avec un courtier passent par les comptes. Le total d'un bordereau peut donc être rapproché d'un montant indépendant. Nous verrons que ce détail change tout.

## 4. Couche 2 : noter chaque bordereau

Chaque bordereau mensuel reçoit une note entre 0 et 1 sur sept dimensions, reprises de la grille du survey et limitées à celles que l'on peut mesurer automatiquement :

| Dimension | Ce que l'on mesure | Tolérance (note nulle) |
|---|---|---|
| Complétude | part de lignes sans date de survenance | 10 % |
| Validité | part de lignes invalides (montant négatif, survenance postérieure au paiement) | 5 % |
| Unicité | part de lignes en double | 10 % |
| Exactitude | écart entre le total du bordereau (sans doublons) et le flux comptable du mois | 10 % |
| Fraîcheur | retard au-delà du délai contractuel de 10 jours | 20 jours de retard |
| Cohérence | dérive de la structure du bordereau par rapport à l'historique validé du partenaire | voir ci-dessous |
| Traçabilité | part de lignes sans identifiant de sinistre | 10 % |

Chaque note suit la même règle simple : `note = max(0, 1 - taux de défaut / tolérance)`. Les tolérances jouent le rôle d'un contrat de données entre l'assureur et son partenaire.

La **cohérence** mérite une explication. On compare chaque bordereau à l'historique du partenaire sur deux indicateurs :
- le niveau des montants, pour repérer par exemple un fichier transmis en milliers au lieu d'unités ;
- la part des paiements dont la survenance tombe le mois même du paiement. Quand un système remplace la date de survenance par la date de paiement, cette part grimpe brutalement.

Point important : **seuls les bordereaux jugés sains alimentent la référence**. Sans cette précaution, une dérive deviendrait en quelques mois « la nouvelle norme » et cesserait d'être détectée.

Les sept notes sont ensuite combinées avec des poids qui dépendent de l'usage, comme le recommande le survey. Pour le provisionnement, les défauts qu'un nettoyage automatique corrige pèsent peu : les doublons exacts (7 %) et les identifiants manquants (5 %). Les défauts qui faussent le triangle sans être visibles ligne à ligne pèsent lourd : l'exactitude (25 %), la complétude (20 %) et la cohérence (20 %). La fraîcheur compte pour 15 % et la validité pour 8 %.

```python
WEIGHTS_RESERVING = {"exactitude": 0.25, "completude": 0.20, "coherence": 0.20,
                     "fraicheur": 0.15, "validite": 0.08, "unicite": 0.07, "tracabilite": 0.05}
score = sum(WEIGHTS_RESERVING[d] * note[d] for d in WEIGHTS_RESERVING)
```

Chaque note est **explicable**. Le code produit automatiquement une phrase du type :

> Note 0,66. Principales causes : Exactitude (écart de 5,2 % avec le flux comptable, -12,9 pts) ; Complétude (5,4 % de lignes sans date de survenance, -10,8 pts).

C'est la couche « explicabilité » du survey, ramenée à sa forme la plus utile : dire au gestionnaire du partenariat **quoi** corriger.

Un incident grave, comme un écart comptable de plus de 30 % ou un montant total cinq fois supérieur ou inférieur à l'habitude, déclenche une **alerte immédiate** et la mise en quarantaine du bordereau, sans attendre l'agrégation.

## 5. Couche 3 : de la note à la réputation

Une note isolée ne suffit pas. Un bon partenaire peut avoir un mauvais mois, et un partenaire qui se dégrade lentement peut rester juste au-dessus des seuils pendant longtemps. Il faut une mémoire.

> **Encadré : le modèle Beta, sans formule superflue**
>
> On cherche à estimer une proportion inconnue : la part des transmissions « bonnes » d'un partenaire. La loi Beta, définie par deux paramètres α et β, décrit ce que l'on sait de cette proportion.
> - α accumule les preuves favorables, β les preuves défavorables.
> - Un bordereau noté 0,9 ajoute 0,9 × poids à α et 0,1 × poids à β.
> - Un nouveau partenaire démarre avec α = β = 1 : on ne sait rien, toutes les valeurs sont également possibles.
>
> Avant chaque mise à jour, α et β sont multipliés par un **facteur d'oubli** (0,85 par mois, soit une mémoire effective d'environ sept mois). Les vieux bordereaux comptent moins, ce qui permet de détecter une dégradation récente et de laisser un partenaire se racheter. Ce mécanisme s'inspire du *Beta Reputation System* de Jøsang et Ismail (2002) et relève de la famille des modèles bayésiens recensés par le survey.

```python
class BetaReputation:
    def update(self, score, critical=False):
        self.alpha *= self.lam            # oubli
        self.beta *= self.lam
        if critical:                      # incident grave : forte preuve défavorable
            self.beta += self.crit_weight
        else:
            self.alpha += self.weight * score
            self.beta += self.weight * (1 - score)
```

Le choix décisif est de **décider sur le quantile à 5 %**, et non sur la moyenne. La raison est simple. Un nouveau partenaire avec deux bons bordereaux a une moyenne excellente, mais une loi encore très étalée : on ne le connaît pas. Son quantile à 5 % reste bas. Le **démarrage à froid**, identifié par le survey comme un problème classique, est ainsi traité sans règle ad hoc.

![Intuition du modèle Beta](images/fig2_intuition_beta.png)

Trois zones en découlent :
- **vert** si le quantile à 5 % dépasse 0,85 ;
- **orange** entre 0,70 et 0,85 ;
- **rouge** en dessous de 0,70.

## 6. Couche 4 : décider

La réputation n'a de valeur que si elle change une décision. Nous en testons trois, de la plus simple à la plus actuarielle.

1. **Alerter et prioriser.** Un partenaire qui passe à l'orange déclenche une relance ciblée, avec l'explication des notes. Un incident critique met le bordereau en quarantaine.
2. **Redresser les montants.** Quand l'exactitude signale un écart avec le flux comptable, le bordereau est remis à l'échelle du montant comptable. Quand un bordereau manque à la clôture, il est estimé à partir du mois précédent, mis à l'échelle du flux comptable du mois.
3. **Pondérer les méthodes de provisionnement.** C'est la traduction actuarielle de la réputation, détaillée dans l'encadré.

> **Encadré : Chain-Ladder, Bornhuetter-Ferguson, Benktander**
>
> **Chain-Ladder (CL)** projette les paiements déjà observés avec des coefficients de développement estimés sur le passé. La méthode suit fidèlement les données, défauts compris : un trou dans le dernier mois est multiplié par le coefficient de développement.
>
> **Bornhuetter-Ferguson (BF)** ajoute au déjà payé une part « restant à payer » calculée à partir d'une sinistralité a priori (sinistres attendus / primes). Elle est moins sensible aux données récentes.
>
> **Benktander** combine les deux : ultime = Z × CL + (1 - Z) × BF, avec une crédibilité Z égale à la part déjà payée. Une survenance ancienne et presque entièrement payée suit ses données ; une survenance récente s'appuie davantage sur l'a priori.
>
> Notre variante multiplie cette crédibilité par un **facteur de confiance** tiré de la réputation du partenaire, Z = confiance × part payée. Le profil de développement de l'a priori est estimé sur les seuls partenaires jugés fiables. Avec une confiance de 1, on retrouve exactement Benktander.

```python
def benktander_trust(tri, f_own, f_pattern, premium, elr, trust=1.0):
    cl = latest(tri) * cdf(f_own)[ages]
    bf = latest(tri) + (1 - 1 / cdf(f_pattern)[ages]) * elr * premium
    z = trust / cdf(f_pattern)[ages]        # confiance x part déjà payée
    return z * cl + (1 - z) * bf
```

## 7. Le protocole de simulation

Toutes les données sont **synthétiques**. Huit partenaires transmettent des bordereaux mensuels pendant 36 mois. Le portefeuille est à maturité au départ, avec quatre années de sinistres antérieurs simulées. Les sinistres sont réglés en un à six paiements, majoritairement dans les deux ans.

Chaque partenaire a un profil de qualité :

| Partenaire | Profil |
|---|---|
| P1 Courtier A | fiable |
| P2 Courtier B | fiable, petits défauts (doublons, identifiants manquants, quelques retards) |
| P3 Bancassureur | dégradation progressive dès le mois 19 : jusqu'à 15 % de lignes manquantes, 6 % de doublons, 8 % de survenances absentes |
| P4 Courtier C | doublons chroniques (6 %) |
| P5 Gestionnaire santé | retards fréquents (35 % des mois, jusqu'à 50 jours) |
| P6 Nouvel apporteur | entrée au mois 25, sans historique |
| P7 Coassureur | un bordereau transmis en milliers au mois 27 |
| P8 Gestionnaire auto | dès le mois 24, 25 % des lignes ont leur survenance remplacée par la date de paiement |

À la clôture (mois 36, plus 15 jours), on estime la charge ultime par trimestre de survenance avec six traitements, ajoutés couche par couche :

1. **Oracle** : données parfaites, Chain-Ladder. C'est l'erreur propre à la méthode, irréductible.
2. **Brut** : bordereaux tels que reçus.
3. **Contrôle** : contrôle ponctuel sans mémoire (doublons exacts, lignes invalides, rejet des incidents critiques).
4. **Redressement** : contrôle et remise à l'échelle par le flux comptable (couche 2).
5. **Benktander** : redressement et méthode de Benktander classique. C'est le **témoin** : la bonne pratique actuarielle sans notion de confiance.
6. **Réputation** : redressement et Benktander pondéré par la confiance du partenaire (couche 3).

L'expérience est répétée **300 fois**. On mesure l'erreur sur la provision, rapportée à la provision réelle, connue puisque les données sont simulées.

Un repère d'ordre de grandeur : dans ce portefeuille à développement court, la provision représente environ 8 % de la charge ultime des trois exercices. Une erreur de 5 % sur les montants payés devient donc une erreur de plusieurs dizaines de pourcents sur la provision.

## 8. Résultats

### La réputation détecte ce que le contrôle ponctuel ne voit pas

![Réputation des partenaires](images/fig3_reputation_partenaires.png)

La figure montre l'évolution de la réputation de chaque partenaire sur une simulation type. Les comportements attendus apparaissent nettement.
- Les partenaires fiables restent au vert.
- Le nouvel apporteur démarre à l'orange, faute d'historique, puis rejoint le vert en quelques mois.
- Le coassureur chute immédiatement au rouge le mois de son erreur d'unité, puis se rachète progressivement.
- Les deux dérives (P3 et P8) font glisser la réputation vers l'orange puis le rouge.

Sur 300 simulations :

| | Début du défaut | Alerte (médiane) | Alerte (10e à 90e centile) | Contrôle ponctuel : au moins un rejet |
|---|---|---|---|---|
| P3, dégradation progressive | mois 19 | mois 23 | mois 22 à 24 | 11 % des simulations |
| P8, survenance mal codée | mois 24 | mois 27 | mois 26 à 29 | jamais |

Les **fausses alertes** sur les partenaires sains restent rares : 0 % des mois pour P1 et P6, 0,9 % pour P4, 3,6 % pour P2, hors période de démarrage.

Le cas de P8 est instructif. Chaque ligne est individuellement valide et le total correspond au flux comptable. Seule la comparaison à l'historique du partenaire révèle le problème. La carte des notes par dimension le montre bien : pour P3, l'exactitude et la complétude se dégradent ; pour P8, seule la cohérence s'allume.

![Notes par dimension](images/fig4_notes_par_dimension.png)

### Sur le montant des provisions, le rapprochement comptable fait l'essentiel

![Erreur de provisionnement](images/fig5_erreur_provision.png)

| Traitement | Biais moyen | Écart-type | Erreur quadratique moyenne |
|---|---|---|---|
| Oracle | 0,0 % | 5,8 % | 5,8 % |
| Brut | -18,3 % | 7,9 % | 20,0 % |
| Contrôle | -31,8 % | 8,1 % | 32,8 % |
| Redressement | -1,3 % | 5,8 % | 5,9 % |
| Benktander | -1,6 % | 5,0 % | 5,3 % |
| Réputation | -0,7 % | 5,6 % | 5,7 % |

*Erreur sur la provision du portefeuille, en % de la provision réelle, 300 simulations.*

Trois enseignements, dont deux contre-intuitifs.

**1. Le contrôle ponctuel aggrave l'erreur.** Les bordereaux bruts sous-estiment la provision de 18 % ; après contrôle, de 32 %. La carte des biais par partenaire en donne la clé.
- Les doublons du courtier C gonflent sa provision de 80 %.
- Les lignes manquantes du bancassureur réduisent la sienne de 98 %.
- Au niveau du portefeuille, les deux erreurs se compensent en partie.

Supprimer les doublons, ce qui est une bonne pratique, fait disparaître la compensation et révèle le trou. **Un total qui paraît juste peut cacher deux erreurs de sens opposé.**

![Biais par partenaire](images/fig6_biais_par_partenaire.png)

**2. Le rapprochement avec le flux comptable fait l'essentiel du travail.** Le biais passe de -32 % à -1 %, et l'erreur quadratique rejoint celle de l'oracle. La dimension « exactitude » est à la fois la plus informative pour détecter et la plus utile pour corriger, ce qui justifie son poids.

**3. Une fois les montants redressés, la pondération par la réputation n'améliore pas significativement le total.** Benktander et sa variante pondérée sont équivalents, aux fluctuations de simulation près. Nous le disons sans détour : sur le montant global, dans ce cadre, la réputation n'est pas un meilleur estimateur.

### Pondérer ne recrée pas une donnée absente

Nous avons rejoué la comparaison sans flux comptable détaillé, c'est-à-dire sans redressement. La pondération par la réputation réduit alors le biais de -30,0 % (Benktander) à -25,8 %. C'est une amélioration, mais elle reste très insuffisante.

La raison est structurelle. Bornhuetter-Ferguson s'appuie moins sur le développement des données récentes, mais il part toujours du montant déjà payé. Si ce montant est faux, aucune pondération ne le corrige.

### Le défaut le plus subtil : un total juste, une répartition fausse

![P8 par trimestre](images/fig7_P8_biais_par_trimestre.png)

Pour le gestionnaire auto (P8), la provision totale est presque exacte, quel que soit le traitement. Le détail par trimestre de survenance est pourtant faux :
- environ -8 à -10 % sur les trimestres 7 et 8 ;
- jusqu'à +16 à +18 % sur le dernier trimestre.

Des paiements de sinistres anciens ont été rattachés à des survenances récentes. Les coefficients de développement s'ajustent et le total se compense.

Pour tout ce qui se pilote par exercice de survenance, c'est un vrai problème : tarification, suivi de la sinistralité par exercice, états réglementaires par exercice. Aucune méthode de provisionnement ne le corrige. Seule une **correction à la source**, déclenchée par l'alerte de cohérence, le peut.

## 9. Ce que cela signifie en pratique

La lecture honnête de ces résultats est la suivante.
- **La couche 2, la note de chaque bordereau, porte la valeur sur le chiffre**, à condition de disposer d'un signal indépendant comme le flux comptable. C'est le message central du survey (« faire confiance à la donnée, pas seulement à l'acteur »), et il se vérifie ici.
- **La couche 3, la réputation, porte la valeur sur la détection, la priorisation et la gouvernance.** Elle signale les dérives lentes en trois à quatre mois et les défauts invisibles au contrôle ligne à ligne. Elle gère le nouvel entrant sans règle ad hoc, et elle produit une trace explicable. C'est exactement ce qu'un contrôleur ou un régulateur demande quand il interroge le dispositif de qualité des données.
- **La couche 4 doit inclure une action humaine.** Le défaut le plus pernicieux de notre simulation ne se corrige qu'en retournant vers le partenaire.

Une mise en œuvre réaliste peut se faire en quatre étapes :
1. **Formaliser un contrat de données** avec chaque partenaire : champs obligatoires, délais, tolérances.
2. **Automatiser le rapprochement** bordereau par bordereau avec la comptabilité. C'est le gain le plus immédiat.
3. **Calculer la note et la réputation** de chaque partenaire, avec un tableau de bord vert, orange, rouge et l'explication des notes.
4. **Brancher les décisions** : relances automatiques, quarantaine, escalade vers la fonction actuarielle et, en option, la pondération des méthodes de provisionnement.

## 10. Limites

- **Les données sont simulées.** Les profils de défauts, les tolérances, les poids et le facteur d'oubli sont illustratifs et devraient être calibrés sur l'historique réel de chaque portefeuille.
- **Le redressement suppose un flux comptable disponible par partenaire et par mois.** Ce n'est pas toujours le cas, notamment quand les règlements avec les intermédiaires passent par des comptes courants agrégés. Sans ce signal, notre propre test montre que les gains chutent fortement.
- **Le redressement suppose que les lignes manquantes ressemblent aux lignes reçues.** Si un partenaire omet systématiquement les gros sinistres, la remise à l'échelle masque une partie du problème.
- **Le modèle ne couvre pas les comportements stratégiques** décrits par le survey : partenaire qui alterne bonnes et mauvaises périodes pour rester sous les seuils, ou qui change d'identité. Dans un contexte de bordereaux contractuels, ces risques sont faibles, mais pas nuls.
- **Ce dispositif ne remplace ni un audit des données ni le jugement de l'actuaire.** Il les oriente.

## Conclusion

Le survey de Wu et Konstantinidis conclut que les systèmes de confiance regardent trop les acteurs et pas assez la donnée. Transposé aux bordereaux d'assurance, ce constat se vérifie, avec une nuance utile.

La donnée se juge d'abord **contre une référence indépendante** : c'est le rapprochement comptable qui ramène l'erreur sur la provision de -32 % à -1 %. La réputation, elle, se juge **dans le temps** : elle repère en quelques mois des dérives qu'aucun contrôle ponctuel ne voit.

Les deux sont complémentaires. Le premier corrige le chiffre, le second dit où regarder.

---

*Le code complet (simulation, notation, réputation, provisionnement et figures) est disponible dans le dépôt associé. Toutes les données sont synthétiques.*

**Références**
- Wu, W. et Konstantinidis, G. (2025). *Trust and Reputation in Data Sharing: A Survey*. arXiv:2508.14028.
- Jøsang, A. et Ismail, R. (2002). *The Beta Reputation System*. 15th Bled Electronic Commerce Conference.
- Directive 2009/138/CE (Solvabilité II), article 82 : qualité des données utilisées pour le calcul des provisions techniques.
- Mack, T. (2000). *Credible claims reserves: the Benktander method*. ASTIN Bulletin, 30(2).

*Esika : Financial Engineering & Process Optimization*
