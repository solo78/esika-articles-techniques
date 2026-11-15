"""Génération d'un corpus synthétique de synthèses de sinistres graves (format courriel, en français).

Chaque dossier a une vérité latente connue (branche, pays, cause, montants, contentieux, facteurs),
puis un texte rédigé à partir de paraphrases tirées au hasard. Aucune donnée réelle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BRANCHES = {
    "Dommages industriels": ["incendie", "explosion", "bris de machine"],
    "RC générale": ["chute d'un tiers", "dommage à un bien confié", "atteinte accidentelle"],
    "RC produits": ["défaut de fabrication", "défaut de conception", "rappel de produit"],
    "Environnement": ["pollution des sols", "pollution des eaux", "émission toxique"],
    "Aviation": ["incident au sol", "atterrissage dur", "collision en vol"],
    "Cyber": ["rançongiciel", "fuite de données", "interruption de service"],
    "Construction": ["effondrement", "malfaçon", "dégât des eaux en chantier"],
    "Maritime": ["avarie de machine", "échouement", "perte de cargaison"],
}
BRANCH_NAMES = list(BRANCHES)
PAYS = ["France", "Allemagne", "Espagne", "Italie", "Royaume-Uni", "États-Unis",
        "Brésil", "Côte d'Ivoire", "Maroc", "Pays-Bas"]
PAYS_P = np.array([.18, .14, .1, .1, .1, .12, .06, .07, .06, .07])

# Facteurs fréquents : (nom, prévalence de base, effet sur le contentieux (logit), paraphrases)
FACTEURS = [
    ("retard de déclaration", .18, .7, ["la déclaration a été faite tardivement",
                                         "le sinistre a été déclaré avec retard",
                                         "déclaration intervenue plusieurs mois après les faits"]),
    ("pluralité d'assureurs", .16, .5, ["plusieurs assureurs interviennent en coassurance",
                                         "le risque est partagé entre plusieurs porteurs",
                                         "coassurance avec trois autres compagnies"]),
    ("expertise contradictoire", .15, 1.0, ["les experts des parties divergent sur l'origine",
                                             "une contre-expertise contredit notre expert",
                                             "désaccord persistant entre experts"]),
    ("sous-traitant impliqué", .14, .4, ["un sous-traitant est mis en cause",
                                          "intervention d'un sous-traitant à l'origine du dommage",
                                          "la responsabilité d'un sous-traitant est évoquée"]),
    ("clause d'exclusion ambiguë", .10, 1.1, ["la portée d'une exclusion est discutée",
                                               "le libellé de l'exclusion prête à interprétation",
                                               "l'assuré conteste l'application d'une exclusion"]),
    ("dommages corporels", .10, .8, ["des blessés sont à déplorer",
                                      "plusieurs victimes corporelles",
                                      "atteintes corporelles graves"]),
    ("perte d'exploitation", .17, .3, ["l'activité est interrompue durablement",
                                        "pertes d'exploitation significatives",
                                        "arrêt de production de plusieurs semaines"]),
    ("pluralité de victimes", .09, .9, ["de nombreux tiers lésés se manifestent",
                                         "réclamations de multiples tiers",
                                         "action groupée de victimes envisagée"]),
    ("sous-assurance", .08, .6, ["la valeur assurée est inférieure à la valeur réelle",
                                  "une sous-assurance est constatée",
                                  "application probable de la règle proportionnelle"]),
]
# Facteurs rares : prévalence 1 à 2 %, en général absents d'un schéma induit sur 200 dossiers
FACTEURS_RARES = [
    ("sanctions internationales", .015, .8, ["une contrepartie figure sur une liste de sanctions",
                                             "vérification de conformité aux sanctions internationales en cours",
                                             "un paiement est bloqué au titre des sanctions"]),
    ("faute inexcusable", .012, 1.2, ["la faute inexcusable de l'employeur est invoquée",
                                      "les ayants droit plaident une faute inexcusable",
                                      "procédure en reconnaissance de faute inexcusable"]),
    ("fraude suspectée", .018, 1.0, ["des incohérences laissent soupçonner une fraude",
                                     "une enquête interne pour fraude est ouverte",
                                     "suspicion de déclaration frauduleuse"]),
]
ALL_FACTEURS = FACTEURS + FACTEURS_RARES
F_NAMES = [f[0] for f in ALL_FACTEURS]
N_FREQ = len(FACTEURS)

LITIGE_OUI = ["Une assignation a été délivrée devant le tribunal compétent.",
              "Le dossier est désormais en contentieux.",
              "La partie adverse a engagé une action en justice.",
              "Une procédure judiciaire est en cours."]
LITIGE_NON = ["Le dossier se dirige vers un règlement amiable.",
              "Aucune action judiciaire n'a été engagée à ce stade.",
              "Une transaction est en cours de négociation.",
              "Les parties privilégient une solution négociée."]
REVISIONS = ["La réserve a été révisée suite au rapport d'expertise.",
             "Réserve ajustée après réception des devis de reconstruction.",
             "La réserve intègre désormais les frais de défense.",
             "Pas de révision de réserve depuis l'ouverture."]

# Multiplicateurs de prévalence par branche (rend les 3 premiers facteurs propres à chaque branche)
_rng0 = np.random.default_rng(12345)
MULT = _rng0.lognormal(0.0, 0.45, size=(len(BRANCH_NAMES), len(ALL_FACTEURS)))
MU_MONTANT = np.array([16.0, 15.0, 15.5, 15.8, 16.5, 15.2, 15.6, 16.2])  # log EUR


def fmt_montant(x: float, rng) -> str:
    """Montant écrit de deux façons, comme dans des courriels réels."""
    if rng.random() < .5:
        return f"{x/1e6:.1f} M EUR".replace(".", ",")
    return f"{int(round(x, -3)):,} EUR".replace(",", " ")


def generate(n: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(BRANCH_NAMES), n)
    pays = rng.choice(len(PAYS), n, p=PAYS_P)
    cause = rng.integers(0, 3, n)
    annee = rng.integers(2015, 2025, n)
    prev = np.array([f[1] for f in ALL_FACTEURS])
    p = np.clip(prev[None, :] * MULT[b], 0, .6)
    F = rng.random((n, len(ALL_FACTEURS))) < p
    montant = np.exp(rng.normal(MU_MONTANT[b], .9))
    eff = np.array([f[2] for f in ALL_FACTEURS])
    logit = -1.9 + F @ eff + .35 * (pays == PAYS.index("États-Unis")) + .25 * (np.log(montant) - 15.7)
    litige = rng.random(n) < 1 / (1 + np.exp(-logit))
    reserve = montant * rng.uniform(.6, 1.1, n)

    rows, texts = [], []
    for i in range(n):
        did = f"SIN-{annee[i]}-{i:05d}"
        bn = BRANCH_NAMES[b[i]]
        cs = BRANCHES[bn][cause[i]]
        facts = [rng.choice(ALL_FACTEURS[j][3]) for j in np.flatnonzero(F[i])]
        rng.shuffle(facts)
        corps = [f"Objet : synthèse du dossier {did} ({bn}, {PAYS[pays[i]]})",
                 "Bonjour à tous,",
                 f"Pour mémoire, sinistre survenu en {annee[i]} (pays : {PAYS[pays[i]]}), branche {bn}. "
                 f"Origine : {cs}.",
                 f"Dommage estimé à {fmt_montant(montant[i], rng)} ; réserve actuelle {fmt_montant(reserve[i], rng)}."]
        if facts:
            corps.append("Points d'attention : " + " ; ".join(facts) + ".")
        corps.append(rng.choice(LITIGE_OUI if litige[i] else LITIGE_NON))
        corps.append(rng.choice(REVISIONS))
        corps.append("Cordialement, le gestionnaire grands risques")
        texts.append("\n".join(corps))
        rows.append(dict(id=did, branche=b[i], pays=pays[i], cause=cause[i], annee=annee[i],
                         montant=montant[i], reserve=reserve[i], litige=bool(litige[i])))
    df = pd.DataFrame(rows)
    for j, name in enumerate(F_NAMES):
        df[f"f{j}"] = F[:, j]
    return df, texts
