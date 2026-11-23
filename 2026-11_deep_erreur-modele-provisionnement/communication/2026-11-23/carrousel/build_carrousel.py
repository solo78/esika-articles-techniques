"""Carrousel LinkedIn Esika (16:9), mode dégradé de esika-slide-builder.

La bibliothèque esika_pptx et le master ne sont pas téléchargeables par le connecteur SharePoint :
on reproduit ici la grille et les tokens de references/charte.md (marges 0,61 ; titre y 0,39 h 0,80 ;
filet y 1,29 ; chapeau y 1,40 ; corps jusqu'à 6,80 ; filet bas y 7,06 ; pied « Esika » + numéro).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

NAVY, GREEN, BROWN, TECH = RGBColor(0x1C, 0x30, 0x41), RGBColor(0x01, 0x43, 0x00), RGBColor(0x77, 0x49, 0x36), RGBColor(0x0B, 0x4F, 0x6C)
GREY_TXT, CARD, NOTE, RULE, WHITE = RGBColor(0x59, 0x59, 0x59), RGBColor(0xF2, 0xF2, 0xF2), RGBColor(0xE4, 0xEA, 0xF0), RGBColor(0xD9, 0xD9, 0xD9), RGBColor(0xFF, 0xFF, 0xFF)
L, W_USE = 0.61, 12.12
IMG = "images/"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, runs, size=12, font="Arial", color=NAVY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None):
    """runs : str, ou liste de paragraphes ; un paragraphe = str ou liste de (texte, gras)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        segs = [(para, bold)] if isinstance(para, str) else para
        for s, b in segs:
            r = p.add_run()
            r.text = s
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = font, Pt(size), b, color
    return tb


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    st = s._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style')
    if st is not None:
        s._element.remove(st)
    return s


def rule(slide, y):
    ln = slide.shapes.add_connector(1, Inches(L), Inches(y), Inches(L + W_USE), Inches(y))
    ln.line.color.rgb = RULE
    ln.line.width = Pt(0.75)


def picture(slide, path, x, y, w, h):
    """Image centrée dans la boîte (x, y, w, h), proportions conservées."""
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def content(n, tracker, title, lead=None, notes=""):
    s = prs.slides.add_slide(BLANK)
    text(s, L + 8.5, 0.11, W_USE - 8.5, 0.2, tracker.upper(), size=7, color=GREY_TXT, align=PP_ALIGN.RIGHT)
    text(s, L, 0.39, W_USE, 0.80, title, size=24, font="Georgia", bold=True, anchor=MSO_ANCHOR.BOTTOM)
    rule(s, 1.29)
    if lead:
        text(s, L, 1.40, W_USE, 0.7, lead, size=15, color=NAVY)
    rule(s, 7.06)
    text(s, L, 7.12, 3, 0.25, "Esika", size=9, color=GREY_TXT)
    text(s, L + W_USE - 1, 7.12, 1, 0.25, f"{n} / 8", size=9, color=GREY_TXT, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


# 1. Couverture
s = prs.slides.add_slide(BLANK)
rect(s, 9.0, 0, 4.333, 7.5, GREEN)
text(s, 9.45, 5.55, 3.5, 0.5, "Esika", size=28, font="Georgia", bold=True, color=WHITE)
text(s, 9.45, 6.15, 3.5, 0.6, "Financial Engineering & Process Optimization", size=12, color=WHITE)
text(s, L, 0.9, 7.8, 0.3, "PROVISIONNEMENT NON VIE  ·  ERREUR DE MODÈLE", size=9, color=GREY_TXT)
text(s, L, 1.6, 7.9, 2.6, "Votre provision dépend du modèle choisi. De combien ?", size=40, font="Georgia", bold=True)
text(s, L, 4.5, 7.6, 1.2, "Faire voter des centaines de modèles pour chiffrer l'erreur de modèle, et ce que ce chiffre ne voit pas.",
     size=16, color=GREY_TXT)
rect(s, L, 4.25, 1.2, 0.06, BROWN)
s.notes_slide.notes_text_frame.text = ("Méthode : Taylor et McGuire, Model Error (or Ambiguity) and Its Estimation, Risks 2023, 11(11), 185. "
                                       "Tous les chiffres du carrousel viennent d'une reproduction Esika sur données simulées.")

# 2. Le problème
s = content(2, "Le problème", "Même passé, avenirs différents : le choix du modèle reste hors de la marge",
            "Mack et le bootstrap chiffrent l'aléa et l'estimation des paramètres. Ils supposent tous deux la forme du modèle juste.",
            "Trois modèles qui collent aux mêmes données passées divergent une fois prolongés : c'est l'erreur de modèle.")
picture(s, IMG + "crop_fig1_intuition.png", L + 0.6, 2.2, W_USE - 1.2, 4.6)

# 3. L'idée
s = content(3, "L'idée", "Plutôt que choisir un modèle, faire voter tous les modèles plausibles",
            "Chaque modèle pèse selon son ajustement et sa simplicité. L'écart entre leurs provisions mesure l'erreur de modèle.",
            "Moyenne bayésienne de modèles : poids = a priori de Laplace (forme issue du LASSO) x vraisemblance Gamma.")
picture(s, IMG + "crop_vote.png", L + 0.4, 2.15, W_USE - 0.8, 4.7)

# 4. Méthode (1)
s = content(4, "La méthode", "Le LASSO fabrique les modèles, Bayes les pèse, des portes écartent les extrapolations",
            None, "Briques 1 à 5 du papier : famille LASSO Poisson, vraisemblance Gamma, a priori de Laplace, portes de la Table 2, vote.")
picture(s, IMG + "crop_fig0_carte_du_papier.png", L, 1.55, W_USE, 2.75)
cards = [("Famille", "Un LASSO produit des centaines de modèles, du plus simple au plus complexe."),
         ("Poids", "Bon ajustement aux données, mais un prix à payer pour la complexité."),
         ("Portes", "Les modèles qui extrapolent de façon déraisonnable sont écartés.")]
cw = (W_USE - 2 * 0.3) / 3
for k, (h, b) in enumerate(cards):
    x = L + k * (cw + 0.3)
    rect(s, x, 4.6, cw, 2.0, CARD)
    text(s, x + 0.25, 4.8, cw - 0.5, 0.4, h, size=17, bold=True, color=BROWN)
    text(s, x + 0.25, 5.35, cw - 0.5, 1.2, b, size=15)

# 5. Méthode (2)
s = content(5, "La méthode", "Le bootstrap sépare l'erreur de modèle de l'erreur de paramètre",
            "On refait tout le vote sur 300 triangles simulés : l'écart dans un vote mesure le modèle, l'écart entre les votes les paramètres.",
            "Section 7 du papier. Dans notre reproduction : résidus de type ODP et portes élargies de 10 %, le protocole littéral n'ayant pas tenu.")
picture(s, IMG + "crop_fig3_bootstrap.png", L + 0.4, 2.35, W_USE - 0.8, 4.4)

# 6. Résultats
s = content(6, "Ce que ça donne", "Sur données simulées, l'erreur mesurée est faible, mais un biais lui échappe",
            "Reproduction sur deux jeux de triangles fictifs du papier.",
            "Sources : results/bootstrap.csv (IMSE 1,3 à 1,8 %, total 7,0 à 9,0 %), results/realisme.csv et complements.json "
            "(écart moyen +23,9 % et +34,7 % sur 20 triangles par jeu ; Mack -0,4 % et +2,6 %, non distinguables de zéro).")
kpis = [("1,3 à 1,8 %", "erreur de modèle interne, en part de la provision", False),
        ("7 à 9 %", "erreur totale mesurée (modèle, paramètre, aléa)", False),
        ("+24 à +35 %", "écart moyen entre la provision et les paiements réellement simulés", True)]
kw = (W_USE - 2 * 0.3) / 3
for k, (v, lab, hi) in enumerate(kpis):
    x = L + k * (kw + 0.3)
    rect(s, x, 2.3, kw, 2.9, GREEN if hi else CARD)
    c = WHITE if hi else NAVY
    text(s, x + 0.25, 2.65, kw - 0.5, 1.0, v, size=36, font="Georgia", bold=True, color=c)
    text(s, x + 0.25, 3.8, kw - 0.5, 1.3, lab, size=16, color=c)
rect(s, L, 5.55, W_USE, 0.95, NOTE)
text(s, L + 0.25, 5.68, W_USE - 0.5, 0.75,
     [[("Simulation : ", True), ("20 triangles indépendants par jeu. Sur les mêmes triangles, l'écart moyen de la méthode de Mack n'est pas distinguable de zéro.", False)]],
     size=15)

# 7. Usage business
s = content(7, "Usage business", "Usage : documenter la marge de risque IFRS 17, à côté d'une méthode de référence",
            "Un chiffrage justifié de l'erreur de modèle, à la place d'un forfait, calculé à chaque clôture.",
            "Exemple fictif de l'article : 500 M€ de provisions, erreur totale 7 %, RA à 75 % d'environ 23,6 M€, dont 0,8 M€ d'erreur de modèle interne.")
picture(s, IMG + "crop_fig9_cas_usage.png", L + 0.5, 2.05, W_USE - 1.0, 3.6)
rect(s, L, 5.85, W_USE, 0.8, NOTE)
text(s, L + 0.25, 6.0, W_USE - 0.5, 0.55,
     [[("Garde-fou : ", True), ("comparer chaque clôture aux paiements observés et chiffrer à part l'inflation future.", False)]], size=15)

# 8. Conclusion
s = content(8, "À retenir", "Mesurer la dispersion ne suffit pas : il faut aussi contrôler la justesse",
            None, "Question ouverte posée par les auteurs eux-mêmes : rendre les portes objectives.")
pts = [("1", "L'erreur de modèle se mesure, en faisant voter les modèles plausibles."),
       ("2", "Le résultat dépend de réglages à publier : résidus, grille, portes."),
       ("3", "Un vote resserré peut être faux : le comparer au réalisé.")]
for k, (n_, t_) in enumerate(pts):
    y = 1.7 + k * 1.05
    text(s, L, y, 0.7, 0.8, n_, size=36, font="Georgia", bold=True, color=BROWN)
    text(s, L + 0.8, y + 0.12, W_USE - 0.8, 0.8, t_, size=20)
rect(s, L, 5.05, W_USE, 0.75, NOTE)
text(s, L + 0.25, 5.22, W_USE - 0.5, 0.5, [[("Question ouverte : ", True), ("comment corriger le biais sans perdre la richesse du vote ?", False)]], size=16)
rect(s, L, 6.0, W_USE, 0.75, GREEN)
text(s, L, 6.0, W_USE, 0.75, "Esika : Financial Engineering & Process Optimization", size=15, bold=True, color=WHITE,
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

out = "Esika - Carrousel - Erreur de modele en provisionnement.pptx"
prs.save(out)
print(out)
