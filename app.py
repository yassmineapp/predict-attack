# -*- coding: utf-8 -*-
"""
Attack Predict - Détection et prédiction des cyberattaques (Streamlit)
Auteur : NAAOUMI Yassmine

Interface en UNE SEULE page, dense et sans espaces vides.
La logique de l'atelier d'origine est conservée : chargement du CSV -> doublons -> NaN ->
X / y -> découpage 70/15/15 -> standardisation -> modèles -> comparaison sur la validation
-> évaluation du meilleur modèle sur le test.

Aucun résultat n'est écrit "en dur" : tout ce qui est affiché est calculé à partir du CSV.
Lancer (depuis ce dossier) :  streamlit run app.py
"""
import os
import time
import math
import base64
import hmac

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix, roc_auc_score)
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Attack Predict", layout="wide", initial_sidebar_state="collapsed")

RANDOM_STATE = 42
AUTEUR = "NAAOUMI Yassmine"
IDENTIFIANT = "admin"
MOT_DE_PASSE = "yassmine"
DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

COLS_EXCLURES = ["attack_type", "is_attack", "timestamp_utc", "src_ip", "dst_ip",
                 "src_port", "sample_id"]
MODELES_DISPO = ["Régression logistique", "KNN", "SVM", "Random Forest", "XGBoost"]
MODELES_DEFAUT = ["Régression logistique", "Random Forest", "XGBoost"]
A_STANDARDISER = {"Régression logistique", "KNN", "SVM"}

# Palette sobre : bleus, cyans, violets, un vert d'eau et un ambre doux
COULEURS = ["#3b82f6", "#22d3ee", "#8b5cf6", "#14b8a6", "#6366f1",
            "#38bdf8", "#a78bfa", "#2dd4bf", "#60a5fa", "#c4b5fd"]
ROUGE_DOUX, VERT_DOUX, AMBRE_DOUX = "#e5566a", "#2fbf9b", "#d9a441"

LOGO_SVG = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 380 92'> <defs> <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#050f24'/><stop offset='.55' stop-color='#0a2352'/><stop offset='1' stop-color='#061431'/></linearGradient> <linearGradient id='tx' x1='0' y1='0' x2='1' y2='0'><stop offset='0' stop-color='#8fdcff'/><stop offset='1' stop-color='#3b9cff'/></linearGradient> <linearGradient id='hd' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#123a7a'/><stop offset='1' stop-color='#0a2352'/></linearGradient> </defs> <rect width='380' height='92' rx='10' fill='url(#bg)'/> <g stroke='#35d0ff' stroke-width='1' fill='none' opacity='.55'><path d='M0 20 H40'/><path d='M0 72 H34'/><path d='M380 20 H350 L340 30 H320'/><path d='M380 74 H356 L346 64 H328'/></g> <g transform='translate(12 6) scale(.8)'> <path d='M 22 92 C 22 80 14 72 12 58 C 8 40 18 14 48 10 C 72 7 88 24 86 42 C 87 46 90 52 93 58 L 85 60 C 86 66 84 70 80 72 L 80 80 C 80 84 74 86 66 88 L 66 92 Z' fill='url(#hd)' stroke='#35d0ff' stroke-width='2.4' stroke-linejoin='round'/> <g stroke='#35d0ff' stroke-width='1.6' fill='none' stroke-linecap='round'> <path d='M30 44 H42 L48 34 H60 L66 26'/><path d='M34 58 H46 L52 48 H70 L76 40'/><path d='M40 30 L46 22 H58'/><path d='M52 62 H64 L70 54'/><path d='M26 52 L32 44'/> </g> <g fill='#8fdcff'><circle cx='30' cy='44' r='2.6'/><circle cx='60' cy='34' r='2.6'/><circle cx='66' cy='26' r='2.6'/><circle cx='70' cy='48' r='2.6'/><circle cx='76' cy='40' r='2.6'/><circle cx='58' cy='22' r='2.6'/><circle cx='64' cy='62' r='2.6'/><circle cx='70' cy='54' r='2.6'/><circle cx='34' cy='58' r='2.6'/></g> </g> <text x='108' y='46' font-family='Segoe UI,Arial,Helvetica,sans-serif' font-size='31' font-weight='800' font-style='italic'><tspan fill='#ffffff'>Attack </tspan><tspan fill='url(#tx)'>Predict</tspan></text> <path d='M110 62 H128' stroke='#3b9cff' stroke-width='1' opacity='.8'/><path d='M300 62 H318' stroke='#3b9cff' stroke-width='1' opacity='.8'/> <text x='214' y='66' text-anchor='middle' font-family='Segoe UI,Arial,Helvetica,sans-serif' font-size='10.5' fill='#c3d4ee' font-style='italic'>Prédiction des Attaques</text> </svg>"
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO_SVG.encode("utf-8")).decode()

COURT = {"Man_in_the_Middle": "MITM", "SQL_Injection": "SQL Inj.", "Port_Scanning": "Port Scan",
         "Brute_Force": "Brute Force"}

# Champs du panneau de test : (colonne, libellé, type)
CHAMPS = [
    ("dst_port", "Port destination", "int"),
    ("flow_duration_ms", "Durée du flux (ms)", "int"),
    ("packet_count", "Nb de paquets", "int"),
    ("packet_rate_pps", "Paquets / s", "float"),
    ("avg_packet_size_bytes", "Taille paquet (o)", "float"),
    ("syn_count", "Paquets SYN", "int"),
    ("unique_dst_ports_5m", "Ports distincts 5 min", "int"),
    ("failed_logins_5m", "Échecs login 5 min", "int"),
    ("http_requests_5m", "Requêtes HTTP 5 min", "int"),
    ("sql_keyword_hits_5m", "Mots-clés SQL 5 min", "int"),
    ("xss_pattern_hits_5m", "Motifs XSS 5 min", "int"),
    ("dns_queries_5m", "Requêtes DNS 5 min", "int"),
    ("arp_spoof_events_5m", "ARP spoof 5 min", "int"),
    ("email_links_count", "Liens e-mail", "int"),
    ("payload_entropy", "Entropie charge utile", "float"),
    ("file_entropy", "Entropie fichiers", "float"),
    ("beacon_interval_cv", "Régularité balises", "float"),
]

# ----------------------------------------------------------------------------
# 2. STYLE : bleu nuit, grille serrée, cartes de même hauteur
# ----------------------------------------------------------------------------
CSS = """
<style>
:root{--bg:#f2f7fc;--card:#ffffff;--line:#dce8f4;--ink:#18324f;--muted:#66809b;--blue:#2878c7;}
.stApp{background:var(--bg);color:var(--ink);}
.block-container{padding:.45rem .65rem .8rem;max-width:1600px;}
#MainMenu,footer,section[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important;}
header[data-testid="stHeader"]{background:transparent;height:0;}
[data-testid="stVerticalBlock"]{gap:.35rem;}
[data-testid="stHorizontalBlock"]{gap:.45rem;align-items:stretch;}
.stApp p,.stApp label,.stApp li{color:var(--ink);}
.stApp span{color:inherit;}
[data-testid="stCaptionContainer"]{color:var(--muted)!important;}
div[data-testid="stVerticalBlockBorderWrapper"],div[data-testid="stExpander"]{
 background:var(--card);border:1px solid var(--line);border-radius:10px;
}
div[data-testid="stSlider"] label p,div[data-testid="stSelectbox"] label p{
 font-size:.66rem;color:var(--muted);margin-bottom:0;
}
div[data-testid="stSlider"]{padding-bottom:0;}
.stButton>button{background:#edf5fd;border:1px solid #cbdced;color:#21496f;border-radius:7px;
 width:100%;padding:.2rem .35rem;font-size:.74rem;min-height:1.9rem;font-weight:650;}
.stButton>button:hover{background:#dceeff;border-color:#80b6e8;color:#174f85;}
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"]{
 background:#2878c7;border:none;color:#fff;}
div[data-baseweb="select"]>div,input{background:#fff!important;color:#18324f!important;border-color:#cbdced!important;}
.topbar{display:flex;align-items:center;gap:10px;background:#fff;border:1px solid var(--line);
 border-radius:10px;padding:5px 10px;box-shadow:0 2px 8px rgba(34,75,115,.05);}
.topbar .logo{height:43px;border-radius:7px;}
.topbar .sep{width:1px;height:32px;background:var(--line);}
.topbar .ttl{display:flex;align-items:center;gap:8px;flex:1;min-width:0;}
.topbar .t1{font-size:1.03rem;font-weight:800;color:#17466f;line-height:1.15;}
.topbar .t2{font-size:.66rem;color:var(--muted);}
.topbar .user{display:flex;align-items:center;gap:6px;border:1px solid var(--line);
 border-radius:7px;padding:3px 7px;font-size:.7rem;color:#31516f;white-space:nowrap;}
.topbar .av{background:#2878c7;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;
 align-items:center;justify-content:center;font-weight:800;font-size:.65rem;}
.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:6px;align-items:stretch;width:100%;min-width:0;}
.c{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:7px 8px;
 min-width:0;max-width:100%;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 1px 4px rgba(34,75,115,.035);}
.c.kpi{border-left:3px solid var(--k);padding:7px 8px;}
.kt{font-size:.60rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.025em;}
.kv{font-size:1.22rem;font-weight:800;color:#183b5c;line-height:1.12;overflow-wrap:anywhere;}
.ks{font-size:.60rem;color:var(--muted);}
.ct{font-size:.68rem;font-weight:800;color:#315b80;margin-bottom:4px;text-transform:uppercase;letter-spacing:.025em;}
.brow{display:flex;align-items:center;gap:5px;margin:2px 0;font-size:.62rem;color:#496985;min-width:0;}
.brow .bl{width:82px;flex:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.brow .bt{flex:1;min-width:10px;background:#e8f0f8;border-radius:4px;height:6px;overflow:hidden;}
.brow .bf{height:100%;border-radius:4px;}
.brow .bv{width:45px;text-align:right;font-weight:700;color:#294b6b;}
.tl .tli{display:flex;gap:6px;align-items:flex-start;margin:4px 0;font-size:.64rem;min-width:0;}
.tl .dot{width:7px;height:7px;border-radius:50%;margin-top:4px;flex:none;}
.tl .sub{color:var(--muted);font-size:.58rem;overflow-wrap:anywhere;}
.tl .tm{margin-left:auto;color:var(--muted);font-size:.57rem;white-space:nowrap;}
table.tb{width:100%;border-collapse:collapse;font-size:.59rem;display:block;overflow-x:auto;}
table.tb th{color:var(--muted);text-align:left;font-weight:700;padding:3px 4px;border-bottom:1px solid var(--line);white-space:nowrap;}
table.tb td{padding:3px 4px;border-bottom:1px solid #edf2f7;color:#294b6b;white-space:nowrap;}
table.tb tr.best td{background:#eaf4ff;}
.badge-a{background:#fff0f1;color:#c74357;border:1px solid #f2c5cb;border-radius:4px;padding:0 5px;font-weight:700;}
.badge-n{background:#eafaf5;color:#168565;border:1px solid #bdebdc;border-radius:4px;padding:0 5px;font-weight:700;}
.res{border-radius:8px;padding:8px 9px;display:flex;align-items:center;gap:10px;}
.res.att{background:#fff1f2;border:1px solid #f0c3ca;}
.res.ok{background:#eafaf5;border:1px solid #bce8d9;}
.res .rt{font-size:.58rem;color:var(--muted);font-weight:700;text-transform:uppercase;}
.res .rv{font-size:1.1rem;font-weight:900;line-height:1.12;}
.chip{display:inline-block;background:#e8f2fc;color:#2368a5;font-weight:700;border-radius:5px;padding:1px 6px;font-size:.68rem;}
.hm{border-collapse:collapse;font-size:.55rem;max-width:100%;}
.hm td,.hm th{padding:2px 3px;text-align:center;border:1px solid #e2ebf4;color:#496985;}
.hm th.v{writing-mode:vertical-rl;transform:rotate(180deg);height:60px;font-weight:600;}
.hm th.l{text-align:right;font-weight:600;white-space:nowrap;}
.login-t{text-align:center;margin:18px 0 10px;}
[data-testid="stAppViewContainer"]{overflow-x:hidden;}
@media(max-width:1100px){
 .topbar .t1{font-size:.88rem}.topbar .logo{height:36px}
 .grid{grid-template-columns:repeat(6,minmax(0,1fr));}
 .grid>.c{grid-column:span 3!important;}
 .grid>.c.kpi{grid-column:span 2!important;}
}
@media(max-width:760px){
 .block-container{padding:.35rem .4rem .6rem;}
 .topbar{flex-wrap:wrap}.topbar .user{margin-left:auto;}
 .grid{grid-template-columns:repeat(2,minmax(0,1fr));}
 .grid>.c,.grid>.c.kpi{grid-column:span 2!important;}
}
.c{border-radius:12px;}
.c.kpi{border-left:none;flex-direction:row;align-items:center;gap:10px;padding:10px 12px;}
.ico{width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;
 background:color-mix(in srgb,var(--k) 14%,#fff);color:var(--k);font-size:1.1rem;flex:none;}
.nav{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px;display:flex;flex-direction:column;gap:3px;}
.nav a{display:block;padding:9px 10px;border-radius:8px;color:#496985!important;font-size:.76rem;font-weight:650;text-decoration:none!important;}
.nav a:hover{background:#edf5fd;}
.nav a.on{background:#2878c7;color:#fff!important;}
.lst{display:flex;justify-content:space-between;align-items:center;font-size:.72rem;margin:6px 0;color:#496985;}
.lst b{color:#183b5c;}
.dotc{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;}
@media(max-width:760px){.nav{flex-direction:row;flex-wrap:wrap;}}
</style>
"""


def html(contenu):
    """Affiche du HTML (retours à la ligne retirés pour éviter les blocs de code Markdown)."""
    st.markdown("".join(l.strip() for l in contenu.splitlines()), unsafe_allow_html=True)


def joli(nom):
    return str(nom).replace("_", " ")


def court(nom):
    return COURT.get(nom, joli(nom))


def nombre(n):
    return f"{int(n):,}".replace(",", " ")


def pct(v, d=2):
    return f"{v * 100:.{d}f} %".replace(".", ",")


# ----------------------------------------------------------------------------
# 3. DONNÉES ET MODÈLES (logique de l'atelier)
# ----------------------------------------------------------------------------
def lire_fichiers(fichiers):
    """Lit un ou plusieurs CSV téléversés (attack_type déduit du nom du fichier si absent)."""
    morceaux = []
    for f in fichiers:
        d = pd.read_csv(f, low_memory=False)
        if "attack_type" not in d.columns:
            d["attack_type"] = os.path.splitext(f.name)[0].split("_", 1)[-1]
        morceaux.append(d)
    return pd.concat(morceaux, ignore_index=True, sort=False)


def nettoyer(df_brut, n_par_classe):
    """Étapes 06 à 08 de l'atelier : doublons et valeurs manquantes."""
    df = df_brut.copy()
    info = {"lignes_brutes": len(df)}
    df = df.drop(columns=["sample_id"], errors="ignore")
    if n_par_classe:
        df = pd.concat([g.sample(min(len(g), n_par_classe), random_state=RANDOM_STATE)
                        for _, g in df.groupby("attack_type")])
    info["doublons"] = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    info["nan_total"] = int(df.isna().sum().sum())
    num = df.select_dtypes(include=np.number).columns
    df[num] = df[num].fillna(0)
    for col in df.columns.difference(num):
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "inconnu")
    df = df.dropna(subset=["attack_type"]).copy()
    df["attack_type"] = df["attack_type"].astype(str)
    return df, info


def vers_X(df, colonnes=None, entrainement=False):
    """Étape 10 : variables explicatives (get_dummies sur le texte). Même fonction pour l'inférence."""
    X = df.drop(columns=COLS_EXCLURES, errors="ignore").copy()
    if entrainement:
        X = X.dropna(axis=1, how="all")
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cat, dtype=int)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    if colonnes is not None:
        X = X.reindex(columns=colonnes, fill_value=0.0)
    return X


def metriques(y_vrai, y_pred):
    return {
        "Accuracy": accuracy_score(y_vrai, y_pred),
        "Précision (macro)": precision_score(y_vrai, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_vrai, y_pred, average="macro", zero_division=0),
        "F1-score (macro)": f1_score(y_vrai, y_pred, average="macro", zero_division=0),
    }


def construire_modele(nom, ponderer):
    cw = "balanced" if ponderer else None
    if nom == "Régression logistique":
        return LogisticRegression(max_iter=300, class_weight=cw, random_state=RANDOM_STATE)
    if nom == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if nom == "SVM":
        return SVC(kernel="rbf", C=1.0, class_weight=cw)
    if nom == "Random Forest":
        return RandomForestClassifier(n_estimators=100, class_weight=cw,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                         objective="multi:softprob", eval_metric="mlogloss",
                         tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)


@st.cache_resource(show_spinner=False)
def entrainer(cle, modeles_choisis, n_par_classe, ponderer, _df):
    """Entraîne les modèles choisis et calcule toutes les métriques (validation puis test)."""
    df, info = nettoyer(_df, n_par_classe)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df["attack_type"]), index=df.index)
    classes = list(le.classes_)
    if y.value_counts().min() < 3:
        raise ValueError("Chaque classe doit contenir au moins 3 lignes.")
    X = vers_X(df, entrainement=True)
    colonnes = X.columns.tolist()

    # Découpage 70 % / 15 % / 15 % stratifié
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp)
    scaler = StandardScaler().fit(X_tr)
    brut = {"train": X_tr, "val": X_va, "test": X_te}
    std = {k: scaler.transform(v) for k, v in brut.items()}
    poids = compute_sample_weight("balanced", y_tr) if ponderer else None

    modeles, lignes = {}, []
    for nom in modeles_choisis:
        m = construire_modele(nom, ponderer)
        d = std if nom in A_STANDARDISER else brut
        t0 = time.time()
        if nom == "XGBoost":
            m.fit(d["train"], y_tr, sample_weight=poids)
        else:
            m.fit(d["train"], y_tr)
        duree = time.time() - t0
        lignes.append({"Modèle": nom, **metriques(y_va, m.predict(d["val"])), "Durée (s)": round(duree, 1)})
        modeles[nom] = m
    tableau = pd.DataFrame(lignes).sort_values("F1-score (macro)", ascending=False).reset_index(drop=True)
    meilleur = tableau.loc[0, "Modèle"]

    # Évaluation finale du meilleur modèle sur le test (+ temps de prédiction mesuré)
    d = std if meilleur in A_STANDARDISER else brut
    m = modeles[meilleur]
    t0 = time.time()
    pred_te = m.predict(d["test"])
    ms_par_ligne = (time.time() - t0) / max(len(pred_te), 1) * 1000
    labels = np.arange(len(classes))
    rapport = classification_report(y_te, pred_te, labels=labels, target_names=classes,
                                    zero_division=0, output_dict=True)
    cm = confusion_matrix(y_te, pred_te, labels=labels)

    proba_te = m.predict_proba(d["test"]) if hasattr(m, "predict_proba") else None
    tot = cm.sum()
    spec = []
    for i in range(len(classes)):
        fp = cm[:, i].sum() - cm[i, i]
        tn = tot - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    try:
        auc = float(roc_auc_score(y_te, proba_te, multi_class="ovr", average="macro", labels=labels))
    except Exception:
        auc = float("nan")
    # Prédictions du meilleur modèle sur le jeu de test (incidents / activités)
    inc = pd.DataFrame(index=X_te.index)
    for c, nom_c in [("timestamp_utc", "Horodatage"), ("src_ip", "IP source"), ("dst_ip", "IP destination")]:
        if c in df.columns:
            inc[nom_c] = df.loc[X_te.index, c].astype(str)
    inc["Classe réelle"] = [classes[i] for i in y_te]
    inc["Classe prédite"] = [classes[i] for i in pred_te]
    inc["Probabilité"] = proba_te.max(axis=1) if proba_te is not None else np.nan

    # Valeurs par défaut du panneau de test : médiane / mode du trafic "Normal"
    brutes = df.drop(columns=COLS_EXCLURES, errors="ignore")
    ref = df[df["attack_type"] == "Normal"] if (df["attack_type"] == "Normal").any() else df
    base = {}
    for c in brutes.columns:
        base[c] = float(ref[c].median()) if pd.api.types.is_numeric_dtype(brutes[c]) else str(ref[c].mode().iloc[0])
    plages = {c: (float(df[c].min()), float(df[c].max())) for c, _, _ in CHAMPS if c in brutes.columns}

    importance, importance_modele = None, None
    for nom in ("Random Forest", "XGBoost"):
        if nom in modeles:
            importance = pd.Series(modeles[nom].feature_importances_, index=colonnes).sort_values(ascending=False)
            importance_modele = nom
            break

    return {"cle": cle, "df": df, "info": info, "classes": classes, "colonnes": colonnes,
            "brutes": list(brutes.columns), "base": base, "plages": plages, "scaler": scaler,
            "modeles": modeles, "tableau": tableau, "meilleur": meilleur,
            "metriques_test": {**metriques(y_te, pred_te), "Spécificité (macro)": float(np.mean(spec)), "AUC (OvR)": auc}, "rapport": rapport, "cm": cm,
            "incidents": inc, "df_test": df.loc[X_te.index], "tailles": (len(X_tr), len(X_va), len(X_te)),
            "importance": importance, "importance_modele": importance_modele,
            "ms_par_ligne": ms_par_ligne,
            "options": {c: sorted(df[c].astype(str).unique()) for c in ("protocol", "service") if c in df.columns}}


def predire(res, nom, lignes):
    """Prédit la classe (et les probabilités si le modèle les fournit) pour des lignes brutes."""
    X = vers_X(lignes, colonnes=res["colonnes"])
    Xm = res["scaler"].transform(X) if nom in A_STANDARDISER else X
    m = res["modeles"][nom]
    pred = m.predict(Xm)
    proba = m.predict_proba(Xm) if hasattr(m, "predict_proba") else None
    return pred, proba


# ----------------------------------------------------------------------------
# 4. COMPOSANTS (renvoient du HTML)
# ----------------------------------------------------------------------------
SHIELD = ("<svg width='26' height='26' viewBox='0 0 24 24' fill='none' stroke='#60a5fa' stroke-width='1.8' "
          "stroke-linejoin='round'><path d='M12 2.5 4.5 5.5v6c0 4.6 3.1 8.4 7.5 10 4.4-1.6 7.5-5.4 7.5-10v-6z'/>"
          "<path d='m8.8 12 2.4 2.4 4.2-4.6'/></svg>")


def cellule(span, titre, corps, extra="", ident=""):
    i = f' id="{ident}"' if ident else ""
    return f'<div class="c"{i} style="grid-column:span {span};{extra}"><div class="ct">{titre}</div>{corps}</div>'


def kpi(span, couleur, titre, valeur, sous="", icone=""):
    ic = f'<div class="ico">{icone}</div>' if icone else ""
    return (f'<div class="c kpi" style="grid-column:span {span};--k:{couleur}">{ic}<div style="min-width:0">'
            f'<div class="kt">{titre}</div><div class="kv">{valeur}</div><div class="ks">{sous}</div></div></div>')


def anneau(valeur, titre, couleur="#3b82f6"):
    """Anneau de progression (comme « Performances du modèle » de l'image de référence)."""
    r = 40
    c = 2 * math.pi * r
    v = max(0.0, min(1.0, 0.0 if valeur != valeur else valeur))
    return (f"<svg viewBox='0 0 100 100' style='width:132px;flex:none'>"
            f"<circle cx='50' cy='50' r='{r}' fill='none' stroke='#e3edf8' stroke-width='11'/>"
            f"<circle cx='50' cy='50' r='{r}' fill='none' stroke='{couleur}' stroke-width='11' stroke-linecap='round' "
            f"stroke-dasharray='{c * v:.1f} {c:.1f}' transform='rotate(-90 50 50)'/>"
            f"<text x='50' y='50' text-anchor='middle' font-size='14' font-weight='800' fill='#183b5c'>{pct(valeur)}</text>"
            f"<text x='50' y='63' text-anchor='middle' font-size='8' fill='#66809b'>{titre}</text></svg>")


NAV = [("Tableau de bord", "#top"), ("Prédiction / Test", "#test"), ("Analyses", "#analyses"),
       ("Incidents", "#incidents"), ("Données & Modèle", "#donnees"), ("Évaluation", "#evaluation"),
       ("Paramètres", "#parametres")]


def sidebar_html():
    liens = "".join(f'<a href="{h}" target="_self"{" class=on" if i == 0 else ""}>{n}</a>' for i, (n, h) in enumerate(NAV))
    return f'<div class="nav">{liens}</div>'


def barres(valeurs, fmt=nombre, numeroter=False):
    m = (max(valeurs.values()) if valeurs else 1) or 1
    out = ""
    for i, (k, v) in enumerate(valeurs.items()):
        num = f"{i + 1}. " if numeroter else ""
        out += (f'<div class="brow"><span class="bl">{num}{court(k)}</span><div class="bt"><div class="bf" '
                f'style="width:{v / m * 100:.1f}%;background:{COULEURS[i % len(COULEURS)]}"></div></div>'
                f'<span class="bv">{fmt(v)}</span></div>')
    return out


def donut(valeurs, centre, sous, taille=118):
    total = sum(valeurs.values()) or 1
    acc, stops, legende = 0.0, [], ""
    for i, (k, v) in enumerate(valeurs.items()):
        c = COULEURS[i % len(COULEURS)]
        d = v / total * 100
        stops.append(f"{c} {acc:.2f}% {acc + d:.2f}%")
        acc += d
        d_txt = f"{d:.1f}".replace(".", ",")
        legende += (f'<div style="display:flex;justify-content:space-between;gap:8px;font-size:.64rem;color:#496985;margin:1px 0">'
                    f'<span><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{c};'
                    f'margin-right:5px"></span>{court(k)}</span><b style="color:#183b5c">{d_txt} %</b></div>')
    t2 = int(taille * 0.66)
    return ('<div style="display:flex;gap:12px;align-items:center;flex:1">'
            f'<div style="width:{taille}px;height:{taille}px;border-radius:50%;background:conic-gradient({", ".join(stops)});'
            'display:flex;align-items:center;justify-content:center;flex:none">'
            f'<div style="width:{t2}px;height:{t2}px;border-radius:50%;background:#ffffff;display:flex;'
            'flex-direction:column;align-items:center;justify-content:center">'
            f'<b style="font-size:.95rem;color:#183b5c">{centre}</b><span style="font-size:.55rem;color:#66809b">{sous}</span></div></div>'
            f'<div style="flex:1;min-width:0">{legende}</div></div>')


def arc(valeur, couleur, titre):
    """Jauge en demi-cercle (comme sur l'image de référence)."""
    cx, cy, r = 60, 56, 44

    def pt(f):
        a = math.pi * (1 - f)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    x0, y0 = pt(0)
    xf, yf = pt(1)
    xv, yv = pt(max(min(valeur, 1.0), 0.0005))
    return (f"<div style='text-align:center'><svg viewBox='0 0 120 72' style='width:100%;max-width:150px'>"
            f"<path d='M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {xf:.1f} {yf:.1f}' fill='none' stroke='#16275a' stroke-width='9' stroke-linecap='round'/>"
            f"<path d='M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {xv:.1f} {yv:.1f}' fill='none' stroke='{couleur}' stroke-width='9' stroke-linecap='round'/>"
            f"<text x='60' y='52' text-anchor='middle' font-size='16' font-weight='800' fill='#ffffff'>{pct(valeur, 1)}</text></svg>"
            f"<div style='font-size:.64rem;color:#9db0d6;margin-top:-4px'>{titre}</div></div>")


def radar(valeurs):
    """Radar du rappel (recall) par type d'attaque, échelle 0 à 100 %."""
    cx, cy, R = 150, 112, 74
    n = len(valeurs)
    noms = list(valeurs.keys())

    def p(i, f):
        a = -math.pi / 2 + 2 * math.pi * i / n
        return cx + R * f * math.cos(a), cy + R * f * math.sin(a)

    s = "<svg viewBox='0 0 300 224' style='width:100%;max-height:215px'>"
    for f in (0.25, 0.5, 0.75, 1.0):
        s += f"<polygon points='{' '.join('%.1f,%.1f' % p(i, f) for i in range(n))}' fill='none' stroke='#d5e3f2' stroke-width='1'/>"
    for i in range(n):
        x, y = p(i, 1)
        s += f"<line x1='{cx}' y1='{cy}' x2='{x:.1f}' y2='{y:.1f}' stroke='#d5e3f2'/>"
    pts = " ".join("%.1f,%.1f" % p(i, max(0.0, min(1.0, valeurs[k]))) for i, k in enumerate(noms))
    s += f"<polygon points='{pts}' fill='rgba(59,130,246,.35)' stroke='#60a5fa' stroke-width='1.6'/>"
    for i, k in enumerate(noms):
        x, y = p(i, max(0.0, min(1.0, valeurs[k])))
        s += f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.4' fill='#93c5fd'/>"
        lx, ly = p(i, 1.2)
        ancre = "middle" if abs(lx - cx) < 8 else ("start" if lx > cx else "end")
        s += f"<text x='{lx:.1f}' y='{ly + 3:.1f}' text-anchor='{ancre}' font-size='9' fill='#66809b'>{court(k)}</text>"
    return s + "</svg>"


def courbes(total, predit):
    """Trafic par heure (jeu de test) : total (aire) et attaques prédites (ligne), en SVG."""
    W, H, L, B, T, R = 760, 190, 36, 22, 8, 8
    m = max(max(total), max(predit), 1)

    def xy(i, v):
        return L + i * (W - L - R) / 23, T + (H - T - B) * (1 - v / m)

    def trace(v):
        return " ".join(f"{'M' if i == 0 else 'L'}{xy(i, x)[0]:.1f} {xy(i, x)[1]:.1f}" for i, x in enumerate(v))

    aire = f"{trace(total)} L{xy(23, 0)[0]:.1f} {xy(23, 0)[1]:.1f} L{xy(0, 0)[0]:.1f} {xy(0, 0)[1]:.1f} Z"
    s = (f"<svg viewBox='0 0 {W} {H}' style='width:100%;max-height:210px'>"
         "<defs><linearGradient id='ar' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#3b82f6' stop-opacity='.35'/>"
         "<stop offset='1' stop-color='#3b82f6' stop-opacity='0'/></linearGradient></defs>")
    for k in range(5):
        y = T + (H - T - B) * k / 4
        s += (f"<line x1='{L}' y1='{y:.1f}' x2='{W - R}' y2='{y:.1f}' stroke='#e6eef7'/>"
              f"<text x='{L - 5}' y='{y + 3:.1f}' text-anchor='end' font-size='9' fill='#66809b'>{round(m * (4 - k) / 4)}</text>")
    for h in range(0, 24, 2):
        s += f"<text x='{xy(h, 0)[0]:.1f}' y='{H - 6}' text-anchor='middle' font-size='9' fill='#66809b'>{h:02d}h</text>"
    s += (f"<path d='{aire}' fill='url(#ar)'/><path d='{trace(total)}' fill='none' stroke='#3b82f6' stroke-width='1.8'/>"
          f"<path d='{trace(predit)}' fill='none' stroke='#8b5cf6' stroke-width='1.8'/></svg>")
    leg = ("<div style='font-size:.64rem;color:#496985'><span style='color:#3b82f6'>&#9644;</span> Trafic total &nbsp; "
           "<span style='color:#8b5cf6'>&#9644;</span> Attaques (prédites)</div>")
    return s + leg


def matrice_html(cm, classes):
    """Matrice de confusion normalisée par ligne (= rappel)."""
    somme = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, somme, out=np.zeros(cm.shape, dtype=float), where=somme != 0)
    out = '<div style="overflow-x:auto"><table class="hm"><tr><th></th>'
    out += "".join(f'<th class="v">{court(c)}</th>' for c in classes) + "</tr>"
    for i, c in enumerate(classes):
        out += f'<tr><th class="l">{court(c)}</th>'
        for j in range(len(classes)):
            v = norm[i, j]
            out += f'<td style="background:rgba(59,130,246,{0.05 + 0.9 * v:.2f});color:{"#fff" if v > .4 else "#6a82b3"}">{v:.2f}</td>'
        out += "</tr>"
    return out + "</table></div>"


def badge(classe):
    return '<span class="badge-n">Normal</span>' if classe == "Normal" else '<span class="badge-a">Attaque</span>'


def heure(txt):
    return str(txt)[:16].replace("T", " ")


# ----------------------------------------------------------------------------
# 5. TABLEAU DE BORD (une seule grille dense)
# ----------------------------------------------------------------------------
def corps_chronologie(res):
    inc = res["incidents"]
    inc = inc[inc["Classe prédite"] != "Normal"]
    if "Horodatage" in inc.columns:
        inc = inc.sort_values("Horodatage", ascending=False)
    classes = [c for c in res["classes"] if c != "Normal"]
    out = '<div class="tl">'
    for _, r in inc.head(6).iterrows():
        i = classes.index(r["Classe prédite"]) if r["Classe prédite"] in classes else 0
        p = pct(r["Probabilité"], 1) if pd.notna(r["Probabilité"]) else "n/d"
        out += (f'<div class="tli"><span class="dot" style="background:{COULEURS[i % len(COULEURS)]}"></span><div>'
                f'<b>{court(r["Classe prédite"])}</b><div class="sub">{r.get("IP source", "")} &rarr; {r.get("IP destination", "")} · {p}</div></div>'
                f'<span class="tm">{heure(r.get("Horodatage", ""))}</span></div>')
    return out + "</div>"


def corps_activites(res):
    inc = res["incidents"]
    if "Horodatage" in inc.columns:
        inc = inc.sort_values("Horodatage", ascending=False)
    out = ('<table class="tb"><tr><th>#</th><th>Horodatage</th><th>Adresse source</th><th>Adresse destination</th>'
           '<th>Classe prédite</th><th>Classe réelle</th><th>Probabilité</th><th>Résultat</th></tr>')
    for n, (_, r) in enumerate(inc.head(8).iterrows(), 1):
        p = pct(r["Probabilité"], 1) if pd.notna(r["Probabilité"]) else "n/d"
        juste = r["Classe réelle"] == r["Classe prédite"]
        coul = "#168565" if juste else "#c74357"
        out += (f'<tr><td>{n}</td><td>{heure(r.get("Horodatage", ""))}</td><td>{r.get("IP source", "")}</td>'
                f'<td>{r.get("IP destination", "")}</td><td style="color:{coul}">{court(r["Classe prédite"])}</td>'
                f'<td>{court(r["Classe réelle"])}</td><td>{p}</td><td>{badge(r["Classe prédite"])}</td></tr>')
    return out + "</table>"


def corps_modeles(res):
    out = '<table class="tb"><tr><th>Modèle</th><th>Accuracy</th><th>F1 macro</th><th>Durée</th></tr>'
    for _, r in res["tableau"].iterrows():
        cls = ' class="best"' if r["Modèle"] == res["meilleur"] else ""
        out += (f'<tr{cls}><td>{r["Modèle"]}</td><td>{pct(r["Accuracy"])}</td><td>{pct(r["F1-score (macro)"])}</td>'
                f'<td>{r["Durée (s)"]} s</td></tr>')
    return out + "</table><div class='ks' style='margin-top:3px'>Validation · ligne en bleu = meilleur modèle</div>"


def corps_donnees(res):
    i, (tr, va, te) = res["info"], res["tailles"]
    lignes = [("Lignes brutes", nombre(i["lignes_brutes"])), ("Doublons retirés", nombre(i["doublons"])),
              ("Valeurs manquantes", nombre(i["nan_total"])), ("Variables", nombre(len(res["colonnes"]))),
              ("Train / Val / Test", f"{nombre(tr)} / {nombre(va)} / {nombre(te)}")]
    return "".join(f'<div style="display:flex;justify-content:space-between;font-size:.68rem;margin:2px 0">'
                   f'<span style="color:#66809b">{a}</span><b>{b}</b></div>' for a, b in lignes)


def dashboard_html(res):
    df, mt, rap = res["df"], res["metriques_test"], res["rapport"]
    att = df[df["attack_type"] != "Normal"]["attack_type"].value_counts()
    inc = res["incidents"]
    n_pred = int((inc["Classe prédite"] != "Normal").sum())
    g = '<div id="top"></div><div class="grid">'
    # Ligne 1 : 4 indicateurs
    g += kpi(3, "#2878c7", "Trafic analysé", nombre(len(df)), "lignes nettoyées", "&#9737;")
    g += kpi(3, "#e5566a", "Attaques prédites", nombre(n_pred), f"sur {nombre(len(inc))} lignes de test", "&#9888;")
    g += kpi(3, "#8b5cf6", "Taux d'attaque", pct(att.sum() / len(df)), "du trafic", "&#9678;")
    g += kpi(3, "#14b8a6", "Précision du modèle", pct(mt["Précision (macro)"]), res["meilleur"], "&#10022;")
    # Ligne 2 : tendance du trafic + performances du modèle
    if "Horodatage" in inc.columns:
        h = pd.to_datetime(inc["Horodatage"], errors="coerce").dt.hour
        ok = h.notna()
        total = np.bincount(h[ok].astype(int), minlength=24)
        pred = np.bincount(h[ok & (inc["Classe prédite"] != "Normal")].astype(int), minlength=24)
        g += cellule(7, "Tendances du trafic réseau (24 h, jeu de test)", courbes(total, pred))
    else:
        g += cellule(7, "Tendances du trafic réseau", "<div class='ks'>timestamp_utc absent</div>")
    auc = mt["AUC (OvR)"]
    items = [("#3b82f6", "Accuracy", pct(mt["Accuracy"])), ("#8b5cf6", "Rappel (macro)", pct(mt["Recall (macro)"])),
             ("#14b8a6", "Spécificité", pct(mt["Spécificité (macro)"])), ("#22d3ee", "F1-score", pct(mt["F1-score (macro)"])),
             ("#6366f1", "AUC (ROC)", pct(auc) if auc == auc else "n/d")]
    liste = "".join(f'<div class="lst"><span><span class="dotc" style="background:{c}"></span>{a}</span><b>{b}</b></div>'
                    for c, a, b in items)
    g += cellule(5, "Performances du modèle (test)",
                 f'<div style="display:flex;align-items:center;gap:14px;flex:1">{anneau(mt["Précision (macro)"], "Précision")}'
                 f'<div style="flex:1;min-width:0">{liste}</div></div>')
    # Ligne 3 : fréquence par classe + répartition
    g += cellule(6, "Fréquence des attaques par classe", barres(att.to_dict(), numeroter=True), ident="analyses")
    g += cellule(6, f"Répartition des attaques ({len(att)} classes)", donut(att.to_dict(), nombre(att.sum()), "attaques"))
    # Ligne 4 : incidents récents
    g += cellule(12, "Incidents récents (prédictions)", corps_activites(res), ident="incidents")
    # Ligne 5 : analyses complémentaires
    rec = {c: rap[c]["recall"] for c in res["classes"] if c != "Normal" and c in rap}
    g += cellule(4, "Rappel par type d'attaque", radar(rec))
    g += cellule(4, "Comparaison des modèles", corps_modeles(res))
    g += cellule(4, "Alertes récentes (test)", corps_chronologie(res))
    # Ligne 6 : matrice, importance + données
    g += cellule(8, "Matrice de confusion (rappel par ligne)", matrice_html(res["cm"], res["classes"]), ident="evaluation")
    if res["importance"] is not None:
        imp = barres(res["importance"].head(6).to_dict(), fmt=lambda v: pct(v, 1))
        titre_imp = f"Variables clés · {res['importance_modele']}"
    else:
        imp, titre_imp = "<div class='ks'>Random Forest ou XGBoost requis</div>", "Variables clés"
    g += ('<div style="grid-column:span 4;display:flex;flex-direction:column;gap:8px">'
          f'<div class="c" style="flex:1"><div class="ct">{titre_imp}</div>{imp}</div>'
          f'<div class="c" id="donnees"><div class="ct">Données</div>{corps_donnees(res)}</div></div>')
    html(g + "</div>")


def bloc_parametres(res):
    html('<div id="parametres"></div>')
    with st.expander("Paramètres, données et entraînement"):
        modeles = st.multiselect("Modèles à entraîner", MODELES_DISPO, default=st.session_state["p_modeles"], key="w_modeles")
        if "SVM" in modeles or "KNN" in modeles:
            st.warning("SVM et KNN peuvent être lents sur de grands jeux de données.")
        n = st.slider("Lignes max. par classe (0 = tout)", 0, 20000, int(st.session_state["p_n"]), step=500, key="w_n")
        pond = st.checkbox("Pondérer les classes", value=st.session_state["p_pond"], key="w_pond")
        if st.button("Appliquer et réentraîner", type="primary"):
            if not modeles:
                st.error("Choisissez au moins un modèle.")
            else:
                st.session_state.update(p_modeles=modeles, p_n=int(n), p_pond=bool(pond))
                st.session_state.pop("modele_actif", None)
                st.rerun()
        dispo = list(res["modeles"].keys())
        st.selectbox("Modèle utilisé pour les tests", dispo, index=dispo.index(res["meilleur"]), key="modele_actif")
        fichiers = st.file_uploader("Vos CSV (un ou plusieurs)", type="csv", accept_multiple_files=True, key="up_data")
        if fichiers and st.button("Utiliser ces fichiers", type="primary"):
            st.session_state["df_brut"] = lire_fichiers(fichiers)
            st.session_state["source"] = "upload:" + ",".join(f.name for f in fichiers)
            st.rerun()
        if st.session_state["source"] != "dataset.csv" and st.button("Revenir à dataset.csv"):
            st.session_state["df_brut"] = pd.read_csv(DATASET, low_memory=False)
            st.session_state["source"] = "dataset.csv"
            st.rerun()
        st.caption(f"Application de {AUTEUR}.")


# ----------------------------------------------------------------------------
# 6. CONNEXION ET EN-TÊTE
# ----------------------------------------------------------------------------
def page_login():
    _, c, _ = st.columns([1, 1.1, 1])
    with c:
        html(f'<div class="login-t"><img src="{LOGO_URI}" style="width:100%;max-width:400px;border-radius:12px"></div>')
        with st.container(border=True):
            st.markdown("##### Connexion")
            with st.form("login"):
                user = st.text_input("Nom d'utilisateur")
                mdp = st.text_input("Mot de passe", type="password")
                ok = st.form_submit_button("Se connecter", type="primary")
            if ok:
                bon_user = hmac.compare_digest(user.strip().lower().encode(), IDENTIFIANT.encode())
                bon_mdp = hmac.compare_digest(mdp.encode(), MOT_DE_PASSE.encode())
                if bon_user and bon_mdp:
                    st.session_state["auth"] = True
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")


def entete():
    initiales = "".join(p[0] for p in AUTEUR.split()[:2]).upper()
    c1, c2 = st.columns([9, 1])
    with c1:
        html(f'<div class="topbar"><img class="logo" src="{LOGO_URI}" alt="Attack Predict"><div class="sep"></div>'
             f'<div class="ttl">{SHIELD}<div><div class="t1">Prédiction des attaques classiques en cyber</div>'
             '<div class="t2">Analyse du trafic · Prédiction · Évaluation · Test de scénarios</div></div></div>'
             f'<div class="user"><div class="av">{initiales}</div><b>{AUTEUR}</b></div></div>')
    with c2:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("Quitter"):
            st.session_state.clear()
            st.rerun()


# ----------------------------------------------------------------------------
# 7. PARTIE DE TEST : curseurs en ligne + boutons de test des attaques
# ----------------------------------------------------------------------------
def plage(res, col, genre):
    lo, hi = res["plages"][col]
    if genre == "int":
        lo, hi = int(np.floor(lo)), int(np.ceil(hi))
        return (lo, hi if hi > lo else lo + 1)
    return (lo, hi if hi > lo else lo + 1.0)


def appliquer_ligne(res, ligne):
    """Copie une ligne (dict) dans les curseurs, ramenée dans la plage de chaque curseur."""
    st.session_state["t_base"] = dict(ligne)
    for col, _, genre in CHAMPS:
        if col in res["plages"] and col in ligne:
            lo, hi = plage(res, col, genre)
            v = min(max(float(ligne[col]), lo), hi)
            st.session_state["t_" + col] = int(round(v)) if genre == "int" else float(v)
    for col in ("protocol", "service"):
        if col in ligne and col in res["options"]:
            st.session_state["t_" + col] = str(ligne[col])


def charger_exemple(res):
    sub = res["df_test"][res["df_test"]["attack_type"] == st.session_state["t_classe_ex"]]
    if len(sub):
        appliquer_ligne(res, sub.sample(1).iloc[0].to_dict())


def reinitialiser(res):
    appliquer_ligne(res, res["base"])


def modele_actif(res):
    nom = st.session_state.get("modele_actif")
    return nom if nom in res["modeles"] else res["meilleur"]


def ligne_courante():
    ligne = dict(st.session_state["t_base"])
    for col, _, _ in CHAMPS:
        if "t_" + col in st.session_state:
            ligne[col] = st.session_state["t_" + col]
    for col in ("protocol", "service"):
        if "t_" + col in st.session_state:
            ligne[col] = st.session_state["t_" + col]
    return ligne


def panneau_test(res):
    if st.session_state.get("t_cle") != res["cle"]:
        appliquer_ligne(res, res["base"])
        st.session_state["t_cle"] = res["cle"]
        st.session_state.pop("r_test", None)
        st.session_state.pop("r_global", None)
    nom = modele_actif(res)
    html('<div id="test"></div>')
    st.markdown("<div class='ct'>Partie de test</div><div class='ks' style='margin-bottom:4px'>"
                "Modifiez les paramètres du trafic et testez la prédiction.</div>", unsafe_allow_html=True)
    e1, e2 = st.columns([2, 1])
    e1.selectbox("Exemple réel du jeu de test", res["classes"], key="t_classe_ex", format_func=joli)
    e2.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
    e2.button("Charger", on_click=charger_exemple, args=(res,))
    s1, s2 = st.columns(2)
    if "protocol" in res["options"]:
        s1.selectbox("Protocole", res["options"]["protocol"], key="t_protocol")
    if "service" in res["options"]:
        s2.selectbox("Service", res["options"]["service"], key="t_service")
    colonnes = st.columns(2)
    visibles = [c for c in CHAMPS if c[0] in res["plages"]]
    for i, (col, lib, genre) in enumerate(visibles):
        with colonnes[i % 2]:
            lo, hi = plage(res, col, genre)
            if genre == "int":
                st.slider(lib, lo, hi, step=1, key="t_" + col)
            else:
                st.slider(lib, lo, hi, step=(hi - lo) / 500, format="%.2f", key="t_" + col)
    lancer = st.button("Prédire", type="primary", key="b_test")
    b2, b3 = st.columns(2)
    glob = b2.button("Test global", key="b_global")
    b3.button("Réinitialiser", on_click=reinitialiser, args=(res,))
    if lancer:
        pred, proba = predire(res, nom, pd.DataFrame([ligne_courante()]))
        top = []
        if proba is not None:
            top = [(res["classes"][i], float(proba[0][i])) for i in np.argsort(proba[0])[::-1][:3]]
        st.session_state["r_test"] = {"cle": res["cle"], "classe": res["classes"][int(pred[0])], "modele": nom,
                                      "conf": float(proba[0].max()) if proba is not None else None, "top": top}
    if glob:
        morceaux = []
        for c in res["classes"]:
            sub = res["df_test"][res["df_test"]["attack_type"] == c]
            if len(sub):
                morceaux.append(sub.sample(1))
        d = pd.concat(morceaux)
        pred, proba = predire(res, nom, d)
        st.session_state["r_global"] = {
            "cle": res["cle"], "modele": nom,
            "lignes": [(d["attack_type"].iloc[i], res["classes"][int(pred[i])],
                        float(proba[i].max()) if proba is not None else None) for i in range(len(d))]}


def resultats_test(res):
    r = st.session_state.get("r_test")
    if r and r["cle"] == res["cle"]:
        att = r["classe"] != "Normal"
        conf = pct(r["conf"]) if r["conf"] is not None else "n/d"
        top = barres({k: v for k, v in r["top"]}, fmt=lambda v: pct(v, 1)) if r["top"] else ""
        html(f'<div class="c" style="margin-top:2px"><div class="ct">Diagnostic · {r["modele"]}</div>'
             f'<div class="res {"att" if att else "ok"}"><div style="flex:1"><div class="rt">Résultat</div>'
             f'<div class="rv" style="color:{ROUGE_DOUX if att else VERT_DOUX}">{"Attaque" if att else "Normal"}</div></div>'
             f'<div style="text-align:right"><div class="rt">Confiance</div><div class="rv" style="color:#183b5c">{conf}</div>'
             f'<div style="height:5px;border-radius:4px;background:#e3edf8;margin-top:3px;min-width:90px"><div style="width:'
             f'{(r["conf"] or 0) * 100:.1f}%;height:100%;border-radius:4px;background:{ROUGE_DOUX if att else VERT_DOUX}"></div></div></div></div>'
             f'<div style="margin:6px 0 4px">Classe prédite <span class="chip">{joli(r["classe"])}</span></div>{top}</div>')
    else:
        html('<div class="c" style="margin-top:2px"><div class="ct">Diagnostic</div>'
             '<div class="ks">Réglez les curseurs puis « Lancer le test ».</div></div>')
    g = st.session_state.get("r_global")
    if g and g["cle"] == res["cle"]:
        bons = sum(1 for a, b, _ in g["lignes"] if a == b)
        t = '<table class="tb"><tr><th>Réelle</th><th>Prédite</th><th>Proba</th><th>Résultat</th></tr>'
        for a, b, p in g["lignes"]:
            ok = a == b
            t += (f'<tr><td>{court(a)}</td><td style="color:{"#168565" if ok else "#c74357"}">{court(b)}</td>'
                  f'<td>{pct(p, 1) if p is not None else "n/d"}</td><td>{"OK" if ok else "Erreur"}</td></tr>')
        html(f'<div class="c" style="margin-top:2px"><div class="ct">Test global · {bons}/{len(g["lignes"])} corrects</div>'
             f'{t}</table><div class="ks" style="margin-top:3px">Un exemple réel tiré au hasard par classe (jeu de test).</div></div>')


# ----------------------------------------------------------------------------
# 8. PROGRAMME PRINCIPAL
# ----------------------------------------------------------------------------
html(CSS)
if not st.session_state.get("auth"):
    page_login()
    st.stop()

st.session_state.setdefault("p_modeles", MODELES_DEFAUT)
st.session_state.setdefault("p_n", 0)
st.session_state.setdefault("p_pond", False)
st.session_state.setdefault("source", "dataset.csv")
if "df_brut" not in st.session_state:
    if not os.path.exists(DATASET):
        st.error("dataset.csv est introuvable à côté de app.py.")
        st.stop()
    st.session_state["df_brut"] = pd.read_csv(DATASET, low_memory=False)

entete()

df_brut = st.session_state["df_brut"]
if "attack_type" not in df_brut.columns:
    st.error("La colonne 'attack_type' est absente du CSV.")
    st.stop()
try:
    with st.spinner("Entraînement des modèles en cours (première ouverture)..."):
        res = entrainer(f"{st.session_state['source']}|{len(df_brut)}", tuple(st.session_state["p_modeles"]),
                        st.session_state["p_n"], st.session_state["p_pond"], df_brut)
except ValueError as e:
    st.error(str(e))
    st.stop()

nav, centre, droite = st.columns([0.5, 2.6, 1.2], gap="small")
with nav:
    html(sidebar_html())
with centre:
    dashboard_html(res)
    bloc_parametres(res)
with droite:
    with st.container(border=True):
        panneau_test(res)
    resultats_test(res)
