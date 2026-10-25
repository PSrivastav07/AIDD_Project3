"""Streamlit dashboard for the production logS predictor.

Paste SMILES (one per line, optionally followed by a name) or upload a CSV, choose the interval level, and get
predicted logS with conformal intervals, solubility class, applicability-domain (AD) flags, the nearest
training compound, molecule depictions, a histogram of predictions and a CSV download. The sidebar shows the
model card (``models/final/metadata.json``).

Run (from the project root)::

    .venv/Scripts/python.exe -m streamlit run app/dashboard.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

from solpred.predict import DEFAULT_MODEL_DIR, SolubilityPredictor  # noqa: E402

MAX_ROWS = 1000
MAX_DEPICT = 24
EXAMPLE = "\n".join([
    "CC(=O)Oc1ccccc1C(=O)O aspirin",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C caffeine",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1 ibuprofen",
    "NC(=O)N1c2ccccc2C=Cc2ccccc21 carbamazepine",
    "O=C([O-])c1ccccc1.[Na+] sodium_benzoate",
    "N[Pt](N)(Cl)Cl cisplatin",
    "not_a_smiles typo",
])
CLASS_COLORS = {"highly_soluble": "#c6e9c6", "soluble": "#e3f4d7", "slightly_soluble": "#fdf0c4", "insoluble": "#f8d0c8"}
CLASS_BOUNDS = [(-4.0, "insoluble | slightly"), (-2.0, "slightly | soluble"), (0.0, "soluble | highly")]


@st.cache_resource(show_spinner="Loading model ...")
def load_predictor(model_dir: str) -> SolubilityPredictor:
    return SolubilityPredictor.load(model_dir)


def parse_text(text: str) -> pd.DataFrame:
    """One record per non-empty line: first whitespace-separated token = SMILES, the rest = name."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if parts:
            rows.append({"name": parts[1] if len(parts) > 1 else "", "smiles": parts[0]})
    return pd.DataFrame(rows, columns=["name", "smiles"])


def model_card(pred: SolubilityPredictor) -> None:
    m = pred.metadata
    st.sidebar.header("Model card")
    if not m:
        st.sidebar.info("metadata.json not found")
        return
    cv = m.get("cv", {}).get("metrics", {}).get("overall", {})
    cf = m.get("conformal", {}).get("alpha_0.1", {})
    st.sidebar.markdown(
        f"**{m['model']['name']} x {m['model']['n_members']} bagged members** on `{m['features']['kind']}` "
        f"({m['features']['n_features']} features)\n\n"
        f"Training set: curated AqSolDB, **{m['dataset']['n_train']:,}** compounds "
        f"(dataset v{m['dataset'].get('dataset_version')}, sha256 `{str(m['dataset'].get('sha256'))[:12]}...`)")
    if cv:
        st.sidebar.markdown("**Scaffold-grouped 5-fold CV (out-of-fold)**")
        c1, c2, c3 = st.sidebar.columns(3)
        c1.metric("MAE", f"{cv['MAE']:.2f}")
        c2.metric("RMSE", f"{cv['RMSE']:.2f}")
        c3.metric("R2", f"{cv['R2']:.2f}")
    if cf:
        lofo = cf.get("normalized_leave_one_fold_out", {})
        st.sidebar.markdown(
            f"**Conformal 90% interval**: coverage {cf['normalized_in_sample']['coverage']:.3f} (OOF), "
            f"{lofo.get('mean', float('nan')):.3f} mean leave-one-fold-out; mean width "
            f"{cf['normalized_in_sample']['mean_width']:.2f} log units")
    ad = m.get("applicability_domain", {})
    st.sidebar.markdown(f"**Applicability domain**: {ad.get('metric')} >= {ad.get('sim_threshold')} "
                        f"({ad.get('fingerprint')})")
    st.sidebar.caption(f"Training logS range: {m['training_logS_range'][0]:.2f} to {m['training_logS_range'][1]:.2f}. "
                       f"Created {m.get('created_utc')}. Classes: >=0 highly soluble, [-2,0) soluble, "
                       "[-4,-2) slightly soluble, <-4 insoluble (log mol/L).")
    with st.sidebar.expander("Full metadata"):
        st.json(m, expanded=False)


def style_table(df: pd.DataFrame):
    def ad_color(v):
        if v is True:
            return "background-color: #c6e9c6"
        if v is False:
            return "background-color: #f8d0c8"
        return ""

    def class_color(v):
        return f"background-color: {CLASS_COLORS[v]}" if v in CLASS_COLORS else ""

    def warn_color(v):
        return "color: #b03a2e" if isinstance(v, str) and v and not v.startswith("in training set") else ""

    return (df.style.map(ad_color, subset=["ad_in_domain"]).map(class_color, subset=["solubility_class"])
            .map(warn_color, subset=["warnings"])
            .format({c: "{:.2f}" for c in ["logS_pred", "logS_std", "pi_low", "pi_high", "ad_maxsim",
                                           "nearest_train_logS"]}, na_rep="-")
            .format({"solubility_mg_per_L": "{:.3g}"}, na_rep="-"))


def histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    v = df["logS_pred"].dropna()
    ax.hist(v, bins=min(30, max(5, len(v))), color="#4C72B0", edgecolor="white")
    for x, _ in CLASS_BOUNDS:
        ax.axvline(x, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("predicted logS (log mol/L)")
    ax.set_ylabel("number of compounds")
    ax.set_title("Distribution of predictions (dashed: class boundaries)")
    fig.tight_layout()
    return fig


def depictions(df: pd.DataFrame):
    ok = df[df["valid"]].head(MAX_DEPICT)
    mols = [Chem.MolFromSmiles(s) for s in ok["standardized_smiles"]]
    legends = [f"{(n or '')[:18]} {p:.2f} [{lo:.1f},{hi:.1f}]{' OOD' if ad is False else ''}"
               for n, p, lo, hi, ad in zip(ok.get("name", [""] * len(ok)), ok["logS_pred"], ok["pi_low"],
                                           ok["pi_high"], ok["ad_in_domain"])]
    return Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 200), legends=legends) if mols else None


def main() -> None:
    st.set_page_config(page_title="solpred - aqueous solubility", layout="wide")
    st.title("Aqueous solubility (logS) predictor")
    st.caption("Bagged LightGBM trained on curated AqSolDB; conformal prediction intervals calibrated on "
               "scaffold-grouped cross-validation; ECFP4 applicability domain. Research tool - not a substitute "
               "for measurement.")
    model_dir = os.environ.get("SOLPRED_MODEL_DIR", DEFAULT_MODEL_DIR)
    try:
        pred = load_predictor(model_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    model_card(pred)
    level = st.sidebar.select_slider("Prediction-interval level", options=[0.5, 0.68, 0.8, 0.9, 0.95], value=0.9)

    tab_paste, tab_csv = st.tabs(["Paste SMILES", "Upload CSV"])
    with tab_paste:
        text = st.text_area("One SMILES per line, optionally followed by a name", value=EXAMPLE, height=180)
        run_text = st.button("Predict pasted SMILES", type="primary")
    with tab_csv:
        up = st.file_uploader("CSV with a SMILES column", type=["csv"])
        run_csv = False
        if up is not None:
            frame = pd.read_csv(up, dtype=str, keep_default_na=False)
            cols = list(frame.columns)
            scol = st.selectbox("SMILES column", cols, index=cols.index("smiles") if "smiles" in cols else 0)
            ncol = st.selectbox("Name column (optional)", ["(none)"] + cols,
                                index=1 + cols.index("name") if "name" in cols else 0)
            run_csv = st.button("Predict uploaded file", type="primary")

    if run_text:
        st.session_state["inputs"] = parse_text(text)
    if run_csv:
        st.session_state["inputs"] = pd.DataFrame({"name": frame[ncol] if ncol != "(none)" else "",
                                                   "smiles": frame[scol]})
    inputs = st.session_state.get("inputs")
    if inputs is None or inputs.empty:
        st.info("Paste SMILES or upload a CSV, then press Predict.")
        return
    if len(inputs) > MAX_ROWS:
        st.warning(f"Only the first {MAX_ROWS} rows are predicted.")
        inputs = inputs.head(MAX_ROWS)

    with st.spinner(f"Predicting {len(inputs)} compounds ..."):
        res = pred.predict(inputs["smiles"].tolist(), alpha=round(1 - level, 4))
    res.insert(0, "name", inputs["name"].to_numpy())

    n_valid = int(res["valid"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("compounds", len(res))
    c2.metric("predicted", n_valid)
    c3.metric("outside AD", int((res["ad_in_domain"] == False).sum()))  # noqa: E712 - nullable boolean
    c4.metric("with warnings", int((res["warnings"] != "").sum()))

    st.subheader(f"Results ({level:.0%} prediction intervals)")
    show = ["name", "input_smiles", "logS_pred", "pi_low", "pi_high", "logS_std", "solubility_class",
            "solubility_mg_per_L", "ad_maxsim", "ad_in_domain", "nearest_train_id", "nearest_train_logS",
            "warnings"]
    st.dataframe(style_table(res[show]), width="stretch", hide_index=True)
    st.download_button("Download full results (CSV)", res.to_csv(index=False).encode(), "solpred_predictions.csv",
                       "text/csv")

    left, right = st.columns([1, 1])
    with left:
        if n_valid:
            st.pyplot(histogram(res), clear_figure=True)
    with right:
        img = depictions(res)
        if img is not None:
            st.image(img, caption=f"First {min(n_valid, MAX_DEPICT)} valid structures: name, predicted logS "
                                  f"[{level:.0%} interval], OOD = outside applicability domain")


main()
