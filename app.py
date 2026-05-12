import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(page_title="Oil Radar 30'", page_icon="🛢️", layout="wide")

REFRESH_MINUTES = 30

YAHOO_TICKERS = {
    "Brent": "BZ=F",
    "WTI": "CL=F",
    "Dollar Index": "DX-Y.NYB",
    "VIX": "^VIX",
}


@st.cache_data(ttl=60 * REFRESH_MINUTES)
def load_market_data():
    frames = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for name, ticker in YAHOO_TICKERS.items():
        try:
            symbol = ticker.replace("^", "%5E").replace("=", "%3D")
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=3mo&interval=1d"

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]

            df = pd.DataFrame({
                "Date": pd.to_datetime(timestamps, unit="s"),
                "Close": closes,
            })

            df = df.dropna()
            df = df.set_index("Date")

            frames[name] = df

        except Exception as e:
            st.warning(f"{name}: erreur de récupération - {e}")
            frames[name] = pd.DataFrame()

    return frames


def pct_change(series, periods):
    try:
        if len(series) <= periods:
            return np.nan
        return (series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100
    except Exception:
        return np.nan


def market_snapshot(frames):
    rows = []

    for name in YAHOO_TICKERS.keys():
        df = frames.get(name, pd.DataFrame())

        if df.empty or "Close" not in df.columns:
            rows.append({
                "Actif": name,
                "Prix": np.nan,
                "1 jour": np.nan,
                "5 jours": np.nan,
                "20 jours": np.nan,
                "Signal": "⚪ N/A",
            })
            continue

        close = df["Close"].dropna()
        last = close.iloc[-1]

        p1 = pct_change(close, 1)
        p5 = pct_change(close, 5)
        p20 = pct_change(close, 20)

        signal = "🟢" if pd.notna(p5) and p5 > 1 else "🔴" if pd.notna(p5) and p5 < -1 else "🟠"

        rows.append({
            "Actif": name,
            "Prix": last,
            "1 jour": p1,
            "5 jours": p5,
            "20 jours": p20,
            "Signal": signal,
        })

    return pd.DataFrame(rows)


def technical_score(snapshot):
    score = 0.0
    factors = []

    try:
        brent = float(snapshot.loc[snapshot["Actif"] == "Brent", "5 jours"].iloc[0])
        wti = float(snapshot.loc[snapshot["Actif"] == "WTI", "5 jours"].iloc[0])
        dollar = float(snapshot.loc[snapshot["Actif"] == "Dollar Index", "5 jours"].iloc[0])
        vix = float(snapshot.loc[snapshot["Actif"] == "VIX", "5 jours"].iloc[0])
    except Exception:
        return 0, []

    oil_momentum = np.nanmean([brent, wti])

    if not math.isnan(oil_momentum):
        s = max(min(oil_momentum * 1.2, 3), -3)
        score += s
        factors.append(("Momentum pétrole 5 jours", round(s, 1)))

    if not math.isnan(dollar):
        s = max(min(-dollar * 1.2, 2), -2)
        score += s
        factors.append(("Dollar Index", round(s, 1)))

    if not math.isnan(vix):
        s = max(min(vix * 0.25, 1.5), -1.5)
        score += s
        factors.append(("Stress marché / VIX", round(s, 1)))

    return round(score, 1), factors


def label_from_score(score):
    if score >= 6:
        return "🟢 BIAIS HAUSSIER FORT"
    if score >= 2:
        return "🟢 BIAIS HAUSSIER"
    if score <= -6:
        return "🔴 BIAIS BAISSIER FORT"
    if score <= -2:
        return "🔴 BIAIS BAISSIER"
    return "🟠 NEUTRE / ATTENTE"


def confidence_from_score(score):
    base = min(abs(score) / 10, 1) * 60 + 25
    return int(max(20, min(90, base)))


def ai_summary(score, top_factors):
    direction = "haussier" if score > 1 else "baissier" if score < -1 else "neutre"
    first = top_factors[0][0] if top_factors else "l'absence de signal dominant"
    return (
        f"Le marché pétrole présente un biais {direction}. "
        f"Le facteur dominant détecté est {first}. "
        f"À confirmer avec les prochaines données macro, stocks et événements géopolitiques."
    )


st.title("🛢️ Oil Radar 30’")
st.caption("Cockpit pétrole en une page — données indicatives, actualisation logique toutes les 30 minutes")

with st.sidebar:
    st.header("Paramètres")
    st.write("La V1 utilise Yahoo Finance direct pour les prix. Les news sont temporairement désactivées.")
    st.button("Rafraîchir maintenant", on_click=st.cache_data.clear)

with st.spinner("Chargement des données marché..."):
    frames = load_market_data()

snapshot = market_snapshot(frames)

if all(df.empty for df in frames.values()):
    st.error("Impossible de récupérer les données marché depuis Yahoo Finance.")

tech_score, tech_factors = technical_score(snapshot)
score = round(max(min(tech_score, 10), -10), 1)
label = label_from_score(score)
confidence = confidence_from_score(score)

c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.6])

for col, name in zip([c1, c2, c3, c4], ["Brent", "WTI", "Dollar Index", "VIX"]):
    asset_rows = snapshot[snapshot["Actif"] == name]

    if asset_rows.empty:
        col.metric(name, "N/A", "N/A")
        continue

    row = asset_rows.iloc[0]
    price = row["Prix"]
    delta = row["1 jour"]

    col.metric(
        name,
        f"{price:,.2f}" if pd.notna(price) else "N/A",
        f"{delta:+.2f}%" if pd.notna(delta) else "N/A",
    )

c5.metric("Oil Bias Score", f"{score:+.1f}/10", label)

st.progress(confidence / 100, text=f"Niveau de confiance indicatif : {confidence}%")

left, center, right = st.columns([1.15, 1.15, 1])

with left:
    st.subheader("📈 Marché")

    show = snapshot.copy()
    for col in ["Prix", "1 jour", "5 jours", "20 jours"]:
        show[col] = show[col].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) and col == "Prix"
            else f"{x:+.2f}%" if pd.notna(x)
            else "N/A"
        )

    st.dataframe(show, use_container_width=True, hide_index=True)

    if "Brent" in frames and not frames["Brent"].empty:
        df = frames["Brent"].tail(90).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Brent"))
        fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title=None,
            yaxis_title="Brent",
        )
        st.plotly_chart(fig, use_container_width=True)

with center:
    st.subheader("🧠 Facteurs de score")

    factor_rows = [{"Facteur": k, "Score": v} for k, v in tech_factors]
    factors_df = pd.DataFrame(factor_rows)

    if not factors_df.empty:
        factors_df = factors_df.sort_values("Score", key=lambda s: s.abs(), ascending=False)
        st.dataframe(factors_df, use_container_width=True, hide_index=True)
    else:
        st.info("Pas assez de données pour scorer.")

    st.subheader("🧾 Synthèse")
    top_factors = [(r["Facteur"], r["Score"]) for _, r in factors_df.head(5).iterrows()] if not factors_df.empty else []
    st.success(ai_summary(score, top_factors))

with right:
    st.subheader("⚡ Top impact news")
    st.warning("News temporairement désactivées dans cette V1 stable.")

st.subheader("🕒 Next risk events")

events = pd.DataFrame([
    {"Quand": "Mercredi 16:30 Paris", "Événement": "Stocks EIA US", "Risque": "Très fort sur WTI / Brent"},
    {"Quand": "Mardi soir / nuit", "Événement": "API Weekly Statistical Bulletin", "Risque": "Signal préliminaire stocks US"},
    {"Quand": "Selon calendrier", "Événement": "OPEP / OPEP+", "Risque": "Production / quotas / discipline"},
    {"Quand": "Mensuel", "Événement": "CPI US / Fed / Dollar", "Risque": "Dollar et appétit pour le risque"},
    {"Quand": "Mensuel", "Événement": "PMI Chine", "Risque": "Demande mondiale"},
])

st.dataframe(events, use_container_width=True, hide_index=True)

st.caption(
    f"Dernière génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
    "— Données indicatives, possiblement différées."
)
