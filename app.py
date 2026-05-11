import os
import re
import math
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Oil Radar 30'", page_icon="🛢️", layout="wide")

REFRESH_MINUTES = 30
TICKERS = {
    "Brent": "BZ=F",
    "WTI": "CL=F",
    "Dollar Index": "DX=F",
    "VIX": "^VIX",
}

NEWS_THEMES = {
    "Hormuz / Iran / Middle East": {
        "query": '(oil OR crude OR brent OR wti) (Iran OR Hormuz OR "Middle East" OR Israel OR Yemen OR Houthi)',
        "direction": 1,
        "weight": 4,
        "keywords": ["iran", "hormuz", "israel", "houthi", "yemen", "middle east", "attack", "missile", "sanction"],
    },
    "OPEC / supply discipline": {
        "query": '(oil OR crude OR brent OR wti) (OPEC OR OPEC+ OR Saudi OR Russia production cut output quota)',
        "direction": 1,
        "weight": 3,
        "keywords": ["opec", "production cut", "quota", "saudi", "russia", "output cut", "supply cut"],
    },
    "Demand weakness / China": {
        "query": '(oil OR crude OR brent OR wti) (China demand slowdown PMI recession weak economy)',
        "direction": -1,
        "weight": 3,
        "keywords": ["china", "demand", "slowdown", "pmi", "recession", "weak", "contraction"],
    },
    "Inventories / EIA / API": {
        "query": '(oil OR crude OR brent OR wti) (EIA API inventories stocks Cushing gasoline distillates)',
        "direction": 0,
        "weight": 3,
        "keywords": ["eia", "api", "inventories", "stocks", "cushing", "gasoline", "distillates"],
    },
    "Macro / Fed / Dollar": {
        "query": '(oil OR crude OR brent OR wti) (Fed dollar rates inflation CPI yields)',
        "direction": 0,
        "weight": 2,
        "keywords": ["fed", "dollar", "rates", "inflation", "cpi", "yields"],
    },
}

@st.cache_data(ttl=60 * REFRESH_MINUTES)
def load_market_data(period="5d", interval="30m"):
    frames = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            frames[name] = df
        except Exception:
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
    for name, df in frames.items():
        if df.empty or "Close" not in df:
            rows.append({"Actif": name, "Prix": np.nan, "30 min": np.nan, "4 h": np.nan, "24 h": np.nan, "Signal": "⚪ N/A"})
            continue
        close = df["Close"].dropna()
        last = close.iloc[-1]
        p30 = pct_change(close, 1)
        p4h = pct_change(close, 8)
        p24h = pct_change(close, 48)
        signal = "🟢" if p4h > 0.4 else "🔴" if p4h < -0.4 else "🟠"
        rows.append({"Actif": name, "Prix": last, "30 min": p30, "4 h": p4h, "24 h": p24h, "Signal": signal})
    return pd.DataFrame(rows)

@st.cache_data(ttl=60 * REFRESH_MINUTES)
def fetch_gdelt_news(query, max_records=20):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=12)
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={quote_plus(query)}"
        "&mode=artlist"
        "&format=json"
        f"&maxrecords={max_records}"
        "&sort=hybridrel"
        f"&startdatetime={start.strftime('%Y%m%d%H%M%S')}"
        f"&enddatetime={end.strftime('%Y%m%d%H%M%S')}"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
    except Exception:
        return pd.DataFrame()

    rows = []
    for a in articles:
        title = a.get("title", "")
        source = a.get("sourceCommonName", "")
        url = a.get("url", "")
        seendate = a.get("seendate", "")
        domain = a.get("domain", "")
        rows.append({"time": seendate, "title": title, "source": source or domain, "url": url})
    return pd.DataFrame(rows).drop_duplicates(subset=["title"])


def keyword_hits(text, keywords):
    low = text.lower()
    return sum(1 for k in keywords if k in low)


def build_news_table():
    all_rows = []
    theme_scores = []
    for theme, cfg in NEWS_THEMES.items():
        df = fetch_gdelt_news(cfg["query"], max_records=12)
        if df.empty:
            theme_scores.append({"Facteur": theme, "Score": 0, "Articles": 0})
            continue
        df["theme"] = theme
        df["hits"] = df["title"].apply(lambda x: keyword_hits(str(x), cfg["keywords"]))
        articles = len(df)
        intensity = min(articles / 6, 1.0)
        keyword_intensity = min(df["hits"].sum() / 6, 1.0)
        raw = cfg["weight"] * (0.55 * intensity + 0.45 * keyword_intensity)
        if cfg["direction"] == 0:
            score = 0
        else:
            score = cfg["direction"] * raw
        theme_scores.append({"Facteur": theme, "Score": round(score, 1), "Articles": articles})
        all_rows.append(df.head(6))
    news = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return pd.DataFrame(theme_scores), news


def technical_score(snapshot):
    score = 0.0
    factors = []
    try:
        brent_4h = float(snapshot.loc[snapshot["Actif"] == "Brent", "4 h"].iloc[0])
        wti_4h = float(snapshot.loc[snapshot["Actif"] == "WTI", "4 h"].iloc[0])
        dollar_4h = float(snapshot.loc[snapshot["Actif"] == "Dollar Index", "4 h"].iloc[0])
        vix_4h = float(snapshot.loc[snapshot["Actif"] == "VIX", "4 h"].iloc[0])
    except Exception:
        return 0, []

    oil_momentum = np.nanmean([brent_4h, wti_4h])
    if not math.isnan(oil_momentum):
        s = max(min(oil_momentum * 1.2, 3), -3)
        score += s
        factors.append(("Momentum pétrole 4 h", round(s, 1)))
    if not math.isnan(dollar_4h):
        s = max(min(-dollar_4h * 1.2, 2), -2)
        score += s
        factors.append(("Dollar Index", round(s, 1)))
    if not math.isnan(vix_4h):
        s = max(min(vix_4h * 0.25, 1.5), -1.5)
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


def confidence_from_score(score, news_count):
    base = min(abs(score) / 10, 1) * 55 + min(news_count / 20, 1) * 25 + 20
    return int(max(20, min(90, base)))


def ai_summary(score, label, top_factors, top_news):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # deterministic local fallback
        direction = "haussier" if score > 1 else "baissier" if score < -1 else "neutre"
        first = top_factors[0][0] if top_factors else "l'absence de signal dominant"
        return f"Le marché pétrole présente un biais {direction}. Le facteur dominant détecté est {first}. À confirmer par le prix et les prochains événements macro/stock."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        news_titles = "\n".join([f"- {x}" for x in top_news[:8]])
        factors = "\n".join([f"- {k}: {v}" for k, v in top_factors])
        prompt = f"""Tu es un analyste pétrole. Résume en français, en 2 phrases maximum, le biais de marché.
Score: {score} / label: {label}
Facteurs:\n{factors}
News:\n{news_titles}
Ne donne pas de conseil d'investissement, seulement une lecture de marché."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Synthèse IA indisponible pour le moment. Le score reste calculé localement."


st.title("🛢️ Oil Radar 30’")
st.caption("Cockpit pétrole en une page — actualisation logique toutes les 30 minutes")

with st.sidebar:
    st.header("Paramètres")
    st.write("La V1 utilise Yahoo Finance/yfinance pour les prix et GDELT pour les news.")
    st.write("Pour activer la synthèse IA, ajoutez `OPENAI_API_KEY` dans `.env`.")
    st.button("Rafraîchir maintenant", on_click=st.cache_data.clear)

frames = load_market_data()
snapshot = market_snapshot(frames)
news_scores, news = build_news_table()
tech_score, tech_factors = technical_score(snapshot)
news_score = round(news_scores["Score"].sum(), 1) if not news_scores.empty else 0
score = round(max(min(tech_score + news_score, 10), -10), 1)
label = label_from_score(score)
confidence = confidence_from_score(score, len(news)) if not news.empty else confidence_from_score(score, 0)

# Top metrics
c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.6])
for col, name in zip([c1, c2, c3, c4], ["Brent", "WTI", "Dollar Index", "VIX"]):
    row = snapshot[snapshot["Actif"] == name].iloc[0]
    price = row["Prix"]
    delta = row["30 min"]
    col.metric(name, f"{price:,.2f}" if pd.notna(price) else "N/A", f"{delta:+.2f}%" if pd.notna(delta) else "N/A")
c5.metric("Oil Bias Score", f"{score:+.1f}/10", label)

st.progress(confidence / 100, text=f"Niveau de confiance indicatif : {confidence}%")

left, center, right = st.columns([1.15, 1.15, 1])

with left:
    st.subheader("📈 Marché")
    show = snapshot.copy()
    for col in ["Prix", "30 min", "4 h", "24 h"]:
        show[col] = show[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and col == "Prix" else f"{x:+.2f}%" if pd.notna(x) else "N/A")
    st.dataframe(show, use_container_width=True, hide_index=True)

    if "Brent" in frames and not frames["Brent"].empty:
        df = frames["Brent"].tail(96).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Brent"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), xaxis_title=None, yaxis_title="Brent")
        st.plotly_chart(fig, use_container_width=True)

with center:
    st.subheader("🧠 Facteurs de score")
    factor_rows = []
    factor_rows.extend([{"Facteur": k, "Score": v} for k, v in tech_factors])
    if not news_scores.empty:
        factor_rows.extend(news_scores[["Facteur", "Score"]].to_dict("records"))
    factors_df = pd.DataFrame(factor_rows)
    if not factors_df.empty:
        factors_df = factors_df.sort_values("Score", key=lambda s: s.abs(), ascending=False)
        st.dataframe(factors_df, use_container_width=True, hide_index=True)
    else:
        st.info("Pas assez de données pour scorer.")

    st.subheader("🧾 Synthèse")
    top_factors = [(r["Facteur"], r["Score"]) for _, r in factors_df.head(5).iterrows()] if not factors_df.empty else []
    top_news_titles = news["title"].head(8).tolist() if not news.empty else []
    st.success(ai_summary(score, label, top_factors, top_news_titles))

with right:
    st.subheader("⚡ Top impact news")
    if news.empty:
        st.warning("Aucune news récupérée. Vérifiez la connexion Internet.")
    else:
        news_view = news.head(8)
        for _, row in news_view.iterrows():
            title = str(row["title"])
            source = str(row.get("source", ""))
            url = str(row.get("url", ""))
            theme = str(row.get("theme", ""))
            st.markdown(f"**{theme}**  \n[{title}]({url})  \n<small>{source}</small>", unsafe_allow_html=True)
            st.divider()

st.subheader("🕒 Next risk events")
events = pd.DataFrame([
    {"Quand": "Mercredi 16:30 Paris", "Événement": "Stocks EIA US", "Risque": "Très fort sur WTI / Brent"},
    {"Quand": "Mardi soir / nuit", "Événement": "API Weekly Statistical Bulletin", "Risque": "Signal préliminaire stocks US"},
    {"Quand": "Selon calendrier", "Événement": "OPEP / OPEP+", "Risque": "Production / quotas / discipline"},
    {"Quand": "Mensuel", "Événement": "CPI US / Fed / Dollar", "Risque": "Dollar et appétit pour le risque"},
    {"Quand": "Mensuel", "Événement": "PMI Chine", "Risque": "Demande mondiale"},
])
st.dataframe(events, use_container_width=True, hide_index=True)

st.caption(f"Dernière génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Données indicatives, possiblement différées.")
