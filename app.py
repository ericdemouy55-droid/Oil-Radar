import math
from datetime import datetime

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Oil Radar 5'",
    page_icon="🛢️",
    layout="wide"
)

REFRESH_MINUTES = 5

st_autorefresh(
    interval=REFRESH_MINUTES * 60 * 1000,
    key="oil_radar_refresh"
)

YAHOO_TICKERS = {
    "Brent": "BZ=F",
    "WTI": "CL=F",
    "Dollar Index": "DX-Y.NYB",
    "VIX": "^VIX",
}

RSS_FEEDS = [
    "https://oilprice.com/rss/main",
    "https://www.eia.gov/rss/todayinenergy.xml",
    "https://www.reutersagency.com/feed/?best-topics=energy&post_type=best",
]

NEWS_RULES = {
    "bullish": {
        "keywords": [
            "iran", "hormuz", "attack", "missile", "houthi", "sanction",
            "supply disruption", "production cut", "opec cut",
            "inventory draw", "stocks fall", "crude draw",
            "refinery outage", "war", "tension"
        ],
        "score": 1.5,
    },
    "bearish": {
        "keywords": [
            "inventory build", "stocks rise", "demand weak", "slowdown",
            "china weak", "recession", "output increase",
            "production increase", "ceasefire", "surplus",
            "dollar strengthens", "rate hike"
        ],
        "score": -1.5,
    },
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=60 * REFRESH_MINUTES)
def load_market_data():
    frames = {}
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for name, ticker in YAHOO_TICKERS.items():
        try:
            symbol = ticker.replace("^", "%5E").replace("=", "%3D")
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{symbol}?range=3mo&interval=1d"
            )

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            result = data.get("chart", {}).get("result", [None])[0]

            if result is None:
                frames[name] = pd.DataFrame()
                continue

            timestamps = result.get("timestamp", [])
            closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])

            if not timestamps or not closes:
                frames[name] = pd.DataFrame()
                continue

            df = pd.DataFrame({
                "Date": pd.to_datetime(timestamps, unit="s"),
                "Close": closes,
            })

            df = df.dropna()
            df = df.set_index("Date")
            frames[name] = df

        except Exception as e:
            st.warning(f"{name}: erreur récupération données marché - {e}")
            frames[name] = pd.DataFrame()

    return frames


@st.cache_data(ttl=60 * REFRESH_MINUTES)
def load_news():
    rows = []

    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)

            for entry in parsed.entries[:10]:
                title = entry.get("title", "")
                link = entry.get("link", "")
                source = feed

                if title:
                    rows.append({
                        "title": title,
                        "link": link,
                        "source": source,
                    })

        except Exception:
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=["title", "link", "source"])

    return df.drop_duplicates(subset=["title"])


# ============================================================
# CALCULS
# ============================================================

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

        if close.empty:
            rows.append({
                "Actif": name,
                "Prix": np.nan,
                "1 jour": np.nan,
                "5 jours": np.nan,
                "20 jours": np.nan,
                "Signal": "⚪ N/A",
            })
            continue

        last = close.iloc[-1]
        p1 = pct_change(close, 1)
        p5 = pct_change(close, 5)
        p20 = pct_change(close, 20)

        if pd.notna(p5) and p5 > 1:
            signal = "🟢"
        elif pd.notna(p5) and p5 < -1:
            signal = "🔴"
        else:
            signal = "🟠"

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
        return 0.0, []

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


def news_sentiment_score(news):
    score = 0.0
    factors = []

    if news.empty:
        return 0.0, []

    for _, row in news.iterrows():
        title = str(row.get("title", "")).lower()

        for keyword in NEWS_RULES["bullish"]["keywords"]:
            if keyword in title:
                score += NEWS_RULES["bullish"]["score"]
                factors.append((
                    f"News haussière : {keyword}",
                    NEWS_RULES["bullish"]["score"]
                ))
                break

        for keyword in NEWS_RULES["bearish"]["keywords"]:
            if keyword in title:
                score += NEWS_RULES["bearish"]["score"]
                factors.append((
                    f"News baissière : {keyword}",
                    NEWS_RULES["bearish"]["score"]
                ))
                break

    score = max(min(score, 4), -4)

    return round(score, 1), factors[:6]


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
    base = (
        min(abs(score) / 10, 1) * 55
        + min(news_count / 20, 1) * 25
        + 20
    )

    return int(max(20, min(90, base)))


def ai_summary(score, top_factors):
    if score > 1:
        direction = "haussier"
    elif score < -1:
        direction = "baissier"
    else:
        direction = "neutre"

    first = top_factors[0][0] if top_factors else "l’absence de signal dominant"

    return (
        f"Le marché pétrole présente actuellement un biais {direction}. "
        f"Le facteur dominant détecté est : {first}. "
        f"Ce signal doit être confirmé avec les prochaines données macro, "
        f"les stocks américains et les événements géopolitiques."
    )


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.title("🛢️ Oil Radar 5’")
st.caption("Cockpit pétrole en une page — refresh automatique toutes les 5 minutes")

with st.sidebar:
    st.header("Paramètres")
    st.write("Prix : Yahoo Finance direct")
    st.write("News : flux RSS énergie")
    st.write("Score : technique + sentiment news")
    st.button("Rafraîchir maintenant", on_click=st.cache_data.clear)


# ============================================================
# CHARGEMENT
# ============================================================

with st.spinner("Chargement des données marché..."):
    frames = load_market_data()

news = load_news()
snapshot = market_snapshot(frames)

if all(df.empty for df in frames.values()):
    st.error("Impossible de récupérer les données marché depuis Yahoo Finance.")


# ============================================================
# SCORES
# ============================================================

tech_score, tech_factors = technical_score(snapshot)
news_score, news_factors = news_sentiment_score(news)

score = round(max(min(tech_score + news_score, 10), -10), 1)
label = label_from_score(score)
confidence = confidence_from_score(score, len(news))


# ============================================================
# KPI PRINCIPAUX
# ============================================================

c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.6])

for col, name in zip(
    [c1, c2, c3, c4],
    ["Brent", "WTI", "Dollar Index", "VIX"]
):
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

st.progress(
    confidence / 100,
    text=f"Niveau de confiance indicatif : {confidence}%"
)


# ============================================================
# CONTENU PRINCIPAL
# ============================================================

left, center, right = st.columns([1.15, 1.15, 1])


with left:
    st.subheader("📈 Marché")

    show = snapshot.copy()

    for col_name in ["Prix", "1 jour", "5 jours", "20 jours"]:
        show[col_name] = show[col_name].apply(
            lambda x: (
                f"{x:,.2f}" if pd.notna(x) and col_name == "Prix"
                else f"{x:+.2f}%" if pd.notna(x)
                else "N/A"
            )
        )

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True
    )

    if "Brent" in frames and not frames["Brent"].empty:
        df_brent = frames["Brent"].tail(90).copy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_brent.index,
                y=df_brent["Close"],
                mode="lines",
                name="Brent"
            )
        )

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
    factor_rows.extend(
        [{"Facteur": k, "Score": v} for k, v in news_factors]
    )

    factors_df = pd.DataFrame(factor_rows)

    st.metric("Score technique", f"{tech_score:+.1f}")
    st.metric("Score news", f"{news_score:+.1f}")

    if not factors_df.empty:
        factors_df = factors_df.sort_values(
            "Score",
            key=lambda s: s.abs(),
            ascending=False
        )

        st.dataframe(
            factors_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Pas assez de données pour scorer.")

    st.subheader("🧾 Synthèse")

    if not factors_df.empty:
        top_factors = [
            (r["Facteur"], r["Score"])
            for _, r in factors_df.head(5).iterrows()
        ]
    else:
        top_factors = []

    st.success(ai_summary(score, top_factors))


with right:
    st.subheader("⚡ Top impact news")

    if news.empty:
        st.warning("Aucune news récupérée.")
    else:
        for _, row in news.head(10).iterrows():
            title = row.get("title", "")
            link = row.get("link", "")

            if link:
                st.markdown(
                    f"**{title}**  \n"
                    f"[Lire l’article]({link})"
                )
            else:
                st.markdown(f"**{title}**")

            st.divider()


# ============================================================
# ÉVÉNEMENTS À RISQUE
# ============================================================

st.subheader("🕒 Next risk events")

events = pd.DataFrame([
    {
        "Quand": "Mercredi 16:30 Paris",
        "Événement": "Stocks EIA US",
        "Risque": "Très fort sur WTI / Brent",
    },
    {
        "Quand": "Mardi soir / nuit",
        "Événement": "API Weekly Statistical Bulletin",
        "Risque": "Signal préliminaire stocks US",
    },
    {
        "Quand": "Selon calendrier",
        "Événement": "OPEP / OPEP+",
        "Risque": "Production / quotas / discipline",
    },
    {
        "Quand": "Mensuel",
        "Événement": "CPI US / Fed / Dollar",
        "Risque": "Dollar et appétit pour le risque",
    },
    {
        "Quand": "Mensuel",
        "Événement": "PMI Chine",
        "Risque": "Demande mondiale",
    },
])

st.dataframe(
    events,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Dernière génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
    "— Données indicatives, possiblement différées. "
    "Ce tableau n’est pas un conseil d’investissement."
)
