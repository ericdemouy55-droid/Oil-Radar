import os
import re
import requests
import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Oil Trading Desk",
    page_icon="🛢️",
    layout="wide"
)

REFRESH_SECONDS = 300  # 5 minutes
DAILY_INVESTMENT = 1000
INITIAL_CAPITAL = 10000
PAPER_TRADES_FILE = "paper_trades.csv"

WTI_TICKER = "CL=F"
BRENT_TICKER = "BZ=F"

NEWS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.oilprice.com/rss/main",
]

# ============================================================
# AUTO REFRESH
# ============================================================

st.markdown(
    f"""
    <meta http-equiv="refresh" content="{REFRESH_SECONDS}">
    """,
    unsafe_allow_html=True
)

# ============================================================
# Pushover optional
# ============================================================

PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")


def send_pushover(title, message):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        return False

    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "title": title,
                "message": message,
            },
            timeout=10
        )
        return True
    except Exception:
        return False


# ============================================================
# MARKET DATA
# ============================================================

def get_oil_data(ticker, period="5d", interval="15m"):
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True
    )

    if data.empty:
        return pd.DataFrame()

    data = data.reset_index()
    return data


def get_last_price(data):
    if data.empty:
        return None
    return float(data["Close"].iloc[-1])


def calculate_technical_score(data):
    if data.empty or len(data) < 30:
        return 0, {}

    df = data.copy()
    close = df["Close"]

    last_price = float(close.iloc[-1])
    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma_20

    momentum = (last_price - float(close.iloc[-10])) / float(close.iloc[-10]) * 100

    volatility = close.pct_change().rolling(20).std().iloc[-1] * 100
    volatility = float(volatility) if not np.isnan(volatility) else 0

    score = 0

    if last_price > sma_20:
        score += 1
    else:
        score -= 1

    if last_price > sma_50:
        score += 1
    else:
        score -= 1

    if momentum > 0.3:
        score += 1
    elif momentum < -0.3:
        score -= 1

    if volatility > 1.5:
        score -= 0.5

    details = {
        "last_price": last_price,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "momentum_pct": momentum,
        "volatility_pct": volatility,
    }

    return score, details


# ============================================================
# NEWS ANALYSIS
# ============================================================

BULLISH_KEYWORDS = [
    "attack", "war", "conflict", "iran", "israel", "russia",
    "sanctions", "supply cut", "production cut", "opec cut",
    "disruption", "pipeline", "refinery fire", "inventory draw",
    "crude draw", "tight supply", "shortage"
]

BEARISH_KEYWORDS = [
    "inventory build", "crude build", "demand weak", "weak demand",
    "recession", "slowdown", "oversupply", "production increase",
    "opec output rise", "ceasefire", "peace deal", "demand concerns"
]

STRONG_NEWS_KEYWORDS = [
    "iran", "israel", "opec", "russia", "sanctions", "attack",
    "war", "strike", "inventory", "eia", "supply cut",
    "production cut", "pipeline", "refinery"
]


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def fetch_news(max_items=20):
    articles = []

    for feed_url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:max_items]:
                title = clean_text(entry.get("title", ""))
                link = entry.get("link", "")
                published = entry.get("published", "")

                if title:
                    articles.append({
                        "title": title,
                        "link": link,
                        "published": published
                    })

        except Exception:
            continue

    seen = set()
    unique_articles = []

    for article in articles:
        if article["title"] not in seen:
            unique_articles.append(article)
            seen.add(article["title"])

    return unique_articles[:max_items]


def analyze_news_sentiment(articles):
    score = 0
    strong_news = []

    for article in articles:
        title = article["title"].lower()

        impact = 0

        for word in BULLISH_KEYWORDS:
            if word in title:
                score += 1
                impact += 1

        for word in BEARISH_KEYWORDS:
            if word in title:
                score -= 1
                impact += 1

        for word in STRONG_NEWS_KEYWORDS:
            if word in title:
                impact += 1

        if impact >= 2:
            strong_news.append({
                "title": article["title"],
                "impact": impact,
                "link": article["link"]
            })

    news_score = max(min(score, 4), -4)
    return news_score, strong_news


# ============================================================
# OIL BIAS SCORE
# ============================================================

def calculate_oil_bias_score(technical_score, news_score):
    final_score = technical_score + news_score
    final_score = max(min(final_score, 10), -10)
    return round(final_score, 1)


def get_signal(score):
    if score >= 2:
        return "BUY"
    elif score <= -2:
        return "SELL"
    return "HOLD"


def get_signal_label(score):
    if score >= 4:
        return "🟢 HAUSSIER FORT"
    elif score >= 2:
        return "🟢 HAUSSIER MODÉRÉ"
    elif score <= -4:
        return "🔴 BAISSIER FORT"
    elif score <= -2:
        return "🔴 BAISSIER MODÉRÉ"
    return "🟠 NEUTRE / ATTENTE"


# ============================================================
# PAPER TRADING
# ============================================================

def init_paper_trading_file():
    if not os.path.exists(PAPER_TRADES_FILE):
        df = pd.DataFrame(columns=[
            "date",
            "open_time",
            "close_time",
            "signal",
            "entry_price",
            "exit_price",
            "invested_amount",
            "quantity",
            "pnl_eur",
            "pnl_pct",
            "capital_after_trade",
            "oil_bias_score",
            "main_news",
            "status"
        ])
        df.to_csv(PAPER_TRADES_FILE, index=False)


def load_trades():
    init_paper_trading_file()
    return pd.read_csv(PAPER_TRADES_FILE)


def save_trades(df):
    df.to_csv(PAPER_TRADES_FILE, index=False)


def get_current_capital():
    df = load_trades()
    closed = df[df["status"] == "CLOSED"]

    if closed.empty:
        return INITIAL_CAPITAL

    return float(closed.iloc[-1]["capital_after_trade"])


def has_open_trade():
    df = load_trades()
    return not df[df["status"] == "OPEN"].empty


def open_paper_trade(signal, current_price, oil_bias_score, main_news):
    df = load_trades()

    if has_open_trade():
        return False, "Une position est déjà ouverte."

    if signal not in ["BUY", "SELL"]:
        return False, "Signal neutre : aucune position ouverte."

    capital = get_current_capital()

    if capital < DAILY_INVESTMENT:
        return False, "Capital insuffisant."

    quantity = DAILY_INVESTMENT / current_price

    new_trade = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "open_time": datetime.now().strftime("%H:%M:%S"),
        "close_time": "",
        "signal": signal,
        "entry_price": round(current_price, 4),
        "exit_price": "",
        "invested_amount": DAILY_INVESTMENT,
        "quantity": quantity,
        "pnl_eur": "",
        "pnl_pct": "",
        "capital_after_trade": "",
        "oil_bias_score": oil_bias_score,
        "main_news": main_news,
        "status": "OPEN"
    }

    df = pd.concat([df, pd.DataFrame([new_trade])], ignore_index=True)
    save_trades(df)

    return True, f"Position virtuelle ouverte : {signal} à {current_price:.2f}$"


def close_paper_trade(current_price):
    df = load_trades()
    open_positions = df[df["status"] == "OPEN"]

    if open_positions.empty:
        return False, "Aucune position ouverte."

    index = open_positions.index[-1]
    trade = df.loc[index]

    entry_price = float(trade["entry_price"])
    quantity = float(trade["quantity"])
    signal = trade["signal"]

    if signal == "BUY":
        pnl_eur = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    elif signal == "SELL":
        pnl_eur = (entry_price - current_price) * quantity
        pnl_pct = ((entry_price - current_price) / entry_price) * 100
    else:
        return False, "Signal invalide."

    previous_capital = get_current_capital()
    capital_after_trade = previous_capital + pnl_eur

    df.at[index, "close_time"] = datetime.now().strftime("%H:%M:%S")
    df.at[index, "exit_price"] = round(current_price, 4)
    df.at[index, "pnl_eur"] = round(pnl_eur, 2)
    df.at[index, "pnl_pct"] = round(pnl_pct, 2)
    df.at[index, "capital_after_trade"] = round(capital_after_trade, 2)
    df.at[index, "status"] = "CLOSED"

    save_trades(df)

    return True, f"Position clôturée. Résultat : {pnl_eur:.2f}€"


# ============================================================
# UI
# ============================================================

st.title("🛢️ Oil Trading Desk — Simulation virtuelle")

st.caption("Dashboard pétrole avec Oil Bias Score, news impactantes et paper trading 1000€ / jour.")

wti_data = get_oil_data(WTI_TICKER)
brent_data = get_oil_data(BRENT_TICKER)

wti_price = get_last_price(wti_data)
brent_price = get_last_price(brent_data)

if wti_price is None:
    st.error("Impossible de récupérer le prix du WTI.")
    st.stop()

technical_score, technical_details = calculate_technical_score(wti_data)

articles = fetch_news()
news_score, strong_news = analyze_news_sentiment(articles)

oil_bias_score = calculate_oil_bias_score(technical_score, news_score)
signal = get_signal(oil_bias_score)
signal_label = get_signal_label(oil_bias_score)

main_news = strong_news[0]["title"] if strong_news else "Aucune news forte détectée"

# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("WTI", f"{wti_price:.2f} $")

with col2:
    if brent_price:
        st.metric("Brent", f"{brent_price:.2f} $")
    else:
        st.metric("Brent", "N/A")

with col3:
    st.metric("Oil Bias Score", f"{oil_bias_score}/10")

with col4:
    st.metric("Signal", signal)

st.subheader(signal_label)

# ============================================================
# CHART
# ============================================================

st.write("### 📈 Prix WTI")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=wti_data[wti_data.columns[0]],
    y=wti_data["Close"],
    mode="lines",
    name="WTI"
))

fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Temps",
    yaxis_title="Prix $"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SCORE DETAILS
# ============================================================

st.write("### 🧠 Détail du score")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Score technique", technical_score)

with c2:
    st.metric("Score news", news_score)

with c3:
    st.metric("Décision", signal)

with st.expander("Voir les détails techniques"):
    st.json(technical_details)

# ============================================================
# STRONG NEWS
# ============================================================

st.write("### 🚨 News fortes détectées")

if strong_news:
    st.error("Une ou plusieurs news potentiellement impactantes ont été détectées.")

    for news in strong_news[:5]:
        st.write(f"**Impact {news['impact']}** — {news['title']}")
        if news["link"]:
            st.write(news["link"])

    if st.button("📲 Envoyer une notification Pushover"):
        sent = send_pushover(
            "🚨 Oil Alert",
            strong_news[0]["title"]
        )

        if sent:
            st.success("Notification envoyée.")
        else:
            st.warning("Pushover non configuré ou erreur d’envoi.")
else:
    st.info("Aucune news forte détectée pour le moment.")

# ============================================================
# NEWS LIST
# ============================================================

with st.expander("📰 Voir toutes les news"):
    if articles:
        for article in articles:
            st.write(f"- **{article['title']}**")
            if article["link"]:
                st.write(article["link"])
    else:
        st.write("Aucune news disponible.")

# ============================================================
# PAPER TRADING UI
# ============================================================

st.write("---")
st.header("🧪 Paper Trading — 1000€ virtuels par jour")

df_trades = load_trades()
current_capital = get_current_capital()

p1, p2, p3 = st.columns(3)

with p1:
    st.metric("Capital simulé", f"{current_capital:,.2f} €")

with p2:
    total_perf = current_capital - INITIAL_CAPITAL
    st.metric("Performance totale", f"{total_perf:,.2f} €")

with p3:
    st.metric("Mise journalière", f"{DAILY_INVESTMENT} €")

open_positions = df_trades[df_trades["status"] == "OPEN"]

if not open_positions.empty:
    st.warning("📌 Une position virtuelle est actuellement ouverte.")

    trade = open_positions.iloc[-1]

    st.write(f"**Signal :** {trade['signal']}")
    st.write(f"**Prix d’entrée :** {trade['entry_price']}")
    st.write(f"**Montant investi :** {trade['invested_amount']} €")
    st.write(f"**Heure d’ouverture :** {trade['open_time']}")
    st.write(f"**News principale :** {trade['main_news']}")

    if st.button("🔒 Clôturer la position maintenant"):
        success, message = close_paper_trade(wti_price)

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

else:
    st.info("Aucune position ouverte actuellement.")

    if signal in ["BUY", "SELL"]:
        if st.button(f"🚀 Ouvrir une position virtuelle {signal} de 1000€"):
            success, message = open_paper_trade(
                signal=signal,
                current_price=wti_price,
                oil_bias_score=oil_bias_score,
                main_news=main_news
            )

            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    else:
        st.warning("Signal neutre : aucune position recommandée pour le moment.")

# ============================================================
# TRADING HISTORY
# ============================================================

st.write("### 📊 Historique des trades")

if df_trades.empty:
    st.info("Aucun trade enregistré.")
else:
    st.dataframe(
        df_trades.sort_values(by=["date", "open_time"], ascending=False),
        use_container_width=True
    )

    closed = df_trades[df_trades["status"] == "CLOSED"].copy()

    if not closed.empty:
        closed["capital_after_trade"] = pd.to_numeric(
            closed["capital_after_trade"],
            errors="coerce"
        )

        closed["pnl_eur"] = pd.to_numeric(
            closed["pnl_eur"],
            errors="coerce"
        )

        st.write("### 📈 Courbe du capital")

        chart_df = closed[["date", "capital_after_trade"]].dropna()
        chart_df = chart_df.set_index("date")

        st.line_chart(chart_df)

        winning_trades = closed[closed["pnl_eur"] > 0]
        win_rate = len(winning_trades) / len(closed) * 100

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Trades clôturés", len(closed))

        with c2:
            st.metric("Trades gagnants", len(winning_trades))

        with c3:
            st.metric("Taux de réussite", f"{win_rate:.1f}%")

# ============================================================
# FOOTER
# ============================================================

st.write("---")
st.caption(
    "Simulation uniquement. Ce dashboard ne constitue pas un conseil financier. "
    "Rafraîchissement automatique toutes les 5 minutes."
)
