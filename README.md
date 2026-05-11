# Oil Radar 30' — V1

Dashboard pétrole en une page : Brent, WTI, Dollar Index, VIX, news énergie/géopolitique, scoring haussier/baissier et synthèse IA optionnelle.

## Installation

```bash
cd oil_radar_v1
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
streamlit run app.py
```

## Clés optionnelles

La V1 fonctionne sans clé payante pour les prix via Yahoo Finance/yfinance et les news via GDELT/RSS.

Pour activer la synthèse IA :

```bash
export OPENAI_API_KEY="votre_cle"
```

Ou créez un fichier `.env` avec :

```env
OPENAI_API_KEY=votre_cle
```

## Sources utilisées

- Prix : Yahoo Finance via yfinance, tickers `BZ=F`, `CL=F`, `DX-Y.NYB`, `^VIX`
- News : GDELT 2.0 DOC API + flux RSS configurables
- EIA : prévu en V2 avec clé gratuite EIA pour données stocks US

## Avertissement

Ce tableau de bord est un outil d'aide à la décision, pas un conseil financier. Les données peuvent être différées, incomplètes ou indisponibles selon les sources.
