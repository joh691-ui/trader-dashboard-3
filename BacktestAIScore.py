import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import re
import os

FILE_PATH = r"G:\Min enhet\Aktiekurser\quantoptimizer_hour\UNIVERSES.txt"

st.set_page_config(page_title="AI Score Backtester", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    .main { background: #0e1117; color: white; }
    h1, h2, h3 { color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATAPARSNING
# ---------------------------------------------------------------------------

def parse_universes(filepath):
    universes = {}
    if not os.path.exists(filepath):
        st.error(f"Kunde inte hitta filen: {filepath}")
        return universes
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    content = re.sub(r'\\', '', content)
    pattern = re.compile(r'(\w+)\s*=\s*\[(.*?)\]', re.DOTALL)
    for name, list_content in pattern.findall(content):
        lines = list_content.split('\n')
        full_string = "".join(l.split('#')[0] for l in lines)
        clean = re.sub(r'["\'\s\\]', '', full_string)
        tickers = [t.strip() for t in clean.split(',') if t.strip()]
        if tickers:
            universes[name] = tickers
    return universes


@st.cache_data(ttl=3600)
def download_data(tickers, start_date):
    """Hämtar Close + Volume, med 1 år extra för att MA200 ska vara klar dag 1."""
    download_start = (pd.to_datetime(start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    df = yf.download(
        tickers, start=download_start,
        group_by='ticker', progress=False,
        auto_adjust=True, threads=True
    )
    return df


# ---------------------------------------------------------------------------
# TEKNISKA INDIKATORER (vectoriserade)
# ---------------------------------------------------------------------------

def compute_rsi(close_df, period=14):
    """Beräknar RSI för varje ticker som en DataFrame."""
    delta = close_df.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def consecutive_true_count(bool_df):
    """
    För varje kolumn: antal på varandra följande True räknat bakifrån till idag.
    Effektiv O(n)-implementation utan looping per rad.
    """
    result = pd.DataFrame(0, index=bool_df.index, columns=bool_df.columns)
    for col in bool_df.columns:
        s = bool_df[col].astype(int)
        # Grupp-id ökar varje gång värdet ändras
        groups = (s != s.shift()).cumsum()
        cumcount = s.groupby(groups).cumsum()
        result[col] = cumcount * s  # nollställ där s=False
    return result


# ---------------------------------------------------------------------------
# AI CONTINUATION SCORE (samma logik som TraderDashboard3.py)
# ---------------------------------------------------------------------------

def compute_ai_scores(close, volume, ma50, ma200, rsi, ret_3m, ret_6m):
    """
    Returnerar en DataFrame (datum × ticker) med score 0–100.
    Faktorer (identiska med dashboardens calculate_continuation_score):
      F1  RSI-läge          ±30
      F2  Trendålder/status ±20
      F3  Volymratio        ±15
      F4  Pris/MA50-avstånd ±15
      F5  Momentum          ±20
    """
    score = pd.DataFrame(50.0, index=close.index, columns=close.columns)

    # --- F1: RSI ---
    score[rsi > 80]  += -30
    score[(rsi > 70) & (rsi <= 80)] += -15
    score[(rsi > 60) & (rsi <= 70)] += -5
    score[(rsi >= 40) & (rsi <= 60)] += 5
    score[(rsi >= 30) & (rsi < 40)] += 10
    score[rsi < 30] += 15
    # pandas boolsk indexering på DataFrame kräver mask
    score = score.where(~(rsi > 80), score - 30)
    # Enklare med apply-per-element via numpy
    rsi_v = rsi.values
    f1 = np.select(
        [rsi_v > 80, rsi_v > 70, rsi_v > 60, rsi_v < 30, rsi_v < 40],
        [-30,         -15,         -5,           15,          10],
        default=5
    )
    score = pd.DataFrame(50.0 + f1, index=close.index, columns=close.columns)

    # --- F2: Trendålder / Golden Trend-status ---
    golden = ma50 > ma200
    trend_age = consecutive_true_count(golden)

    not_golden_mask = ~golden
    score[not_golden_mask] -= 15

    score[(golden) & (trend_age < 30)]  += 20
    score[(golden) & (trend_age >= 30) & (trend_age < 90)]  += 10
    score[(golden) & (trend_age >= 90) & (trend_age < 180)] += 0
    score[(golden) & (trend_age >= 180) & (trend_age < 365)] -= 10
    score[(golden) & (trend_age >= 365)] -= 20

    # --- F3: Volymratio (10d / 50d) ---
    vol10  = volume.rolling(10).mean()
    vol50  = volume.rolling(50).mean()
    vol_ratio = (vol10 / vol50.replace(0, np.nan)).values
    f3 = np.select(
        [vol_ratio > 1.5, vol_ratio > 1.2, vol_ratio < 0.7, vol_ratio < 0.9],
        [15,               8,               -15,              -5],
        default=0
    )
    score += pd.DataFrame(f3, index=close.index, columns=close.columns)

    # --- F4: Pris / MA50-avstånd ---
    dist = ((close - ma50) / ma50.replace(0, np.nan)).values
    f4 = np.select(
        [dist > 0.15, dist > 0.08, dist > 0.03, dist < -0.05],
        [-15,          -5,           10,           15],
        default=5
    )
    score += pd.DataFrame(f4, index=close.index, columns=close.columns)

    # --- F5: Momentum (3m + 6m kombinerat) ---
    mom = (ret_3m + ret_6m).values * 100  # i procent
    f5 = np.select(
        [mom >= 60, mom >= 40, mom >= 20, mom >= 10, mom >= 5, mom < 5],
        [20,         15,         8,          0,         -10,    -20],
        default=0
    )
    score += pd.DataFrame(f5, index=close.index, columns=close.columns)

    return score.clip(0, 100)


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------

def run_backtest(df_raw, start_date, initial_capital, commission_pct, top_n, use_ai_score):
    # --- Extrahera close + volume ---
    try:
        close  = df_raw.xs('Close',  level=1, axis=1).ffill()
        volume = df_raw.xs('Volume', level=1, axis=1).ffill()
    except KeyError:
        close  = df_raw['Close'].to_frame().ffill()
        volume = df_raw['Volume'].to_frame().ffill()

    tickers = close.columns.tolist()

    # --- Indikatorer ---
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi   = compute_rsi(close, 14)
    ret_3m = close.pct_change(60,  fill_method=None)
    ret_6m = close.pct_change(120, fill_method=None)
    momentum = ret_3m + ret_6m  # för ren momentum-ranking

    if use_ai_score:
        ai_scores = compute_ai_scores(close, volume, ma50, ma200, rsi, ret_3m, ret_6m)

    # --- Rebalanseringsdatum (månadsslutet) ---
    rebalance_dates = close.resample('ME').last().index
    rebalance_dates = [d for d in rebalance_dates if d >= pd.to_datetime(start_date)]

    portfolio_history = []
    current_capital   = initial_capital
    current_holdings  = {}
    last_selection    = []

    for date in rebalance_dates:
        idx = close.index.get_indexer([date], method='pad')[0]
        today = close.index[idx]

        today_prices = close.iloc[idx]
        ma50_today   = ma50.iloc[idx]
        ma200_today  = ma200.iloc[idx]

        # --- Uppdatera portföljvärde ---
        if current_holdings:
            pv = sum(
                shares * today_prices[t]
                for t, shares in current_holdings.items()
                if t in today_prices and not pd.isna(today_prices[t])
            )
            current_capital = pv

        # --- Filter: Pris > MA200 (mjukt) ---
        passes_filter = (today_prices > ma200_today)

        if use_ai_score:
            ranking = ai_scores.iloc[idx]
        else:
            ranking = momentum.iloc[idx]

        candidates = ranking[passes_filter].dropna()
        selected   = candidates.sort_values(ascending=False).head(top_n).index.tolist()
        last_selection = selected

        # --- Handel ---
        if not selected:
            cost = current_capital * commission_pct
            current_capital -= cost
            current_holdings = {}
            portfolio_history.append({'Date': today, 'Value': current_capital, 'Holdings': 0})
            continue

        target_val = current_capital / len(selected)
        all_involved = set(current_holdings) | set(selected)
        turnover = 0.0
        new_holdings = {}

        for t in all_involved:
            price = today_prices.get(t, 0)
            if pd.isna(price) or price == 0:
                continue
            old_val = current_holdings.get(t, 0) * price
            new_val = target_val if t in selected else 0.0
            if t in selected:
                new_holdings[t] = new_val / price
            turnover += abs(new_val - old_val)

        current_capital -= turnover * commission_pct
        current_holdings = new_holdings
        portfolio_history.append({'Date': today, 'Value': current_capital, 'Holdings': len(selected)})

    res = pd.DataFrame(portfolio_history).set_index('Date')
    return res, last_selection


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("⚙️ Inställningar")

    universes = parse_universes(FILE_PATH)
    if not universes:
        return

    selected_universe = st.sidebar.selectbox("Välj Lista", list(universes.keys()))
    tickers = universes[selected_universe]

    st.sidebar.markdown("---")
    top_n          = st.sidebar.selectbox("Antal innehav (Top N)", [3, 5, 6, 8, 10], index=2)
    start_date     = st.sidebar.date_input("Startdatum", value=pd.to_datetime("2016-01-01"))
    initial_cap    = st.sidebar.number_input("Startkapital", value=100_000, step=10_000)
    commission_pct = st.sidebar.number_input("Courtage (%)", value=0.4, step=0.01) / 100
    compare        = st.sidebar.checkbox("Jämför AI Score vs Ren Momentum", value=True)

    if not st.sidebar.button("🚀 Kör Backtest"):
        st.title("🤖 AI Continuation Score — Backtester")
        st.info("Konfigurera inställningar i sidopanelen och klicka på **Kör Backtest**.")
        st.markdown("""
**Strategi:**
- Filter: `Pris > MA200` (mjukare än klassisk Golden Trend)
- Ranking: AI Continuation Score (RSI, trendålder, volym, pris/MA50, momentum)
- Rebalansering: en gång per månad
- Väljer Top N instrument med högst score

**Jämförelse:** samma universe och filter men ranking på ren 3m+6m momentum.
        """)
        return

    st.title(f"🤖 AI Score Backtest — {selected_universe} (Top {top_n})")

    with st.spinner("Hämtar historisk data..."):
        df_raw = download_data(tickers, str(start_date))

    if df_raw.empty:
        st.error("Ingen data kunde hämtas.")
        return

    with st.spinner("Kör AI Score-strategi..."):
        res_ai, picks_ai = run_backtest(
            df_raw, str(start_date), initial_cap, commission_pct, top_n, use_ai_score=True
        )

    res_mom, picks_mom = None, []
    if compare:
        with st.spinner("Kör Momentum-strategi (jämförelse)..."):
            res_mom, picks_mom = run_backtest(
                df_raw, str(start_date), initial_cap, commission_pct, top_n, use_ai_score=False
            )

    # --- Nyckeltal ---
    def metrics(res):
        fv   = res['Value'].iloc[-1]
        ret  = (fv / initial_cap) - 1
        yrs  = (res.index[-1] - res.index[0]).days / 365.25
        cagr = (fv / initial_cap) ** (1 / yrs) - 1 if yrs > 0 else 0
        res  = res.copy()
        res['Peak']     = res['Value'].cummax()
        res['Drawdown'] = (res['Value'] - res['Peak']) / res['Peak']
        mdd  = res['Drawdown'].min()
        return fv, ret, cagr, mdd, res

    fv_ai, ret_ai, cagr_ai, mdd_ai, res_ai = metrics(res_ai)

    cols = st.columns(4)
    cols[0].metric("Slutvärde (AI)",        f"{fv_ai:,.0f} kr")
    cols[1].metric("Total avkastning (AI)", f"{ret_ai*100:.1f}%")
    cols[2].metric("CAGR (AI)",             f"{cagr_ai*100:.2f}%")
    cols[3].metric("Max Drawdown (AI)",     f"{mdd_ai*100:.2f}%")

    if compare and res_mom is not None:
        fv_m, ret_m, cagr_m, mdd_m, res_mom = metrics(res_mom)
        cols2 = st.columns(4)
        cols2[0].metric("Slutvärde (Momentum)",        f"{fv_m:,.0f} kr",  delta=f"{(fv_ai-fv_m):,.0f} kr")
        cols2[1].metric("Total avkastning (Momentum)", f"{ret_m*100:.1f}%")
        cols2[2].metric("CAGR (Momentum)",             f"{cagr_m*100:.2f}%", delta=f"{(cagr_ai-cagr_m)*100:.2f}%")
        cols2[3].metric("Max Drawdown (Momentum)",     f"{mdd_m*100:.2f}%")

    # --- Equity Curve ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res_ai.index, y=res_ai['Value'],
        name="AI Score", line=dict(color='#00F0FF', width=2)
    ))
    if compare and res_mom is not None:
        fig.add_trace(go.Scatter(
            x=res_mom.index, y=res_mom['Value'],
            name="Momentum", line=dict(color='#FFB347', width=2, dash='dot')
        ))
    fig.update_layout(
        title="Equity Curve", template="plotly_dark",
        xaxis_title="Datum", yaxis_title="Portföljvärde (kr)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Drawdown ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=res_ai.index, y=res_ai['Drawdown'] * 100,
        name="AI Score", line=dict(color='#FF4444', width=1), fill='tozeroy'
    ))
    if compare and res_mom is not None:
        fig_dd.add_trace(go.Scatter(
            x=res_mom.index, y=res_mom['Drawdown'] * 100,
            name="Momentum", line=dict(color='#FFA500', width=1, dash='dot')
        ))
    fig_dd.update_layout(
        title="Drawdown (%)", template="plotly_dark",
        xaxis_title="Datum", yaxis_title="Drawdown %"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Senaste innehav ---
    st.subheader(f"📌 Senaste innehav AI Score ({res_ai.index[-1].strftime('%Y-%m-%d')})")
    if picks_ai:
        st.table(pd.DataFrame({"Ticker": picks_ai}))
    else:
        st.info("Cash — inga instrument passade filtret.")

    with st.expander("Visa rådata (AI Score)"):
        st.dataframe(res_ai)

    if compare and res_mom is not None:
        with st.expander("Visa rådata (Momentum)"):
            st.dataframe(res_mom)


if __name__ == "__main__":
    main()
