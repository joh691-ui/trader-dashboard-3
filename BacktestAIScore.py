import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import differential_evolution
import re
import os

FILE_PATH = r"G:\Min enhet\Aktiekurser\quantoptimizer_hour\UNIVERSES.txt"
FACTOR_NAMES = ['RSI', 'Trend', 'Volym', 'Avstånd MA50', 'Momentum']
DEFAULT_WEIGHTS = [0.2, 0.45, 1.9, 0.35, 2.1]  # optimerade vikter (Sharpe, 2016– och 2020–genomsnitt)

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
    for name, list_content in re.compile(r'(\w+)\s*=\s*\[(.*?)\]', re.DOTALL).findall(content):
        full = "".join(l.split('#')[0] for l in list_content.split('\n'))
        clean = re.sub(r'["\'\s\\]', '', full)
        tickers = [t for t in clean.split(',') if t]
        if tickers:
            universes[name] = tickers
    return universes


@st.cache_data(ttl=3600)
def download_data(tickers, start_date):
    dl_start = (pd.to_datetime(start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    return yf.download(
        tickers, start=dl_start,
        group_by='ticker', progress=False,
        auto_adjust=True, threads=True
    )


# ---------------------------------------------------------------------------
# INDIKATORER
# ---------------------------------------------------------------------------

def compute_rsi(close_df, period=14):
    delta = close_df.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def consecutive_true_count(bool_df):
    """Antal på varandra följande True räknat bakifrån per kolumn."""
    result = pd.DataFrame(0, index=bool_df.index, columns=bool_df.columns)
    for col in bool_df.columns:
        s = bool_df[col].astype(int)
        groups = (s != s.shift()).cumsum()
        result[col] = s.groupby(groups).cumsum() * s
    return result


# ---------------------------------------------------------------------------
# FAKTORER — varje faktor returnerar en 0-100 score-DataFrame
# ---------------------------------------------------------------------------

def compute_factor_scores(close, volume, ma50, ma200, rsi, ret_3m, ret_6m):
    """
    Returnerar en dict med fem separata faktor-DataFrames (0-100).
    Lägre RSI = bättre (mer utrymme uppåt).
    Ny Golden Cross = bättre. Hög volym = bättre. Nära MA50 = bättre.
    Hög momentum = bättre.
    """
    idx, cols = close.index, close.columns

    # --- F1: RSI (lågt RSI = hög score) ---
    rsi_v = rsi.values
    f1 = np.select(
        [rsi_v > 80, rsi_v > 70, rsi_v > 60, rsi_v < 30, rsi_v < 40],
        [5,           20,          35,          90,          70],
        default=50
    )
    rsi_score = pd.DataFrame(f1.astype(float), index=idx, columns=cols)

    # --- F2: Trendstatus + ålder ---
    golden = ma50 > ma200
    trend_age = consecutive_true_count(golden)
    trend_score = pd.DataFrame(25.0, index=idx, columns=cols)  # ej golden = 25
    trend_score[golden & (trend_age < 30)]  = 100
    trend_score[golden & (trend_age >= 30)  & (trend_age < 90)]  = 75
    trend_score[golden & (trend_age >= 90)  & (trend_age < 180)] = 55
    trend_score[golden & (trend_age >= 180) & (trend_age < 365)] = 35
    trend_score[golden & (trend_age >= 365)] = 20

    # --- F3: Volymratio (10d vs 50d) ---
    vol_ratio = volume.rolling(10).mean() / volume.rolling(50).mean().replace(0, np.nan)
    vr = vol_ratio.values
    f3 = np.select(
        [vr > 1.5, vr > 1.2, vr < 0.7, vr < 0.9],
        [90,        70,        15,        35],
        default=50
    )
    vol_score = pd.DataFrame(f3.astype(float), index=idx, columns=cols)

    # --- F4: Pris/MA50-avstånd (nära men ovanför = bäst) ---
    dist = ((close - ma50) / ma50.replace(0, np.nan)).values
    f4 = np.select(
        [dist > 0.15, dist > 0.08, dist > 0.03, dist < -0.05],
        [15,           35,           75,           90],
        default=60
    )
    dist_score = pd.DataFrame(f4.astype(float), index=idx, columns=cols)

    # --- F5: Momentum (3m + 6m, i %) ---
    mom = (ret_3m + ret_6m).values * 100
    f5 = np.select(
        [mom >= 60, mom >= 40, mom >= 20, mom >= 10, mom >= 5, mom < 5],
        [95,         80,         65,         50,         30,        10],
        default=50
    )
    mom_score = pd.DataFrame(f5.astype(float), index=idx, columns=cols)

    return {
        'RSI':          rsi_score,
        'Trend':        trend_score,
        'Volym':        vol_score,
        'Avstånd MA50': dist_score,
        'Momentum':     mom_score,
    }


def combine_scores(factor_dict, weights):
    """Viktat medelvärde av faktorer → kombinerad score 0-100."""
    w_arr = np.array(weights, dtype=float)
    total_w = w_arr.sum()
    if total_w == 0:
        return None
    combined = sum(w * factor_dict[name] for w, name in zip(w_arr, FACTOR_NAMES))
    return (combined / total_w).clip(0, 100)


# ---------------------------------------------------------------------------
# SNABB BACKTEST (numpy) — används av optimeraren
# ---------------------------------------------------------------------------

def fast_monthly_backtest(scores_monthly, close_monthly, filter_monthly, top_n, commission_pct):
    """
    scores_monthly  : (n_months, n_tickers) float — NaN om data saknas
    close_monthly   : (n_months, n_tickers) float
    filter_monthly  : (n_months, n_tickers) bool
    Returnerar array med månadsavkastningar (längd n_months-1).
    """
    n_months, n_tickers = scores_monthly.shape
    port_returns = np.empty(n_months - 1)
    prev_sel = frozenset()

    for i in range(n_months - 1):
        masked = np.where(filter_monthly[i], scores_monthly[i], -np.inf)
        valid  = np.where(np.isfinite(masked))[0]

        if len(valid) == 0:
            port_returns[i] = 0.0
            prev_sel = frozenset()
            continue

        k = min(top_n, len(valid))
        top_idx = valid[np.argsort(masked[valid])[-k:]]
        curr_sel = frozenset(top_idx.tolist())

        # Courtage på omsättning
        changed = len(curr_sel.symmetric_difference(prev_sel))
        cost = (changed / (2 * max(top_n, 1))) * commission_pct

        # Avkastning (lika viktad)
        c_prices = close_monthly[i]
        n_prices = close_monthly[i + 1]
        rets = [
            n_prices[t] / c_prices[t] - 1
            for t in curr_sel
            if not np.isnan(c_prices[t]) and not np.isnan(n_prices[t]) and c_prices[t] > 0
        ]
        port_returns[i] = (np.mean(rets) if rets else 0.0) - cost
        prev_sel = curr_sel

    return port_returns


def sharpe_ratio(returns, rf_annual=0.02):
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    excess = returns - rf_monthly
    std = excess.std()
    if std < 1e-9:
        return 0.0
    return (excess.mean() / std) * np.sqrt(12)


# ---------------------------------------------------------------------------
# FULL BACKTEST (med equity curve) — används för visning
# ---------------------------------------------------------------------------

def run_full_backtest(close, factor_dict, filter_df, start_date, initial_capital,
                      commission_pct, top_n, weights, use_momentum_only=False):
    """
    Simulerar portföljvärde med korrekt kapital-tracking.
    use_momentum_only=True → rankar på ren momentum-score istället för AI score.
    """
    if use_momentum_only:
        ranking = factor_dict['Momentum']
    else:
        ranking = combine_scores(factor_dict, weights)

    rebalance_dates = close.resample('ME').last().index
    rebalance_dates = [d for d in rebalance_dates if d >= pd.to_datetime(start_date)]

    history       = []
    capital       = initial_capital
    holdings      = {}

    for date in rebalance_dates:
        idx = close.index.get_indexer([date], method='pad')[0]
        today = close.index[idx]
        prices = close.iloc[idx]

        if holdings:
            capital = sum(
                sh * prices[t]
                for t, sh in holdings.items()
                if t in prices and not pd.isna(prices[t])
            )

        filt_today  = filter_df.iloc[idx]
        score_today = ranking.iloc[idx]
        candidates  = score_today[filt_today].dropna()
        selected    = candidates.nlargest(top_n).index.tolist()

        if not selected:
            capital -= capital * commission_pct
            holdings = {}
            history.append({'Date': today, 'Value': capital})
            continue

        target_val    = capital / len(selected)
        all_involved  = set(holdings) | set(selected)
        turnover      = 0.0
        new_holdings  = {}

        for t in all_involved:
            p = prices.get(t, 0)
            if pd.isna(p) or p == 0:
                continue
            old_val = holdings.get(t, 0) * p
            new_val = target_val if t in selected else 0.0
            if t in selected:
                new_holdings[t] = new_val / p
            turnover += abs(new_val - old_val)

        capital  -= turnover * commission_pct
        holdings  = new_holdings
        history.append({'Date': today, 'Value': capital})

    res = pd.DataFrame(history).set_index('Date')
    res['Peak']     = res['Value'].cummax()
    res['Drawdown'] = (res['Value'] - res['Peak']) / res['Peak']
    return res


def metrics_from_res(res, initial_capital):
    fv   = res['Value'].iloc[-1]
    ret  = fv / initial_capital - 1
    yrs  = (res.index[-1] - res.index[0]).days / 365.25
    cagr = (fv / initial_capital) ** (1 / yrs) - 1 if yrs > 0 else 0
    mdd  = res['Drawdown'].min()
    mon_ret = res['Value'].pct_change().dropna()
    sr  = sharpe_ratio(mon_ret.values)
    return {'Slutvärde': fv, 'Avkastning': ret, 'CAGR': cagr, 'Max DD': mdd, 'Sharpe': sr}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("⚙️ Inställningar")

    universes = parse_universes(FILE_PATH)
    if not universes:
        return

    sel_universe = st.sidebar.selectbox("Välj Lista", list(universes.keys()))
    tickers      = universes[sel_universe]

    st.sidebar.markdown("---")
    top_n          = st.sidebar.selectbox("Antal innehav", [3, 5, 6, 8, 10], index=2)
    start_date     = st.sidebar.date_input("Startdatum", value=pd.to_datetime("2016-01-01"))
    initial_cap    = st.sidebar.number_input("Startkapital (kr)", value=100_000, step=10_000)
    commission_pct = st.sidebar.number_input("Courtage (%)", value=0.4, step=0.01) / 100

    tab_bt, tab_opt = st.tabs(["📊 Backtest", "🔬 Viktoptimering (Sharpe)"])

    # === Gemensam datahämtning ===
    @st.cache_data(ttl=3600)
    def get_prepared_data(tickers_key, start):
        df_raw = download_data(list(tickers_key), start)
        if df_raw.empty:
            return None
        try:
            close  = df_raw.xs('Close',  level=1, axis=1).ffill()
            volume = df_raw.xs('Volume', level=1, axis=1).ffill()
        except KeyError:
            close  = df_raw['Close'].to_frame().ffill()
            volume = df_raw['Volume'].to_frame().ffill()

        ma50   = close.rolling(50).mean()
        ma200  = close.rolling(200).mean()
        rsi    = compute_rsi(close)
        ret_3m = close.pct_change(60,  fill_method=None)
        ret_6m = close.pct_change(120, fill_method=None)

        factors    = compute_factor_scores(close, volume, ma50, ma200, rsi, ret_3m, ret_6m)
        filter_df  = (close > ma200)

        return close, volume, factors, filter_df

    # ===========================
    #  TAB 1: BACKTEST
    # ===========================
    with tab_bt:
        st.title(f"🤖 AI Score Backtest — {sel_universe} (Top {top_n})")
        compare_mom = st.checkbox("Jämför mot ren Momentum-ranking", value=True)

        if st.button("🚀 Kör Backtest", key="bt_run"):
            with st.spinner("Hämtar data..."):
                prepared = get_prepared_data(tuple(tickers), str(start_date))
            if prepared is None:
                st.error("Ingen data kunde hämtas.")
                return
            close, _, factors, filter_df = prepared

            with st.spinner("Simulerar AI Score-strategi..."):
                res_ai = run_full_backtest(
                    close, factors, filter_df, str(start_date),
                    initial_cap, commission_pct, top_n, DEFAULT_WEIGHTS
                )
            m_ai = metrics_from_res(res_ai, initial_cap)

            res_mom = None
            if compare_mom:
                with st.spinner("Simulerar Momentum-strategi..."):
                    res_mom = run_full_backtest(
                        close, factors, filter_df, str(start_date),
                        initial_cap, commission_pct, top_n, DEFAULT_WEIGHTS,
                        use_momentum_only=True
                    )
                m_mom = metrics_from_res(res_mom, initial_cap)

            # Metrics
            labels = ["Slutvärde", "Avkastning", "CAGR", "Max Drawdown", "Sharpe"]
            fmts   = ["{:,.0f} kr", "{:.1%}", "{:.2%}", "{:.2%}", "{:.2f}"]
            keys   = ['Slutvärde', 'Avkastning', 'CAGR', 'Max DD', 'Sharpe']
            st.markdown("#### AI Continuation Score")
            cols = st.columns(5)
            for col, lbl, fmt, key in zip(cols, labels, fmts, keys):
                col.metric(lbl, fmt.format(m_ai[key]))

            if compare_mom and res_mom is not None:
                st.markdown("#### Ren Momentum")
                cols2 = st.columns(5)
                for col, lbl, fmt, key in zip(cols2, labels, fmts, keys):
                    delta = None
                    if key in ['CAGR', 'Sharpe']:
                        delta = f"{m_ai[key] - m_mom[key]:+.2f}"
                    elif key == 'Max DD':
                        delta = f"{(m_ai[key] - m_mom[key])*100:+.2f}%"
                    col.metric(lbl, fmt.format(m_mom[key]), delta=delta)

            # Equity Curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_ai.index, y=res_ai['Value'],
                                     name="AI Score", line=dict(color='#00F0FF', width=2)))
            if res_mom is not None:
                fig.add_trace(go.Scatter(x=res_mom.index, y=res_mom['Value'],
                                         name="Momentum", line=dict(color='#FFB347', width=2, dash='dot')))
            fig.update_layout(title="Equity Curve", template="plotly_dark",
                              xaxis_title="Datum", yaxis_title="Portföljvärde (kr)")
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=res_ai.index, y=res_ai['Drawdown'] * 100,
                                        name="AI Score", line=dict(color='#FF4444', width=1),
                                        fill='tozeroy'))
            if res_mom is not None:
                fig_dd.add_trace(go.Scatter(x=res_mom.index, y=res_mom['Drawdown'] * 100,
                                            name="Momentum", line=dict(color='#FFA500', width=1, dash='dot')))
            fig_dd.update_layout(title="Drawdown (%)", template="plotly_dark")
            st.plotly_chart(fig_dd, use_container_width=True)

    # ===========================
    #  TAB 2: VIKTOPTIMERING
    # ===========================
    with tab_opt:
        st.title("🔬 Viktoptimering mot Sharpe-ratio")
        st.markdown("""
Optimeraren hittar de **vikter** för de fem faktorerna som historiskt gav **högst Sharpe-ratio**.

Metod: `differential_evolution` (scipy) — global, gradientfri optimering.
Varje vikt sätts i intervallet **[0, 3]** (relativa vikter, normaliseras automatiskt).
Resultaten speglar *historisk* anpassning — testa att styla fram- och bakåt.
        """)

        max_iter  = st.slider("Max iterationer (fler = mer exakt men långsammare)", 20, 200, 60, 10)
        rf_annual = st.number_input("Riskfri ränta (% årsvis)", value=2.0, step=0.5) / 100

        if st.button("🔬 Starta optimering", key="opt_run"):
            with st.spinner("Hämtar data..."):
                prepared = get_prepared_data(tuple(tickers), str(start_date))
            if prepared is None:
                st.error("Ingen data.")
                return
            close, _, factors, filter_df = prepared

            # Förbered månadsdata (numpy) för snabb optimering
            rebal_dates   = close.resample('ME').last().index
            rebal_dates   = [d for d in rebal_dates if d >= pd.to_datetime(start_date)]

            idx_list = [close.index.get_indexer([d], method='pad')[0] for d in rebal_dates]

            close_arr  = close.values   # (days, tickers)
            filter_arr = filter_df.values

            # Faktorer samplade vid varje månadsslut
            factor_monthly = {
                name: df.values[idx_list]   # (n_months, n_tickers)
                for name, df in factors.items()
            }
            filter_monthly = filter_arr[idx_list]
            close_monthly  = close_arr[idx_list]

            # Stableisera NaN
            for name in factor_monthly:
                factor_monthly[name] = np.nan_to_num(factor_monthly[name], nan=50.0)

            progress_bar = st.progress(0.0)
            status_txt   = st.empty()
            best_state   = {'sharpe': -np.inf, 'weights': None, 'calls': 0}
            # Gissning av max anrop (popsize * maxiter * n_factors)
            est_calls    = 15 * max_iter * len(FACTOR_NAMES)

            def objective(weights):
                w = np.abs(weights)
                total_w = w.sum()
                if total_w < 1e-9:
                    return 10.0  # straff
                # Kombinera faktorer
                scores = sum(w[i] * factor_monthly[n] for i, n in enumerate(FACTOR_NAMES))
                scores /= total_w

                rets = fast_monthly_backtest(scores, close_monthly, filter_monthly,
                                            top_n, commission_pct)
                rf_m = (1 + rf_annual) ** (1 / 12) - 1
                excess = rets - rf_m
                std = excess.std()
                sr  = (excess.mean() / std * np.sqrt(12)) if std > 1e-9 else 0.0

                best_state['calls'] += 1
                if sr > best_state['sharpe']:
                    best_state['sharpe']  = sr
                    best_state['weights'] = w.copy()
                progress = min(best_state['calls'] / est_calls, 1.0)
                progress_bar.progress(float(progress))
                status_txt.text(
                    f"Anrop {best_state['calls']} — Bästa Sharpe: {best_state['sharpe']:.3f}"
                )
                return -sr

            result = differential_evolution(
                objective,
                bounds=[(0.0, 3.0)] * len(FACTOR_NAMES),
                seed=42, maxiter=max_iter, popsize=10,
                tol=0.005, mutation=(0.5, 1.5), recombination=0.7,
                workers=1
            )
            progress_bar.progress(1.0)

            opt_weights = np.abs(result.x)
            opt_weights_norm = opt_weights / opt_weights.sum() * len(FACTOR_NAMES)  # skala till sum=N

            st.success(f"Optimering klar! Bästa Sharpe: **{best_state['sharpe']:.3f}**")

            # Visa vikter
            weight_df = pd.DataFrame({
                'Faktor':          FACTOR_NAMES,
                'Defaultvikt':     DEFAULT_WEIGHTS,
                'Optimerad vikt':  [round(w, 2) for w in opt_weights_norm],
            })
            st.table(weight_df.set_index('Faktor'))

            # Kör fullständiga backtests med optimerade vikter + default + momentum
            with st.spinner("Kör jämförande backtests..."):
                res_opt = run_full_backtest(
                    close, factors, filter_df, str(start_date),
                    initial_cap, commission_pct, top_n, opt_weights_norm.tolist()
                )
                res_def = run_full_backtest(
                    close, factors, filter_df, str(start_date),
                    initial_cap, commission_pct, top_n, DEFAULT_WEIGHTS
                )
                res_mom = run_full_backtest(
                    close, factors, filter_df, str(start_date),
                    initial_cap, commission_pct, top_n, DEFAULT_WEIGHTS,
                    use_momentum_only=True
                )

            m_opt = metrics_from_res(res_opt, initial_cap)
            m_def = metrics_from_res(res_def, initial_cap)
            m_mom = metrics_from_res(res_mom, initial_cap)

            labels = ["Slutvärde", "CAGR", "Max Drawdown", "Sharpe"]
            fmts   = ["{:,.0f} kr", "{:.2%}", "{:.2%}", "{:.2f}"]
            keys   = ['Slutvärde', 'CAGR', 'Max DD', 'Sharpe']

            for title, m, ref in [
                ("🏆 Optimerad AI Score", m_opt, m_def),
                ("⚙️ Default AI Score (lika vikter)", m_def, m_mom),
                ("📈 Ren Momentum", m_mom, None),
            ]:
                st.markdown(f"#### {title}")
                cols = st.columns(4)
                for col, lbl, fmt, key in zip(cols, labels, fmts, keys):
                    delta = None
                    if ref and key in ['CAGR', 'Sharpe']:
                        delta = f"{m[key] - ref[key]:+.2f}"
                    elif ref and key == 'Max DD':
                        delta = f"{(m[key] - ref[key])*100:+.2f}%"
                    col.metric(lbl, fmt.format(m[key]), delta=delta)

            # Equity Curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_opt.index, y=res_opt['Value'],
                                     name="Optimerad AI", line=dict(color='#00FF88', width=2)))
            fig.add_trace(go.Scatter(x=res_def.index, y=res_def['Value'],
                                     name="Default AI", line=dict(color='#00F0FF', width=2, dash='dot')))
            fig.add_trace(go.Scatter(x=res_mom.index, y=res_mom['Value'],
                                     name="Momentum", line=dict(color='#FFB347', width=2, dash='dash')))
            fig.update_layout(title="Equity Curve — Optimerad vs Default vs Momentum",
                              template="plotly_dark",
                              xaxis_title="Datum", yaxis_title="Portföljvärde (kr)")
            st.plotly_chart(fig, use_container_width=True)

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=res_opt.index, y=res_opt['Drawdown'] * 100,
                                        name="Optimerad AI", line=dict(color='#00FF88', width=1),
                                        fill='tozeroy'))
            fig_dd.add_trace(go.Scatter(x=res_def.index, y=res_def['Drawdown'] * 100,
                                        name="Default AI", line=dict(color='#00F0FF', width=1, dash='dot')))
            fig_dd.add_trace(go.Scatter(x=res_mom.index, y=res_mom['Drawdown'] * 100,
                                        name="Momentum", line=dict(color='#FFA500', width=1, dash='dash')))
            fig_dd.update_layout(title="Drawdown (%)", template="plotly_dark")
            st.plotly_chart(fig_dd, use_container_width=True)

            st.caption(
                "⚠️ Optimerade vikter är anpassade till historisk data (in-sample). "
                "Testa robusthet genom att flytta startdatumet och se om vikterna håller."
            )


if __name__ == "__main__":
    main()
