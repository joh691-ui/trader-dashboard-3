import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import re
import os

# --- KONFIGURATION ---
ST_PAGE_TITLE = "Monthly Momentum Backtester"
FILE_PATH = r"G:\Min enhet\Aktiekurser\quantoptimizer_hour\UNIVERSES.txt"

st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide", page_icon="📈")

# Dark mode CSS
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; color: white; }
    h1, h2, h3 { color: #f0f2f6; }
    .stTextInput > div > div > input { color: white; }
    /* Fix för selectbox textfärg */
    .stSelectbox > div > div > div { color: white; }
</style>
""", unsafe_allow_html=True)

# --- FUNKTIONER ---

def parse_universes(filepath):
    universes = {}
    if not os.path.exists(filepath):
        st.error(f"Kunde inte hitta filen: {filepath}")
        return universes

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Rensa backslashes (Minnesregel 2025-12-19)
    content = re.sub(r'\\', '', content)
    
    pattern = re.compile(r'(\w+)\s*=\s*\[(.*?)\]', re.DOTALL)
    matches = pattern.findall(content)

    for name, list_content in matches:
        lines = list_content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.split('#')[0]
            cleaned_lines.append(line)
        
        full_string = "".join(cleaned_lines)
        clean_content = re.sub(r'["\'\s]', '', full_string)
        clean_content = re.sub(r'\\', '', clean_content)
        
        tickers = [t.strip() for t in clean_content.split(',') if t.strip()]
        if tickers:
            universes[name] = tickers
    return universes

@st.cache_data
def download_historical_data(tickers, start_date="2015-01-01"):
    if not tickers:
        return pd.DataFrame()
    try:
        # Hämta data 1 år innan startdatum för att MA200 ska finnas dag 1
        download_start = (pd.to_datetime(start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        df = yf.download(tickers, start=download_start, group_by='ticker', progress=False, auto_adjust=True, threads=True)
        return df
    except Exception as e:
        st.error(f"Datafel: {e}")
        return pd.DataFrame()

def run_backtest(df, start_date, initial_capital, commission_pct, top_n=10, w_3m=0.5, w_6m=0.5):
    """
    Kör backtestet med dynamiskt antal innehav (top_n).
    """
    # 1. Förbered Data
    try:
        close_prices = df.xs('Close', level=1, axis=1)
    except KeyError:
        close_prices = df['Close'].to_frame()
        close_prices.columns = [df.columns.name if df.columns.name else "Ticker"]

    # Fyll hål (Forward Fill) - Minnesregel 2025-12-18
    close_prices = close_prices.ffill()

    # 2. Beräkna Indikatorer
    ma50 = close_prices.rolling(window=50).mean()
    ma200 = close_prices.rolling(window=200).mean()
    
    # Momentum: 3 mån (63 dagar) + 6 mån (126 dagar)
    # Använd fill_method=None - Minnesregel 2025-12-27
    ret_3m = close_prices.pct_change(60, fill_method=None)
    ret_6m = close_prices.pct_change(120, fill_method=None)
    
    momentum_score = (ret_3m * w_3m) + (ret_6m * w_6m)

    # 3. Rebalanserings-schema (Månadsslut)
    rebalance_dates = close_prices.resample('ME').last().index
    rebalance_dates = [d for d in rebalance_dates if d >= pd.to_datetime(start_date)]
    
    # 4. Loop
    portfolio_history = []
    current_capital = initial_capital
    current_holdings = {} 
    last_selection = []
    
    for date in rebalance_dates:
        # Hitta data för datumet
        idx_loc = close_prices.index.get_indexer([date], method='pad')[0]
        current_date = close_prices.index[idx_loc]
        today_prices = close_prices.iloc[idx_loc]
        
        # --- Uppdatera portföljvärde ---
        if current_holdings:
            portfolio_value_before_trade = 0
            for ticker, shares in current_holdings.items():
                if ticker in today_prices and not pd.isna(today_prices[ticker]):
                    portfolio_value_before_trade += shares * today_prices[ticker]
            current_capital = portfolio_value_before_trade
        
        # --- Urval (Ranking) ---
        # Data för dagen
        prices_at_date = close_prices.iloc[idx_loc]
        ma50_at_date = ma50.iloc[idx_loc]
        ma200_at_date = ma200.iloc[idx_loc]
        mom_at_date = momentum_score.iloc[idx_loc]
        
        # Filter: Pris > MA50 > MA200
        trend_filter = (prices_at_date > ma50_at_date) & (ma50_at_date > ma200_at_date)
        candidates = mom_at_date[trend_filter].dropna()
        
        # Välj Top N (5, 8 eller 10 beroende på input)
        selected_tickers = candidates.sort_values(ascending=False).head(top_n).index.tolist()
        last_selection = selected_tickers
        
        # --- Handel ---
        if not selected_tickers:
            # Gå till cash
            turnover_sell = current_capital
            commission_cost = turnover_sell * commission_pct
            current_capital -= commission_cost
            current_holdings = {}
            portfolio_history.append({'Date': current_date, 'Value': current_capital, 'Holdings': 0})
            continue

        # Target allocation
        target_weight = 1.0 / len(selected_tickers)
        target_value_per_stock = current_capital * target_weight
        
        new_holdings = {}
        
        # Beräkna turnover för courtage
        all_involved_tickers = set(current_holdings.keys()).union(set(selected_tickers))
        turnover_value = 0
        
        for ticker in all_involved_tickers:
            price = today_prices.get(ticker, 0)
            if pd.isna(price) or price == 0:
                continue 
                
            old_shares = current_holdings.get(ticker, 0)
            old_val = old_shares * price
            
            if ticker in selected_tickers:
                new_val = target_value_per_stock
                new_shares = new_val / price
                new_holdings[ticker] = new_shares
            else:
                new_val = 0
                new_shares = 0
                
            turnover_value += abs(new_val - old_val)
            
        commission_cost = turnover_value * commission_pct
        current_capital -= commission_cost
        current_holdings = new_holdings
        
        portfolio_history.append({'Date': current_date, 'Value': current_capital, 'Holdings': len(selected_tickers)})

    results_df = pd.DataFrame(portfolio_history).set_index('Date')
    return results_df, last_selection

def get_full_name(ticker):
    """Hämtar långt namn för en ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return info.get('longName') or info.get('shortName') or ticker
    except Exception:
        return ticker

# --- MAIN APP ---

def main():
    st.sidebar.title("⚙️ Inställningar")
    
    universes = parse_universes(FILE_PATH)
    if not universes:
        st.warning("Inga listor funna.")
        return
        
    selected_universe = st.sidebar.selectbox("Välj Lista", list(universes.keys()))
    tickers = universes[selected_universe]
    
    st.sidebar.markdown("---")
    
    # NYTT: Välj antal instrument
    top_n = st.sidebar.selectbox("Antal instrument i portföljen", options=[4, 5, 8, 10], index=2)
    
    start_date = st.sidebar.date_input("Startdatum", value=pd.to_datetime("2016-01-01"))
    initial_capital = st.sidebar.number_input("Startkapital", value=100000, step=10000)
    commission_pct = st.sidebar.number_input("Courtage (%)", value=0.4, step=0.01) / 100

    st.sidebar.markdown("### Momentum Vikter")
    w_3m = st.sidebar.slider("Vikt 3 mån (a)", 0.0, 1.0, 0.5, 0.1)
    w_6m = st.sidebar.slider("Vikt 6 mån (b)", 0.0, 1.0, 0.5, 0.1)

    if st.sidebar.button("🧠 Optimera (Sharpe)"):
        st.info("Optimerar vikter (a + b = 1.0)...")
        with st.spinner("Kör simuleringar..."):
            df_opt = download_historical_data(tickers, start_date=str(start_date))
            if not df_opt.empty:
                best_sharpe = -np.inf
                best_params = (0.5, 0.5)
                results = []
                
                for a in np.linspace(0.0, 1.0, 11):
                    b = 1.0 - a
                    res, _ = run_backtest(df_opt, str(start_date), initial_capital, commission_pct, top_n=top_n, w_3m=a, w_6m=b)
                    if not res.empty:
                        daily_ret = res['Value'].pct_change().dropna()
                        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
                        results.append({'a (3m)': a, 'b (6m)': b, 'Sharpe': sharpe})
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = (a, b)
                
                st.success(f"Bästa resultat: a={best_params[0]:.1f}, b={best_params[1]:.1f} -> Sharpe: {best_sharpe:.2f}")
                st.dataframe(pd.DataFrame(results).sort_values('Sharpe', ascending=False).style.format("{:.2f}"))
    
    if st.sidebar.button("🚀 Kör Backtest"):
        st.title(f"📊 Resultat: {selected_universe} (Top {top_n})")
        
        with st.spinner("Simulerar..."):
            df = download_historical_data(tickers, start_date=str(start_date))
            
            if df.empty:
                st.error("Ingen data kunde hämtas.")
                return
            
            # Skickar med top_n till funktionen
            res, latest_picks = run_backtest(df, str(start_date), initial_capital, commission_pct, top_n=top_n, w_3m=w_3m, w_6m=w_6m)
            
            if res.empty:
                st.error("Inga affärer kunde göras.")
                return
            
            # --- Metrics ---
            final_val = res['Value'].iloc[-1]
            total_ret = (final_val / initial_capital) - 1
            days = (res.index[-1] - res.index[0]).days
            years = days / 365.25
            cagr = (final_val / initial_capital) ** (1/years) - 1 if years > 0 else 0
            
            # Drawdown
            res['Peak'] = res['Value'].cummax()
            res['Drawdown'] = (res['Value'] - res['Peak']) / res['Peak']
            max_dd = res['Drawdown'].min()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Slutvärde", f"{final_val:,.0f}")
            col2.metric("Total Avkastning", f"{total_ret*100:.2f}%")
            col3.metric("CAGR", f"{cagr*100:.2f}%")
            col4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
            
            # --- Grafer (med width='stretch' fix) ---
            
            # Equity Curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.index, y=res['Value'], mode='lines', name='Portfölj',
                                     line=dict(color='#00F0FF', width=2)))
            fig.update_layout(title="Utveckling (Equity Curve)", template="plotly_dark",
                              xaxis_title="Datum", yaxis_title="Värde")
            st.plotly_chart(fig, width="stretch")
            
            # Drawdown Plot
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=res.index, y=res['Drawdown']*100, mode='lines', name='Drawdown',
                                        line=dict(color='#FF0000', width=1), fill='tozeroy'))
            fig_dd.update_layout(title="Drawdown (%)", template="plotly_dark",
                                 xaxis_title="Datum", yaxis_title="Nedgång %")
            st.plotly_chart(fig_dd, width="stretch")

            with st.expander("Visa transaktionsdata"):
                st.dataframe(res)

            st.subheader(f"📌 Senaste valda innehav ({res.index[-1].strftime('%Y-%m-%d')})")
            if latest_picks:
                with st.spinner("Hämtar namn på innehav..."):
                    holdings_data = []
                    for t in latest_picks:
                        holdings_data.append({"Ticker": t, "Namn": get_full_name(t)})
                st.table(pd.DataFrame(holdings_data))
            else:
                st.info("Inga innehav valdes vid senaste ombalansering (Cash).")

if __name__ == "__main__":
    main()