import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
import time

# --- KONFIGURATION ---
ST_PAGE_TITLE = "Trader Dashboard - Top 20 Momentum"
FILE_PATH = "UNIVERSES.txt"
DEFAULT_LOOKBACK = "2y"

st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide", page_icon="🏆")

# Dark mode CSS
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; color: white; }
    h1, h2, h3 { color: #f0f2f6; }
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

    # Regex-fix för backslashes enligt minnesregel [2025-12-19]
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
        # Dubbelkoll av backslashes även inuti listorna
        clean_content = re.sub(r'\\', '', clean_content)
        
        tickers = [t.strip() for t in clean_content.split(',') if t.strip()]
        if tickers:
            universes[name] = tickers
            
    return universes

@st.cache_data(ttl=300)
def fetch_data(tickers, period="2y"):
    if not tickers:
        return pd.DataFrame()
    try:
        # Fonder kan vara långsamma på Yahoo, vi ökar timeout och stänger av progress
        data = yf.download(tickers, period=period, group_by='ticker', progress=False, auto_adjust=True, threads=True)
        return data
    except Exception as e:
        st.error(f"Fel vid hämtning av data: {e}")
        return pd.DataFrame()

def calculate_rsi(series, period=5):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's Smoothing (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ---------------------------------------------------------------------------
# Universe-specifika vikter (Sharpe-optimerade, differential_evolution)
# Varje faktor normaliseras till [-1, +1] och viktas sedan ihop.
# ---------------------------------------------------------------------------
UNIVERSE_WEIGHTS = {
    'ETF': {'RSI': 0.20, 'Trend': 0.45, 'Volym': 1.90, 'MA50': 0.35, 'Momentum': 2.10},
    'SWE': {'RSI': 2.30, 'Trend': 0.35, 'Volym': 0.80, 'MA50': 0.10, 'Momentum': 1.40},
    'USA': {'RSI': 0.27, 'Trend': 0.80, 'Volym': 1.17, 'MA50': 0.02, 'Momentum': 2.76},
}

def get_universe_weights(universe_name):
    """Returnerar vikter baserat på universumtyp (ETF / SWE-aktier / USA-aktier)."""
    name = universe_name.upper()
    if 'ETF' in name:
        return UNIVERSE_WEIGHTS['ETF'], 'ETF'
    elif 'SWE' in name:
        return UNIVERSE_WEIGHTS['SWE'], 'SWE'
    elif 'USA' in name:
        return UNIVERSE_WEIGHTS['USA'], 'USA'
    return UNIVERSE_WEIGHTS['ETF'], 'ETF'  # default


def calculate_continuation_score(t_data, rsi_val, momentum_val=0, weights=None):
    """
    Regelbaserad sannolikhet (0–100) för att trenden fortsätter uppåt.

    Varje faktor normaliseras till [-1, +1] och viktas sedan:
      score = 50 + 50 * (Σ w_i * f_i) / Σ w_i

    Vikterna är universe-specifika (optimerade mot Sharpe-ratio):
      Universe  RSI   Trend  Volym  MA50  Momentum
      ETF       0.20  0.45   1.90   0.35  2.10
      SWE       2.30  0.35   0.80   0.10  1.40
      USA       0.27  0.80   1.17   0.02  2.76
    """
    if weights is None:
        weights = UNIVERSE_WEIGHTS['ETF']

    w_rsi  = weights['RSI']
    w_trnd = weights['Trend']
    w_vol  = weights['Volym']
    w_ma50 = weights['MA50']
    w_mom  = weights['Momentum']
    total_w = w_rsi + w_trnd + w_vol + w_ma50 + w_mom
    if total_w == 0:
        return 50

    # --- F1: RSI [-1, +1] ---
    if   rsi_val >= 90: f1 = -1.0
    elif rsi_val >= 80: f1 = -0.6
    elif rsi_val >= 70: f1 = -0.2
    elif rsi_val <= 30: f1 =  1.0
    elif rsi_val <= 50: f1 =  0.6
    else:               f1 =  0.0

    # --- F2: Trendålder + Golden Trend-status [-1, +1] ---
    f2 = 0.0
    if 'MA50' in t_data.columns and 'MA200' in t_data.columns:
        ma50_now  = t_data['MA50'].iloc[-1]
        ma200_now = t_data['MA200'].iloc[-1]
        if ma50_now < ma200_now:
            f2 = -0.8
        else:
            above = (t_data['MA50'] > t_data['MA200']).values
            trend_age = 0
            for val in reversed(above):
                if val: trend_age += 1
                else:   break
            if   trend_age < 30:  f2 =  1.0
            elif trend_age < 90:  f2 =  0.5
            elif trend_age < 180: f2 =  0.0
            elif trend_age < 365: f2 = -0.5
            else:                 f2 = -1.0

    # --- F3: Volymratio 10d/50d [-1, +1] ---
    f3 = 0.0
    if 'Volume' in t_data.columns:
        vol = t_data['Volume'].replace(0, pd.NA).dropna()
        if len(vol) >= 50:
            ratio = vol.iloc[-10:].mean() / vol.iloc[-50:].mean()
            if   ratio > 1.5: f3 =  1.0
            elif ratio > 1.2: f3 =  0.5
            elif ratio > 0.8: f3 =  0.0
            elif ratio > 0.5: f3 = -0.5
            else:             f3 = -1.0

    # --- F4: Pris/MA50-avstånd [-1, +1] ---
    f4 = 0.0
    close_val = t_data['Close'].iloc[-1]
    ma50_val  = t_data['MA50'].iloc[-1]
    if ma50_val > 0:
        dist_pct = (close_val - ma50_val) / ma50_val * 100
        if   dist_pct < 3:  f4 =  1.0
        elif dist_pct < 8:  f4 =  0.4
        elif dist_pct < 15: f4 =  0.0
        elif dist_pct < 25: f4 = -0.6
        else:               f4 = -1.0

    # --- F5: Momentum 3m+6m [-1, +1] ---
    if   momentum_val >= 60: f5 =  1.0
    elif momentum_val >= 40: f5 =  0.7
    elif momentum_val >= 20: f5 =  0.3
    elif momentum_val >= 10: f5 =  0.0
    elif momentum_val >= 5:  f5 = -0.5
    else:                    f5 = -1.0

    weighted = w_rsi*f1 + w_trnd*f2 + w_vol*f3 + w_ma50*f4 + w_mom*f5
    return max(0, min(100, int(50 + 50 * weighted / total_w)))


def analyze_ticker(ticker, df, weights=None):
    t_data = None
    
    # Robust data-extrahering
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns:
            t_data = df[ticker].copy()
        elif ticker.upper() in df.columns:
            t_data = df[ticker.upper()].copy()
    else:
        t_data = df.copy()

    # Rensa rader där Close är NaN
    if t_data is not None and not t_data.empty:
        t_data = t_data.dropna(subset=['Close'])

    # Krav: Minst 200 datapunkter för MA200
    if t_data is None or t_data.empty or len(t_data) < 200: 
        return None 

    close = t_data['Close']
    
    # Beräkna indikatorer
    t_data['MA50'] = close.rolling(window=50).mean()
    t_data['MA200'] = close.rolling(window=200).mean()
    t_data['RSI5'] = calculate_rsi(close, period=5)
    
    # Sista giltiga värdet
    current = t_data.iloc[-1]
    
    # Säkerhetscheck för NaN i indikatorer
    if pd.isna(current['MA200']) or pd.isna(current['MA50']) or pd.isna(current['Close']):
        return None

    # 1. TREND FILTER: Pris > MA200 (mjukare än Golden Trend — fångar tidig återhämtning)
    is_uptrend = current['Close'] > current['MA200']

    if not is_uptrend:
        return None

    # 2. RÖRELSE (MOMENTUM): 3m + 6m
    try:
        price_now = current['Close']
        if len(t_data) > 130:
            price_3m = t_data['Close'].iloc[-63]
            price_6m = t_data['Close'].iloc[-126]
            
            ret_3m = (price_now / price_3m) - 1
            ret_6m = (price_now / price_6m) - 1
            
            momentum_score = (ret_3m * 100) + (ret_6m * 100)
        else:
            momentum_score = 0 
            
    except (IndexError, ZeroDivisionError):
        return None 

    # 3. TIMING (RSI)
    rsi_val = current['RSI5']
    if pd.isna(rsi_val): rsi_val = 50.0

    cont_score = calculate_continuation_score(t_data, rsi_val, momentum_score, weights)

    return {
        'ticker': ticker,
        'momentum_val': momentum_score,
        'price': current['Close'],
        'rsi': rsi_val,
        'ma50': current['MA50'],
        'ma200': current['ma200'] if 'ma200' in current else current['MA200'],
        'cont_score': cont_score
    }

@st.cache_data(ttl=3600)
def get_full_name(ticker):
    """
    Hämtar bolagsnamn via yfinance. Cachas i 1 timme för att undvika
    rate limiting vid auto-refresh var 60:e sekund.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        # Prova longName → shortName → displayName → ticker som sista utväg
        name = (info.get('longName')
                or info.get('shortName')
                or info.get('displayName')
                or ticker)
        # Om svaret ser ut som en ticker (inga mellanslag, versaler) — returnera ändå
        return name
    except Exception:
        return ticker

def cont_score_label(score):
    if score >= 75:
        return f"<span style='color:#00FF88'>AI-Score: {score} ✅ Hög</span>"
    elif score >= 50:
        return f"<span style='color:#FFD700'>AI-Score: {score} 🟡 Medel</span>"
    elif score >= 25:
        return f"<span style='color:#FFA500'>AI-Score: {score} ⚠️ Låg</span>"
    else:
        return f"<span style='color:#FF4444'>AI-Score: {score} 🔴 Varning</span>"


def plot_chart(metrics, data_frame):
    df = data_frame
    display_name = metrics.get('full_name', metrics['ticker'])
    rsi = metrics['rsi']
    cont_score = metrics.get('cont_score', 50)

    display_df = df.tail(126)

    rsi_color = "green" if rsi < 60 else "red"
    rsi_text = "DIP" if rsi < 60 else "HÖG"

    title_html = (f"<b>{display_name}</b><br>"
                  f"<span style='font-size: 14px;'>Pris: {metrics['price']:.2f} | "
                  f"Mom: {metrics['momentum_val']:.2f} | "
                  f"<span style='color:{rsi_color}'>RSI: {rsi:.2f} ({rsi_text})</span> | "
                  f"{cont_score_label(cont_score)}</span>")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # Pris och MA
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Close'], name='Pris', 
                             line=dict(color='#00F0FF', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['MA50'], name='MA50', 
                             line=dict(color='#FFA500', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['MA200'], name='MA200', 
                             line=dict(color='#FF0000', width=1)), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['RSI5'], name='RSI(5)', 
                             line=dict(color='#ADFF2F', width=1.5)), row=2, col=1)
    
    fig.add_hline(y=90, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=50, line_dash="solid", line_color="white", row=2, col=1)
    fig.add_hline(y=10, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(
        title=dict(text=title_html, x=0, font=dict(size=16)),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        height=450,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# --- MAIN ---

def main():
    st.sidebar.title("⚙️ Inställningar")
    
    universes = parse_universes(FILE_PATH)
    
    if not universes:
        st.warning("Inga listor hittades.")
        return

    universe_list = list(universes.keys())
    default_universe = "UNIVERSE_STOCKS_LARGE_CAP_SWE"
    default_index = universe_list.index(default_universe) if default_universe in universe_list else 0
    selected_universe_name = st.sidebar.selectbox("Välj Instrument-lista", universe_list, index=default_index)
    selected_tickers = universes[selected_universe_name]
    
    st.sidebar.markdown(f"**Antal instrument:** {len(selected_tickers)}")
    
    auto_refresh = st.sidebar.checkbox("Aktivera Auto-Refresh (60s)", value=True)

    if st.sidebar.button("🔄 Uppdatera Nu"):
        st.cache_data.clear()
        st.rerun()

    universe_weights, universe_type = get_universe_weights(selected_universe_name)
    weight_labels = {'ETF': '📊 ETF-profil', 'SWE': '🇸🇪 SWE Aktier-profil', 'USA': '🇺🇸 USA Aktier-profil'}
    weight_desc   = {
        'ETF': 'Volym (1.9) + Momentum (2.1) dominerar. RSI har liten vikt.',
        'SWE': 'RSI (2.3) + Momentum (1.4) dominerar. Mean-reversion-karaktär.',
        'USA': 'Momentum (2.76) + Volym (1.17) dominerar. Trend (0.8) bidrar.',
    }

    st.title(f"🚀 Top 20 Momentum: {selected_universe_name}")
    st.markdown(f"Visar instrument där **Pris > MA200** rankade efter **AI Continuation Score** — {weight_labels[universe_type]}.")
    st.info(f"💡 **Viktprofil ({weight_labels[universe_type]}):** {weight_desc[universe_type]} "
            f"Vikter: RSI={universe_weights['RSI']}, Trend={universe_weights['Trend']}, "
            f"Volym={universe_weights['Volym']}, MA50={universe_weights['MA50']}, "
            f"Momentum={universe_weights['Momentum']}.")

    with st.spinner('Hämtar prisdata...'):
        data = fetch_data(selected_tickers, period=DEFAULT_LOOKBACK)

    if data.empty:
        st.error("Ingen data kunde hämtas.")
        return

    analyzed_stocks = []
    progress_bar = st.progress(0)
    
    loop_tickers = selected_tickers if len(selected_tickers) > 1 else [selected_tickers[0]]
    
    for i, ticker in enumerate(loop_tickers):
        try:
            metrics = analyze_ticker(ticker, data, universe_weights)
            if metrics:
                analyzed_stocks.append(metrics)
        except Exception:
            pass
        
        if len(loop_tickers) > 0:
            progress_bar.progress((i + 1) / len(loop_tickers))
            
    progress_bar.empty()

    # Sortera och välj Top 20
    analyzed_stocks.sort(key=lambda x: x['momentum_val'], reverse=True)
    top_stocks = analyzed_stocks[:20]

    if not top_stocks:
        st.warning(f"Inga instrument uppfyller kriterierna (Pris > MA50 > MA200) för {selected_universe_name}.")
        return

    # --- NYTT STEG: Hämta riktiga namn för topplistan ---
    # time.sleep(0.3) mellan anrop förhindrar Yahoo Finance rate-limiting
    with st.spinner(f'Hämtar beskrivande namn för de {len(top_stocks)} bästa kandidaterna...'):
        for stock in top_stocks:
            stock['full_name'] = get_full_name(stock['ticker'])
            time.sleep(0.3)

    # --- Sammanfattningstabell med AI Continuation Score ---
    st.markdown("### 📊 Rankning: AI Continuation Score")
    st.markdown(f"*AI Continuation Score — {weight_labels[universe_type]}: {weight_desc[universe_type]}*")

    def score_bar(score):
        color = "#00FF88" if score >= 75 else "#FFD700" if score >= 50 else "#FFA500" if score >= 25 else "#FF4444"
        filled = int(score / 5)
        bar = "█" * filled + "░" * (20 - filled)
        return f"<span style='color:{color};font-family:monospace'>{bar}</span> <b style='color:{color}'>{score}</b>"

    table_rows = ""
    for rank, s in enumerate(sorted(top_stocks, key=lambda x: x['cont_score'], reverse=True), 1):
        full_name = s.get('full_name', s['ticker'])
        # Visa alltid bolagsnamn + ticker inom parentes
        name = f"{full_name} <span style='color:#888;font-size:12px'>({s['ticker']})</span>"
        rsi_col = f"<span style='color:{'#FF4444' if s['rsi'] >= 80 else '#FFD700' if s['rsi'] >= 70 else '#00FF88'}'>{s['rsi']:.1f}</span>"
        table_rows += (
            f"<tr>"
            f"<td style='padding:4px 10px;color:#aaa'>{rank}</td>"
            f"<td style='padding:4px 10px'><b>{name}</b></td>"
            f"<td style='padding:4px 10px;text-align:right'>{s['momentum_val']:.1f}</td>"
            f"<td style='padding:4px 10px;text-align:right'>{rsi_col}</td>"
            f"<td style='padding:4px 10px'>{score_bar(s['cont_score'])}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""<table style='width:100%;border-collapse:collapse;font-size:14px'>
        <thead><tr style='border-bottom:1px solid #444;color:#888'>
          <th style='padding:4px 10px;text-align:left'>#</th>
          <th style='padding:4px 10px;text-align:left'>Instrument</th>
          <th style='padding:4px 10px;text-align:right'>Momentum</th>
          <th style='padding:4px 10px;text-align:right'>RSI(5)</th>
          <th style='padding:4px 10px;text-align:left'>AI Continuation Score</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
        </table>""",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Sortera graferna efter AI Continuation Score (högst först)
    top_stocks_sorted = sorted(top_stocks, key=lambda x: x['cont_score'], reverse=True)

    # Visa grafer
    col1, col2 = st.columns(2)

    for i, stock in enumerate(top_stocks_sorted):
        ticker = stock['ticker']
        # Extrahera data för grafen här för att spara minne (late binding)
        if isinstance(data.columns, pd.MultiIndex):
            stock_data = data[ticker].copy().dropna(subset=['Close'])
        else:
            stock_data = data.copy().dropna(subset=['Close'])
        
        # Beräkna indikatorer igen för grafen (billigt jämfört med minne)
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
        stock_data['RSI5'] = calculate_rsi(stock_data['Close'], period=5)
        
        fig = plot_chart(stock, stock_data)
        # ÄNDRING: use_container_width=True är deprecated. Använder width="stretch" istället.
        if i % 2 == 0:
            with col1: st.plotly_chart(fig, width="stretch")
        else:
            with col2: st.plotly_chart(fig, width="stretch")

    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()