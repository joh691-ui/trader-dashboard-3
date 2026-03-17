import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
import time

# --- KONFIGURATION ---
ST_PAGE_TITLE = "Hourly Trader Dashboard - Top 12"
FILE_PATH = r"G:\Min enhet\Aktiekurser\quantoptimizer_hour\UNIVERSES.txt"

# För timdata ("1h") tillåter Yahoo max 730 dagar bakåt. 
# Vi väljer "1y" för att säkert täcka 6 månaders momentum (ca 900 timmar).
DEFAULT_PERIOD = "1y" 
DEFAULT_INTERVAL = "1h"

st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide", page_icon="⏱️")

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

@st.cache_data(ttl=300)
def fetch_data(tickers, period="1y", interval="1h"):
    if not tickers:
        return pd.DataFrame()
    try:
        # VIKTIGT: interval="1h" för timdata
        data = yf.download(
            tickers, 
            period=period, 
            interval=interval, 
            group_by='ticker', 
            progress=False, 
            auto_adjust=True, 
            threads=True
        )
        return data
    except Exception as e:
        st.error(f"Fel vid hämtning av data: {e}")
        return pd.DataFrame()

def calculate_rsi(series, period=5):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_ticker(ticker, df):
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

    # Krav: Minst 200 timmar för MA200 (Hours)
    if t_data is None or t_data.empty or len(t_data) < 205: 
        return None 

    close = t_data['Close']
    
    # Beräkna indikatorer (TIMMAR)
    t_data['MA50'] = close.rolling(window=50).mean()   # 50 timmar
    t_data['MA200'] = close.rolling(window=200).mean() # 200 timmar
    t_data['RSI5'] = calculate_rsi(close, period=5)
    
    current = t_data.iloc[-1]
    
    # Säkerhetscheck
    if pd.isna(current['MA200']) or pd.isna(current['MA50']) or pd.isna(current['Close']):
        return None

    # 1. TREND FILTER: Pris > MA50 > MA200 (På TIM-grafen)
    is_uptrend = (current['Close'] > current['MA50']) and (current['MA50'] > current['MA200'])
    
    if not is_uptrend:
        return None

    # 2. RÖRELSE (MOMENTUM): 3m + 6m
    # Konvertering: En handelsdag har ca 7 timmar (varierar lite, men 7 är bra snitt).
    # 3 månader = 63 dagar * 7 timmar ≈ 441 timmar
    # 6 månader = 126 dagar * 7 timmar ≈ 882 timmar
    
    try:
        price_now = current['Close']
        
        # Vi måste ha tillräckligt med historik (ca 900 timmar)
        if len(t_data) > 900:
            price_3m = t_data['Close'].iloc[-441]
            price_6m = t_data['Close'].iloc[-882]
            
            ret_3m = (price_now / price_3m) - 1
            ret_6m = (price_now / price_6m) - 1
            
            momentum_score = (ret_3m * 100) + (ret_6m * 100)
        else:
            # Fallback om instrumentet är nynoterat eller saknar djup data
            # Vi straffar det lite genom att sätta 0 om vi inte kan mäta långa trenden
            momentum_score = 0 
            
    except (IndexError, ZeroDivisionError):
        return None 

    # 3. TIMING (RSI)
    rsi_val = current['RSI5']
    if pd.isna(rsi_val): rsi_val = 50.0

    return {
        'ticker': ticker,
        'data': t_data,
        'momentum_val': momentum_score,
        'price': current['Close'],
        'rsi': rsi_val,
        'ma50': current['MA50'],
        'ma200': current['MA200']
    }

def get_full_name(ticker):
    """ Hämtar långt namn för Top 12 """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = info.get('longName') or info.get('shortName') or ticker
        return name
    except Exception:
        return ticker

def plot_chart(metrics):
    df = metrics['data']
    display_name = metrics.get('full_name', metrics['ticker'])
    rsi = metrics['rsi']
    
    # Visa senaste ~126 timmarna (motsvarar ca 3-4 veckors handel)
    display_df = df.tail(126) 

    rsi_color = "green" if rsi < 50 else "red"
    rsi_text = "DIP" if rsi < 50 else "HÖG"
    
    # Titel
    title_html = (f"<b>{display_name}</b> (1h)<br>"
                  f"<span style='font-size: 14px;'>Pris: {metrics['price']:.2f} | "
                  f"Mom(3m+6m): {metrics['momentum_val']:.2f} | "
                  f"<span style='color:{rsi_color}'>RSI: {rsi:.2f} ({rsi_text})</span></span>")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # Pris och MA (Timmar)
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Close'], name='Pris', 
                             line=dict(color='#00F0FF', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['MA50'], name='MA50h', 
                             line=dict(color='#FFA500', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['MA200'], name='MA200h', 
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
    # Stäng av range slider för renare look
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# --- MAIN ---

def main():
    st.sidebar.title("⚙️ Inställningar")
    
    universes = parse_universes(FILE_PATH)
    
    if not universes:
        st.warning("Inga listor hittades.")
        return

    selected_universe_name = st.sidebar.selectbox("Välj Instrument-lista", list(universes.keys()))
    selected_tickers = universes[selected_universe_name]
    
    st.sidebar.markdown(f"**Antal instrument:** {len(selected_tickers)}")
    st.sidebar.info("Obs: Endast instrument med tim-data visas (Aktier/ETF/Crypto).")
    
    auto_refresh = st.sidebar.checkbox("Aktivera Auto-Refresh (60s)", value=True)

    if st.sidebar.button("🔄 Uppdatera Nu"):
        st.cache_data.clear()
        st.rerun()

    st.title(f"⏱️ Hourly Top 12: {selected_universe_name}")
    st.markdown("Visar **Golden Trend (Pris > MA50h > MA200h)** på **1h-data**.")

    with st.spinner('Hämtar 1h-data (1 år bakåt för momentum)...'):
        data = fetch_data(selected_tickers, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL)

    if data.empty:
        st.error("Ingen data kunde hämtas.")
        return

    analyzed_stocks = []
    progress_bar = st.progress(0)
    
    loop_tickers = selected_tickers if len(selected_tickers) > 1 else [selected_tickers[0]]
    
    for i, ticker in enumerate(loop_tickers):
        try:
            metrics = analyze_ticker(ticker, data)
            if metrics:
                analyzed_stocks.append(metrics)
        except Exception:
            pass
        
        if len(loop_tickers) > 0:
            progress_bar.progress((i + 1) / len(loop_tickers))
            
    progress_bar.empty()

    # Sortera Top 12 Momentum
    analyzed_stocks.sort(key=lambda x: x['momentum_val'], reverse=True)
    top_stocks = analyzed_stocks[:12]

    if not top_stocks:
        st.warning(f"Inga instrument uppfyller kriterierna (Pris > MA50h > MA200h) just nu.")
        return

    # Hämta namn för vinnarna
    with st.spinner(f'Hämtar namn för {len(top_stocks)} instrument...'):
        for stock in top_stocks:
            stock['full_name'] = get_full_name(stock['ticker'])

    col1, col2 = st.columns(2)
    
    for i, stock in enumerate(top_stocks):
        fig = plot_chart(stock)
        if i % 2 == 0:
            with col1: st.plotly_chart(fig, use_container_width=True)
        else:
            with col2: st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()