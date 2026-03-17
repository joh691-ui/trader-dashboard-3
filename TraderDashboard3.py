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

    # 1. TREND FILTER: Pris > MA50 > MA200
    is_uptrend = (current['Close'] > current['MA50']) and (current['MA50'] > current['MA200'])
    
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
    """
    Försöker hämta det långa namnet för en ticker via yfinance Ticker-objektet.
    Används bara på topplistan för att spara tid.
    """
    try:
        t = yf.Ticker(ticker)
        # Försök hämta longName, fallback till shortName, fallback till ticker
        info = t.info
        name = info.get('longName') or info.get('shortName') or ticker
        return name
    except Exception:
        return ticker

def plot_chart(metrics):
    df = metrics['data']
    # Använd det fina namnet om vi har hämtat det, annars ticker
    display_name = metrics.get('full_name', metrics['ticker'])
    rsi = metrics['rsi']
    
    display_df = df.tail(126) 

    rsi_color = "green" if rsi < 60 else "red"
    rsi_text = "DIP" if rsi < 60 else "HÖG"
    
    # Titel med Fullständigt namn
    title_html = (f"<b>{display_name}</b><br>"
                  f"<span style='font-size: 14px;'>Pris: {metrics['price']:.2f} | "
                  f"Mom: {metrics['momentum_val']:.2f} | "
                  f"<span style='color:{rsi_color}'>RSI: {rsi:.2f} ({rsi_text})</span></span>")

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

    selected_universe_name = st.sidebar.selectbox("Välj Instrument-lista", list(universes.keys()))
    selected_tickers = universes[selected_universe_name]
    
    st.sidebar.markdown(f"**Antal instrument:** {len(selected_tickers)}")
    
    auto_refresh = st.sidebar.checkbox("Aktivera Auto-Refresh (60s)", value=True)

    if st.sidebar.button("🔄 Uppdatera Nu"):
        st.cache_data.clear()
        st.rerun()

    st.title(f"🚀 Top 20 Momentum: {selected_universe_name}")
    st.markdown("Visar **Golden Trend** instrument rankade efter **3m + 6m avkastning**.")
    st.info("💡 **Golden Trend innebär:** Instrumentet befinner sig i en stark uppåtgående trend där priset är högre än 50-dagars glidande medelvärde (MA50), och MA50 i sin tur är högre än 200-dagars glidande medelvärde (MA200).")

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
            metrics = analyze_ticker(ticker, data)
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
    with st.spinner(f'Hämtar beskrivande namn för de {len(top_stocks)} bästa kandidaterna...'):
        for stock in top_stocks:
            stock['full_name'] = get_full_name(stock['ticker'])

    # Visa grafer
    col1, col2 = st.columns(2)
    
    for i, stock in enumerate(top_stocks):
        fig = plot_chart(stock)
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