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
# ÄNDRING: Relativ sökväg för molnet (filen måste ligga i samma mapp på GitHub)
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
        # Visar tydligt felmeddelande om filen saknas i repot
        st.error(f"Kunde inte hitta filen: {filepath}. Se till att 'UNIVERSES.txt' ligger i GitHub-repot.")
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
        data = yf.download(tickers, period=period, group_by='ticker', progress=False, auto_adjust=True, threads=True)
        return data
    except Exception as e:
        st.error(f"Fel vid hämtning av data: {e}")
        return pd.DataFrame()

def calculate_rsi(series, period=5):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_continuation_score(t_data, rsi_val, momentum_val=0):
    """
    Regelbaserad sannolikhet (0–100) för att en Golden Trend fortsätter uppåt.

    Faktorer (max ±poäng):
      1. RSI-läge        ±30 — överhettad RSI straffas hårt
      2. Trendålder      ±20 — ny trend har mer utrymme kvar
      3. Volymratio      ±15 — stark volym bekräftar trenden
      4. Pris/MA50-avst. ±15 — för utsträckt = rekylrisk
      5. Momentum (3m+6m)±20 — svagt momentum (t.ex. obligationer) straffas
    Summan klipps till [0, 100].
    """
    score = 50  # Neutralt startläge

    # --- Faktor 1: RSI-justering (±30) ---
    if rsi_val >= 90:
        score -= 30
    elif rsi_val >= 80:
        score -= 20
    elif rsi_val >= 70:
        score -= 10
    elif rsi_val <= 30:
        score += 15   # Dip inom upptrend = bra ingång
    elif rsi_val <= 50:
        score += 10

    # --- Faktor 2: Trendålder – dagar sedan MA50 > MA200 gällt (±20) ---
    if 'MA50' in t_data.columns and 'MA200' in t_data.columns:
        above = (t_data['MA50'] > t_data['MA200']).values
        trend_age = 0
        for val in reversed(above):
            if val:
                trend_age += 1
            else:
                break
        if trend_age < 30:
            score += 20   # Ny trend – stor uppåtpotential
        elif trend_age < 90:
            score += 10
        elif trend_age < 180:
            score += 0
        elif trend_age < 365:
            score -= 10
        else:
            score -= 20   # Gammal trend – reverteringsrisk ökar

    # --- Faktor 3: Volymratio – senaste 10d / 50d snitt (±15) ---
    if 'Volume' in t_data.columns:
        vol = t_data['Volume'].replace(0, pd.NA).dropna()
        if len(vol) >= 50:
            ratio = vol.iloc[-10:].mean() / vol.iloc[-50:].mean()
            if ratio > 1.5:
                score += 15   # Accelererande volym – stark bekräftelse
            elif ratio > 1.2:
                score += 8
            elif ratio > 0.8:
                score += 0
            elif ratio > 0.5:
                score -= 8
            else:
                score -= 15   # Volymtorka – svagt trendintresse

    # --- Faktor 4: Pris-MA50-avstånd i % (±15) ---
    close_val = t_data['Close'].iloc[-1]
    ma50_val = t_data['MA50'].iloc[-1]
    if ma50_val > 0:
        dist_pct = (close_val - ma50_val) / ma50_val * 100
        if dist_pct < 3:
            score += 15   # Priset tätt intill MA50 – stabilt läge
        elif dist_pct < 8:
            score += 8
        elif dist_pct < 15:
            score += 0
        elif dist_pct < 25:
            score -= 10
        else:
            score -= 15   # Kraftigt utsträckt – rekylrisk

    # --- Faktor 5: Momentum 3m+6m (±20) ---
    # Lågt momentum (t.ex. obligationer ~1–5) straffas hårt
    if momentum_val >= 60:
        score += 20
    elif momentum_val >= 40:
        score += 15
    elif momentum_val >= 20:
        score += 8
    elif momentum_val >= 10:
        score += 0
    elif momentum_val >= 5:
        score -= 10
    else:
        score -= 20   # Momentum < 5 = obligationer/korta räntor, ej trend-värdigt

    return max(0, min(100, int(score)))


def analyze_ticker(ticker, df):
    t_data = None
    
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns:
            t_data = df[ticker].copy()
        elif ticker.upper() in df.columns:
            t_data = df[ticker.upper()].copy()
    else:
        t_data = df.copy()

    if t_data is not None and not t_data.empty:
        t_data = t_data.dropna(subset=['Close'])

    if t_data is None or t_data.empty or len(t_data) < 200: 
        return None 

    close = t_data['Close']
    
    t_data['MA50'] = close.rolling(window=50).mean()
    t_data['MA200'] = close.rolling(window=200).mean()
    t_data['RSI5'] = calculate_rsi(close, period=5)
    
    current = t_data.iloc[-1]
    
    if pd.isna(current['MA200']) or pd.isna(current['MA50']) or pd.isna(current['Close']):
        return None

    is_uptrend = (current['Close'] > current['MA50']) and (current['MA50'] > current['MA200'])
    
    if not is_uptrend:
        return None

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

    rsi_val = current['RSI5']
    if pd.isna(rsi_val): rsi_val = 50.0

    cont_score = calculate_continuation_score(t_data, rsi_val, momentum_score)

    return {
        'ticker': ticker,
        'data': t_data,
        'momentum_val': momentum_score,
        'price': current['Close'],
        'rsi': rsi_val,
        'ma50': current['MA50'],
        'ma200': current['MA200'],
        'cont_score': cont_score
    }

def get_full_name(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = info.get('longName') or info.get('shortName') or ticker
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


def plot_chart(metrics):
    df = metrics['data']
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

    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Close'], name='Pris', 
                             line=dict(color='#00F0FF', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['MA50'], name='MA50', 
                             line=dict(color='#FFA500', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['MA200'], name='MA200', 
                             line=dict(color='#FF0000', width=1)), row=1, col=1)

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
        # Om filen finns men är tom eller felaktig
        st.warning("Hittade filen men inga listor kunde läsas in.")
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

    analyzed_stocks.sort(key=lambda x: x['momentum_val'], reverse=True)
    top_stocks = analyzed_stocks[:20]

    if not top_stocks:
        st.warning(f"Inga instrument uppfyller kriterierna (Pris > MA50 > MA200) för {selected_universe_name}.")
        return

    with st.spinner(f'Hämtar beskrivande namn för de {len(top_stocks)} bästa kandidaterna...'):
        for stock in top_stocks:
            stock['full_name'] = get_full_name(stock['ticker'])

    # --- Sammanfattningstabell med AI Continuation Score ---
    st.markdown("### 📊 Rankning: AI Continuation Score")
    st.markdown("*Regelbaserad sannolikhet (0–100) att trenden fortsätter. Viktade faktorer: RSI-läge, trendålder, volym och pris/MA50-avstånd.*")

    def score_bar(score):
        color = "#00FF88" if score >= 75 else "#FFD700" if score >= 50 else "#FFA500" if score >= 25 else "#FF4444"
        filled = int(score / 5)  # 0–20 block
        bar = "█" * filled + "░" * (20 - filled)
        return f"<span style='color:{color};font-family:monospace'>{bar}</span> <b style='color:{color}'>{score}</b>"

    table_rows = ""
    for rank, s in enumerate(sorted(top_stocks, key=lambda x: x['cont_score'], reverse=True), 1):
        name = s.get('full_name', s['ticker'])
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

    col1, col2 = st.columns(2)

    for i, stock in enumerate(top_stocks):
        fig = plot_chart(stock)
        # ÄNDRING: width="stretch" fungerar ej. use_container_width=True är korrekt.
        if i % 2 == 0:
            with col1: st.plotly_chart(fig, use_container_width=True)
        else:
            with col2: st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()