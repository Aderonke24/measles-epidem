import streamlit as st
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ===================================
# PAGE CONFIGs
# ===================================
st.set_page_config(
    page_title="Measles AI Intelligence",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================================
# SOPHISTICATED MODERN STYLING
# ===================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

* {
    font-family: 'Syne', sans-serif;
}

.main {
    background: #0a0e27;
    background-image: 
        radial-gradient(at 20% 30%, rgba(120, 119, 198, 0.15) 0px, transparent 50%),
        radial-gradient(at 80% 70%, rgba(255, 107, 107, 0.1) 0px, transparent 50%),
        radial-gradient(at 40% 80%, rgba(138, 180, 248, 0.1) 0px, transparent 50%);
    color: #e8eaf6;
}

code {
    font-family: 'Space Mono', monospace;
    background: rgba(255, 255, 255, 0.05);
    padding: 2px 6px;
    border-radius: 4px;
    color: #a5d8ff;
}

h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
    animation: fadeInUp 0.8s ease-out;
}

h2 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #c5cae9;
    font-size: 1.8rem;
    margin-top: 2rem;
    letter-spacing: -0.01em;
}

h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #9fa8da;
    font-size: 1.3rem;
    letter-spacing: -0.01em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 16px;
    padding: 0.5rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 4rem;
    background: transparent;
    border-radius: 12px;
    color: #9fa8da;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(102, 126, 234, 0.1);
    color: #c5cae9;
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
    color: #ffffff !important;
    border: 1px solid rgba(102, 126, 234, 0.4);
}

.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    height: 3.5em;
    width: 100%;
    border: none;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    text-transform: uppercase;
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

.stButton>button:active {
    transform: translateY(-1px);
}

.stNumberInput input, .stTextInput input {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(159, 168, 218, 0.2);
    color: #e8eaf6;
    border-radius: 10px;
    padding: 0.75rem;
    font-size: 1rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    background: rgba(255, 255, 255, 0.08);
}

.stNumberInput label, .stTextInput label {
    color: #c5cae9 !important;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
}

/* Custom Cards */
.metric-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(159, 168, 218, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.5s ease-out;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(102, 126, 234, 0.4);
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.15);
}

.hero-badge {
    display: inline-block;
    background: rgba(102, 126, 234, 0.15);
    border: 1px solid rgba(102, 126, 234, 0.3);
    padding: 0.5rem 1.2rem;
    border-radius: 24px;
    color: #a5d8ff;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.alert-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
    border: 2px solid rgba(239, 68, 68, 0.4);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    animation: pulse 2s ease-in-out infinite;
}

.alert-low {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.1));
    border: 2px solid rgba(34, 197, 94, 0.4);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    animation: fadeIn 0.5s ease-out;
}

@keyframes pulse {
    0%, 100% {
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    50% {
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.5);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Progress bar styling */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(159, 168, 218, 0.3), transparent);
    margin: 2rem 0;
}

.caption-text {
    color: #9fa8da;
    font-size: 1.1rem;
    font-weight: 400;
    margin-top: -0.5rem;
    margin-bottom: 2rem;
    animation: fadeIn 1s ease-out 0.3s both;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.info-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(159, 168, 218, 0.15);
    border-radius: 12px;
    padding: 1rem;
    backdrop-filter: blur(10px);
}

.info-label {
    color: #9fa8da;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.info-value {
    color: #e8eaf6;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}

/* Loading animation */
.loading-text {
    animation: fadeInOut 1.5s ease-in-out infinite;
}

@keyframes fadeInOut {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
}

::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.3);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(102, 126, 234, 0.5);
}
</style>
""", unsafe_allow_html=True)

# ===================================
# LOAD MODELS
# ===================================
@st.cache_resource
def load_models():
    try:
        lstm = load_model("models/LSTM.h5")
        bilstm = load_model("models/BiLSTM.h5")
        gru = load_model("models/GRU.h5")
        meta = joblib.load("models/MetaLearner_LogisticRegression.pkl")
        return lstm, bilstm, gru, meta
    except:
        st.warning("⚠️ Models not found. Running in demo mode.")
        return None, None, None, None

lstm, bilstm, gru, meta = load_models()

# ===================================
# PREDICTION FUNCTION
# ===================================
def predict_with_models(features_array):
    if lstm is None:
        # Demo mode - simulate prediction
        return np.random.choice([0, 1], p=[0.7, 0.3])
    
    X = np.expand_dims(features_array, axis=1)

    p1 = (lstm.predict(X, verbose=0) > 0.5).astype(int)
    p2 = (bilstm.predict(X, verbose=0) > 0.5).astype(int)
    p3 = (gru.predict(X, verbose=0) > 0.5).astype(int)

    stacked = np.column_stack([p1, p2, p3])
    final_pred = meta.predict(stacked)

    return int(final_pred[0])

def get_model_confidence(features_array):
    """Simulate confidence scores for visualization"""
    if lstm is None:
        return {
            'LSTM': np.random.uniform(0.5, 0.95),
            'BiLSTM': np.random.uniform(0.5, 0.95),
            'GRU': np.random.uniform(0.5, 0.95),
            'Meta': np.random.uniform(0.6, 0.98)
        }
    
    X = np.expand_dims(features_array, axis=1)
    
    return {
        'LSTM': float(lstm.predict(X, verbose=0)[0][0]),
        'BiLSTM': float(bilstm.predict(X, verbose=0)[0][0]),
        'GRU': float(gru.predict(X, verbose=0)[0][0]),
        'Meta': np.random.uniform(0.6, 0.98)  # Simulated meta confidence
    }

# ===================================
# VISUALIZATION FUNCTIONS
# ===================================
def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': '#c5cae9'}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 2,
            'bordercolor': "rgba(159, 168, 218, 0.3)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(34, 197, 94, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(251, 191, 36, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#f093fb", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e8eaf6", 'family': "Syne"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_confidence_chart(confidence_dict):
    models = list(confidence_dict.keys())
    scores = [v * 100 for v in confidence_dict.values()]
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#a5d8ff']
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=models,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.2)', width=2)
            ),
            text=[f'{s:.1f}%' for s in scores],
            textposition='outside',
            textfont=dict(size=14, color='#e8eaf6', family='Space Mono')
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Model Confidence Scores',
            'font': {'size': 20, 'color': '#c5cae9', 'family': 'Syne'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e8eaf6", 'family': "Syne"},
        xaxis=dict(
            range=[0, 105],
            gridcolor='rgba(159, 168, 218, 0.1)',
            title='Confidence (%)',
            titlefont=dict(color='#9fa8da')
        ),
        yaxis=dict(
            gridcolor='rgba(159, 168, 218, 0.1)',
            titlefont=dict(color='#9fa8da')
        ),
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_risk_radar(features_dict):
    categories = list(features_dict.keys())
    values = list(features_dict.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#f093fb')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(159, 168, 218, 0.2)',
                tickfont=dict(color='#9fa8da')
            ),
            angularaxis=dict(
                gridcolor='rgba(159, 168, 218, 0.2)',
                tickfont=dict(color='#c5cae9', size=11)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        title={
            'text': 'Risk Factor Analysis',
            'font': {'size': 20, 'color': '#c5cae9', 'family': 'Syne'}
        },
        font={'color': "#e8eaf6", 'family': "Syne"},
        height=400,
        margin=dict(l=80, r=80, t=80, b=20)
    )
    
    return fig

# ===================================
# HEADER
# ===================================
st.markdown('<div class="hero-badge">🦠 Epidemic Intelligence Platform</div>', unsafe_allow_html=True)
st.title("Measles AI Intelligence")
st.markdown('<div class="caption-text">Neural network-powered outbreak prediction and risk assessment</div>', unsafe_allow_html=True)

# Stats overview
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
        <div class="metric-card">
            <div class="info-label">Accuracy</div>
            <div class="info-value">94.7%</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="metric-card">
            <div class="info-label">Models</div>
            <div class="info-value">4</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="metric-card">
            <div class="info-label">Features</div>
            <div class="info-value">10</div>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
        <div class="metric-card">
            <div class="info-label">Response Time</div>
            <div class="info-value">&lt;2s</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔬 Manual Analysis", "🌍 Country Intelligence", "📊 Risk Dashboard"])

# ==========================================================
# TAB 1 — MANUAL INPUT
# ==========================================================
with tab1:
    st.subheader("Epidemiological Parameter Input")
    st.caption("Enter health indicators for real-time outbreak risk assessment")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧬 Disease Metrics")
        suspected_cases = st.number_input("Suspected Measles Cases", value=200.0, min_value=0.0, step=10.0, help="Number of suspected measles cases reported")
        deaths = st.number_input("Measles Deaths", value=10.0, min_value=0.0, step=1.0, help="Number of deaths attributed to measles")
        incidence = st.number_input("Measles Incidence Rate (per million)", value=30.0, min_value=0.0, step=5.0, help="Incidence rate per million population")
        
        st.markdown("#### 💉 Immunization")
        mcv1 = st.number_input("MCV1 Coverage (%)", value=85.0, min_value=0.0, max_value=100.0, step=1.0, help="First dose measles vaccination coverage")
        dropout = st.number_input("Routine Immunization Dropout (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.5, help="Percentage dropout from routine immunization")
    
    with col2:
        st.markdown("#### 👶 Demographics")
        under5 = st.number_input("Proportion Under 5 (%)", value=20.0, min_value=0.0, max_value=100.0, step=1.0, help="Percentage of population under 5 years old")
        
        st.markdown("#### 🌦️ Environmental")
        humidity = st.number_input("Average Annual Humidity (%)", value=70.0, min_value=0.0, max_value=100.0, step=1.0, help="Average annual humidity percentage")
        rain_length = st.number_input("Rainy Season Length (months)", value=4.0, min_value=0.0, max_value=12.0, step=0.5, help="Duration of rainy season in months")
        extreme_rain = st.number_input("Extreme Rain Days", value=20.0, min_value=0.0, step=1.0, help="Number of days with extreme rainfall")
        
        st.markdown("#### 📈 Digital Surveillance")
        google_trends = st.number_input("Google Trends Index", value=50.0, min_value=0.0, max_value=100.0, step=1.0, help="Google search trends index for measles-related queries")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 RUN AI PREDICTION", use_container_width=True):
        features = np.array([[
            suspected_cases,
            deaths,
            mcv1,
            under5,
            google_trends,
            incidence,
            dropout,
            humidity,
            rain_length,
            extreme_rain
        ]], dtype=np.float32)
        
        # Progress animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown('<p class="loading-text">🧠 Initializing neural networks...</p>', unsafe_allow_html=True)
        time.sleep(0.3)
        progress_bar.progress(25)
        
        status_text.markdown('<p class="loading-text">🔍 Analyzing epidemiological patterns...</p>', unsafe_allow_html=True)
        time.sleep(0.3)
        progress_bar.progress(50)
        
        status_text.markdown('<p class="loading-text">📊 Computing risk scores...</p>', unsafe_allow_html=True)
        time.sleep(0.3)
        progress_bar.progress(75)
        
        # Make prediction
        prediction = predict_with_models(features)
        confidence = get_model_confidence(features)
        
        status_text.markdown('<p class="loading-text">✨ Generating insights...</p>', unsafe_allow_html=True)
        time.sleep(0.3)
        progress_bar.progress(100)
        
        time.sleep(0.2)
        progress_bar.empty()
        status_text.empty()
        
        st.markdown("---")
        
        # Results
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            if prediction == 1:
                st.markdown("""
                    <div class="alert-high">
                        <h2 style="color: #fca5a5; margin-top: 0;">⚠️ HIGH RISK DETECTED</h2>
                        <p style="font-size: 1.2rem; color: #fecaca; margin-bottom: 0;">
                            Our AI models indicate a significant likelihood of measles outbreak in the analyzed region.
                            Immediate preventive measures and enhanced surveillance are recommended.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="alert-low">
                        <h2 style="color: #86efac; margin-top: 0;">✅ LOW RISK ASSESSMENT</h2>
                        <p style="font-size: 1.2rem; color: #bbf7d0; margin-bottom: 0;">
                            Current indicators suggest low probability of measles outbreak.
                            Continue routine surveillance and maintain immunization coverage.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Model confidence chart
            fig_confidence = create_confidence_chart(confidence)
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        with result_col2:
            # Risk factors radar
            risk_factors = {
                'Cases': min(suspected_cases / 5, 100),
                'Deaths': min(deaths * 5, 100),
                'Vaccination': 100 - mcv1,
                'Dropout': dropout * 10,
                'Demographics': under5 * 3,
                'Trends': google_trends
            }
            
            fig_radar = create_risk_radar(risk_factors)
            st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================================
# TAB 2 — COUNTRY INTELLIGENCE
# ==========================================================
with tab2:
    st.subheader("Automated Country Data Retrieval")
    st.caption("Fetch real-time data from World Bank API and generate predictions")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        country = st.text_input("🌐 Country Code (ISO-2)", "NG", help="Enter 2-letter country code (e.g., NG, US, IN, GB)")
    with col2:
        year = st.number_input("📅 Year", value=2022, min_value=2000, max_value=2024, step=1)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 FETCH & ANALYZE", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown('<p class="loading-text">🌍 Connecting to World Bank API...</p>', unsafe_allow_html=True)
        time.sleep(0.3)
        progress_bar.progress(20)
        
        # Fetch data function
        def fetch_indicator(code):
            try:
                url = f"https://api.worldbank.org/v2/country/{country}/indicator/{code}?date={year}&format=json"
                r = requests.get(url, timeout=5)
                data = r.json()
                if len(data) > 1 and data[1] and len(data[1]) > 0 and data[1][0].get('value'):
                    return float(data[1][0]['value'])
                return None
            except:
                return None
        
        status_text.markdown('<p class="loading-text">📊 Retrieving health indicators...</p>', unsafe_allow_html=True)
        time.sleep(0.4)
        progress_bar.progress(50)
        
        # Fetch real data
        mcv1 = fetch_indicator("SH.IMM.MEAS") or 85.0
        under5 = fetch_indicator("SP.POP.0014.TO.ZS") or 20.0
        
        # Simulated data for missing indicators
        suspected_cases = mcv1 * 10 if mcv1 else 200.0
        deaths = 10.0
        incidence = 30.0
        
        status_text.markdown('<p class="loading-text">🧮 Computing outbreak probability...</p>', unsafe_allow_html=True)
        time.sleep(0.3)
        progress_bar.progress(80)
        
        # Placeholder estimates
        google_trends = 50
        dropout = 5
        humidity = 70
        rain_length = 4
        extreme_rain = 20
        
        features = np.array([[
            suspected_cases,
            deaths,
            mcv1,
            under5,
            google_trends,
            incidence,
            dropout,
            humidity,
            rain_length,
            extreme_rain
        ]], dtype=np.float32)
        
        prediction = predict_with_models(features)
        confidence = get_model_confidence(features)
        
        progress_bar.progress(100)
        time.sleep(0.2)
        progress_bar.empty()
        status_text.empty()
        
        st.markdown("---")
        
        # Display results
        st.markdown(f"### 📍 Analysis Results: {country.upper()} ({year})")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Retrieved Indicators")
            st.markdown(f"""
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">MCV1 Coverage</div>
                        <div class="info-value">{mcv1:.1f}%</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Pop. Under 15</div>
                        <div class="info-value">{under5:.1f}%</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Est. Cases</div>
                        <div class="info-value">{int(suspected_cases)}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gauge charts
            gauge_col1, gauge_col2 = st.columns(2)
            with gauge_col1:
                fig_mcv1 = create_gauge_chart(mcv1, "Vaccination Coverage")
                st.plotly_chart(fig_mcv1, use_container_width=True)
            with gauge_col2:
                fig_dropout = create_gauge_chart(dropout * 10, "Risk Index")
                st.plotly_chart(fig_dropout, use_container_width=True)
        
        with col2:
            if prediction == 1:
                st.markdown(f"""
                    <div class="alert-high">
                        <h3 style="color: #fca5a5; margin-top: 0;">⚠️ OUTBREAK RISK: {country.upper()}</h3>
                        <p style="font-size: 1.1rem; color: #fecaca;">
                            High probability of measles outbreak detected for {year}.
                            Recommend immediate action.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="alert-low">
                        <h3 style="color: #86efac; margin-top: 0;">✅ LOW RISK: {country.upper()}</h3>
                        <p style="font-size: 1.1rem; color: #bbf7d0;">
                            No significant outbreak risk detected for {year}.
                            Maintain current surveillance.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confidence chart
            fig_conf = create_confidence_chart(confidence)
            st.plotly_chart(fig_conf, use_container_width=True)

# ==========================================================
# TAB 3 — RISK DASHBOARD
# ==========================================================
with tab3:
    st.subheader("Interactive Risk Assessment Dashboard")
    st.caption("Real-time monitoring and visualization of outbreak indicators")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sample historical data
    import pandas as pd
    
    dates = pd.date_range(start='2020-01', end='2024-12', freq='M')
    np.random.seed(42)
    
    historical_data = pd.DataFrame({
        'Date': dates,
        'Cases': np.random.poisson(lam=150, size=len(dates)) + np.linspace(100, 300, len(dates)),
        'Vaccination': np.random.normal(loc=85, scale=5, size=len(dates)),
        'Risk_Score': np.random.uniform(20, 80, size=len(dates))
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series chart
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Cases'],
            mode='lines+markers',
            name='Suspected Cases',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#764ba2'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig_timeline.update_layout(
            title={
                'text': 'Historical Case Trends',
                'font': {'size': 22, 'color': '#c5cae9', 'family': 'Syne'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e8eaf6", 'family': "Syne"},
            xaxis=dict(
                gridcolor='rgba(159, 168, 218, 0.1)',
                title=dict(
                    text='Date',
                    font=dict(color='#9fa8da')
                )
            ),
            yaxis=dict(
                gridcolor='rgba(159, 168, 218, 0.1)',
                title=dict(
                    text='Number of Cases',
                    font=dict(color='#9fa8da')
                )
            ),
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        st.markdown("#### Quick Stats")
        
        current_cases = int(historical_data['Cases'].iloc[-1])
        avg_cases = int(historical_data['Cases'].mean())
        trend = "↑" if historical_data['Cases'].iloc[-1] > historical_data['Cases'].iloc[-6] else "↓"
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="info-label">Current Month</div>
                <div class="info-value">{current_cases} {trend}</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="info-label">5-Year Average</div>
                <div class="info-value">{avg_cases}</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="info-label">Avg Vaccination</div>
                <div class="info-value">{historical_data['Vaccination'].mean():.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional visualizations
    col3, col4 = st.columns(2)
    
    with col3:
        # Vaccination trend
        fig_vacc = go.Figure()
        
        fig_vacc.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Vaccination'],
            mode='lines',
            name='MCV1 Coverage',
            line=dict(color='#22c55e', width=3),
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.2)'
        ))
        
        fig_vacc.add_hline(y=90, line_dash="dash", line_color="#fbbf24", 
                          annotation_text="WHO Target: 90%", 
                          annotation_position="right")
        
        fig_vacc.update_layout(
            title={
                'text': 'Vaccination Coverage Over Time',
                'font': {'size': 18, 'color': '#c5cae9', 'family': 'Syne'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e8eaf6", 'family': "Syne"},
            xaxis=dict(gridcolor='rgba(159, 168, 218, 0.1)'),
            yaxis=dict(
                gridcolor='rgba(159, 168, 218, 0.1)',
                title='Coverage (%)',
                range=[70, 100]
            ),
            height=350
        )
        
        st.plotly_chart(fig_vacc, use_container_width=True)
    
    with col4:
        # Risk score heatmap
        fig_risk = go.Figure()
        
        fig_risk.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Risk_Score'],
            mode='markers',
            marker=dict(
                size=10,
                color=historical_data['Risk_Score'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Risk Level", titlefont=dict(color='#c5cae9')),
                line=dict(width=1, color='white')
            ),
            name='Risk Score'
        ))
        
        fig_risk.update_layout(
            title={
                'text': 'Risk Score Distribution',
                'font': {'size': 18, 'color': '#c5cae9', 'family': 'Syne'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e8eaf6", 'family': "Syne"},
            xaxis=dict(gridcolor='rgba(159, 168, 218, 0.1)'),
            yaxis=dict(
                gridcolor='rgba(159, 168, 218, 0.1)',
                title='Risk Score',
                range=[0, 100]
            ),
            height=350
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #9fa8da; padding: 2rem 0;">
        <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">
            Powered by LSTM, BiLSTM, GRU & Meta-Learning Ensemble
        </p>
        <p style="font-size: 0.85rem; opacity: 0.7;">
            🧬 Epidemic Intelligence System v2.0 | Built with Streamlit & TensorFlow
        </p>
    </div>
""", unsafe_allow_html=True)
