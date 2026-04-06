"""
dashboard/app.py — ClearSignal Analytics
Cinematic, interactive UI with hero screen + scroll-reveal animations.
"""

import os, sys, json, requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="ClearSignal", layout="wide",
    initial_sidebar_state="collapsed" if not st.session_state.get("entered") else "expanded",
)

API_BASE    = "http://localhost:8000"
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics")

if "entered"    not in st.session_state: st.session_state["entered"]    = False
if "result"     not in st.session_state: st.session_state["result"]     = None
if "transcript" not in st.session_state: st.session_state["transcript"] = ""
if "page"       not in st.session_state: st.session_state["page"]       = "Operational"

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:       #03060d;
  --surface:  #07111e;
  --border:   #0d2035;
  --accent:   #00b4d8;
  --accent2:  #0077b6;
  --green:    #06d6a0;
  --amber:    #ffd166;
  --red:      #ef476f;
  --text:     #ccd6f6;
  --muted:    #3d5a80;
  --bright:   #e8f4fd;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
  background: var(--bg);
}
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: rgba(7,17,30,0.95) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] > div { padding: 2rem 1.5rem !important; }

/* ── Scroll animations ── */
.reveal {
  opacity: 0;
  transform: translateY(28px);
  transition: opacity 0.65s cubic-bezier(.22,.68,0,1.2), transform 0.65s cubic-bezier(.22,.68,0,1.2);
}
.reveal.visible { opacity: 1; transform: translateY(0); }
.reveal-left  { opacity:0; transform:translateX(-30px); transition: opacity 0.6s ease, transform 0.6s ease; }
.reveal-left.visible  { opacity:1; transform:translateX(0); }
.reveal-right { opacity:0; transform:translateX(30px);  transition: opacity 0.6s ease, transform 0.6s ease; }
.reveal-right.visible { opacity:1; transform:translateX(0); }

/* delays */
.d1 { transition-delay: 0.1s !important; }
.d2 { transition-delay: 0.2s !important; }
.d3 { transition-delay: 0.3s !important; }
.d4 { transition-delay: 0.4s !important; }

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.4rem 1.6rem;
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.card:hover {
  border-color: var(--accent2);
  box-shadow: 0 0 20px rgba(0,119,182,0.12);
}

/* ── Metric cards ── */
.m-label { font-size:0.65rem; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem; }
.m-value { font-family:'DM Mono',monospace; font-size:2.2rem; font-weight:500; color:var(--bright); line-height:1; }
.m-sub   { font-family:'DM Mono',monospace; font-size:0.72rem; color:var(--muted); margin-top:0.5rem; }

/* ── Badge ── */
.badge {
  display:inline-block; padding:0.22rem 0.7rem; border-radius:20px;
  font-size:0.67rem; font-weight:500; letter-spacing:0.08em; text-transform:uppercase;
  font-family:'DM Mono',monospace;
}
.badge-r { background:rgba(239,71,111,0.12); color:#ef476f; border:1px solid rgba(239,71,111,0.25); }
.badge-a { background:rgba(255,209,102,0.12); color:#ffd166; border:1px solid rgba(255,209,102,0.25); }
.badge-g { background:rgba(6,214,160,0.12);  color:#06d6a0; border:1px solid rgba(6,214,160,0.25); }

/* ── Section header ── */
.sec-header {
  font-size:0.65rem; letter-spacing:0.14em; text-transform:uppercase;
  color:var(--muted); padding-bottom:0.6rem;
  border-bottom:1px solid var(--border); margin-bottom:1.4rem;
}

/* ── Phrase tags ── */
.tag { display:inline-block; padding:0.25rem 0.65rem; border-radius:4px; font-size:0.72rem;
  font-family:'DM Mono',monospace; margin:0.2rem 0.15rem 0.2rem 0; }
.tag-p { background:rgba(6,214,160,0.08);  color:#06d6a0; border:1px solid rgba(6,214,160,0.2); }
.tag-n { background:rgba(239,71,111,0.08); color:#ef476f; border:1px solid rgba(239,71,111,0.2); }

/* ── Coaching box ── */
.coaching-box {
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:1.2rem 1.5rem; font-size:0.84rem; line-height:1.75; color:#8da9c4;
}

/* ── Transcript ── */
.transcript-box {
  background:#020810; border:1px solid var(--border); border-radius:6px;
  padding:1rem 1.2rem; font-family:'DM Mono',monospace; font-size:0.72rem;
  color:var(--muted); line-height:1.9; white-space:pre-wrap;
  max-height:280px; overflow-y:auto;
}

/* ── KPI bar ── */
.kpi-bar {
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:1rem 1.8rem; display:flex; align-items:center; gap:2.8rem; margin-bottom:1.6rem;
}
.kpi-v { font-family:'DM Mono',monospace; font-size:1.5rem; font-weight:400; color:var(--bright); }
.kpi-l { font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-top:0.2rem; }
.kpi-sep { width:1px; height:32px; background:var(--border); flex-shrink:0; }

/* ── Insight card ── */
.insight {
  background:var(--surface); border:1px solid var(--border);
  border-left:2px solid var(--accent2); border-radius:6px;
  padding:1rem 1.1rem; font-size:0.8rem; color:#8da9c4; line-height:1.65;
}

/* ── Page tabs in sidebar ── */
.nav-btn {
  display:block; width:100%; padding:0.6rem 0.9rem; margin-bottom:0.3rem;
  background:transparent; border:1px solid transparent; border-radius:5px;
  font-family:'DM Mono',monospace; font-size:0.75rem; color:var(--muted);
  cursor:pointer; text-align:left; transition:all 0.2s ease; letter-spacing:0.05em;
}
.nav-btn:hover  { background:var(--surface); border-color:var(--border); color:var(--text); }
.nav-btn.active { background:rgba(0,119,182,0.15); border-color:var(--accent2); color:var(--accent); }

/* ── Input styling ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextArea"] textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.78rem !important;
  transition: border-color 0.2s ease !important;
}
div[data-testid="stSelectbox"]:focus-within > div > div,
div[data-testid="stTextArea"]:focus-within textarea {
  border-color: var(--accent2) !important;
  box-shadow: 0 0 0 3px rgba(0,119,182,0.15) !important;
}
div[data-testid="stRadio"] label span { font-size:0.78rem !important; color:var(--muted) !important; }

/* ── Primary button ── */
div[data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(135deg, #0077b6, #00b4d8) !important;
  border: none !important; border-radius: 6px !important;
  font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important;
  letter-spacing: 0.05em !important; color: #fff !important; font-weight: 400 !important;
  transition: opacity 0.2s ease, transform 0.15s ease !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
  opacity: 0.88 !important; transform: translateY(-1px) !important;
}
div[data-testid="stButton"] button {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 6px !important; font-family: 'DM Mono', monospace !important;
  font-size: 0.75rem !important; color: var(--muted) !important;
  transition: border-color 0.2s, color 0.2s !important;
}
div[data-testid="stButton"] button:hover {
  border-color: var(--accent2) !important; color: var(--accent) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 8px !important; overflow:hidden; }
div[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:8px !important; }

/* status dot */
.dot-on  { color:#06d6a0; font-size:0.68rem; }
.dot-off { color:#ef476f; font-size:0.68rem; }

/* ── Expander ── */
details { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:6px !important; }
details summary { font-size:0.8rem !important; color:var(--muted) !important; }

/* Plotly */
.js-plotly-plot .plotly { border-radius:8px; }

/* padding inside main content area */
.main-wrap { padding: 2.5rem 3rem; }
</style>

<script>
// Scroll-reveal engine — runs once, re-checks on Streamlit re-render
function initReveal() {
  var obs = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting){ e.target.classList.add('visible'); }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach(function(el){
    if(!el.dataset.observed){ el.dataset.observed='1'; obs.observe(el); }
  });
}
// Run now and whenever DOM updates
initReveal();
var _mutObs = new MutationObserver(initReveal);
_mutObs.observe(document.body, {childList:true, subtree:true});
</script>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    paper_bgcolor="#07111e", plot_bgcolor="#07111e",
    font=dict(family="DM Sans", color="#3d5a80", size=11),
    margin=dict(l=10,r=10,t=36,b=10),
    xaxis=dict(gridcolor="#0d2035", linecolor="#0d2035", tickcolor="#0d2035", zeroline=False),
    yaxis=dict(gridcolor="#0d2035", linecolor="#0d2035", tickcolor="#0d2035", zeroline=False),
    title_font=dict(size=11, color="#3d5a80", family="DM Mono"),
    hoverlabel=dict(bgcolor="#0d2035", font_size=11, font_family="DM Mono"),
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_test_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df["_label"] = (
        df["call_id"].astype(str) + "  ·  " +
        df["issue_type"].str.replace("_"," ").str.title() + "  ·  " +
        df["csat_score"].round(1).astype(str)
    )
    return df

@st.cache_data
def load_metrics():
    try:
        return pd.read_csv(os.path.join(METRICS_DIR, "test_metrics_table.csv"))
    except Exception:
        return pd.DataFrame({
            "Model":     ["Naive Baseline","Ridge","Random Forest","DistilBERT","Ensemble"],
            "MAE":       [1.134,0.635,0.600,1.141,0.609],
            "RMSE":      [1.357,0.726,0.687,1.357,0.693],
            "Pearson r": ["—",0.845,0.862,-0.054,0.860],
            "F1 (≥3.0)":[0.798,0.863,0.863,0.798,0.863],
        })

@st.cache_data(ttl=60)
def load_agg():
    try:
        r = requests.get(f"{API_BASE}/aggregate", timeout=3)
        r.raise_for_status(); return r.json()
    except Exception:
        p = os.path.join(METRICS_DIR,"aggregate_stats.json")
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ISSUE_TYPES = ["billing_error","broadband","account_access","payment",
               "technical_support","roaming","contract","new_connection",
               "tv_service","general_enquiry"]

FEAT_LABELS = {
    "repeat_contact":"Repeat Contact","transfer_count":"Transfers",
    "empathy_density":"Empathy Density","last_20_sentiment":"Closing Sentiment",
    "mean_sentiment":"Mean Sentiment","std_sentiment":"Sentiment Volatility",
    "talk_time_ratio":"Agent Talk Ratio","apology_count":"Apologies",
    "resolution_flag":"Resolution Signal","interruption_count":"Interruptions",
    "duration_ordinal":"Duration","duration_deviation":"Duration vs Avg",
    "avg_agent_words":"Agent Words/Turn","avg_customer_words":"Customer Words/Turn",
}

ARC_DESC = {"rise":"Sentiment improved","fall":"Sentiment declined",
            "flat":"Consistent tone","v_shape":"Dipped then recovered"}

def sc(s):
    return "#ef476f" if s<2.5 else "#ffd166" if s<3.5 else "#06d6a0"

def badge(s):
    if s<2.5: return "AT RISK","badge-r"
    if s<3.5: return "SATISFACTORY","badge-a"
    return "EXCELLENT","badge-g"

def call_api(transcript, issue_type, call_duration, repeat_contact, resolution_status="resolved"):
    r = requests.post(f"{API_BASE}/predict",
        json={"transcript":transcript,"call_metadata":{"issue_type":issue_type,
              "call_duration":call_duration,"repeat_contact":int(repeat_contact),
              "resolution_status":resolution_status}},
        params={"skip_bert":"true"}, timeout=30)
    r.raise_for_status(); return r.json()

def mk_insights(agg):
    out=[]
    issue=agg.get("avg_csat_by_issue",{})
    if issue:
        avg=sum(issue.values())/len(issue); worst=min(issue,key=issue.get)
        pct=round((1-issue[worst]/avg)*100)
        out.append(f"{worst.replace('_',' ').title()} scores {pct}% below the portfolio average.")
    rc=agg.get("repeat_contact_rate_by_issue",{})
    if rc:
        top=max(rc,key=rc.get)
        out.append(f"{top.replace('_',' ').title()} has a {rc[top]:.0%} repeat-contact rate — first-call resolution is failing.")
    arcs=agg.get("avg_csat_by_arc",{})
    if "rise" in arcs and "fall" in arcs:
        out.append(f"Rising calls score {arcs['rise']-arcs['fall']:.1f} pts higher than falling — call endings drive satisfaction.")
    return out or ["No insights available."]

def wrap(html, cls="reveal", delay=""):
    d = f' class="{cls} {delay}"' if delay else f' class="{cls}"'
    return f'<div{d}>{html}</div>'


# ══════════════════════════════════════════════════════════════════════════════
# HERO SCREEN
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state["entered"]:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display:none !important; }
    .block-container { padding:0 !important; }

    .hero {
        position: relative;
        width: 100vw;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background: #03060d;
    }

    /* animated mesh background */
    .hero::before {
        content: '';
        position: absolute; inset: 0;
        background:
          radial-gradient(ellipse 80% 60% at 20% 40%, rgba(0,119,182,0.12) 0%, transparent 60%),
          radial-gradient(ellipse 60% 50% at 80% 60%, rgba(0,180,216,0.08) 0%, transparent 55%),
          radial-gradient(ellipse 40% 40% at 50% 80%, rgba(6,214,160,0.06) 0%, transparent 50%);
        animation: meshShift 12s ease-in-out infinite alternate;
    }

    @keyframes meshShift {
        0%   { opacity:0.6; transform:scale(1); }
        100% { opacity:1;   transform:scale(1.05); }
    }

    /* grid overlay */
    .hero::after {
        content: '';
        position: absolute; inset: 0;
        background-image:
          linear-gradient(rgba(0,180,216,0.035) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0,180,216,0.035) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: radial-gradient(ellipse 80% 80% at center, black 20%, transparent 80%);
    }

    .hero-inner {
        position: relative; z-index: 2;
        text-align: center; padding: 2rem;
        animation: heroFadeIn 1.2s cubic-bezier(.22,.68,0,1.1) both;
    }
    @keyframes heroFadeIn {
        from { opacity:0; transform:translateY(24px); }
        to   { opacity:1; transform:translateY(0); }
    }

    .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
        color: #0077b6; margin-bottom: 1.2rem;
        animation: heroFadeIn 1s 0.2s both;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(3.5rem, 8vw, 7rem);
        font-weight: 800; line-height: 0.95;
        color: #e8f4fd; letter-spacing: -0.03em;
        margin-bottom: 1.5rem;
        animation: heroFadeIn 1s 0.35s both;
    }
    .hero-title span {
        background: linear-gradient(135deg, #00b4d8, #06d6a0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .hero-sub {
        font-size: 1rem; font-weight: 300; color: #3d5a80;
        max-width: 440px; margin: 0 auto 2.8rem; line-height: 1.7;
        animation: heroFadeIn 1s 0.5s both;
    }

    .hero-stats {
        display: flex; gap: 3rem; justify-content: center; margin-bottom: 3rem;
        animation: heroFadeIn 1s 0.65s both;
    }
    .hero-stat-val {
        font-family: 'DM Mono', monospace; font-size: 1.6rem;
        color: #e8f4fd; font-weight: 400;
    }
    .hero-stat-lbl { font-size: 0.65rem; color: #3d5a80; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }

    .hero-sep { width: 1px; background: #0d2035; }

    .hero-btn-wrap { animation: heroFadeIn 1s 0.8s both; }

    /* floating data particles */
    .particle {
        position: absolute; border-radius: 50%;
        background: rgba(0,180,216,0.15);
        animation: float linear infinite;
        pointer-events: none;
    }
    @keyframes float {
        0%   { transform: translateY(100vh) scale(0); opacity:0; }
        10%  { opacity:1; }
        90%  { opacity:0.4; }
        100% { transform: translateY(-20vh) scale(1); opacity:0; }
    }
    </style>

    <div class="hero">
      <!-- floating particles -->
      <div class="particle" style="width:4px;height:4px;left:15%;animation-duration:14s;animation-delay:0s;"></div>
      <div class="particle" style="width:3px;height:3px;left:30%;animation-duration:18s;animation-delay:3s;background:rgba(6,214,160,0.2)"></div>
      <div class="particle" style="width:5px;height:5px;left:55%;animation-duration:11s;animation-delay:1s;"></div>
      <div class="particle" style="width:3px;height:3px;left:70%;animation-duration:16s;animation-delay:5s;background:rgba(6,214,160,0.15)"></div>
      <div class="particle" style="width:4px;height:4px;left:85%;animation-duration:13s;animation-delay:2s;"></div>
      <div class="particle" style="width:2px;height:2px;left:45%;animation-duration:20s;animation-delay:7s;"></div>

      <div class="hero-inner">
        <div class="hero-eyebrow">Call Centre Intelligence Platform</div>
        <div class="hero-title">Clear<span>Signal</span></div>
        <div class="hero-sub">
          Predict customer satisfaction from every call.<br>
          Understand what drives it. Act on it.
        </div>
        <div class="hero-stats">
          <div>
            <div class="hero-stat-val">0.86</div>
            <div class="hero-stat-lbl">Pearson r</div>
          </div>
          <div class="hero-sep"></div>
          <div>
            <div class="hero-stat-val">MAE 0.60</div>
            <div class="hero-stat-lbl">Accuracy</div>
          </div>
          <div class="hero-sep"></div>
          <div>
            <div class="hero-stat-val">1,500</div>
            <div class="hero-stat-lbl">Calls Analysed</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("Enter Dashboard →", type="primary", use_container_width=True):
            st.session_state["entered"] = True
            st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # brand
    st.markdown("""
    <div style="margin-bottom:1.8rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;
                  color:#e8f4fd;letter-spacing:-0.01em">ClearSignal</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#3d5a80;
                  letter-spacing:0.1em;text-transform:uppercase;margin-top:0.2rem">
        v1.0 · Analytics</div>
    </div>
    <div style="height:1px;background:var(--border, #0d2035);margin-bottom:1.5rem"></div>
    """, unsafe_allow_html=True)

    # page nav
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#3d5a80;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.7rem">Navigation</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Operational", use_container_width=True):
            st.session_state["page"] = "Operational"
    with col_b:
        if st.button("Technical", use_container_width=True):
            st.session_state["page"] = "Technical"

    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#0077b6;
                margin:0.4rem 0 1.4rem;letter-spacing:0.05em">
      ▸ {st.session_state['page']}
    </div>
    <div style="height:1px;background:var(--border,#0d2035);margin-bottom:1.5rem"></div>
    """, unsafe_allow_html=True)

    # input panel
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#3d5a80;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.9rem">Call Input</div>', unsafe_allow_html=True)

    mode = st.radio("", ["From test set","Custom transcript"], label_visibility="collapsed")

    if mode == "From test set":
        df = load_test_data()
        c1,c2 = st.columns([1,5])
        with c1:
            st.write("")
            if st.button("⟳", help="Random"):
                st.session_state["rand"] = df.sample(1).iloc[0].to_dict()
        with c2:
            default = st.session_state.get("rand",{}).get("_label", df["_label"].iloc[0])
            idx = int(df[df["_label"]==default].index[0]) if default in df["_label"].values else 0
            sel = st.selectbox("", df["_label"], index=idx, label_visibility="collapsed")

        row  = df[df["_label"]==sel].iloc[0]
        transcript    = str(row.get("transcript_text",""))
        issue_type    = str(row.get("issue_type","billing_error"))
        secs          = row.get("call_duration_seconds",300)
        call_duration = "short" if secs<=240 else "long" if secs>390 else "medium"
        _rc           = str(row.get("repeat_contact","0")).strip().lower()
        repeat_contact= 1 if _rc in("yes","true","1","1.0") else 0
        res_status    = str(row.get("resolution_status","resolved"))

        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.67rem;color:#3d5a80;
                    line-height:1.9;margin-top:0.5rem;padding:0.6rem 0.8rem;
                    background:#07111e;border:1px solid #0d2035;border-radius:5px">
          <span style="color:#0077b6">{issue_type.replace("_"," ").title()}</span>
          &nbsp;·&nbsp; {call_duration} ({int(secs)}s)
          &nbsp;·&nbsp; repeat: {"yes" if repeat_contact else "no"}
          &nbsp;·&nbsp; <span style="color:#ccd6f6">CSAT {row.get("csat_score","?")}</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        transcript     = st.text_area("", height=100,
                            placeholder="Turn 1: AGENT: Hello...\nTurn 2: CUSTOMER: Hi...",
                            label_visibility="collapsed")
        issue_type     = st.selectbox("Issue type", ISSUE_TYPES)
        call_duration  = st.selectbox("Duration", ["short","medium","long"])
        repeat_contact = st.selectbox("Repeat contact", [0,1],
                            format_func=lambda x: "Yes" if x else "No")
        res_status     = "resolved"

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Analysis", type="primary", use_container_width=True)
    if run:
        if not transcript.strip():
            st.error("Transcript is empty.")
        else:
            with st.spinner(""):
                try:
                    res = call_api(transcript, issue_type, call_duration, repeat_contact, res_status)
                    st.session_state["result"]     = res
                    st.session_state["transcript"] = transcript
                    st.markdown(f"""
                    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                                color:#06d6a0;margin-top:0.5rem;text-align:center">
                      ✓ CSAT {res['csat_score']:.1f} · {res.get('inference_time','—')}s
                    </div>""", unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.error("API offline.\n\n`python -m uvicorn src.api.main:app --reload`")
                except Exception as e:
                    st.error(str(e))

    st.markdown("""
    <div style="height:1px;background:#0d2035;margin:1.5rem 0 1rem"></div>
    """, unsafe_allow_html=True)

    try:
        requests.get(f"{API_BASE}/aggregate", timeout=1)
        st.markdown('<span class="dot-on">● API connected</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="dot-off">● API offline</span>', unsafe_allow_html=True)

    if st.button("← Back to home", use_container_width=False):
        st.session_state["entered"] = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

page = st.session_state["page"]

st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OPERATIONAL
# ══════════════════════════════════════════════════════════════════════════════

if page == "Operational":

    # Page header
    st.markdown(wrap("""
    <div style="margin-bottom:2.5rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:700;
                  color:#e8f4fd;letter-spacing:-0.02em;margin-bottom:0.4rem">
        Operational Overview
      </div>
      <div style="font-size:0.82rem;color:#3d5a80">
        Per-call prediction · Coaching signals · Portfolio health
      </div>
    </div>
    """, "reveal"), unsafe_allow_html=True)

    # ── Call result ────────────────────────────────────────────────────────────
    if st.session_state["result"]:
        res   = st.session_state["result"]
        score = res["csat_score"]
        arc   = res.get("emotional_arc","flat")
        ci    = res.get("confidence_interval",[score-0.4,score+0.4])
        lbl,bcls = badge(score)
        col   = sc(score)
        t     = res.get("inference_time","—")

        st.markdown(wrap('<div class="sec-header">Call Result</div>', "reveal"), unsafe_allow_html=True)

        # metric cards
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(wrap(f"""
            <div class="card">
              <div class="m-label">Predicted CSAT</div>
              <div class="m-value" style="color:{col}">{score:.1f}</div>
              <div class="m-sub">/ 5.0 &nbsp;·&nbsp; CI {ci[0]}–{ci[1]}</div>
            </div>""","reveal d1"), unsafe_allow_html=True)
        with c2:
            st.markdown(wrap(f"""
            <div class="card">
              <div class="m-label">Outcome</div>
              <div style="margin-top:0.8rem"><span class="badge {bcls}">{lbl}</span></div>
            </div>""","reveal d2"), unsafe_allow_html=True)
        with c3:
            st.markdown(wrap(f"""
            <div class="card">
              <div class="m-label">Emotional Arc</div>
              <div class="m-value" style="font-size:1.4rem;padding-top:0.3rem">
                {arc.replace("_"," ").title()}
              </div>
              <div class="m-sub">{ARC_DESC.get(arc,"—")}</div>
            </div>""","reveal d3"), unsafe_allow_html=True)
        with c4:
            st.markdown(wrap(f"""
            <div class="card">
              <div class="m-label">Inference</div>
              <div class="m-value" style="font-size:1.4rem;padding-top:0.3rem">{t}s</div>
              <div class="m-sub">CPU · Ridge + RF</div>
            </div>""","reveal d4"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # coaching
        st.markdown(wrap('<div class="sec-header">Coaching</div>',"reveal"), unsafe_allow_html=True)
        left, right = st.columns([1,2])

        with left:
            pos = res.get("top_positive_phrases",[])
            neg = res.get("top_negative_phrases",[])
            tags = "".join(f'<span class="tag tag-p">{p}</span>' for p in pos)
            tags += "".join(f'<span class="tag tag-n">{n}</span>' for n in neg)
            st.markdown(wrap(
                f'<div style="line-height:2.4">{tags or "<span style=\'color:#3d5a80;font-size:0.8rem\'>No phrases extracted.</span>"}</div>',
                "reveal-left"), unsafe_allow_html=True)

        with right:
            coaching = res.get("coaching_summary","")
            st.markdown(wrap(
                f'<div class="coaching-box">{coaching or "No coaching summary available."}</div>',
                "reveal-right"), unsafe_allow_html=True)

        with st.expander("View full transcript"):
            st.markdown(
                f'<div class="transcript-box">{st.session_state.get("transcript","")}</div>',
                unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:3rem 0;text-align:center;color:#1e3a5f;
                    font-family:'DM Mono',monospace;font-size:0.8rem;letter-spacing:0.08em">
          SELECT A CALL AND RUN ANALYSIS TO BEGIN
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    st.markdown(wrap('<div class="sec-header">Portfolio Trends</div>',"reveal"), unsafe_allow_html=True)

    agg = load_agg()
    if agg is None:
        st.info("Run `python scripts/generate_aggregate_stats.py` to generate aggregate data.")
    else:
        oa  = agg.get("overall_avg_csat","—")
        tot = agg.get("total_calls_in_training","—")
        dist= agg.get("csat_distribution",{})
        risk= dist.get("low_1_to_2.5","—")
        exc = dist.get("high_3.5_to_5","—")

        st.markdown(wrap(f"""
        <div class="kpi-bar">
          <div><div class="kpi-v">{oa}</div><div class="kpi-l">Avg CSAT</div></div>
          <div class="kpi-sep"></div>
          <div><div class="kpi-v">{tot}</div><div class="kpi-l">Training Calls</div></div>
          <div class="kpi-sep"></div>
          <div><div class="kpi-v" style="color:#ef476f">{risk}%</div><div class="kpi-l">At Risk</div></div>
          <div class="kpi-sep"></div>
          <div><div class="kpi-v" style="color:#06d6a0">{exc}%</div><div class="kpi-l">Excellent</div></div>
        </div>""","reveal"), unsafe_allow_html=True)

        col1,col2,col3 = st.columns(3)

        with col1:
            issue = agg.get("avg_csat_by_issue",{})
            if issue:
                idf = pd.DataFrame(issue.items(),columns=["Issue","CSAT"]).sort_values("CSAT")
                idf["Issue"] = idf["Issue"].str.replace("_"," ").str.title()
                idf["col"] = idf["CSAT"].apply(lambda x:"#ef476f" if x<2.5 else "#ffd166" if x<3.5 else "#06d6a0")
                fig = go.Figure(go.Bar(x=idf["CSAT"],y=idf["Issue"],orientation="h",
                                       marker_color=idf["col"],marker_line_width=0))
                fig.update_layout(**PLOT_THEME,title="CSAT by Issue Type",height=310,xaxis_range=[1,5])
                st.markdown(wrap("", "reveal d1"), unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            arc_d = agg.get("avg_csat_by_arc",{})
            if arc_d:
                adf = pd.DataFrame(arc_d.items(),columns=["Arc","CSAT"])
                adf["Arc"] = adf["Arc"].str.replace("_"," ").str.title()
                adf["col"] = ["#06d6a0","#ef476f","#3d5a80","#00b4d8"][:len(adf)]
                fig2 = go.Figure(go.Bar(x=adf["Arc"],y=adf["CSAT"],
                                        marker_color=adf["col"],marker_line_width=0))
                fig2.update_layout(**PLOT_THEME,title="CSAT by Emotional Arc",height=310,yaxis_range=[1,5])
                st.markdown(wrap("","reveal d2"), unsafe_allow_html=True)
                st.plotly_chart(fig2, use_container_width=True)

        with col3:
            rc = agg.get("repeat_contact_rate_by_issue",{})
            if rc:
                rdf = pd.DataFrame(rc.items(),columns=["Issue","Rate"]).sort_values("Rate",ascending=False)
                rdf["Issue"] = rdf["Issue"].str.replace("_"," ").str.title()
                rdf["Pct"]   = (rdf["Rate"]*100).round(1)
                fig3 = go.Figure(go.Bar(x=rdf["Issue"],y=rdf["Pct"],
                                        marker_color="#0077b6",marker_line_width=0))
                fig3.update_layout(**PLOT_THEME,title="Repeat Contact Rate (%)",
                                   height=310,xaxis_tickangle=-30)
                st.markdown(wrap("","reveal d3"), unsafe_allow_html=True)
                st.plotly_chart(fig3, use_container_width=True)

        # insights
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(wrap('<div class="sec-header">Key Insights</div>',"reveal"), unsafe_allow_html=True)
        ins = mk_insights(agg)
        cols = st.columns(len(ins))
        for i,c in enumerate(cols):
            with c:
                st.markdown(wrap(f'<div class="insight">{ins[i]}</div>',
                                  f"reveal d{i+1}"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TECHNICAL
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Technical":

    st.markdown(wrap("""
    <div style="margin-bottom:2.5rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:700;
                  color:#e8f4fd;letter-spacing:-0.02em;margin-bottom:0.4rem">
        Technical Analysis
      </div>
      <div style="font-size:0.82rem;color:#3d5a80">
        Emotion trajectory · Feature contributions · Model evaluation
      </div>
    </div>""","reveal"), unsafe_allow_html=True)

    if not st.session_state["result"]:
        st.markdown("""
        <div style="padding:3rem 0;text-align:center;color:#1e3a5f;
                    font-family:'DM Mono',monospace;font-size:0.8rem;letter-spacing:0.08em">
          SELECT A CALL AND RUN ANALYSIS TO BEGIN
        </div>""", unsafe_allow_html=True)
    else:
        res  = st.session_state["result"]
        arc  = res.get("emotional_arc","flat")
        cl,cr = st.columns(2)

        with cl:
            st.markdown(wrap('<div class="sec-header">Emotion Trajectory</div>',"reveal"), unsafe_allow_html=True)
            series = res.get("sentiment_series",[])
            if series and len(series)>1:
                n  = len(series)
                x  = [round(i/(n-1)*100) for i in range(n)]
                lc = {"rise":"#06d6a0","fall":"#ef476f","v_shape":"#00b4d8","flat":"#3d5a80"}.get(arc,"#3d5a80")
                fc = {"rise":"rgba(6,214,160,0.06)","fall":"rgba(239,71,111,0.06)",
                      "v_shape":"rgba(0,180,216,0.06)","flat":"rgba(61,90,128,0.06)"}.get(arc,"rgba(61,90,128,0.06)")
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=x, y=series, mode="lines",
                    line=dict(color=lc, width=1.5), fill="tozeroy", fillcolor=fc, name="Sentiment"))
                fig_e.add_hline(y=0, line_dash="dot", line_color="#0d2035")
                fig_e.add_annotation(x=x[-1],y=series[-1],
                    text=arc.replace("_"," ").upper(),
                    font=dict(size=9,color=lc,family="DM Mono"),showarrow=False,xanchor="right")
                fig_e.update_layout(**PLOT_THEME, height=290, showlegend=False,
                    xaxis_title="Call Progress (%)", yaxis_title="Sentiment",
                    yaxis_range=[-1.1,1.1])
                st.markdown(wrap("","reveal-left"), unsafe_allow_html=True)
                st.plotly_chart(fig_e, use_container_width=True)
            else:
                st.markdown('<div style="color:#3d5a80;font-size:0.8rem">Sentiment series unavailable.</div>',
                            unsafe_allow_html=True)

        with cr:
            st.markdown(wrap('<div class="sec-header">Feature Contributions (SHAP)</div>',"reveal"), unsafe_allow_html=True)
            shap = res.get("shap_features",{})
            if shap:
                sdf = pd.DataFrame([
                    {"Feature":FEAT_LABELS.get(k,k.replace("_"," ").title()),
                     "Val":v, "Col":"#06d6a0" if v>=0 else "#ef476f"}
                    for k,v in shap.items()
                ]).sort_values("Val")
                fig_s = go.Figure(go.Bar(x=sdf["Val"],y=sdf["Feature"],orientation="h",
                                          marker_color=sdf["Col"],marker_line_width=0))
                fig_s.add_vline(x=0,line_color="#0d2035",line_width=1)
                fig_s.update_layout(**PLOT_THEME, height=290, showlegend=False,
                                    xaxis_title="SHAP contribution")
                st.markdown(wrap("","reveal-right"), unsafe_allow_html=True)
                st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # model table
    st.markdown(wrap('<div class="sec-header">Model Performance — Held-Out Test Set (225 calls)</div>',"reveal"),
                unsafe_allow_html=True)

    mdf = load_metrics()
    def style_tbl(row):
        n = str(row.get("Model",""))
        if "Ensemble" in n:   return ["background:#0a1f33;color:#e8f4fd;font-weight:600"]*len(row)
        if "Naive" in n or "Baseline" in n: return ["color:#3d5a80"]*len(row)
        return [""]*len(row)

    st.markdown(wrap("","reveal"), unsafe_allow_html=True)
    st.dataframe(mdf.style.apply(style_tbl,axis=1), use_container_width=True, hide_index=True)
    st.markdown('<div style="font-size:0.7rem;color:#3d5a80;margin-top:0.4rem">Labels not seen during training. Ensemble highlighted.</div>',
                unsafe_allow_html=True)

    with st.expander("Why does DistilBERT underperform Ridge?"):
        st.markdown("""
DistilBERT (Pearson r = −0.054) underperforms because the transcripts are synthetic
placeholder text. Attention heads have nothing real to learn from.

**Finding:** structural and metadata features carry all predictive signal.
Ridge and RF achieve r ≈ 0.86 without any NLP. CSAT can be predicted from
call metadata alone — in real time, without text processing overhead.
        """)

st.markdown('</div>', unsafe_allow_html=True)