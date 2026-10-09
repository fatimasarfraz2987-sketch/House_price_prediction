import base64
import io
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
APP_NAME = "AI Powered House Price Predictor"                      
TAGLINE = "Smart property price estimates for Pakistan."
MODEL_PATH = Path(__file__).resolve().parent / "Random_forest_model.pkl"
SCALER_PATH = Path(__file__).resolve().parent / "scaler.pkl"
FEATURES_PATH = Path(__file__).resolve().parent / "feature_columns.pkl" 

CITY_INFO = {
    "Islamabad": (33.721484, 73.043289, "Islamabad Capital"),
    "Karachi": (24.860800, 67.010400, "Sindh"),
    "Lahore": (31.557996, 74.350713, "Punjab"),
    "Rawalpindi": (33.597331, 73.047904, "Punjab"),
    "Faisalabad": (31.415536, 73.089693, "Punjab"),
}
LOG_TARGET = False

PROPERTY_TYPES = ["House", "Flat", "Upper Portion", "Lower Portion", "Room", "Farm House"]
CITIES = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad"]

BASE_DIR = Path(__file__).resolve().parent   
HERO_DIR = Path("house_images")
HERO_FILES = ["House.png", "Room.png", "Farmhouse.png"]
HERO_IMAGE_URLS: list[str] = []
SLIDE_SECONDS = 5

AUTHOR = {
    "name": "Fatima Sarfraz",
    "role": "Machine Learning Engineer",
    "github": "https://github.com/fatimasarfraz2987-sketch",
    "email": "fatimasarfraz2987@gmail.com",
    "repo": "https://github.com/fatimasarfraz2987-sketch/house-price-prediction",
}

st.set_page_config(
    page_title=f"{APP_NAME} | House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Source+Sans+3:wght@400;500;600&display=swap');

:root{
  --ink:#16233A; --ink-soft:#586377; --paper:#F7F4EE; --card:#FFFFFF;
  --pine:#13294B; --pine-2:#1F3D6B; --marigold:#C9A24B; --line:#E2DCCF; --mist:#EEE9DE;
}

/* Hide Streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="collapsedControl"]{display:none !important;}

html, body, .stApp{background:var(--paper) !important; color:var(--ink); font-family:'Source Sans 3',sans-serif;}
.block-container{max-width:1120px; padding:0 1.25rem 0 1.25rem !important;}
h1,h2,h3,h4{font-family:'Bricolage Grotesque',sans-serif !important; color:var(--ink) !important; letter-spacing:-0.01em;}
p, li, label, span{font-family:'Source Sans 3',sans-serif;}

/* Navbar */
.nav{display:flex; align-items:center; justify-content:space-between; gap:1rem;
     padding:1rem 0; border-bottom:1px solid var(--line); flex-wrap:wrap;}
.brand{font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.4rem;
       color:var(--pine) !important; text-decoration:none !important; display:flex; align-items:center; gap:.5rem;}
.brand-mark{width:30px; height:30px; background:var(--pine); border-radius:8px 8px 8px 2px;
            display:inline-flex; align-items:center; justify-content:center; font-size:1rem;}
.nav-links{display:flex; gap:.35rem; align-items:center; flex-wrap:wrap;}
.nav-links a{color:var(--ink-soft) !important; text-decoration:none !important; font-weight:600;
             padding:.5rem .9rem; border-radius:999px; transition:background .15s, color .15s;}
.nav-links a:hover{background:var(--mist); color:var(--pine) !important;}
.nav-links a.active{background:var(--pine); color:#fff !important;}
.nav-links a.cta{background:var(--marigold); color:var(--ink) !important;}
.nav-links a:focus-visible, .btn:focus-visible, .foot a:focus-visible{outline:3px solid var(--marigold); outline-offset:2px;}

/* Hero slideshow */
.hero-wrap{position:relative; height:clamp(420px,64vh,620px); border-radius:24px; overflow:hidden;
           margin:1.5rem 0 1rem; isolation:isolate; background:var(--pine);}
.hero-slide{position:absolute; inset:0; background-size:cover; background-position:center; opacity:0;}
.hero-slide:first-child{opacity:1;}
.hero-shade{position:absolute; inset:0; z-index:1;
            background:linear-gradient(90deg, rgba(10,22,44,.88) 0%, rgba(10,22,44,.55) 50%, rgba(10,22,44,.08) 100%);}
.hero-text{position:absolute; inset:0; z-index:2; display:flex; flex-direction:column; justify-content:center;
           padding:0 clamp(1.2rem,5vw,4rem); max-width:46rem; color:#fff;}
.hero-text h1{color:#fff !important; font-size:clamp(2rem,5vw,3.5rem); line-height:1.07; font-weight:800; margin:0 0 1rem;}
.hero-text p.lead{font-size:1.15rem; color:#E6EAF2; max-width:34rem; margin:0 0 1.6rem; line-height:1.55;}
.btn{display:inline-block; padding:.8rem 1.4rem; border-radius:12px; font-weight:600; text-decoration:none !important;
     background:var(--pine); color:#fff !important; margin-right:.6rem; transition:transform .15s, background .15s;}
.btn:hover{background:var(--pine-2); transform:translateY(-1px);}
.btn.ghost{background:transparent; color:var(--pine) !important; border:1.5px solid var(--pine);}
.btn.gold{background:var(--marigold); color:var(--ink) !important;}
.btn.gold:hover{background:#D8B25C;}
.btn.light{background:transparent; color:#fff !important; border:1.5px solid rgba(255,255,255,.8);}
.btn.light:hover{background:rgba(255,255,255,.14);}
.type-row{display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.2rem;}
.type-chip{background:var(--card); border:1px solid var(--line); border-radius:999px; padding:.4rem .9rem; color:var(--ink-soft); font-weight:500;}

/* Sections */
.section{padding:2rem 0;}
.section h2{font-size:clamp(1.6rem,3vw,2.1rem); margin-bottom:.4rem;}
.sub{color:var(--ink-soft); margin-bottom:1.4rem; max-width:40rem;}
.grid3{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;}
.card{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.3rem;}
.card h3{font-size:1.15rem; margin:.3rem 0 .4rem;}
.card p{color:var(--ink-soft); margin:0; line-height:1.5;}
.step-n{display:inline-flex; width:2rem; height:2rem; border-radius:50%; background:var(--marigold);
        align-items:center; justify-content:center; font-weight:700; color:var(--ink);}
.page-title{padding:2.2rem 0 .5rem;}
.page-title h1{font-size:clamp(2rem,4vw,2.8rem); margin:0 0 .3rem;}
.page-title p{color:var(--ink-soft); font-size:1.08rem;}

/* Result */
.result{background:var(--pine); color:#fff; border-radius:20px; padding:1.8rem; margin-top:1rem;}
.result .lbl{opacity:.75; margin:0;}
.result .price{font-family:'Bricolage Grotesque',sans-serif; font-size:clamp(2rem,5vw,3rem); font-weight:800; color:var(--marigold); margin:.1rem 0;}
.result .range{opacity:.9; margin:0 0 1rem;}
.stat-row{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.7rem;}
.stat{background:rgba(255,255,255,.1); border-radius:12px; padding:.7rem .9rem;}
.stat b{display:block; font-size:1.05rem;}
.stat span{opacity:.75; font-size:.85rem;}

/* Streamlit widgets */
[data-testid="stForm"]{background:var(--card); border:1px solid var(--line) !important; border-radius:20px; padding:1.4rem;}
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input{border-radius:10px !important;}
.stButton > button, [data-testid="stFormSubmitButton"] > button{
  background:var(--pine); color:#fff; border:0; border-radius:12px; padding:.7rem 1.4rem; font-weight:600; width:100%;}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover{background:var(--pine-2); color:#fff;}

/* Footer */
.foot{margin-top:3rem; background:var(--pine); color:#D5DCE8; border-radius:24px 24px 0 0; padding:2.2rem 1.8rem 1.2rem;}
.foot-grid{display:grid; grid-template-columns:1.6fr 1fr 1fr 1.2fr; gap:2rem;}
.foot h4{color:#fff !important; font-size:1rem; margin:0 0 .7rem;}
.foot a{color:#D5DCE8 !important; text-decoration:none !important; display:block; padding:.18rem 0;}
.foot a:hover{color:var(--marigold) !important; text-decoration:underline !important;}
.foot p{margin:0 0 .5rem; line-height:1.5; font-size:.95rem;}
.foot-left{display:flex; flex-direction:column; gap:.25rem;}
.foot-bottom{align-items:flex-end;}
.foot-author{margin-left:auto; text-align:right; line-height:1.3;}
.foot-author .a-name{display:block; color:#fff !important; font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1rem;}
.foot-author a.a-mail{display:block; font-size:.72rem; padding:0; opacity:.85;}
/* Responsive */
@media (max-width:860px){
  .hero-shade{background:linear-gradient(180deg, rgba(10,22,44,.8) 0%, rgba(10,22,44,.65) 100%);}
  .grid3{grid-template-columns:1fr;}
  .foot-grid{grid-template-columns:1fr 1fr;}
}
@media (max-width:520px){
  .nav{justify-content:center;}
  .foot-grid{grid-template-columns:1fr;}
  .btn{margin-bottom:.6rem;}
}
@media (prefers-reduced-motion:reduce){*{transition:none !important;} .hero-slide{animation:none !important;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render raw HTML. Leading whitespace is stripped so Markdown doesn't treat it as a code block."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 3. MODEL + PREDICTION HELPERS
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, scaler and training columns once per server process."""
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURES_PATH.exists()):
        return None
    return {
        "model": joblib.load(MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "columns": list(joblib.load(FEATURES_PATH)),
    }


def area_category(area_marla: float) -> str | None:
    """Same bins as training. Returns None for the dropped baseline (0-5 Marla)."""
    if area_marla <= 5:
        return None
    if area_marla <= 20:                               # Marla bins
        for lo, hi in [(5, 10), (10, 15), (15, 20)]:
            if lo < area_marla <= hi:
                return f"{lo}-{hi} Marla"
    kanal = area_marla / 20
    edges = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo < kanal <= hi:
            return f"{lo}-{hi} Kanal"
    return None


def build_model_input(row: dict, columns: list, scaler) -> pd.DataFrame:
    """Rebuild every training feature from the form values, in the training column order."""
    lat0, lon0, province = CITY_INFO[row["city"]]
    area = row["area"]                                  
    is_marla = area <= 20                               # 1 Kanal = 20 Marla
    today = datetime.now()

    values = dict.fromkeys(columns, 0.0)                
    values.update({
        "latitude": lat0,
        "longitude": lon0,
        "baths": row["bathrooms"],
        "bedrooms": row["bedrooms"],
        "Area Size": area if is_marla else area / 20,  
        "Area_Marla": area,
        "year": today.year, "month": today.month, "day": today.day,
        "Area Type_Marla": 1.0 if is_marla else 0.0,
    })
    one_hot = [f"property_type_{row['property_type']}", f"city_{row['city']}",
               f"province_name_{province}"]
    category = area_category(area)
    if category:
        one_hot.append(f"Area Category_{category}")
    for name in one_hot:
        if name in values:                             
            values[name] = 1.0

    X = pd.DataFrame([values])[columns].astype(float)
    if hasattr(scaler, "feature_names_in_"):            
        cols = list(scaler.feature_names_in_)
        X[cols] = scaler.transform(X[cols])
    else:
        X = pd.DataFrame(scaler.transform(X), columns=columns)
    return X


def demo_estimate(row: dict) -> float:
    """Rough rule-of-thumb used ONLY when model.pkl is not found, so the UI still works."""
    per_marla = {"Lahore": 4.0e6, "Karachi": 3.2e6, "Islamabad": 5.0e6, "Rawalpindi": 3.0e6, "Faisalabad": 2.2e6}
    type_mult = {"House": 1.0, "Flat": 0.7, "Upper Portion": 0.55, "Lower Portion": 0.55,
                 "Room": 0.2, "Penthouse": 1.2, "Farm House": 1.4}
    base = per_marla.get(row["city"], 3e6) * row["area"] * type_mult.get(row["property_type"], 1.0)
    return base * (1 + 0.03 * row["bedrooms"] + 0.02 * row["bathrooms"])


def predict_price(row: dict) -> tuple[float, bool]:
    """Return (price_in_PKR, used_real_model)."""
    art = load_artifacts()
    if art is None:
        return demo_estimate(row), False
    X = build_model_input(row, art["columns"], art["scaler"])
    value = float(art["model"].predict(X)[0])
    if LOG_TARGET:
        import numpy as np
        value = float(np.expm1(value))
    return value, True


def fmt_pkr(value: float) -> str:
    if value >= 1e7:
        return f"PKR {value / 1e7:,.2f} Crore"
    if value >= 1e5:
        return f"PKR {value / 1e5:,.2f} Lakh"
    return f"PKR {value:,.0f}"

PAGES = {"home": "Home", "predict": "Predict", "about": "About"}


def navbar(active: str) -> None:
    links = "".join(
        f'<a href="?page={key}" target="_self" class="{"active" if key == active else ""}">{label}</a>'
        for key, label in PAGES.items() if key != "predict"
    )
    html(f"""
    <nav class="nav" aria-label="Main navigation">
      <a class="brand" href="?page=home" target="_self"><span class="brand-mark">🏠</span>{APP_NAME}</a>
      <div class="nav-links">
        {links}
        <a href="?page=predict" target="_self" class="cta">Get a price estimate</a>
      </div>
    </nav>
    """)


def footer() -> None:
    year = datetime.now().year
    html(f"""
    <footer class="foot">
      <div class="foot-grid">
        <div>
          <h4>{APP_NAME}</h4>
          <p>{TAGLINE}</p>
          <p>Built with Python, scikit-learn and Streamlit.</p>
        </div>
        <div>
          <h4>Pages</h4>
          <a href="?page=home" target="_self">Home</a>
          <a href="?page=predict" target="_self">Predict price</a>
          <a href="?page=about" target="_self">About</a>
        </div>
        <div>
          <h4>Resources</h4>
          <a href="{AUTHOR['repo']}" target="_blank" rel="noopener">Source code</a>
          <a href="https://docs.streamlit.io" target="_blank" rel="noopener">Streamlit docs</a>
          <a href="https://scikit-learn.org" target="_blank" rel="noopener">scikit-learn</a>
        </div>
        <div>
          <h4>Contact</h4>
          <a href="mailto:{AUTHOR['email']}">{AUTHOR['email']}</a>
          <a href="{AUTHOR['github']}" target="_blank" rel="noopener">GitHub</a>
        </div>
      </div>
            <div class="foot-bottom">
        <div class="foot-left">
          <span>© {year} {APP_NAME}. All rights reserved.</span>
          <span>Estimates are indicative only and are not a formal valuation. Inputs are not stored.</span>
        </div>
        <div class="foot-author">
          <span class="a-name">{AUTHOR['name']}</span>
          <a class="a-mail" href="mailto:{AUTHOR['email']}">{AUTHOR['email']}</a>
        </div>
      </div>
    </footer>
    """)

def _svg_uri(inner: str, defs: str = "") -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 700" preserveAspectRatio="xMidYMid slice">'
           f'<defs>{defs}</defs>{inner}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _tree(x: int, y: int, r: int, color: str = "#3f7d3c") -> str:
    return (f'<rect x="{x - 8}" y="{y}" width="16" height="{r}" fill="#5a3a2a"/>'
            f'<circle cx="{x}" cy="{y - 6}" r="{r}" fill="{color}"/>'
            f'<circle cx="{x - r // 2}" cy="{y + 14}" r="{int(r * .7)}" fill="{color}"/>'
            f'<circle cx="{x + r // 2}" cy="{y + 14}" r="{int(r * .7)}" fill="{color}"/>')


def _scene_villa() -> str:
    defs = ('<linearGradient id="s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#35568a"/>'
            '<stop offset=".55" stop-color="#f0b27a"/><stop offset="1" stop-color="#fbe3b5"/></linearGradient>'
            '<radialGradient id="sun"><stop offset="0" stop-color="#fff6d6"/><stop offset="1" stop-color="#fff6d6" stop-opacity="0"/></radialGradient>'
            '<linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7a9a5c"/><stop offset="1" stop-color="#46663a"/></linearGradient>')
    p = ['<rect width="1600" height="700" fill="url(#s)"/>',
         '<circle cx="1250" cy="360" r="190" fill="url(#sun)"/><circle cx="1250" cy="360" r="62" fill="#fff3c9"/>',
         '<path d="M0 470 Q300 390 620 455 T1220 440 T1600 430 V700 H0Z" fill="#6c8470"/>',
         '<rect y="525" width="1600" height="175" fill="url(#g)"/>',
         '<polygon points="760,700 850,700 900,540 740,540" fill="#d2c9b8"/>',
         _tree(250, 400, 70, "#3a6b3a"), _tree(1190, 420, 60, "#3a6b3a"), _tree(1400, 410, 75, "#33602f"),
         '<rect x="360" y="322" width="680" height="14" fill="#2b3446"/>',
         '<rect x="380" y="336" width="640" height="204" fill="#efe8dc"/>',
         '<rect x="380" y="336" width="170" height="204" fill="#a06f45"/>',
         '<rect x="520" y="222" width="400" height="14" fill="#2b3446"/>',
         '<rect x="540" y="236" width="360" height="86" fill="#dcd2c1"/>']
    p += [f'<line x1="395" x2="535" y1="{366 + k * 30}" y2="{366 + k * 30}" stroke="#7e5233" stroke-width="4"/>' for k in range(5)]
    for i in range(3):
        x = 580 + i * 140
        p.append(f'<rect x="{x}" y="366" width="110" height="130" fill="#ffd98a" stroke="#2b3446" stroke-width="5"/>'
                 f'<line x1="{x + 55}" x2="{x + 55}" y1="366" y2="496" stroke="#2b3446" stroke-width="4"/>')
        x2 = 565 + i * 110
        p.append(f'<rect x="{x2}" y="252" width="90" height="56" fill="#ffd98a" stroke="#2b3446" stroke-width="4"/>')
    p.append('<rect x="360" y="536" width="680" height="10" fill="#cfc6b6"/>')
    p += [f'<ellipse cx="{x}" cy="545" rx="38" ry="22" fill="#3f6e3a"/>' for x in (420, 1000, 1060)]
    return _svg_uri("".join(p), defs)


def _scene_house() -> str:
    defs = ('<linearGradient id="s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6fb1ee"/>'
            '<stop offset="1" stop-color="#e2f1ff"/></linearGradient>')
    p = ['<rect width="1600" height="700" fill="url(#s)"/>',
         '<circle cx="200" cy="130" r="52" fill="#fff6c0"/>',
         '<g fill="#fff" opacity=".92"><ellipse cx="520" cy="140" rx="110" ry="30"/><ellipse cx="600" cy="125" rx="80" ry="30"/>'
         '<ellipse cx="1180" cy="90" rx="120" ry="30"/><ellipse cx="1260" cy="75" rx="80" ry="28"/></g>',
         '<rect y="540" width="1600" height="160" fill="#7fb35a"/>',
         '<polygon points="770,700 830,700 850,540 750,540" fill="#d9cfbd"/>',
         _tree(300, 380, 80), _tree(1330, 390, 85, "#3a7438"),
         '<rect x="520" y="330" width="560" height="210" fill="#f3e6d0"/>',
         '<polygon points="480,340 800,160 1120,340" fill="#a4422f"/>',
         '<rect x="950" y="200" width="40" height="90" fill="#7d3223"/>',
         '<circle cx="800" cy="270" r="30" fill="#bfe0f5" stroke="#fff" stroke-width="6"/>',
         '<rect x="770" y="420" width="60" height="120" fill="#5a3a2a"/><circle cx="820" cy="485" r="4" fill="#C9A24B"/>']
    for x in (580, 690, 880, 980):
        p.append(f'<rect x="{x}" y="400" width="80" height="80" fill="#bfe0f5" stroke="#fff" stroke-width="6"/>'
                 f'<line x1="{x + 40}" x2="{x + 40}" y1="400" y2="480" stroke="#fff" stroke-width="4"/>'
                 f'<line x1="{x}" x2="{x + 80}" y1="440" y2="440" stroke="#fff" stroke-width="4"/>')
    p.append('<rect x="520" y="535" width="560" height="10" fill="#d9cfbd"/>')
    p += [f'<rect x="{x}" y="500" width="14" height="60" rx="5" fill="#fff"/>' for x in range(0, 460, 40)]
    p += [f'<rect x="{x}" y="500" width="14" height="60" rx="5" fill="#fff"/>' for x in range(1140, 1600, 40)]
    p += ['<rect x="0" y="520" width="440" height="8" fill="#fff"/><rect x="1140" y="520" width="460" height="8" fill="#fff"/>']
    p += [f'<circle cx="{x}" cy="552" r="9" fill="{c}"/>' for x, c in
          [(560, "#e85d75"), (600, "#f5c542"), (640, "#e85d75"), (960, "#f5c542"), (1000, "#e85d75"), (1040, "#f5c542")]]
    return _svg_uri("".join(p), defs)


def _windows(x: int, y: int, w: int, h: int, cols: int, rows: int, seed: int) -> str:
    cw, ch, out = w / cols, h / rows, []
    for r in range(rows):
        for c in range(cols):
            lit = (r * 7 + c * 13 + seed) % 5 != 0
            out.append(f'<rect x="{x + c * cw + cw * .22:.0f}" y="{y + r * ch + ch * .25:.0f}" '
                       f'width="{cw * .56:.0f}" height="{ch * .5:.0f}" fill="{"#ffd98a" if lit else "#3a4666"}"/>')
    return "".join(out)


def _scene_towers() -> str:
    defs = ('<linearGradient id="s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#121a38"/>'
            '<stop offset=".6" stop-color="#5b4a7e"/><stop offset="1" stop-color="#f2a46f"/></linearGradient>')
    p = ['<rect width="1600" height="700" fill="url(#s)"/>', '<circle cx="250" cy="120" r="40" fill="#f6f0d8"/>']
    p += [f'<circle cx="{x}" cy="{y}" r="2" fill="#fff" opacity=".8"/>' for x, y in
          [(500, 70), (700, 130), (900, 60), (1100, 110), (1350, 80), (1500, 150), (80, 200), (420, 190)]]
    ground = 610
    for x, w, h in [(0, 140, 250), (150, 120, 330), (1250, 150, 300), (1410, 190, 360), (700, 110, 280), (820, 90, 330)]:
        p.append(f'<rect x="{x}" y="{ground - h}" width="{w}" height="{h}" fill="#26304f"/>')
    for i, (x, w, h) in enumerate([(300, 230, 430), (560, 260, 520), (850, 230, 380), (1110, 200, 460)]):
        p.append(f'<rect x="{x}" y="{ground - h}" width="{w}" height="{h}" fill="{"#1d2745" if i % 2 == 0 else "#2b3760"}"/>')
        p.append(f'<rect x="{x + 20}" y="{ground - h - 14}" width="{w - 40}" height="14" fill="#1d2745"/>')
        p.append(_windows(x + 16, ground - h + 24, w - 32, h - 60, 6, h // 36, i + 1))
    p.append(f'<rect y="{ground}" width="1600" height="{700 - ground}" fill="#161d33"/>')
    p.append(f'<rect y="{ground + 40}" width="1600" height="6" fill="#C9A24B" opacity=".6"/>')
    p += [_tree(x, 585, 34, "#1e3a33") for x in (120, 240, 1050, 1230, 1380)]
    return _svg_uri("".join(p), defs)


_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _hero_sources() -> list[Path]:
    """Image files for the slideshow: house_images/ folder first, else HERO_FILES next to app.py."""
    folder = BASE_DIR / HERO_DIR
    if folder.is_dir():
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in _IMG_EXT)
        if files:
            return files
    by_name = {f.name.lower(): f for f in BASE_DIR.iterdir() if f.is_file()}
    return [by_name[n.lower()] for n in HERO_FILES if n.lower() in by_name]


@st.cache_data(show_spinner=False)
def _encode_image(path: str, mtime: float) -> str:
    """Resize once and return a data URI. mtime is part of the cache key, so replacing a file refreshes it."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((1400, 900))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=78, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def hero_images() -> list[str]:
    """Your photos if found, otherwise the built-in illustrations."""
    uris: list[str] = []
    for item in HERO_IMAGE_URLS:                     
        if str(item).startswith("http"):
            uris.append(item)
        elif Path(item).is_file():
            uris.append(_encode_image(str(item), Path(item).stat().st_mtime))
    if not uris:
        for f in _hero_sources():
            try:
                uris.append(_encode_image(str(f), f.stat().st_mtime))
            except Exception:
                continue
    return uris or [_scene_villa(), _scene_house(), _scene_towers()]


def hero_slideshow_css(n: int, secs: int) -> str:
    if n < 2:
        return ""
    total, visible = n * secs, 100 / n
    fade = min(visible / 4, 8)
    css = (f"@keyframes heroFade{{0%{{opacity:0}}{fade:.1f}%{{opacity:1}}{visible:.1f}%{{opacity:1}}"
           f"{visible + fade:.1f}%{{opacity:0}}100%{{opacity:0}}}}"
           f".hero-slide{{animation:heroFade {total}s infinite;}}")
    css += "".join(f".hero-slide:nth-child({i + 1}){{animation-delay:{i * secs}s;}}" for i in range(n))
    return css


def page_home() -> None:
    images = hero_images()
    slides = "".join(f'<div class="hero-slide" style="background-image:url(\'{u}\')"></div>' for u in images)
    chips = "".join(f'<span class="type-chip">{t}</span>' for t in PROPERTY_TYPES)
    html(f"""
    <style>{hero_slideshow_css(len(images), SLIDE_SECONDS)}</style>
    <section class="hero-wrap" aria-label="Featured homes">
      {slides}
      <div class="hero-shade"></div>
      <div class="hero-text">
        <h1>Know a home's market price before you make an offer.</h1>
        <p class="lead">Enter the city, size and rooms. {APP_NAME} compares them with thousands of real listings
        and gives you an estimated price with a likely range.</p>
        <div>
          <a class="btn gold" href="?page=predict" target="_self">Estimate a price</a>
          <a class="btn light" href="?page=about" target="_self">How it works</a>
        </div>
      </div>
    </section>
    <div class="type-row">{chips}</div>

    <section class="section">
      <h2>From details to estimate in three steps</h2>
      <p class="sub">No sign-up and nothing to install.</p>
      <div class="grid3">
        <div class="card"><span class="step-n">1</span><h3>Describe the property</h3>
          <p>Pick the type and city, then enter the area in Marla, bedrooms and bathrooms.</p></div>
        <div class="card"><span class="step-n">2</span><h3>Run the model</h3>
          <p>A machine learning model trained on past listings predicts the market price.</p></div>
        <div class="card"><span class="step-n">3</span><h3>Read your estimate</h3>
          <p>See the price in Lakh and Crore, a likely range, and the price per Marla.</p></div>
      </div>
    </section>

    <section class="section">
      <h2>Why use it</h2>
      <div class="grid3">
        <div class="card"><h3>Fast</h3><p>Results appear in a moment, with nothing to fill in beyond the basics.</p></div>
        <div class="card"><h3>Built for Pakistan</h3><p>Areas are in Marla and prices are shown in Lakh and Crore.</p></div>
        <div class="card"><h3>Works on any screen</h3><p>The layout adapts to phones, tablets and desktops.</p></div>
      </div>
    </section>
    """)

DARK_CSS = """
<style>
:root{--ink:#EAF0FA; --ink-soft:#A9B6CC; --paper:#0A1730; --card:#12294D; --line:#27426E;
      --mist:#1B3560; --pine:#0F2142; --pine-2:#1F3D6B;}
html, body, .stApp{background:#0A1730 !important;}
.stApp, .stApp p, .stApp label, .stApp li, .stApp span, .stApp small{color:var(--ink);}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] *{color:var(--ink-soft) !important;}
.nav{border-bottom-color:var(--line);}
.brand{color:#fff !important;}
.brand-mark{background:var(--marigold);}
.nav-links a:hover{background:var(--mist); color:#fff !important;}
.nav-links a.active{background:var(--mist); color:#fff !important; box-shadow:inset 0 -2px 0 var(--marigold);}
[data-testid="stForm"]{background:var(--card); border-color:var(--line) !important;}
[data-baseweb="select"] > div, [data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="base-input"], [data-testid="stNumberInput"] input{
  background:#0A1730 !important; color:#fff !important; border-color:var(--line) !important;}
[data-baseweb="select"] *{color:#fff !important;}
[data-baseweb="select"] svg{fill:#fff !important;}
[data-testid="stNumberInput"] button{background:var(--mist) !important; color:#fff !important;}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li{background:var(--card) !important; color:#fff !important;}
[data-baseweb="popover"] li:hover{background:var(--mist) !important;}
.stButton > button, [data-testid="stFormSubmitButton"] > button{background:var(--marigold); color:#0A1730 !important;}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover{background:#D8B25C; color:#0A1730 !important;}
/* buttons, nav and cards on the dark background */
.btn, .btn.gold{background:var(--marigold); color:#0A1730 !important;}
.btn:hover, .btn.gold:hover{background:#D8B25C;}
.btn.ghost{background:transparent; color:#fff !important; border:1.5px solid rgba(255,255,255,.55);}
.btn.ghost:hover{background:rgba(255,255,255,.1);}
.btn.light{background:transparent; color:#fff !important; border:1.5px solid rgba(255,255,255,.8);}
.nav-links a.cta{background:var(--marigold); color:#0A1730 !important;}
.stApp .step-n{color:#0A1730;}
.stApp .card p, .stApp .sub, .stApp .page-title p, .stApp .type-chip{color:var(--ink-soft);}
.card, .type-chip{background:var(--card); border-color:var(--line);}
[data-testid="stFormSubmitButton"] > button p{color:#0A1730 !important;}
[data-testid="stAlert"]{background:var(--card) !important; border:1px solid var(--line);}
[data-testid="stAlert"] *{color:var(--ink) !important;}
.result{background:var(--card); border:1px solid rgba(201,162,75,.55);}
.stat{background:rgba(255,255,255,.07);}
.foot{background:#0F2142; border:1px solid var(--line); border-bottom:0;}
</style>
"""

def page_predict() -> None:
    html("""
    <div class="page-title">
      <h1>Predict a property price</h1>
      <p>Please provide the following details.</p>
    </div>
    """)

    with st.form("property_form"):
        c1, c2 = st.columns(2)
        with c1:
            property_type = st.selectbox("Property Type", PROPERTY_TYPES, index=0)
            area = st.number_input("Property Area (Marla)", min_value=1.0, max_value=2000.0,
                                   value=10.0, step=0.5, format="%.2f")
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=4, step=1)
        with c2:
            city = st.selectbox("City", CITIES, index=0)
            bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=4, step=1)
            # st.caption("1 Marla is about 225 sq ft. 20 Marla = 1 Kanal.")
        submitted = st.form_submit_button("Predict price")

    if not submitted:
        st.info("Fill in the details above and select **Predict price**.")
        return

    if bathrooms > bedrooms + 2:
        st.warning("That is a lot of bathrooms for the number of bedrooms. Check the values, or continue if they are correct.")

    row = {"property_type": property_type, "city": city, "area": float(area),
           "bedrooms": int(bedrooms), "bathrooms": int(bathrooms)}

    try:
        with st.spinner("Estimating price..."):
            price, real_model = predict_price(row)
    except Exception as exc:  # column mismatch, bad pickle, etc.
        st.error("The model could not make a prediction. Check that feature_columns.pkl "
                 "matches the model and scaler.")
        with st.expander("Technical details"):
            st.code(repr(exc))
        return

    low, high = price * 0.90, price * 1.10
    html(f"""
    <div class="result" role="status">
      <p class="lbl">Estimated market price</p>
      <p class="price">{fmt_pkr(price)}</p>
      <p class="range">Likely range: {fmt_pkr(low)} to {fmt_pkr(high)}</p>
      <div class="stat-row">
        <div class="stat"><b>{fmt_pkr(price / row['area'])}</b><span>Price per Marla</span></div>
        <div class="stat"><b>{row['area']:g} Marla</b><span>{row['property_type']} in {row['city']}</span></div>
        <div class="stat"><b>{row['bedrooms']} bed / {row['bathrooms']} bath</b><span>Rooms</span></div>
        <div class="stat"><b>PKR {price:,.0f}</b><span>Exact figure</span></div>
      </div>
    </div>
    """)
    if not real_model:
        st.warning("Model files were not found, so this figure comes from a rough demo formula. "
                   "Keep Random_forest_model.pkl, scaler.pkl and feature_columns.pkl next to app.py.")
    st.caption("The range is a simple ±10% band around the estimate, not a statistical confidence interval.")


def page_about() -> None:
    html(f"""
    <div class="page-title">
      <h1>About {APP_NAME}</h1>
      <p>Quick, data-driven price estimates for homes in major Pakistani cities.</p>
    </div>

    <section class="section">
      <div class="grid3">
        <div class="card"><h3>What it does</h3>
          <p>You describe a property and {APP_NAME} estimates what similar homes are selling for.</p></div>
        <div class="card"><h3>How it works</h3>
          <p>A machine learning model learns from past property listings and applies those patterns to your inputs.</p></div>
        <div class="card"><h3>Good to know</h3>
          <p>Estimates are a guide, not a valuation. Condition, exact street and recent market changes are not included.</p></div>
      </div>
    </section>

    <section class="section">
      <p class="sub">Feedback and ideas are welcome.</p>
      <a class="btn" href="{AUTHOR['github']}" target="_blank" rel="noopener">GitHub</a>
    </section>
     <section class="section">
     <h4>Built by {AUTHOR['name']}</h4>
     <h4>Email: <a href="mailto:{AUTHOR['email']}">{AUTHOR['email']}</a></h4>
    </section>
    """)

ROUTES = {"home": page_home, "predict": page_predict, "about": page_about}

def main() -> None:
    current = st.query_params.get("page", "home")
    if current not in ROUTES:
        current = "home"
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    navbar(current)
    ROUTES[current]()
    footer()

main()
