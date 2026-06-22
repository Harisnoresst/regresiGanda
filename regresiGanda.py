import io, base64, json, os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from flask import Flask, render_template_string, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- BACA DATASET ISPU JAKARTA ---
CSV_PATH = os.path.join(os.path.dirname(__file__), "ispu_jakarta_clean.csv")
df_raw = pd.read_csv(CSV_PATH)
# Buang data dengan kolom yang kosong
df_raw = df_raw.dropna(subset=['pm10', 'o3', 'co', 'max', 'categori', 'critical', 'stasiun'])
# Buang data yang tidak valid di kolom critical
df_raw = df_raw[~df_raw['critical'].isin(['1', '2', '3', '5', 'NO2'])]
# Rename agar mudah diakses
df_raw.rename(columns={'max': 'ispu_max', 'categori': 'kategori'}, inplace=True)

df = df_raw.copy()

# X = Prediktor | Y = Target
X_COLS = ["pm10", "o3", "co"]
Y_COL  = "ispu_max"

X = df[X_COLS].values
y = df[Y_COL].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_global = model.predict(X_test)

r2_global   = r2_score(y_test, y_pred_global)
rmse_global = np.sqrt(mean_squared_error(y_test, y_pred_global))
mae_global  = mean_absolute_error(y_test, y_pred_global)
n_g = len(y_test); p_g = len(X_COLS)
r2_adj_global = 1 - (1 - r2_global) * (n_g - 1) / (n_g - p_g - 1)

# Fitur dummy untuk Model Penuh (Prediksi Form)
df["is_kat_sedang"] = (df["kategori"] == "SEDANG").astype(int)
df["is_kat_tidak_sehat"] = (df["kategori"] == "TIDAK SEHAT").astype(int)
df["is_kat_sangat"] = (df["kategori"] == "SANGAT TIDAK SEHAT").astype(int)
df["is_kat_bahaya"] = (df["kategori"] == "BERBAHAYA").astype(int)

df["is_crit_o3"] = (df["critical"] == "O3").astype(int)
df["is_crit_pm10"] = (df["critical"] == "PM10").astype(int)
df["is_crit_pm25"] = (df["critical"] == "PM2.5").astype(int)
df["is_crit_so2"] = (df["critical"] == "SO2").astype(int)

df["is_dki2"] = (df["stasiun"] == "DKI2 Kelapa Gading").astype(int)
df["is_dki3"] = (df["stasiun"] == "DKI3 Jagakarsa").astype(int)
df["is_dki4"] = (df["stasiun"] == "DKI4 Lubang Buaya").astype(int)
df["is_dki5"] = (df["stasiun"] == "DKI5 Kebon Jeruk").astype(int)

X_pred_cols = [
    "pm10", "o3", "co", 
    "is_kat_sedang", "is_kat_tidak_sehat", "is_kat_sangat", "is_kat_bahaya",
    "is_crit_o3", "is_crit_pm10", "is_crit_pm25", "is_crit_so2",
    "is_dki2", "is_dki3", "is_dki4", "is_dki5"
]

X_pred_full = df[X_pred_cols].values
scaler_full = StandardScaler()
X_pred_scaled = scaler_full.fit_transform(X_pred_full)

model_full = LinearRegression()
model_full.fit(X_pred_scaled, y)


def get_theme_colors(is_dark):
    if is_dark:
        return "#1E1E1E", "#F3F4F6", "#9CA3AF"
    return "#ffffff", "#1C1C1C", "#6B7280"

def fig_to_b64(fig, bg_color):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data

def make_heatmap(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    corr = df_sub[X_COLS + [Y_COL]].corr()
    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    cmap = sns.diverging_palette(250, 20, as_cmap=True) 
    annot_kws_color = "#ffffff" if is_dark else "#1C1C1C"
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.8, linecolor=bg_color, ax=ax,
                annot_kws={"size": 9, "weight": "bold", "color": annot_kws_color},
                cbar_kws={"shrink": 0.7}, vmin=-1, vmax=1)
                
    ax.set_title(f"Heatmap Korelasi  (n={len(df_sub)})", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=mut_color, labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", color=mut_color)
    plt.setp(ax.get_yticklabels(), rotation=0, color=mut_color)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=mut_color, labelsize=7.5)
    fig.tight_layout(pad=1.0)
    return fig_to_b64(fig, bg_color)

def make_pls_scores(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    LINE_COLOR = "#4B5563" if is_dark else "#e2e8f0"
    if len(df_sub) < 5: return fig_to_b64(plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)[0], bg_color)

    X_s = scaler.transform(df_sub[X_COLS].values)
    y_s = df_sub[Y_COL].values
    pls = PLSRegression(n_components=2)
    pls.fit(X_s, y_s)
    X_scores, _ = pls.transform(X_s, y_s)

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.scatter(X_scores[:, 0], X_scores[:, 1], alpha=0.6, s=16, color="#FF671D", linewidths=0)
    
    ax.set_title(f"PLS Scores Plot  (n={len(df_sub)})", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("PLS Component 1", color=mut_color, fontsize=9)
    ax.set_ylabel("PLS Component 2", color=mut_color, fontsize=9)
    ax.tick_params(colors=mut_color, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig, bg_color)

def make_pls_loadings(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    LINE_COLOR = "#4B5563" if is_dark else "#e2e8f0"
    if len(df_sub) < 5: return fig_to_b64(plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)[0], bg_color)

    X_s = scaler.transform(df_sub[X_COLS].values)
    y_s = df_sub[Y_COL].values
    pls = PLSRegression(n_components=2)
    pls.fit(X_s, y_s)
    
    x_loadings = pls.x_loadings_

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    ax.axhline(0, color=LINE_COLOR, linestyle='--', linewidth=1)
    ax.axvline(0, color=LINE_COLOR, linestyle='--', linewidth=1)

    for i, col in enumerate(X_COLS):
        ax.arrow(0, 0, x_loadings[i, 0], x_loadings[i, 1], color="#22C55E", 
                 alpha=0.8, head_width=0.03, head_length=0.03, linewidth=1.5)
        ax.text(x_loadings[i, 0] * 1.15, x_loadings[i, 1] * 1.15, col.upper(),
                color=text_color, ha='center', va='center', fontweight='bold', fontsize=9)
    
    max_val = np.max(np.abs(x_loadings)) * 1.4
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.set_title(f"PLS Loadings Plot", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("PLS Component 1", color=mut_color, fontsize=9)
    ax.set_ylabel("PLS Component 2", color=mut_color, fontsize=9)
    ax.tick_params(colors=mut_color, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig, bg_color)

def make_scatter(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=bg_color)
    LINE_COLOR = "#4B5563" if is_dark else "#cbd5e1"
    colors = ["#FF671D", "#9CA3AF" if is_dark else "#0A0A0A", "#22C55E"]
    labels = ["PM10", "O3 (Ozon)", "CO"]

    for i, (col, clr, lbl) in enumerate(zip(X_COLS, colors, labels)):
        ax = axes[i]
        ax.set_facecolor(bg_color)
        x_vals = df_sub[col].values
        y_vals = df_sub[Y_COL].values
        ax.scatter(x_vals, y_vals, alpha=0.5, s=16, color=clr, linewidths=0)
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(x_line, m * x_line + b, color=LINE_COLOR, linewidth=1.8, alpha=0.9, linestyle="--")
        r_val = np.corrcoef(x_vals, y_vals)[0, 1]
        ax.set_title(f"{lbl}  (r={r_val:.3f})", color=text_color, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(col.upper(), color=mut_color, fontsize=9)
        ax.set_ylabel("ISPU Max" if i == 0 else "", color=mut_color, fontsize=9)
        ax.tick_params(colors=mut_color, labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
        for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)

    fig.suptitle(f"Scatter Plot: X vs ISPU Max  (n={len(df_sub)})", color=text_color, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(pad=1.4)
    return fig_to_b64(fig, bg_color)

def make_actual_vs_pred(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    LINE_COLOR = "#4B5563" if is_dark else "#e2e8f0"
    if len(df_sub) < 5: return fig_to_b64(plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)[0], bg_color)

    X_s = scaler.transform(df_sub[X_COLS].values)
    y_s = df_sub[Y_COL].values
    n_sub = len(df_sub)
    test_size = max(0.2, 5 / n_sub) if n_sub >= 10 else 0.5
    Xtr, Xte, ytr, yte = train_test_split(X_s, y_s, test_size=test_size, random_state=42)
    m_sub = LinearRegression().fit(Xtr, ytr)
    ypred = m_sub.predict(Xte)

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.scatter(yte, ypred, alpha=0.6, s=16, color="#FF671D", linewidths=0)
    
    lim = [min(yte.min(), ypred.min()) - 10, max(yte.max(), ypred.max()) + 10]
    PERFECT_LINE = "#9CA3AF" if is_dark else "#0A0A0A"
    ax.plot(lim, lim, color=PERFECT_LINE, linewidth=1.5, linestyle="--", label="Perfect fit")
    ax.set_xlim(lim); ax.set_ylim(lim)
    r2_s = r2_score(yte, ypred)
    
    ax.set_title(f"Actual vs Predicted  (R²={r2_s:.3f})", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Actual ISPU", color=mut_color, fontsize=9)
    ax.set_ylabel("Predicted ISPU", color=mut_color, fontsize=9)
    ax.tick_params(colors=mut_color, labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(fontsize=8, labelcolor=mut_color, frameon=False)
    for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig, bg_color)

def make_residual(df_sub, is_dark=False):
    bg_color, text_color, mut_color = get_theme_colors(is_dark)
    LINE_COLOR = "#4B5563" if is_dark else "#e2e8f0"
    if len(df_sub) < 5: return fig_to_b64(plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)[0], bg_color)

    X_s = scaler.transform(df_sub[X_COLS].values)
    y_s = df_sub[Y_COL].values
    n_sub = len(df_sub)
    test_size = max(0.2, 5 / n_sub) if n_sub >= 10 else 0.5
    Xtr, Xte, ytr, yte = train_test_split(X_s, y_s, test_size=test_size, random_state=42)
    m_sub = LinearRegression().fit(Xtr, ytr)
    ypred = m_sub.predict(Xte)
    residuals = yte - ypred

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    SCATTER_COLOR = "#9CA3AF" if is_dark else "#0A0A0A"
    ax.scatter(ypred, residuals, alpha=0.6, s=16, color=SCATTER_COLOR, linewidths=0)
    ax.axhline(0, color="#FF671D", linewidth=1.5, linestyle="--")
    
    ax.set_title(f"Residual Plot  (n={len(df_sub)})", color=text_color, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted ISPU", color=mut_color, fontsize=9)
    ax.set_ylabel("Residuals", color=mut_color, fontsize=9)
    ax.tick_params(colors=mut_color, labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    for spine in ax.spines.values(): spine.set_edgecolor(LINE_COLOR)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig, bg_color)


HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Regresi Berganda & PLS — ISPU Jakarta Dataset</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,500;1,600;1,700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #FDF7EC; 
  --card: #FFFFFF;
  --nav-bg: rgba(15, 15, 15, 0.65); 
  --nav-border: rgba(255, 255, 255, 0.1);
  --nav-text: #FFFFFF;
  --accent: #FF671D; 
  --text: #1C1C1C;
  --muted: #6B7280;
  --border: #EAE0D3;
  --shadow: rgba(0,0,0,0.04);
  --pill-bg: #F1F5F9;
  --table-hover: #F8FAFC;
}
[data-theme="dark"] {
  --bg: #121212;
  --card: #1E1E1E;
  --nav-bg: rgba(25, 25, 25, 0.7); 
  --nav-border: rgba(255, 255, 255, 0.05);
  --nav-text: #F3F4F6;
  --accent: #FF671D;
  --text: #F3F4F6;
  --muted: #9CA3AF;
  --border: #374151;
  --shadow: rgba(0,0,0,0.3);
  --pill-bg: #374151;
  --table-hover: #292929;
}
*{box-sizing:border-box;margin:0;padding:0}
html { scroll-behavior: smooth; scroll-padding-top: 100px; }
body{background:var(--bg);color:var(--text);font-family:"Plus Jakarta Sans",sans-serif;min-height:100vh; transition: background 0.3s, color 0.3s;}

.nav-wrapper { position: sticky; top: 1.5rem; z-index: 100; padding: 0 2rem; }
nav { background: var(--nav-bg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid var(--nav-border); color: var(--nav-text); max-width: 1200px; margin: 0 auto; border-radius: 99px; padding: 0.8rem 1.5rem 0.8rem 2rem; display: flex; align-items: center; box-shadow: 0 10px 30px rgba(0,0,0,0.15); transition: all 0.3s ease; }
.nav-logo { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
.nav-logo span { font-weight: 400; }
.nav-links { display: flex; gap: 2rem; margin-left: auto; margin-right: 2rem; align-items: center; }
.nav-links a { font-size: 0.9rem; font-weight: 500; color: #E5E5E5; text-decoration: none; transition: color 0.2s; }
.nav-links a:hover { color: var(--accent); }
.nav-actions { display: flex; align-items: center; gap: 0.8rem; }
.nav-btn { background: linear-gradient(135deg, #FF7B00, #F95000); color: white; border: none; border-radius: 99px; padding: 0.5rem 1.2rem; font-family: inherit; font-weight: 700; font-size: 0.85rem; cursor: pointer; }
.theme-toggle { background: transparent; border: 1px solid rgba(255,255,255,0.2); color: white; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 1rem; transition: background 0.2s; }
.theme-toggle:hover { background: rgba(255,255,255,0.1); }

.hero{text-align: center; padding: 5rem 2rem 3rem; max-width:900px; margin:0 auto}
.hero h1{font-size:clamp(2.5rem, 5vw, 4.2rem); font-weight:800; letter-spacing:-0.03em; line-height:1.15; color:var(--text);}
.hero h1 .serif { font-family: 'Playfair Display', serif; font-style: italic; font-weight: 600; color: var(--accent); }
.hero p{margin-top:1.5rem; color:var(--muted); font-size:1.1rem; line-height:1.6; font-weight: 500;}
.badge-row{display:flex;flex-wrap:wrap;justify-content: center;gap:0.6rem;margin-top:2rem}
.badge{background:transparent;border:1.5px solid var(--border);border-radius:99px; padding:0.4rem 1rem;font-size:0.85rem;color:var(--text);font-weight:600;}

.wrap{max-width:1200px;margin:0 auto;padding:0 2rem 5rem}
.section-title{font-size:1.5rem;font-weight:700;margin-bottom:1.2rem;color:var(--text); display:flex;align-items:center;gap:0.6rem; letter-spacing: -0.02em; margin-top:1.5rem;}
.section-title i { font-family: 'Playfair Display', serif; font-style: italic; color: var(--accent); }

.filter-bar{background:var(--card);border:none;border-radius:100px; padding:0.8rem 1.5rem;margin-bottom:2.5rem;display:flex;flex-wrap:wrap; align-items:center;gap:1rem;box-shadow:0 8px 30px var(--shadow)}
.filter-bar label{font-size:0.9rem;color:var(--text);font-weight:600}
.sel, .search-box{ background:var(--bg);border:1px solid var(--border);border-radius:99px; color:var(--text);padding:0.6rem 1.2rem;font-size:0.9rem;cursor:pointer;outline:none; font-family:inherit; transition: all 0.2s; font-weight: 500; }
.sel:focus, .search-box:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(255,103,29,0.15)}
.search-box{width:220px}
.search-box::placeholder{color:var(--muted)}
.filter-info{margin-left:auto;font-size:0.9rem;color:var(--muted); font-weight: 500;}
.filter-info strong{color:var(--text)}
.btn-refresh{background:var(--text);border:none;border-radius:99px;color:var(--bg); padding:0.6rem 1.5rem;font-size:0.9rem;font-weight:600;cursor:pointer; transition:all 0.2s;white-space:nowrap; font-family: inherit;}
.btn-refresh:hover{opacity: 0.8;}

.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1.5rem;margin-bottom:3.5rem}
.metric{background:var(--card);border:none;border-radius:24px;padding:1.8rem; box-shadow:0 8px 30px var(--shadow); text-align: center;}
.metric-label{font-size:0.8rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:0.05em}
.metric-value{font-size:2.2rem;font-weight:800;margin:0.5rem 0 0.2rem;color:var(--text); font-family:"JetBrains Mono",monospace; letter-spacing: -1px;}
.metric-desc{font-size:0.85rem;color:var(--muted); font-weight: 500;}

.eq-card{background:var(--card);border:none;border-radius:24px; padding:2rem;margin-bottom:3.5rem;overflow-x:auto;box-shadow:0 8px 30px var(--shadow)}
.eq-title{font-size:0.85rem;color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing: 0.05em; margin-bottom:1rem}
.eq-formula{font-family:"JetBrains Mono",monospace;font-size:clamp(1rem,2vw,1.15rem); color:var(--text);line-height:1.8;background:var(--bg);padding:1.5rem;border-radius:16px; border:1px solid var(--border); font-weight: 500;}
.eq-formula .var{color:var(--text);font-weight:700}
.eq-formula .coef{color:var(--accent)}

.charts-2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:3.5rem}
.charts-1{margin-bottom:3.5rem}
.chart-card{background:var(--card);border:none;border-radius:24px;overflow:hidden; box-shadow:0 8px 30px var(--shadow); padding: 1rem; transition: background 0.3s;}
.chart-card img{width:100%;border-radius: 12px;display:block;}
.chart-card .chart-label{padding:1rem 1rem 0.5rem;font-size:0.9rem;color:var(--muted); text-align:center; font-weight: 500;}

.tbl-wrap{background:var(--card);border:none;border-radius:24px;overflow:hidden; box-shadow:0 8px 30px var(--shadow); padding: 1rem;}
.tbl-scroll{overflow-x:auto;max-height:500px;overflow-y:auto; border-radius: 12px;}
.tbl-scroll::-webkit-scrollbar{width:8px;height:8px}
.tbl-scroll::-webkit-scrollbar-track{background:var(--card)}
.tbl-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
table{width:100%;border-collapse:collapse;font-size:0.9rem}
thead{position:sticky;top:0;z-index:5}
thead th{background:var(--bg);color:var(--text);font-weight:700; padding:1rem 1.2rem;text-align:left;white-space:nowrap; border-bottom:2px solid var(--border)}
tbody tr{border-bottom:1px solid var(--border);transition:background 0.2s}
tbody tr:hover{background:var(--table-hover)}
tbody td{padding:0.9rem 1.2rem;color:var(--text);white-space:nowrap; font-weight: 500;}
tbody td.num{font-family:"JetBrains Mono",monospace;color:var(--muted)}
tbody td.charge{color:var(--text);font-weight:700;font-family:"JetBrains Mono",monospace}
.pill{display:inline-block;border-radius:99px;padding:0.25rem 0.8rem;font-size:0.8rem;font-weight:700}
.pill-yes{background:rgba(255,103,29,0.15);color:var(--accent);}
.pill-no{background:var(--pill-bg);color:var(--muted);}
.pill-m{background:rgba(34,197,94,0.15);color:#22C55E;}
.pill-f{background:rgba(217,119,6,0.15);color:#D97706;}
.tbl-footer{padding:1.2rem 1rem 0.5rem;font-size:0.9rem;color:var(--muted); font-weight: 600; display:flex;justify-content:space-between;align-items:center}

.predict-card{background:var(--card);border:none;border-radius:24px; padding:2.5rem;margin-bottom:3.5rem;box-shadow:0 8px 30px var(--shadow)}
.input-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1.5rem;margin-bottom:2rem}
.inp-group label{display:block;font-size:0.9rem;color:var(--text);margin-bottom:0.6rem;font-weight:700}
.inp-group input, .inp-group select {width:100%;background:var(--bg);border:1px solid var(--border); border-radius:12px;color:var(--text);padding:0.8rem 1rem;font-size:1rem; outline:none;transition:all 0.2s;font-family:"JetBrains Mono",monospace; font-weight: 600;}
.inp-group input:focus, .inp-group select:focus {border-color:var(--accent);box-shadow:0 0 0 3px rgba(255,103,29,0.15)}
.btn-orange{background:linear-gradient(135deg, #FF7B00, #F95000);border:none; border-radius:99px;color:#fff;padding:0.8rem 2rem;font-size:1rem;font-weight:700; cursor:pointer;transition:all 0.2s; font-family: inherit; box-shadow: 0 4px 15px rgba(249, 80, 0, 0.25);}
.btn-orange:hover{transform: translateY(-2px); box-shadow: 0 6px 20px rgba(249, 80, 0, 0.35);}
.pred-result{margin-top:2rem;padding:1.5rem 2rem;background:var(--bg); border-radius:16px;display:none; text-align: center; border: 1px solid var(--border);}
.pred-result .pred-label{font-size:0.9rem;color:var(--muted);font-weight:700;text-transform:uppercase; letter-spacing: 0.05em;}
.pred-result .pred-val{font-size:2.5rem;font-weight:800;color:var(--accent); font-family:"JetBrains Mono",monospace;margin-top:0.5rem; letter-spacing: -1px;}

.coef-table{width:100%;border-collapse:collapse;font-size:0.9rem;margin-top:2rem}
.coef-table th{background:transparent;color:var(--muted);font-size:0.8rem;font-weight:700; text-transform:uppercase;padding:1rem;text-align:left;border-bottom:2px solid var(--border)}
.coef-table td{padding:1rem;border-bottom:1px solid var(--border);font-family:"JetBrains Mono",monospace; font-weight: 500;}
.coef-table td:first-child{font-family:"Plus Jakarta Sans",sans-serif;font-weight:700;color:var(--text)}
.coef-bar{height:8px;border-radius:4px;background:var(--accent);margin-top:8px;transition:width 0.5s}

.chart-spinner{display:flex;align-items:center;justify-content:center; height:160px;color:var(--muted);font-size:0.9rem;gap:0.8rem; font-weight: 600;}
.spin{width:22px;height:22px;border:3px solid var(--border); border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

@media(max-width:800px){ .charts-2{grid-template-columns:1fr} .wrap{padding:0 1rem 3rem} .nav-wrapper { padding: 0 1rem; } .nav-links { display: none; } .hero h1 { font-size: 2.2rem; } .filter-bar { border-radius: 20px; } }

/* --- CSS UNTUK POP-UP MATEMATIKA --- */
.modal-overlay {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(5px);
  z-index: 9999; display: none; align-items: center; justify-content: center;
  opacity: 0; transition: opacity 0.3s ease;
}
.modal-overlay.active { display: flex; opacity: 1; }
.modal-content {
  background: var(--card); border-radius: 24px; padding: 2rem;
  width: 95%; max-width: 1000px; max-height: 90vh; overflow-y: auto;
  box-shadow: 0 20px 40px rgba(0,0,0,0.4); border: 1px solid var(--border);
  position: relative; transform: translateY(20px); transition: transform 0.3s ease;
}
.modal-overlay.active .modal-content { transform: translateY(0); }
.modal-close {
  position: absolute; top: 1.5rem; right: 1.5rem;
  background: var(--pill-bg); border: none; color: var(--text);
  width: 36px; height: 36px; border-radius: 50%; font-size: 1.2rem;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: all 0.2s;
}
.modal-close:hover { background: var(--accent); color: white; }
.modal-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1rem; }
.modal-img-container img { width: 100%; border-radius: 12px; border: 1px solid var(--border); position: sticky; top: 0;}
.modal-math-container { font-size: 0.95rem; line-height: 1.8; color: var(--text); }
.modal-math-container h3 { color: var(--accent); margin-bottom: 1rem; font-size: 1.4rem; }
.math-box { background: var(--bg); padding: 1.2rem; border-radius: 12px; font-family: "JetBrains Mono", monospace; border: 1px solid var(--border); margin-bottom: 1rem; color: var(--accent); font-weight: 700; overflow-x: auto; font-size: 1rem;}
.modal-math-container ul { margin-left: 1.5rem; margin-bottom: 1rem; }
.modal-math-container li { margin-bottom: 0.5rem; }
.modal-math-container strong { color: var(--text); }
.modal-math-container code { background: var(--pill-bg); padding: 2px 6px; border-radius: 4px; font-family: "JetBrains Mono", monospace; font-size: 0.9em; color: var(--accent); }

@media(max-width: 800px) { .modal-grid { grid-template-columns: 1fr; } .modal-img-container img {position: static;} }
</style>
</head>
<body>

<div class="nav-wrapper">
  <nav>
    <div class="nav-logo">Regresi<span>Ganda</span></div>
    <div class="nav-links">
      <a href="#model">Matriks OLS</a>
      <a href="#prediksi">Prediksi</a>
      <a href="#dataset">Dataset</a>
      <a href="#pls">Grafik PLS</a>
    </div>
    <div class="nav-actions">
      <button class="nav-btn">n = {{ n_total }}</button>
      <button class="theme-toggle" id="theme-btn" title="Toggle Dark Mode">🌙</button>
    </div>
  </nav>
</div>

<div class="hero">
  <h1>Analisis <span class="serif">Regresi Berganda</span><br>&amp; <span class="serif">PLS</span> ISPU Jakarta</h1>
  <p>Belajar memprediksi nilai puncak harian polusi (<strong>ISPU Max</strong>) berdasarkan indikator PM10, O3, dan CO dengan pendekatan OLS (Regresi Linear) dan Regresi PLS, Data bersumber dari Open Data Jakarta.<br><br><span style="color:var(--accent); font-weight:800; letter-spacing:1px; font-size:0.95rem;">— KELOMPOK 2 —</span></p>
  <div class="badge-row">
    <span class="badge">X₁: PM10</span>
    <span class="badge">X₂: O3</span>
    <span class="badge">X₃: CO</span>
    <span class="badge" style="color:var(--accent); border-color:var(--accent)">Y: ISPU Max</span>
  </div>
</div>

<div class="wrap">

  <div class="section-title" id="model">Metrik <i>Model OLS</i></div>
  <div class="metrics">
    <div class="metric">
      <div class="metric-label">R² Score</div>
      <div class="metric-value">{{ "%.4f"|format(r2) }}</div>
      <div class="metric-desc">Koefisien determinasi</div>
    </div>
    <div class="metric">
      <div class="metric-label">Adj. R²</div>
      <div class="metric-value">{{ "%.4f"|format(r2_adj) }}</div>
      <div class="metric-desc">Adjusted R-squared</div>
    </div>
    <div class="metric">
      <div class="metric-label">RMSE</div>
      <div class="metric-value" style="color:var(--accent)">{{ "{:,.2f}".format(rmse) }}</div>
      <div class="metric-desc">Root Mean Sq. Error</div>
    </div>
    <div class="metric">
      <div class="metric-label">MAE</div>
      <div class="metric-value">{{ "{:,.2f}".format(mae) }}</div>
      <div class="metric-desc">Mean Absolute Error</div>
    </div>
  </div>

  <div class="eq-card">
    <div class="eq-title">Model Regresi Linear: Y = b₀ + b₁X₁ + b₂X₂ + b₃X₃</div>
    <div class="eq-formula">
      <span class="var">ISPU Max</span> = 
      <span class="coef">{{ "{:,.2f}".format(intercept) }}</span> +
      (<span class="coef">{{ "{:,.2f}".format(coefs[0]) }}</span> × <span class="var">pm10</span>) +
      (<span class="coef">{{ "{:,.2f}".format(coefs[1]) }}</span> × <span class="var">o3</span>) +
      (<span class="coef">{{ "{:,.2f}".format(coefs[2]) }}</span> × <span class="var">co</span>)
    </div>
    <table class="coef-table">
      <thead><tr><th>Variabel</th><th>Koefisien (Scaled)</th><th>Kontribusi Relatif</th></tr></thead>
      <tbody>
        {% for v, c, b in coef_rows %}
        <tr>
          <td>{{ v }}</td>
          <td>{{ "{:,.4f}".format(c) }}</td>
          <td>
            <div>{{ "%.1f"|format(b) }}%</div>
            <div class="coef-bar" style="width:{{ b }}%"></div>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="section-title" id="prediksi">Prediksi <i>Nilai ISPU</i></div>
  <div class="predict-card">
    <div class="input-grid">
      <div class="inp-group">
        <label>PM10</label>
        <input type="number" id="p-pm10" placeholder="e.g. 50" min="0" max="300" value="50"/>
      </div>
      <div class="inp-group">
        <label>Kategori</label>
        <select id="p-kategori">
          <option value="BAIK">BAIK</option>
          <option value="SEDANG">SEDANG</option>
          <option value="TIDAK SEHAT">TIDAK SEHAT</option>
          <option value="SANGAT TIDAK SEHAT">SANGAT TIDAK SEHAT</option>
          <option value="BERBAHAYA">BERBAHAYA</option>
        </select>
      </div>
      <div class="inp-group">
        <label>O3 (Ozon)</label>
        <input type="number" id="p-o3" placeholder="e.g. 100" step="0.1" min="0" max="300" value="100"/>
      </div>
      <div class="inp-group">
        <label>CO (Karbon Monoksida)</label>
        <input type="number" id="p-co" placeholder="e.g. 20" min="0" max="100" value="20"/>
      </div>
      <div class="inp-group">
        <label>Polutan Kritis</label>
        <select id="p-critical">
          <option value="O3">O3</option>
          <option value="PM10">PM10</option>
          <option value="PM2.5">PM2.5</option>
          <option value="CO">CO</option>
          <option value="SO2">SO2</option>
        </select>
      </div>
      <div class="inp-group">
        <label>Stasiun</label>
        <select id="p-stasiun">
          <option value="DKI1 Bunderan HI">DKI1 Bunderan HI</option>
          <option value="DKI2 Kelapa Gading">DKI2 Kelapa Gading</option>
          <option value="DKI3 Jagakarsa">DKI3 Jagakarsa</option>
          <option value="DKI4 Lubang Buaya">DKI4 Lubang Buaya</option>
          <option value="DKI5 Kebon Jeruk">DKI5 Kebon Jeruk</option>
        </select>
      </div>
    </div>
    <button class="btn-orange" onclick="predict()">Apply Prediction</button>
    <div class="pred-result" id="pred-result">
      <div class="pred-label">Estimasi Nilai ISPU Puncak</div>
      <div class="pred-val" id="pred-val">–</div>
    </div>
  </div>

  <div class="section-title" id="dataset">Dataset <i>&amp; Filter</i></div>
  <div class="filter-bar">
    <label>Show:</label>
    <select class="sel" id="row-sel" onchange="applyFilters()">
      <option value="100">100 baris</option>
      <option value="200">200 baris</option>
      <option value="500">500 baris</option>
      <option value="1000">1000 baris</option>
      <option value="all">Semua ({{ n_total }})</option>
    </select>

    <input class="search-box" id="search" placeholder="Cari data..." oninput="applyFilters()"/>

    <select class="sel" id="kategori-sel" onchange="applyFilters()">
      <option value="all">Kategori: All</option>
      <option value="BAIK">Kategori: BAIK</option>
      <option value="SEDANG">Kategori: SEDANG</option>
      <option value="TIDAK SEHAT">Kategori: TIDAK SEHAT</option>
      <option value="SANGAT TIDAK SEHAT">Kategori: SANGAT TIDAK SEHAT</option>
      <option value="BERBAHAYA">Kategori: BERBAHAYA</option>
    </select>

    <select class="sel" id="sort-col" onchange="applyFilters()">
      <option value="ispu_max-desc">Sort: ISPU Max ↓</option>
      <option value="ispu_max-asc">Sort: ISPU Max ↑</option>
      <option value="pm10-desc">Sort: PM10 ↓</option>
      <option value="pm10-asc">Sort: PM10 ↑</option>
      <option value="o3-desc">Sort: O3 ↓</option>
    </select>

    <button class="btn-refresh" id="btn-update" onclick="updateCharts()">Sync Charts</button>
    <span class="filter-info">Aktif: <strong id="active-count">–</strong> baris</span>
  </div>

  <div class="tbl-wrap" style="margin-bottom:3.5rem">
    <div class="tbl-scroll">
      <table id="main-table">
        <thead>
          <tr>
            <th>#</th>
            <th>PM10</th><th>Kategori</th><th>O3</th>
            <th>CO</th><th>Kritis</th><th>Stasiun</th><th>ISPU Max</th>
          </tr>
        </thead>
        <tbody id="tbl-body"></tbody>
      </table>
    </div>
    <div class="tbl-footer">
      <span id="tbl-info">Memuat data…</span>
    </div>
  </div>

  <div class="section-title" id="pls">Eksplorasi Heatmap <i>&amp; Model PLS</i></div>
  <div class="charts-2">
    <div class="chart-card" id="card-heatmap" style="cursor: pointer;" onclick="openModal('heatmap')">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-heatmap">1. Heatmap Korelasi Pearson</div>
    </div>
    <div class="chart-card" id="card-pls-scores" style="cursor: pointer;" onclick="openModal('pls_scores')">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-pls-scores">2. PLS Scores Plot (Persebaran Data)</div>
    </div>
  </div>
  
  <div class="charts-2">
    <div class="chart-card" id="card-pls-loadings" style="cursor: pointer;" onclick="openModal('pls_loadings')">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-pls-loadings">3. PLS Loadings (Arah/Pengaruh Variabel X)</div>
    </div>
    <div class="chart-card" style="display:flex; align-items:center; justify-content:center; padding:2rem; text-align:left; background: transparent; box-shadow:none;">
      <p style="color:var(--text); line-height:1.7; font-size:0.95rem;">
        <strong>Interpretasi PLS Components:</strong><br><br>
        <span style="color:var(--accent); font-weight:700;">1. PLS Component 1</span><br>
        Bukan hanya Ozon (O3) atau PM10 secara terpisah, melainkan "variabel buatan/adonan" hasil kombinasi linear dari <code style="color:var(--accent)">pm10, o3, co</code> yang diracik khusus agar <strong>paling kuat/jago</strong> menebak besaran ispu (ISPU Max).<br><br>
        <span style="color:var(--accent); font-weight:700;">2. PLS Component 2</span><br>
        Ini adalah kombinasi linear sisa yang tegak lurus (ortogonal) dari Komponen 1. Berfungsi untuk menangkap <strong>varians/detail pola udara yang belum terjelaskan</strong> oleh Komponen 1 demi meningkatkan presisi model.
      </p>
    </div>
  </div>

  <div class="section-title" id="diagnostik">Diagnostik Regresi <i>(OLS Scatter &amp; Error)</i></div>
  <div class="charts-1">
    <div class="chart-card" id="card-scatter" style="cursor: pointer;" onclick="openModal('scatter')">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-scatter">4. Hubungan X vs ISPU Max beserta Garis Regresi Linier</div>
    </div>
  </div>

  <div class="charts-2">
    <div class="chart-card" id="card-avp" style="cursor: pointer;" onclick="openModal('avp')">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-avp">5. Actual vs Predicted (OLS)</div>
    </div>
    <div class="chart-card" id="card-res" style="cursor: pointer;" onclick="openModal('residual')">
      <div class="chart-spinner"><div class="spin"></div> Memproses...</div>
      <div class="chart-label" id="lbl-res">6. Residual Plot (Homoskedastisitas)</div>
    </div>
  </div>

</div>

<div class="modal-overlay" id="math-modal" onclick="closeModal(event)">
  <div class="modal-content" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal(event)">×</button>
    <div class="modal-grid">
      <div class="modal-img-container" id="modal-img-target">
        </div>
      <div class="modal-math-container">
        <h3 id="modal-title">Judul Analisis</h3>
        <div class="math-box" id="modal-formula">Y = mx + c</div>
        <div id="modal-explanation">Penjelasan matematis akan muncul di sini...</div>
      </div>
    </div>
  </div>
</div>

<script>
const RAW = {{ table_data | safe }};
let currentFiltered = [];

const themeBtn = document.getElementById("theme-btn");
const currentTheme = localStorage.getItem("theme") || "light";

if (currentTheme === "dark") {
  document.documentElement.setAttribute("data-theme", "dark");
  themeBtn.textContent = "☀️";
}

themeBtn.addEventListener("click", () => {
  let theme = document.documentElement.getAttribute("data-theme");
  if (theme === "dark") {
    document.documentElement.removeAttribute("data-theme");
    localStorage.setItem("theme", "light");
    themeBtn.textContent = "🌙";
  } else {
    document.documentElement.setAttribute("data-theme", "dark");
    localStorage.setItem("theme", "dark");
    themeBtn.textContent = "☀️";
  }
  updateCharts(); 
});


async function predict(){
  const pm10 = document.getElementById("p-pm10").value;
  const kategori = document.getElementById("p-kategori").value;
  const o3 = document.getElementById("p-o3").value;
  const co = document.getElementById("p-co").value;
  const critical = document.getElementById("p-critical").value;
  const stasiun = document.getElementById("p-stasiun").value;
  
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({pm10, kategori, o3, co, critical, stasiun}) // Dikirim ke backend
    });
    const data = await res.json();
    document.getElementById("pred-result").style.display = "block";
    document.getElementById("pred-val").textContent = 
      data.prediction.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2});
  } catch(e) { alert("Gagal memprediksi"); }
}

function applyFilters(){
  const limit    = document.getElementById("row-sel").value;
  const search   = document.getElementById("search").value.toLowerCase();
  const kategori = document.getElementById("kategori-sel").value;
  const [sKey, sDir] = document.getElementById("sort-col").value.split("-");

  let data = RAW.filter(r => {
    if(kategori !== "all" && r.kategori !== kategori) return false;
    if(search){
      if(!Object.values(r).join(" ").toLowerCase().includes(search)) return false;
    }
    return true;
  });

  data.sort((a,b) => {
    const va = parseFloat(a[sKey]), vb = parseFloat(b[sKey]);
    return sDir === "asc" ? va-vb : vb-va;
  });

  const total = data.length;
  if(limit !== "all") data = data.slice(0, parseInt(limit));
  currentFiltered = data; 

  const tbody = document.getElementById("tbl-body");
  tbody.innerHTML = data.map((r,i) => `
    <tr>
      <td class="num">${i+1}</td>
      <td class="num">${r.pm10}</td>
      <td><span class="pill pill-m">${r.kategori}</span></td>
      <td class="num">${parseFloat(r.o3).toFixed(2)}</td>
      <td class="num">${r.co}</td>
      <td><span class="pill pill-yes">${r.critical}</span></td>
      <td style="color:var(--muted)">${r.stasiun}</td>
      <td class="charge">${parseFloat(r.ispu_max).toFixed(0)}</td>
    </tr>`).join("");

  document.getElementById("tbl-info").textContent = `Menampilkan ${data.length} dari ${total} baris`;
  document.getElementById("active-count").textContent = data.length;
}

function setCardImg(cardId, b64){
  const card = document.getElementById(cardId);
  const lbl  = card.querySelector(".chart-label");
  const old = card.querySelector(".chart-spinner, img");
  if(old) old.remove();
  const img = document.createElement("img");
  img.src = "data:image/png;base64," + b64;
  card.insertBefore(img, lbl);
}

function setSpinner(cardId){
  const card = document.getElementById(cardId);
  const lbl  = card.querySelector(".chart-label");
  const old  = card.querySelector("img");
  if(old) old.remove();
  if(!card.querySelector(".chart-spinner")){
    const s = document.createElement("div");
    s.className = "chart-spinner";
    s.innerHTML = '<div class="spin"></div> Memproses Visualisasi…';
    card.insertBefore(s, lbl);
  }
}

async function updateCharts(){
  const btn = document.getElementById("btn-update");
  btn.disabled = true;
  btn.textContent = "Syncing...";

  const rows = currentFiltered.map(r => ({
    pm10: parseFloat(r.pm10), o3: parseFloat(r.o3),
    co: parseFloat(r.co), ispu_max: parseFloat(r.ispu_max)
  }));
  
  const currentThemeStr = document.documentElement.getAttribute("data-theme") || "light";

  ["card-heatmap", "card-pls-scores", "card-pls-loadings", "card-scatter", "card-avp", "card-res"].forEach(setSpinner);

  try {
    const res = await fetch("/charts", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ rows: rows, theme: currentThemeStr })
    });
    const data = await res.json();
    if(data.error){ alert("Error: " + data.error); return; }

    setCardImg("card-heatmap", data.heatmap);
    setCardImg("card-pls-scores", data.pls_scores);
    setCardImg("card-pls-loadings", data.pls_loadings); 
    setCardImg("card-scatter", data.scatter);
    setCardImg("card-avp",     data.act_vs_pred);
    setCardImg("card-res",     data.residual);

  } catch(e){
    alert("Gagal memuat grafik: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Sync Charts";
  }
}

// --- FUNGSI DINAMIS MATEMATIKA ---
// Kalkulasi ulang secara otomatis mengikuti jumlah baris (n) yang sedang di-filter
function getMathExplanation(chartType, currentData) {
    const n = currentData.length;
    if (n === 0) return { title: "Error", formula: "-", desc: "Tidak ada data untuk dihitung." };

    let sum_pm10 = 0, sum_o3 = 0, sum_co = 0, sum_ispu = 0;
    currentData.forEach(r => {
        sum_pm10 += parseFloat(r.pm10);
        sum_o3 += parseFloat(r.o3);
        sum_co += parseFloat(r.co);
        sum_ispu += parseFloat(r.ispu_max);
    });
    
    // Menghitung Rata-rata dari n baris yang sedang aktif
    const avg_pm10 = (sum_pm10 / n).toFixed(1);
    const avg_o3 = (sum_o3 / n).toFixed(1);
    const avg_co = (sum_co / n).toFixed(1);
    const avg_ispu = (sum_ispu / n).toFixed(1);

    const dict = {
        'heatmap': {
            title: '1. Heatmap Korelasi Pearson',
            formula: 'r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)² Σ(yi - ȳ)²]',
            desc: `<strong>Forensik Algoritma (n=${n}):</strong><br>
                   Di layar tertera angka korelasi. Kalkulasi pastinya saat ini didapat dari perhitungan dinamis:<br>
                   <ul>
                     <li>Komputer mengunci tepat <strong>${n} baris data</strong> yang sedang Anda filter.</li>
                     <li>Titik keseimbangan rata-rata saat ini berada di: PM10 (x̄) = <strong>${avg_pm10}</strong>, O3 = <strong>${avg_o3}</strong>, CO = <strong>${avg_co}</strong>, dan ISPU Max (ȳ) = <strong>${avg_ispu}</strong>.</li>
                     <li>Rumus Pearson menghitung deviasi setiap data harian terhadap rata-rata di atas, lalu dikalikan dan diakumulasikan.</li>
                   </ul>
                   Warna kotak ini murni representasi matematis mutlak dari <strong>${n} data</strong> terpilih. Matriks terhitung ulang otomatis jika jumlah/filter baris diubah!`
        },
        'pls_scores': {
            title: '2. PLS Scores Plot (Koordinat Harian Udara)',
            formula: 'T₁ = (Z_pm10 × W_pm10) + (Z_o3 × W_o3) + (Z_co × W_co)',
            desc: `<strong>Forensik Algoritma (n=${n}):</strong><br>
                   Setiap titik (dot) di plot grafik ini persis mewakili 1 hari pengamatan (total ada <strong>${n} titik</strong>).
                   <ul>
                     <li><strong>Z-Score:</strong> Data asli terlebih dahulu diseragamkan ukurannya (standardized) berdasarkan perilaku kelompok data sebanyak n=${n} ini.</li>
                     <li>Titik-titik yang terpental menjauh dari gerombolan pusat (0,0) membuktikan adanya hari-hari "anomali" di mana tingkat polusi melampaui kebiasaan harian dari ${n} sampel ini.</li>
                   </ul>`
        },
        'pls_loadings': {
            title: '3. PLS Loadings Plot (Arah Dominasi Polutan)',
            formula: 'p₁ = (Xᵀ t₁) / (t₁ᵀ t₁)',
            desc: `<strong>Forensik Algoritma (n=${n}):</strong><br>
                   Arah panah ini bukan sekadar ilustrasi statis, melainkan proyeksi vektor kekuatan asli dari <strong>${n} data historis</strong> Anda.<br>
                   <ul>
                     <li>Algoritma menghitung varians tarikan dari pusat nilai PM10 (${avg_pm10}), O3 (${avg_o3}), dan CO (${avg_co}).</li>
                     <li>Panah yang menukik paling jauh secara horizontal/vertikal secara harfiah adalah polutan "Bos Utama" yang paling banyak menyebabkan kenaikan ISPU pada <strong>${n} sampel</strong> ini.</li>
                   </ul>`
        },
        'scatter': {
            title: '4. Scatter Plot OLS (Polutan vs ISPU Max)',
            formula: 'm = r × (Sy / Sx)<br>C = ȳ - m(x̄)',
            desc: `<strong>Forensik Algoritma (n=${n}):</strong><br>
                   Garis putus-putus abu-abu digambar menggunakan fungsi regresi linear.<br>
                   <ul>
                     <li>Garis ini dihitung oleh model agar melewati jalur tengah paling membelah <strong>${n} titik observasi</strong>.</li>
                     <li>Sebagai bukti perhitungan dinamis: Jantung dari garis miring tersebut dipastikan akan selalu memotong koordinat Rata-rata. Contoh untuk PM10: Ia melewati X = <strong>${avg_pm10}</strong> dan Y = <strong>${avg_ispu}</strong>.</li>
                   </ul>
                   Jika Anda mengubah filter n dari ${n} ke angka lain, kemiringan lereng (Slope/m) otomatis akan merespons persebaran baru.`
        },
        'avp': {
            title: '5. Actual vs Predicted (Akurasi Prediksi)',
            formula: 'R² = 1 - (SSR / SST)<br>RMSE = √(SSR / n)',
            desc: `<strong>Forensik Algoritma (n=${n}):</strong><br>
                   Grafik ini mengadu langsung Nilai ISPU Asli (Sumbu X) vs Hasil Tebakan Mesin (Sumbu Y) untuk tepat <strong>${n} sampel data</strong>.<br>
                   <ul>
                     <li>Titik oranye yang melayang menjauhi garis diagonal hitam adalah Error/Residu tebakan.</li>
                     <li>Karena saat ini n=<strong>${n}</strong>, komputer mengakumulasikan jarak Error dari seluruh ${n} titik ini untuk merumuskan nilai R² final di layar. Semakin n ditambah, sebaran bias cuaca biasanya membuat error semakin melebar.</li>
                   </ul>`
        },
        'residual': {
            title: '6. Residual Plot (Uji Homoskedastisitas)',
            formula: 'e_i = Y_aktual - Y_prediksi',
            desc: `<strong>Forensik Algoritma (n=${n}):</strong><br>
                   Sumbu vertikal ini menampilkan murni selisih (Residual) tebakan dari tebakan model regresi terhadap <strong>${n} baris data CSV</strong>.<br>
                   <ul>
                     <li>Titik yang berada di atas garis nol (Nol Merah) artinya mesin menebak kerendahan. Titik di bawah garis artinya mesin menebak terlalu tinggi.</li>
                     <li>Dari sebaran <strong>${n} titik</strong> tersebut, uji statistik mencari apakah pola titik melebar menjadi corong ke arah kanan. Jika ya, itu membuktikan secara matematis bahwa akurasi linear model selalu pecah saat indeks udara semakin kotor.</li>
                   </ul>`
        }
    };
    return dict[chartType];
}


function openModal(chartType) {
  const modal = document.getElementById('math-modal');
  const imgTarget = document.getElementById('modal-img-target');
  const title = document.getElementById('modal-title');
  const formula = document.getElementById('modal-formula');
  const explanation = document.getElementById('modal-explanation');
  
  let cardId = '';
  if(chartType === 'heatmap') cardId = 'card-heatmap';
  if(chartType === 'pls_scores') cardId = 'card-pls-scores';
  if(chartType === 'pls_loadings') cardId = 'card-pls-loadings';
  if(chartType === 'scatter') cardId = 'card-scatter';
  if(chartType === 'avp') cardId = 'card-avp';
  if(chartType === 'residual') cardId = 'card-res';
  
  const sourceImg = document.querySelector(`#${cardId} img`);
  
  if(sourceImg) {
    imgTarget.innerHTML = `<img src="${sourceImg.src}" alt="Grafik Detail">`;
  } else {
    imgTarget.innerHTML = `<p style="color:var(--muted)">Memuat grafik...</p>`;
  }

  // Menggunakan kalkulasi dinamis
  const data = getMathExplanation(chartType, currentFiltered);
  
  title.innerHTML = data.title;
  formula.innerHTML = data.formula;
  
  const currentN = currentFiltered.length;
  explanation.innerHTML = `<div style="margin-bottom:12px; display:inline-block; background:rgba(255,103,29,0.15); color:var(--accent); padding:6px 12px; border-radius:8px; font-weight:700; font-size:0.9rem; font-family:'JetBrains Mono', monospace;">Data Terfilter: n = ${currentN} baris</div><br>` + data.desc;

  modal.classList.add('active');
}

function closeModal(e) {
  if(e.target.classList.contains('modal-overlay') || e.target.classList.contains('modal-close')) {
    document.getElementById('math-modal').classList.remove('active');
  }
}

applyFilters(); 
updateCharts(); 
</script>
</body>
</html>
"""

@app.route("/")
def index():
    coefs_raw    = model.coef_
    intercept_raw = model.intercept_
    coefs_orig   = coefs_raw / scaler.scale_
    intercept_orig = intercept_raw - np.dot(coefs_orig, scaler.mean_)

    abs_coefs = np.abs(coefs_raw)
    pct = abs_coefs / abs_coefs.sum() * 100
    coef_rows = list(zip(X_COLS, coefs_orig, pct))

    table_data = df_raw[["pm10","kategori","o3","co","critical","stasiun","ispu_max"]].to_dict("records")

    return render_template_string(
        HTML,
        r2=r2_global, r2_adj=r2_adj_global, rmse=rmse_global, mae=mae_global,
        n_total=len(df_raw), n_train=len(X_train), n_test=len(X_test),
        intercept=intercept_orig,
        coefs=coefs_orig,
        coef_rows=coef_rows,
        table_data=json.dumps(table_data),
    )

@app.route("/charts", methods=["POST"])
def charts_route():
    try:
        body = request.get_json()
        rows = body.get("rows", [])
        is_dark = body.get("theme", "light") == "dark"

        if not rows:
            df_sub = df[X_COLS + [Y_COL]].copy()
        else:
            df_sub = pd.DataFrame(rows, columns=["pm10", "o3", "co", "ispu_max"])

        if len(df_sub) < 2:
            return jsonify({"error": "Data terlalu sedikit."}), 400

        return jsonify({
            "n":             len(df_sub),
            "heatmap":       make_heatmap(df_sub, is_dark),
            "pls_scores":    make_pls_scores(df_sub, is_dark),
            "pls_loadings":  make_pls_loadings(df_sub, is_dark),
            "scatter":       make_scatter(df_sub, is_dark),
            "act_vs_pred":   make_actual_vs_pred(df_sub, is_dark),
            "residual":      make_residual(df_sub, is_dark),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_route():
    body     = request.get_json()
    pm10     = float(body.get("pm10", 50))
    kategori = body.get("kategori", "BAIK")
    o3       = float(body.get("o3", 100))
    co       = float(body.get("co", 20))
    critical = body.get("critical", "O3")
    stasiun  = body.get("stasiun", "DKI1 Bunderan HI")

    # Encode dummynya
    is_kat_sedang = 1 if kategori == "SEDANG" else 0
    is_kat_tidak_sehat = 1 if kategori == "TIDAK SEHAT" else 0
    is_kat_sangat = 1 if kategori == "SANGAT TIDAK SEHAT" else 0
    is_kat_bahaya = 1 if kategori == "BERBAHAYA" else 0

    is_crit_o3 = 1 if critical == "O3" else 0
    is_crit_pm10 = 1 if critical == "PM10" else 0
    is_crit_pm25 = 1 if critical == "PM2.5" else 0
    is_crit_so2 = 1 if critical == "SO2" else 0

    is_dki2 = 1 if stasiun == "DKI2 Kelapa Gading" else 0
    is_dki3 = 1 if stasiun == "DKI3 Jagakarsa" else 0
    is_dki4 = 1 if stasiun == "DKI4 Lubang Buaya" else 0
    is_dki5 = 1 if stasiun == "DKI5 Kebon Jeruk" else 0

    X_in = scaler_full.transform([[
        pm10, o3, co, 
        is_kat_sedang, is_kat_tidak_sehat, is_kat_sangat, is_kat_bahaya,
        is_crit_o3, is_crit_pm10, is_crit_pm25, is_crit_so2,
        is_dki2, is_dki3, is_dki4, is_dki5
    ]])
    pred = model_full.predict(X_in)[0]
    
    pred = max(0, pred)
    return jsonify({"prediction": round(pred, 2)})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Regresi Berganda & PLS — ISPU Jakarta Dataset")
    print("  Buka browser: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)