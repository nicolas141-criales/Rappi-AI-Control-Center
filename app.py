import base64
import os
import re

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.data_loader import (
    load_dataset,
    get_countries,
    get_metrics,
    get_zone_types,
    get_prioritizations,
    get_top_zones,
    get_weekly_trend,
    get_country_averages,
    get_zone_comparison,
    get_orders_trend,
    get_orders_by_country,
    get_metric_by_zone_type,
    get_dataset_summary,
    get_wow_zones,
)
from src.insights import (
    generate_insights, prioritize_insights,
    count_by_severity, count_by_category,
    SEVERITY_CONFIG, CATEGORY_CONFIG,
)
from src.chat import (
    build_analytics_context,
    build_dynamic_context,
    build_full_context,
    compute_intent_chart,
    detect_question_intent,
    extract_suggested_question,
    load_system_prompt,
    parse_chart_spec,
    render_chart_from_spec,
    stream_response,
    strip_chart_block,
)
from src.providers import get_provider
from src.providers.base import AuthError, NetworkError, ProviderError, RateLimitError
from src.report import (
    generate_html_report, generate_markdown_report, build_email_body,
    generate_pdf_report, generate_insights_csv,
    generate_chat_csv, generate_chat_pdf,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rappi Operations Analytics",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
RAPPI_RED   = "#FF441F"
DARK_NAV    = "#1C1C28"
CARD_BG     = "#FFFFFF"
PAGE_BG     = "#F5F6FA"
BORDER      = "#E4E8F0"
TEXT_PRI    = "#1C1C28"
TEXT_SEC    = "#6B7280"
GREEN       = "#10B981"
AMBER       = "#F59E0B"
RED_ALERT   = "#EF4444"
BLUE        = "#3B82F6"
CHART_COLORS = [RAPPI_RED, DARK_NAV, BLUE, GREEN, AMBER, "#8B5CF6", "#EC4899", "#06B6D4"]

# HTML-escape helper — prevents XSS from raw zone/metric names in HTML cards
_H = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

/* ── Fixed-height SPA shell ────────────────────────────────────────────────────
   html/body/stApp are locked at 100dvh — the document never page-scrolls.
   stMain gets an explicit height so it never grows beyond the viewport.
   Scroll is handled per page:
     • Insights / content pages  →  stMain overridden to overflow-y:auto (page CSS)
     • Copiloto                  →  stMain stays hidden; st.container(540) scrolls */
html, body {{
    height: 100% !important;
    overflow: hidden !important;
}}
[data-testid="stApp"] {{
    height: 100dvh !important;
    overflow: hidden !important;
}}
[data-testid="stMain"] {{
    overflow: hidden !important;
    height: 100dvh !important;
}}

/* ── Layout ── */
.main .block-container {{
    padding: 0.5rem 1.25rem 0 1.25rem !important;
    max-width: 100% !important;
    height: 100dvh !important;
    box-sizing: border-box !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}}

/* ── KPI metric cards ── */
[data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1.5px solid {BORDER};
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(28,28,40,0.05);
}}
[data-testid="stMetricLabel"] p {{
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.7px !important;
    text-transform: uppercase !important;
    color: {TEXT_SEC} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 28px !important;
    font-weight: 800 !important;
    color: {TEXT_PRI} !important;
    letter-spacing: -0.5px !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 12px !important;
    font-weight: 600 !important;
}}

/* ── Sidebar nav buttons — layout & sizing ── */
section[data-testid="stSidebar"] [data-testid="baseButton-primary"] {{
    background: {RAPPI_RED} !important;
    background-color: {RAPPI_RED} !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(255,68,31,0.35) !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    border-radius: 6px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 6px 12px !important;
    min-height: 30px !important;
    height: auto !important;
    opacity: 1 !important;
    filter: none !important;
}}
section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    font-size: 12.5px !important;
    border-radius: 6px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 6px 12px !important;
    min-height: 30px !important;
    height: auto !important;
    opacity: 1 !important;
    filter: none !important;
    transition: background 0.1s !important;
}}
section[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {{
    background: rgba(255,68,31,0.12) !important;
}}

/* Collapse gap between nav items */
section[data-testid="stSidebar"] .stButton {{
    margin-bottom: -3px !important;
    margin-top: 0 !important;
}}

/* ── Sidebar — narrower to give content more space ── */
section[data-testid="stSidebar"] {{
    min-width: 210px !important;
    max-width: 210px !important;
    width:     210px !important;
}}
section[data-testid="stSidebar"] > div {{
    width: 210px !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
    background: {DARK_NAV} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {{
    color: rgba(255,255,255,0.82) !important;
}}

/* ── Nav button TEXT — placed after general p rule to win cascade ── */
:root section[data-testid="stSidebar"] [data-testid="baseButton-primary"],
:root section[data-testid="stSidebar"] [data-testid="baseButton-primary"] *,
:root section[data-testid="stSidebar"] button[kind="primary"],
:root section[data-testid="stSidebar"] button[kind="primary"] * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    filter: none !important;
}}
:root section[data-testid="stSidebar"] [data-testid="baseButton-secondary"],
:root section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] *,
:root section[data-testid="stSidebar"] button[kind="secondary"],
:root section[data-testid="stSidebar"] button[kind="secondary"] * {{
    color: {RAPPI_RED} !important;
    -webkit-text-fill-color: {RAPPI_RED} !important;
    opacity: 1 !important;
    filter: none !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: rgba(255,255,255,0.96) !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stMultiSelect > div {{
    background: rgba(255,255,255,0.07) !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.07) !important;
    margin: 6px 0 !important;
}}

/* ── Sidebar action buttons (e.g. Nueva sesion) ── */
section[data-testid="stSidebar"] [data-testid="baseButton-tertiary"] {{
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: rgba(255,255,255,0.42) !important;
    border-radius: 6px !important;
    font-size: 11.5px !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 5px 12px !important;
    min-height: 28px !important;
    height: auto !important;
    transition: background 0.1s, color 0.1s !important;
}}
section[data-testid="stSidebar"] [data-testid="baseButton-tertiary"]:hover {{
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.75) !important;
    border-color: rgba(255,255,255,0.18) !important;
}}
/* ── Sidebar expander ── */
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
    background: rgba(255,255,255,0.02) !important;
    margin-top: 2px !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
    color: rgba(255,255,255,0.45) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
}}
section[data-testid="stSidebar"] .stCaption p {{
    color: rgba(255,255,255,0.28) !important;
    font-size: 10px !important;
}}

/* ── Chat: base defaults (enhanced in tab_chat scope) ── */
[data-testid="stChatMessageContent"] {{
    line-height: 1.65 !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
}}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
    overflow: hidden;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{ border-radius: 10px !important; font-size: 13.5px !important; }}

/* ── Dividers ── */
hr {{ border-color: {BORDER} !important; margin: 0.75rem 0 !important; }}

/* ── Section headers ── */
.section-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {TEXT_SEC};
    margin-bottom: 10px;
    margin-top: 4px;
}}
</style>
""", unsafe_allow_html=True)


# ── Load data + AI provider ───────────────────────────────────────────────────
metrics_df, orders_df = load_dataset("data/Data1.xlsx")
summary = get_dataset_summary(metrics_df, orders_df)
ai_provider = get_provider()

all_countries = get_countries(metrics_df)
all_metrics   = get_metrics(metrics_df)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand mark
    st.markdown(
        f"""<div style="padding:16px 4px 14px 4px;border-bottom:1px solid rgba(255,255,255,0.07);
                        margin-bottom:10px;">
          <div style="display:flex;align-items:center;gap:9px;">
            <div style="width:26px;height:26px;background:{RAPPI_RED};border-radius:7px;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              <span style="font-size:13px;font-weight:900;color:white;letter-spacing:-0.5px;
                           font-family:Inter,-apple-system,sans-serif;line-height:1;">R</span>
            </div>
            <div>
              <div style="font-size:14px;font-weight:800;color:#FFFFFF;letter-spacing:-0.2px;
                          line-height:1.15;">rappi</div>
              <div style="font-size:8.5px;font-weight:500;color:rgba(255,255,255,0.38);
                          letter-spacing:1.6px;text-transform:uppercase;margin-top:1px;">operations</div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Navigation ────────────────────────────────────────────────────────────
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Copiloto IA"

    for _lbl in ["Copiloto IA", "Insights Operacionales"]:
        _active = st.session_state.get("active_page", "Copiloto IA") == _lbl
        st.button(
            _lbl,
            key=f"nav_{abs(hash(_lbl)) % 99991}",
            use_container_width=True,
            type="primary" if _active else "secondary",
            on_click=lambda lbl=_lbl: st.session_state.update({"active_page": lbl}),
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("Nueva sesión", key="sidebar_new_session", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.api_history   = []
        st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Late CSS injection — overrides Streamlit emotion-CSS button text ──────
    st.markdown(f"""<style>
:root section[data-testid="stSidebar"] [data-testid="baseButton-secondary"],
:root section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] p,
:root section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] span,
:root section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] div,
:root section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] em,
:root section[data-testid="stSidebar"] button[kind="secondary"],
:root section[data-testid="stSidebar"] button[kind="secondary"] * {{
    color: {RAPPI_RED} !important;
    -webkit-text-fill-color: {RAPPI_RED} !important;
    opacity: 1 !important;
    filter: none !important;
}}
:root section[data-testid="stSidebar"] [data-testid="baseButton-primary"],
:root section[data-testid="stSidebar"] [data-testid="baseButton-primary"] p,
:root section[data-testid="stSidebar"] [data-testid="baseButton-primary"] span,
:root section[data-testid="stSidebar"] [data-testid="baseButton-primary"] div,
:root section[data-testid="stSidebar"] [data-testid="baseButton-primary"] em,
:root section[data-testid="stSidebar"] button[kind="primary"],
:root section[data-testid="stSidebar"] button[kind="primary"] * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    filter: none !important;
}}
</style>""", unsafe_allow_html=True)

    # ── Status footer ──────────────────────────────────────────────────────────
    _prov_key  = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(ai_provider.name, "ANTHROPIC_API_KEY")
    api_status = "Activo" if os.environ.get(_prov_key) else "Sin clave API"
    api_color  = "#10B981" if os.environ.get(_prov_key) else "#F59E0B"
    st.markdown(
        f"<div style='font-size:10px;color:rgba(255,255,255,0.30);line-height:1.9;padding-top:2px;'>"
        f"<span style='color:{api_color};font-size:7px;vertical-align:middle;'>&#9679;</span>"
        f" IA {api_status} &nbsp;·&nbsp; "
        f"<span style='color:rgba(255,255,255,0.22);'>Rappi 2025</span></div>",
        unsafe_allow_html=True,
    )


# ── Global defaults — each tab defines its own filter context ────────────────
selected_country = None
metric_filter    = "Perfect Orders" if "Perfect Orders" in all_metrics else all_metrics[0]
top_n            = 10
filtered_df      = metrics_df



# ── Insight card renderer ─────────────────────────────────────────────────────

def _card_html(insight: dict) -> str:
    """Compact insight card HTML — collapsed by default, full detail on expand."""
    sev_cfg   = SEVERITY_CONFIG[insight["severity"]]
    cat_cfg   = CATEGORY_CONFIG.get(insight.get("category", "anomaly"), {})
    color     = sev_cfg["color"]
    label     = sev_cfg["label"]
    cat_lbl   = cat_cfg.get("label", "")
    cat_emoji = cat_cfg.get("emoji", "")
    cat_col   = cat_cfg.get("color", TEXT_SEC)
    delta     = insight["delta_pct"]
    sev       = insight["severity"]

    if sev == "opportunity":
        delta_str, delta_color = f"▼ {abs(delta):.1f}%", RED_ALERT
    elif sev == "positive":
        delta_str, delta_color = f"▲ {delta:.1f}%", GREEN
    else:
        delta_str  = f"{delta:+.1f}%"
        delta_color = RED_ALERT if delta < 0 else GREEN

    score_str = f"{insight.get('score', 0):.0f}"
    title     = _H(insight["title"])
    finding   = _H(insight["finding"])
    expl      = _H(insight["explanation"])
    action    = _H(insight["action"])
    meta      = (f"{_H(insight['country'])} · {_H(insight['city'])} · "
                 f"{_H(insight['zone'])} · {_H(insight['metric'])}")

    return (
        f'<div style="border-left:3px solid {color};background:{CARD_BG};'
        f'border-radius:0 6px 6px 0;padding:7px 12px 5px 11px;margin-bottom:4px;'
        f'box-shadow:0 1px 2px rgba(28,28,40,0.04);">'
        # — header row: badges + title + delta
        f'<div style="display:flex;align-items:center;gap:5px;overflow:hidden;">'
        f'<span style="background:{color};color:#fff;font-size:9px;font-weight:700;'
        f'letter-spacing:.4px;padding:2px 7px;border-radius:100px;flex-shrink:0;">{label}</span>'
        f'<span style="background:{cat_col}1A;color:{cat_col};font-size:9px;font-weight:600;'
        f'padding:2px 6px;border-radius:100px;flex-shrink:0;">{cat_emoji} {cat_lbl}</span>'
        f'<span style="background:#F1F3F9;color:{TEXT_SEC};font-size:9px;font-weight:600;'
        f'padding:2px 6px;border-radius:100px;flex-shrink:0;">S {score_str}</span>'
        f'<span style="font-weight:600;font-size:12.5px;color:{TEXT_PRI};flex:1;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{title}</span>'
        f'<span style="background:{delta_color}1A;color:{delta_color};font-size:9.5px;'
        f'font-weight:700;padding:2px 7px;border-radius:100px;flex-shrink:0;'
        f'white-space:nowrap;">{delta_str}</span>'
        f'</div>'
        # — one-line finding preview
        f'<div style="font-size:11px;color:{TEXT_SEC};margin-top:2px;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{finding}</div>'
        # — expandable full detail
        f'<details>'
        f'<summary style="cursor:pointer;font-size:10px;color:{BLUE};list-style:none;'
        f'outline:none;user-select:none;display:inline-block;margin-top:2px;">'
        f'&#9656; Ver detalle</summary>'
        f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid {BORDER};'
        f'font-size:12px;line-height:1.5;">'
        f'<p style="margin:0 0 4px;color:{TEXT_PRI};"><b>Hallazgo:</b> {finding}</p>'
        f'<p style="margin:0 0 4px;color:#4B5563;"><b>Por qué importa:</b> {expl}</p>'
        f'<p style="margin:0 0 4px;color:{BLUE};"><b>Acción:</b> {action}</p>'
        f'<p style="margin:4px 0 0;color:#9CA3AF;font-size:10px;">{meta}</p>'
        f'</div>'
        f'</details>'
        f'</div>'
    )


def _cards_html(items: list[dict]) -> None:
    """Render a list of insight cards as a single HTML block (enables native <details> toggle)."""
    if not items:
        return
    block = '<div style="display:flex;flex-direction:column;">'
    for ins in items:
        block += _card_html(ins)
    block += "</div>"
    st.markdown(block, unsafe_allow_html=True)


# ── Shared chart style helper ─────────────────────────────────────────────────
_FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

def _chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family=_FONT, size=11, color=TEXT_SEC),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=12, r=12, t=40, b=12),
        title_font=dict(size=13, color=TEXT_PRI, family=_FONT),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, family=_FONT),
        ),
        hoverlabel=dict(
            bgcolor="white", bordercolor=BORDER,
            font=dict(size=11, family=_FONT),
        ),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor="rgba(228,232,240,0.4)",
        tickfont=dict(size=11, color=TEXT_SEC),
    )
    fig.update_yaxes(
        gridcolor="rgba(228,232,240,0.55)", gridwidth=0.5,
        zeroline=False, linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color=TEXT_SEC),
    )
    return fig


# ── Page routing ──────────────────────────────────────────────────────────────
_page = st.session_state.get("active_page", "Copiloto IA")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE · Centro de Mando
# ═══════════════════════════════════════════════════════════════════════════════
if _page == "Centro de Mando":

    # ── Local filters ─────────────────────────────────────────────────────────
    _fc1, _fc2, _fc3, _fsp = st.columns([2, 3, 1, 4])
    with _fc1:
        _cf = st.selectbox("País", ["Todos"] + all_countries, key="cmd_country")
        selected_country = None if _cf == "Todos" else _cf
    with _fc2:
        metric_filter = st.selectbox(
            "Métrica", all_metrics, key="cmd_metric",
            index=all_metrics.index("Perfect Orders") if "Perfect Orders" in all_metrics else 0,
        )
    with _fc3:
        top_n = st.number_input("Top N", min_value=5, max_value=30, value=10, step=5, key="cmd_topn")
    filtered_df = metrics_df

    # ── Page header + KPI row ─────────────────────────────────────────────────
    scope_badge = selected_country if selected_country else f"{summary['countries']} paises"
    st.markdown(
        f"""<div style="display:flex;align-items:center;justify-content:space-between;
                        padding:14px 0 18px 0;border-bottom:2px solid {BORDER};margin-bottom:18px;">
          <div>
            <div style="font-size:22px;font-weight:900;color:{TEXT_PRI};letter-spacing:-0.4px;">
              Centro de Operaciones
              <span style="color:{RAPPI_RED};">·</span> LATAM
            </div>
            <div style="font-size:12.5px;color:{TEXT_SEC};margin-top:2px;font-weight:500;">
              {summary['zones']:,} zonas &nbsp;·&nbsp; {summary['cities']:,} ciudades
              &nbsp;·&nbsp; {summary['metrics_count']} metricas &nbsp;·&nbsp; {scope_badge}
            </div>
          </div>
          <div style="display:flex;gap:6px;align-items:center;">
            <span style="background:{RAPPI_RED};color:white;font-size:10.5px;font-weight:700;
                         padding:5px 13px;border-radius:100px;letter-spacing:0.3px;">LIVE</span>
            <span style="background:#F1F3F9;color:{TEXT_SEC};font-size:10.5px;font-weight:600;
                         padding:5px 13px;border-radius:100px;">{metric_filter}</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
    _k1, _k2, _k3, _k4, _k5 = st.columns(5)
    _k1.metric("Paises",   summary["countries"])
    _k2.metric("Ciudades", summary["cities"])
    _k3.metric("Zonas",    summary["zones"])
    _k4.metric("Metricas", summary["metrics_count"])
    _k5.metric(
        "Pedidos (ultima semana)",
        f"{summary['total_orders']:,}",
        f"{summary['orders_wow_pct']:+.1f}% SaS",
    )
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

    # Quick critical alert banner
    quick_insights = generate_insights(filtered_df, country=selected_country, metric=metric_filter)
    q_counts = count_by_severity(quick_insights)
    n_crit = q_counts.get("critical", 0)
    n_warn = q_counts.get("warning", 0)

    if n_crit:
        st.error(
            f"⚠️ **{n_crit} alerta{'s' if n_crit > 1 else ''} critica{'s' if n_crit > 1 else ''}** "
            f"detectada{'s' if n_crit > 1 else ''} en {metric_filter} — revision operativa inmediata recomendada."
        )
    elif n_warn:
        st.warning(
            f"**{n_warn} alerta{'s' if n_warn > 1 else ''}** en {metric_filter} "
            "requiere{'n' if n_warn > 1 else ''} seguimiento esta semana."
        )

    # Country benchmark + network trend
    col_bench, col_trend = st.columns([3, 2])

    with col_bench:
        st.markdown("<div class='section-label'>Benchmark de paises</div>", unsafe_allow_html=True)
        avg_df = get_country_averages(filtered_df, metric_filter)
        # Color code: above network avg = green, below = red
        net_avg = avg_df["avg_value"].mean()
        avg_df["_color"] = avg_df["avg_value"].apply(
            lambda v: GREEN if v >= net_avg else RED_ALERT
        )
        fig_bench = px.bar(
            avg_df, x="COUNTRY", y="avg_value",
            color="_color",
            color_discrete_map={GREEN: GREEN, RED_ALERT: RED_ALERT},
            template=None,
            labels={"avg_value": "Valor prom. W-0", "COUNTRY": ""},
            text="avg_value",
        )
        fig_bench.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            textfont=dict(size=10, color=TEXT_SEC, family=_FONT),
            marker_line_width=0,
        )
        fig_bench.update_layout(showlegend=False)
        fig_bench.add_hline(
            y=net_avg, line_dash="dot", line_color=TEXT_SEC, line_width=1.5,
            annotation_text=f"  Prom. red {net_avg:.3f}",
            annotation_font_size=10, annotation_font_color=TEXT_SEC,
        )
        _chart(fig_bench)
        st.plotly_chart(fig_bench, width='stretch')

    with col_trend:
        st.markdown("<div class='section-label'>Tendencia de red (8 semanas)</div>", unsafe_allow_html=True)
        scope_lbl = selected_country if selected_country else "Todos los paises"
        trend = get_weekly_trend(filtered_df, metric_filter, country=selected_country)
        fig_trend = px.line(
            trend, x="week", y="value", markers=True,
            template=None,
            labels={"value": metric_filter, "week": ""},
            title=scope_lbl,
            color_discrete_sequence=[RAPPI_RED],
        )
        fig_trend.update_traces(
            line=dict(width=2, color=RAPPI_RED),
            marker=dict(size=5, color=RAPPI_RED, line=dict(color="white", width=1.5)),
        )
        _chart(fig_trend)
        st.plotly_chart(fig_trend, width='stretch')

    # WoW signals row
    col_down, col_up = st.columns(2)

    with col_down:
        st.markdown(
            f"<div class='section-label' style='color:{RED_ALERT};'>&#9660; Zonas en deterioro "
            f"(SaS) &nbsp;·&nbsp; {metric_filter}</div>",
            unsafe_allow_html=True,
        )
        worst_df = get_wow_zones(filtered_df, metric_filter, country=selected_country, n=top_n, ascending=True)
        if worst_df.empty:
            st.info("Sin datos SaS disponibles para la seleccion actual.")
        else:
            show_cols = [c for c in ["COUNTRY", "CITY", "ZONE", "wow_pct", "L0W_ROLL"] if c in worst_df.columns]
            st.dataframe(
                worst_df[show_cols].rename(columns={"wow_pct": "Cambio SaS %", "L0W_ROLL": "Valor W-0"}),
                width='stretch', hide_index=True,
            )

    with col_up:
        st.markdown(
            f"<div class='section-label' style='color:{GREEN};'>&#9650; Zonas en mejora "
            f"(SaS) &nbsp;·&nbsp; {metric_filter}</div>",
            unsafe_allow_html=True,
        )
        best_df = get_wow_zones(filtered_df, metric_filter, country=selected_country, n=top_n, ascending=False)
        if best_df.empty:
            st.info("Sin datos SaS disponibles para la seleccion actual.")
        else:
            show_cols = [c for c in ["COUNTRY", "CITY", "ZONE", "wow_pct", "L0W_ROLL"] if c in best_df.columns]
            st.dataframe(
                best_df[show_cols].rename(columns={"wow_pct": "Cambio SaS %", "L0W_ROLL": "Valor W-0"}),
                width='stretch', hide_index=True,
            )

    # Zone type breakdown
    zt_df = get_metric_by_zone_type(filtered_df, metric_filter, country=selected_country)
    if not zt_df.empty:
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Distribucion por tipo de zona</div>", unsafe_allow_html=True)
        fig_zt = px.bar(
            zt_df, x="ZONE_TYPE", y="avg_value", color="ZONE_TYPE",
            template=None,
            labels={"avg_value": "Valor prom.", "ZONE_TYPE": ""},
            color_discrete_sequence=CHART_COLORS,
            text="avg_value",
        )
        fig_zt.update_traces(
            texttemplate="%{text:.3f}", textposition="outside",
            textfont_size=10, marker_line_width=0,
        )
        fig_zt.update_layout(showlegend=False)
        _chart(fig_zt)
        st.plotly_chart(fig_zt, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE · Analisis de Metricas
# ═══════════════════════════════════════════════════════════════════════════════
if _page == "Analisis de Metricas":
    _fm1, _fm2, _fm3, _fmsp = st.columns([2, 3, 1, 4])
    with _fm1:
        _cf = st.selectbox("País", ["Todos"] + all_countries, key="met_country")
        selected_country = None if _cf == "Todos" else _cf
    with _fm2:
        metric_filter = st.selectbox(
            "Métrica", all_metrics, key="met_metric",
            index=all_metrics.index("Perfect Orders") if "Perfect Orders" in all_metrics else 0,
        )
    with _fm3:
        top_n = st.number_input("Top N", min_value=5, max_value=30, value=10, step=5, key="met_topn")
    filtered_df = metrics_df

    col_trend2, col_type2 = st.columns([2, 1])

    with col_trend2:
        st.subheader(f"Tendencia semanal · {metric_filter}")
        scope_lbl2 = selected_country if selected_country else "Todos los paises"
        trend2 = get_weekly_trend(filtered_df, metric_filter, country=selected_country)
        fig_t2 = px.line(
            trend2, x="week", y="value", markers=True,
            template=None,
            labels={"value": "Valor prom.", "week": ""},
            title=scope_lbl2,
            color_discrete_sequence=[RAPPI_RED],
        )
        fig_t2.update_traces(
            line=dict(width=2, color=RAPPI_RED),
            marker=dict(size=5, color=RAPPI_RED, line=dict(color="white", width=1.5)),
        )
        _chart(fig_t2)
        st.plotly_chart(fig_t2, width='stretch')

    with col_type2:
        st.subheader("Por tipo de zona")
        zt2 = get_metric_by_zone_type(filtered_df, metric_filter, country=selected_country)
        fig_zt2 = px.bar(
            zt2, x="ZONE_TYPE", y="avg_value", color="ZONE_TYPE",
            template=None,
            labels={"avg_value": "Valor prom.", "ZONE_TYPE": ""},
            color_discrete_sequence=CHART_COLORS,
        )
        fig_zt2.update_layout(showlegend=False)
        _chart(fig_zt2)
        st.plotly_chart(fig_zt2, width='stretch')

    st.subheader(f"Comparacion de zonas · {metric_filter}")
    if selected_country:
        zone_cmp = get_zone_comparison(filtered_df, metric_filter, selected_country, n=top_n)
        color_col    = "ZONE_TYPE" if "ZONE_TYPE" in zone_cmp.columns else None
        hover_extras = [c for c in ["CITY", "ZONE_PRIORITIZATION"] if c in zone_cmp.columns]
        fig_zc = px.bar(
            zone_cmp, x="ZONE", y="L0W_ROLL", color=color_col,
            template=None,
            labels={"L0W_ROLL": "Valor (ultima semana)", "ZONE": ""},
            hover_data=hover_extras or None,
            color_discrete_sequence=CHART_COLORS,
        )
        fig_zc.update_layout(xaxis_tickangle=-40)
        _chart(fig_zc)
        st.plotly_chart(fig_zc, width='stretch')
        st.dataframe(zone_cmp, width='stretch', hide_index=True)
    else:
        st.info("Selecciona un país específico en el filtro superior para ver la comparación por zona.")

    st.subheader("Tabla de promedios por pais")
    st.dataframe(
        get_country_averages(filtered_df, metric_filter),
        width='stretch', hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE · Pedidos
# ═══════════════════════════════════════════════════════════════════════════════
if _page == "Pedidos":
    _fo1, _fo2, _fosp = st.columns([2, 1, 7])
    with _fo1:
        _cf = st.selectbox("País", ["Todos"] + all_countries, key="ord_country")
        selected_country = None if _cf == "Todos" else _cf
    with _fo2:
        top_n = st.number_input("Top N", min_value=5, max_value=30, value=10, step=5, key="ord_topn")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Tendencia semanal de pedidos")
        scope_ord = selected_country if selected_country else "Todos los paises"
        ord_trend = get_orders_trend(orders_df, country=selected_country)
        fig_ot = px.line(
            ord_trend, x="week", y="orders", markers=True,
            template=None,
            labels={"orders": "Total pedidos", "week": ""},
            title=scope_ord,
            color_discrete_sequence=[RAPPI_RED],
        )
        fig_ot.update_traces(
            line=dict(width=2, color=RAPPI_RED),
            marker=dict(size=5, color=RAPPI_RED, line=dict(color="white", width=1.5)),
        )
        _chart(fig_ot)
        st.plotly_chart(fig_ot, width='stretch')

    with col_b:
        st.subheader("Participacion por pais (ultima semana)")
        obc2 = get_orders_by_country(orders_df)
        fig_pie = px.pie(
            obc2, names="COUNTRY", values="total_orders",
            template=None, hole=0.5,
            color_discrete_sequence=CHART_COLORS,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        _chart(fig_pie)
        st.plotly_chart(fig_pie, width='stretch')

    st.subheader(f"Top {top_n} zonas por pedidos (ultima semana)")
    top_ord_zones = get_top_zones(
        orders_df, metric="Orders", country=selected_country, n=top_n, week_col="L0W",
    )
    st.dataframe(top_ord_zones, width='stretch', hide_index=True)

    st.subheader("Pedidos por pais — barra comparativa")
    obc3 = get_orders_by_country(orders_df)
    fig_obc3 = px.bar(
        obc3, x="COUNTRY", y="total_orders", color="COUNTRY",
        template="plotly_white",
        labels={"total_orders": "Pedidos W-0", "COUNTRY": ""},
        color_discrete_sequence=CHART_COLORS,
        text="total_orders",
    )
    fig_obc3.update_traces(
        texttemplate="%{text:,.0f}", textposition="outside",
        textfont_size=10, marker_line_width=0,
    )
    fig_obc3.update_layout(showlegend=False)
    _chart(fig_obc3)
    st.plotly_chart(fig_obc3, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE · Insights Operacionales
# ═══════════════════════════════════════════════════════════════════════════════
if _page == "Insights Operacionales":

    st.markdown("""<style>
/* ── Insights scroll fix ────────────────────────────────────────────────────────
   stMain becomes the scroll viewport for this page.
   block-container height is released so it grows with content naturally.
   stApp/html/body stay overflow:hidden — no document scroll, no double scrollbar. */
[data-testid="stMain"] {
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
[data-testid="stMainBlockContainer"],
.main .block-container {
    height: auto !important;
    min-height: 0 !important;
    overflow: visible !important;
    padding-bottom: 80px !important;
}
/* Scrollbar on stMain */
[data-testid="stMain"]::-webkit-scrollbar       { width: 4px; }
[data-testid="stMain"]::-webkit-scrollbar-track { background: transparent; }
[data-testid="stMain"]::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.14); border-radius: 2px; }
[data-testid="stMain"] { scrollbar-width: thin; scrollbar-color: rgba(0,0,0,0.12) transparent; }
</style>""", unsafe_allow_html=True)

    # ── Compact filter bar ─────────────────────────────────────────────────────
    _fi1, _fi2, _fi_sp = st.columns([2.5, 2, 4.5])
    with _fi1:
        ins_metric = st.selectbox(
            "Métrica", ["Todas"] + all_metrics, key="ins_metric",
            index=all_metrics.index(metric_filter) + 1,
        )
    with _fi2:
        ins_country = st.selectbox(
            "País", ["Todos"] + all_countries, key="ins_country",
            index=(all_countries.index(selected_country) + 1) if selected_country else 0,
        )

    ins_country_arg = None if ins_country == "Todos" else ins_country
    ins_metric_arg  = None if ins_metric == "Todas" else ins_metric
    insights = generate_insights(filtered_df, country=ins_country_arg, metric=ins_metric_arg)
    counts   = count_by_severity(insights)
    _crit    = counts.get("critical", 0)
    _warn    = counts.get("warning", 0)
    _opp     = counts.get("opportunity", 0)
    _pos     = counts.get("positive", 0)
    _total   = len(insights)

    # Executive selection — top-10 by default; full pool on demand.
    # Key includes filter values so changing filters resets to executive view.
    _show_all_key  = f"ins_show_all_{ins_country}_{ins_metric}"
    _show_all      = st.session_state.get(_show_all_key, False)
    _exec_insights = insights if _show_all else prioritize_insights(insights, top_n=10)

    # ── Severity summary + total — compact pill strip ──────────────────────────
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;
                        padding:8px 0 10px;border-bottom:1px solid {BORDER};">
          <span style="background:{RED_ALERT};color:#fff;font-size:11px;font-weight:700;
                       padding:3px 11px;border-radius:100px;">{_crit} Critico</span>
          <span style="background:{AMBER};color:#fff;font-size:11px;font-weight:700;
                       padding:3px 11px;border-radius:100px;">{_warn} Alerta</span>
          <span style="background:{BLUE};color:#fff;font-size:11px;font-weight:700;
                       padding:3px 11px;border-radius:100px;">{_opp} Oportunidad</span>
          <span style="background:{GREEN};color:#fff;font-size:11px;font-weight:700;
                       padding:3px 11px;border-radius:100px;">{_pos} Positivo</span>
          <span style="margin-left:auto;font-size:12px;color:{TEXT_SEC};">
            <b style="color:{TEXT_PRI};font-size:16px;">{_total}</b> insights</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Category breakdown strip ────────────────────────────────────────────────
    cat_counts = count_by_category(insights)
    _cat_cols  = st.columns(5)
    for _ci, (_cat_key, _cat_cfg) in enumerate(CATEGORY_CONFIG.items()):
        _n   = cat_counts.get(_cat_key, 0)
        _col = _cat_cfg["color"]
        _cat_cols[_ci].markdown(
            f"""<div style="background:{_col}10;border:1px solid {_col}28;border-radius:8px;
                            padding:6px 8px;text-align:center;margin-top:4px;">
              <div style="font-size:16px;font-weight:800;color:{_col};line-height:1.1;">{_n}</div>
              <div style="font-size:9px;font-weight:600;color:{_col};letter-spacing:.3px;
                          text-transform:uppercase;margin-top:1px;opacity:.85;">
                {_cat_cfg['emoji']} {_cat_cfg['label']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Executive view toggle ──────────────────────────────────────────────────
    _n_exec = len(_exec_insights)
    _info_c, _btn_c = st.columns([5, 2])
    with _info_c:
        _mode_desc = (
            f"Top <b>{_n_exec}</b> de <b>{_total}</b> &nbsp;·&nbsp; selección ejecutiva"
            if not _show_all else
            f"Mostrando todos los <b>{_total}</b> insights"
        )
        st.markdown(
            f"<div style='font-size:11.5px;color:{TEXT_SEC};padding:3px 0;'>{_mode_desc}</div>",
            unsafe_allow_html=True,
        )
    with _btn_c:
        _toggle_lbl = f"Ver todos ({_total})" if not _show_all else "Vista ejecutiva"
        st.button(
            _toggle_lbl,
            key="ins_toggle",
            on_click=lambda k=_show_all_key, v=_show_all: st.session_state.update({k: not v}),
        )

    # ── Alert banner ───────────────────────────────────────────────────────────
    if _crit:
        st.error(
            f"**{_crit} alerta{'s' if _crit > 1 else ''} critica{'s' if _crit > 1 else ''}** "
            f"— revision operativa inmediata recomendada."
        )
    elif _warn:
        st.warning(
            f"**{_warn} alerta{'s' if _warn > 1 else ''}** "
            f"requiere{'n' if _warn > 1 else ''} seguimiento esta semana."
        )

    # ── Insight cards ──────────────────────────────────────────────────────────
    if not _exec_insights:
        st.info("Sin insights para la seleccion actual.")
    else:
        left_items  = [i for i in _exec_insights if i["severity"] in ("critical", "warning")]
        right_items = [i for i in _exec_insights if i["severity"] in ("opportunity", "positive")]

        if left_items and right_items:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(
                    f"<div class='section-label' style='color:{RED_ALERT};'>"
                    f"Alertas · {len(left_items)}</div>",
                    unsafe_allow_html=True,
                )
                _cards_html(left_items)
            with col_r:
                st.markdown(
                    f"<div class='section-label' style='color:{GREEN};'>"
                    f"Oportunidades y Logros · {len(right_items)}</div>",
                    unsafe_allow_html=True,
                )
                _cards_html(right_items)
        else:
            _cards_html(_exec_insights)

    # ── Category detail tabs ────────────────────────────────────────────────────
    st.markdown(
        f"<div style='border-top:1px solid {BORDER};margin-top:14px;padding-top:10px;'>"
        f"<span style='font-size:10.5px;font-weight:700;letter-spacing:1.1px;"
        f"text-transform:uppercase;color:{TEXT_SEC};'>Detalle por categoría</span></div>",
        unsafe_allow_html=True,
    )

    _COLS = ["severity", "category", "score", "country", "city", "zone", "metric", "value", "delta_pct"]

    (atab_anom, atab_trend, atab_bench,
     atab_corr, atab_opp_t, atab_pos) = st.tabs([
        "🔺 Anomalías", "📉 Tendencias", "🌎 Benchmarking",
        "🔗 Correlaciones", "💡 Oportunidades", "🟢 Mejoras",
    ])

    def _cat_table(tab, cat_key: str) -> None:
        with tab:
            rows = [i for i in insights if i.get("category") == cat_key]
            if rows:
                _df = pd.DataFrame(rows)[[c for c in _COLS if c in pd.DataFrame(rows).columns]]
                st.dataframe(_df.sort_values("score", ascending=False), width='stretch', hide_index=True)
            else:
                st.info(f"Sin insights de categoría '{CATEGORY_CONFIG[cat_key]['label']}' en la seleccion actual.")

    _cat_table(atab_anom,  "anomaly")
    _cat_table(atab_trend, "trend")
    _cat_table(atab_bench, "benchmark")
    _cat_table(atab_corr,  "correlation")
    _cat_table(atab_opp_t, "opportunity")

    with atab_pos:
        rows = [i for i in insights if i["severity"] == "positive"]
        if rows:
            _df = pd.DataFrame(rows)[[c for c in _COLS if c in pd.DataFrame(rows).columns]]
            st.dataframe(_df.sort_values("score", ascending=False), width='stretch', hide_index=True)
        else:
            st.info("Sin mejoras positivas en la seleccion actual.")

    # ── Exportación de resultados ────────────────────────────────────────────────
    st.markdown(
        f"<div style='border-top:1px solid {BORDER};margin-top:14px;padding-top:12px;'></div>",
        unsafe_allow_html=True,
    )

    _exp_hdr, _exp_csv, _exp_gen = st.columns([3, 1.5, 1.5])
    with _exp_hdr:
        st.markdown(
            f"<div style='font-size:14px;font-weight:700;color:{TEXT_PRI};margin-bottom:2px;'>"
            f"Exportación de resultados</div>"
            f"<div style='font-size:12px;color:{TEXT_SEC};'>"
            f"CSV disponible de inmediato · HTML, PDF y Markdown requieren generar el informe."
            f"</div>",
            unsafe_allow_html=True,
        )
    with _exp_csv:
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        _csv_data = generate_insights_csv(insights).encode("utf-8")
        _csv_scope = (ins_country_arg or "latam").lower().replace(" ", "_")
        st.download_button(
            "Exportar CSV",
            data=_csv_data,
            file_name=f"insights_rappi_{_csv_scope}.csv",
            mime="text/csv",
            width="stretch",
            key="dl_csv_quick",
            help=f"Descarga los {len(insights)} insights actuales como CSV (Excel/Sheets)",
        )
    with _exp_gen:
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        if st.button("Generar Informe", type="primary", key="btn_gen_report", width="stretch"):
            with st.spinner("Generando informe (HTML · PDF · MD)…"):
                _trend_df     = get_weekly_trend(filtered_df, ins_metric_arg or metric_filter,
                                                 country=ins_country_arg)
                _avg_df       = get_country_averages(filtered_df, ins_metric_arg or metric_filter)
                _rep_insights = prioritize_insights(insights, top_n=15)
                _rep_metric   = ins_metric_arg or metric_filter

                st.session_state["_report_html"] = generate_html_report(
                    _rep_insights, summary, _rep_metric, ins_country_arg, _avg_df, _trend_df,
                )
                st.session_state["_report_md"] = generate_markdown_report(
                    _rep_insights, summary, _rep_metric, ins_country_arg, _avg_df,
                )
                try:
                    st.session_state["_report_pdf"] = generate_pdf_report(
                        _rep_insights, summary, _rep_metric, ins_country_arg, _avg_df,
                    )
                except Exception as _pdf_err:
                    st.session_state.pop("_report_pdf", None)
                    st.warning(f"PDF no generado: {_pdf_err}. Instala fpdf2: `pip install fpdf2`")
                st.session_state["_report_email"] = build_email_body(
                    _rep_insights, summary, _rep_metric, ins_country_arg,
                )

    if st.session_state.get("_report_html"):
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

        # ── Download row ──────────────────────────────────────────────────────
        _dl_html_b = st.session_state["_report_html"].encode("utf-8")
        _dl_md_b   = st.session_state["_report_md"].encode("utf-8")
        _dl_pdf_b  = st.session_state.get("_report_pdf")

        _n_dl = 4 if _dl_pdf_b else 3
        _dl_cols = st.columns([1] * _n_dl + [1])
        with _dl_cols[0]:
            st.download_button(
                "Informe HTML",
                data=_dl_html_b,
                file_name="informe_ejecutivo_rappi.html",
                mime="text/html",
                width="stretch",
                key="dl_html",
            )
        with _dl_cols[1]:
            st.download_button(
                "Informe MD",
                data=_dl_md_b,
                file_name="informe_ejecutivo_rappi.md",
                mime="text/markdown",
                width="stretch",
                key="dl_md",
            )
        if _dl_pdf_b:
            with _dl_cols[2]:
                st.download_button(
                    "Informe PDF",
                    data=_dl_pdf_b,
                    file_name="informe_ejecutivo_rappi.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="dl_pdf",
                )
        with _dl_cols[-1]:
            if st.button("Limpiar", width="stretch", key="clear_report"):
                for k in ("_report_html", "_report_md", "_report_pdf", "_report_email"):
                    st.session_state.pop(k, None)
                st.rerun()

        # HTML preview
        with st.expander("Vista previa del informe", expanded=True):
            components.html(st.session_state["_report_html"], height=720, scrolling=True)

        # Email demo
        st.markdown(
            f"<div style='margin-top:12px;font-size:13px;font-weight:600;color:{TEXT_PRI};'>"
            "Enviar por correo (demo)</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Configurar envio de correo"):
            em_col1, em_col2 = st.columns([3, 1])
            with em_col1:
                email_to = st.text_input(
                    "Destinatario",
                    value="operations@rappi.com",
                    key="email_to",
                    placeholder="correo@empresa.com",
                )
            with em_col2:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                send_clicked = st.button(
                    "Simular envio",
                    width="stretch",
                    key="btn_send_email",
                    disabled=not email_to,
                )
            if send_clicked and email_to:
                st.success(
                    f"Simulacion exitosa: el informe seria enviado a **{email_to}** "
                    "via SMTP configurado."
                )
                st.info(
                    "Para activar el envio real, configura estas variables de entorno: "
                    "`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`. "
                    "La funcion `build_email_body()` ya genera el HTML del correo."
                )
            st.markdown(
                "<div style='margin-top:12px;font-size:12px;font-weight:600;"
                f"color:{TEXT_SEC};margin-bottom:4px;'>Vista previa del correo:</div>",
                unsafe_allow_html=True,
            )
            components.html(st.session_state["_report_email"], height=380, scrolling=True)

    # stMain overflow is now CSS-controlled (no stale JS state to clean up).
    # Only need to release the chat input's fixed positioning if stBottom persists in DOM.
    components.html("""<script>
(function() {
    function releaseFixedInput() {
        var ci = window.parent.document.querySelector('.stChatFloatingInputContainer');
        if (ci) { ci.style.position = ''; ci.style.left = ''; ci.style.right = ''; ci.style.bottom = ''; ci.style.zIndex = ''; }
    }
    releaseFixedInput();
    setTimeout(releaseFixedInput, 80);
})();
</script>""", height=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE · Copiloto IA
# ═══════════════════════════════════════════════════════════════════════════════
if _page == "Copiloto IA":

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "api_history" not in st.session_state:
        st.session_state.api_history = []

    _provider_key_map = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
    _key_name   = _provider_key_map.get(ai_provider.name, "ANTHROPIC_API_KEY")
    api_key_set = bool(os.environ.get(_key_name))
    # ── Chat-specific CSS ─────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    /* ── FLEX CHAIN: every ancestor shrinks, only the message div scrolls ── */
    html, body {{
        height: 100dvh !important;
        overflow: hidden !important;
    }}
    [data-testid="stApp"] {{
        height: 100dvh !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
    }}
    [data-testid="stAppViewContainer"] {{
        flex: 1 1 0% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: row !important;
    }}
    [data-testid="stMain"] {{
        flex: 1 1 0% !important;
        min-height: 0 !important;
        min-width: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }}
    [data-testid="stMainBlockContainer"] {{
        flex: 1 1 0% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        padding-top: 0.4rem !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }}
    .main .block-container {{
        flex: 1 1 0% !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        padding-top: 0.4rem !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }}
    /* ── SCROLL CONTAINER ──
       The outer stVerticalBlock (direct child of .block-container) is the ONLY
       scroll container. Messages render inside it directly — no st.container()
       wrapper. JS refines the height precisely after paint.                    */
    .main .block-container > [data-testid="stVerticalBlock"] {{
        height: calc(100dvh - 100px) !important;
        max-height: calc(100dvh - 100px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-bottom: 100px !important;
        box-sizing: border-box !important;
        /* DEBUG — remove after confirming scroll works */
        outline: 2px solid red !important;
        background: rgba(255, 0, 0, 0.04) !important;
        scrollbar-width: thin;
        scrollbar-color: rgba(0,0,0,0.10) transparent;
    }}
    .main .block-container > [data-testid="stVerticalBlock"]::-webkit-scrollbar {{ width: 5px; }}
    .main .block-container > [data-testid="stVerticalBlock"]::-webkit-scrollbar-track {{ background: transparent; }}
    .main .block-container > [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {{
        background: rgba(0,0,0,0.14); border-radius: 3px;
    }}
    /* All nested stVerticalBlocks grow freely — tables, charts, markdown don't clip */
    .main .block-container > [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] {{
        overflow: visible !important;
        height: auto !important;
        min-height: 0 !important;
    }}

    /* ── User bubble — wide, right-aligned ── */
    .umsg {{
        display: flex;
        justify-content: flex-end;
        margin: 5px 0 5px 8%;
    }}
    .ububble {{
        background: {RAPPI_RED};
        color: #FFFFFF;
        border-radius: 20px 20px 4px 20px;
        padding: 11px 17px;
        font-size: 14px;
        line-height: 1.6;
        word-break: break-word;
        overflow-wrap: break-word;
        max-width: 86%;
        box-shadow: 0 3px 10px rgba(255,68,31,0.22);
    }}

    /* ── Assistant message — wide, full-wrap ── */
    [data-testid="stChatMessage"] {{
        margin: 5px 3% 5px 0 !important;
        gap: 10px !important;
        max-width: 97% !important;
        align-items: flex-start !important;
    }}
    [data-testid="stChatMessageContent"] {{
        background: #FFFFFF !important;
        border: 1px solid #EAECF4 !important;
        border-radius: 4px 20px 20px 20px !important;
        padding: 13px 18px 11px !important;
        box-shadow: 0 2px 8px rgba(28,28,40,0.06) !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        width: 100% !important;
    }}
    [data-testid="stChatMessageContent"] p {{
        font-size: 14px !important;
        line-height: 1.7 !important;
        margin-bottom: 8px !important;
        color: {TEXT_PRI} !important;
        overflow-wrap: break-word !important;
    }}
    [data-testid="stChatMessageContent"] strong {{
        font-weight: 700 !important;
        color: {TEXT_PRI} !important;
    }}
    [data-testid="stChatMessageContent"] li {{
        font-size: 14px !important;
        line-height: 1.65 !important;
        color: {TEXT_PRI} !important;
        margin-bottom: 4px !important;
    }}
    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3 {{
        font-size: 15px !important;
        font-weight: 700 !important;
        margin: 14px 0 6px !important;
        color: {TEXT_PRI} !important;
    }}
    [data-testid="stChatMessageContent"] code {{
        font-size: 12.5px !important;
        background: #F4F5F7 !important;
        padding: 2px 7px !important;
        border-radius: 5px !important;
    }}
    [data-testid="stChatMessageContent"] table {{
        font-size: 13px !important;
        width: 100% !important;
        border-collapse: collapse !important;
    }}
    [data-testid="stChatMessageContent"] th,
    [data-testid="stChatMessageContent"] td {{
        padding: 6px 10px !important;
        border-bottom: 1px solid {BORDER} !important;
    }}
    /* Charts inside bubbles: full width */
    [data-testid="stChatMessageContent"] [data-testid="stPlotlyChart"],
    [data-testid="stChatMessageContent"] .stPlotlyChart,
    [data-testid="stChatMessageContent"] [data-testid="element-container"] {{
        width: 100% !important;
    }}

    /* ── Chat input — full width, pinned bottom, ChatGPT style ── */
    .stChatFloatingInputContainer {{
        border-top: none !important;
        background: linear-gradient(to top, {PAGE_BG} 72%, rgba(245,246,250,0)) !important;
        padding: 0 20px 18px !important;
        box-sizing: border-box !important;
    }}
    .stChatFloatingInputContainer textarea {{
        font-size: 14.5px !important;
        border-radius: 28px !important;
        border: 1.5px solid rgba(220,225,238,0.95) !important;
        background: #FFFFFF !important;
        box-shadow: 0 4px 22px rgba(28,28,40,0.08) !important;
        padding: 13px 22px !important;
        line-height: 1.55 !important;
        min-height: 52px !important;
        transition: box-shadow 0.18s, border-color 0.18s !important;
    }}
    .stChatFloatingInputContainer textarea:focus {{
        border-color: rgba(255,68,31,0.38) !important;
        box-shadow: 0 4px 22px rgba(28,28,40,0.10), 0 0 0 3px rgba(255,68,31,0.09) !important;
        outline: none !important;
    }}

    /* ── Follow-up suggestion chip ── */
    .followup .stButton > button {{
        background: #EFF6FF;
        border: 1.5px solid #BFDBFE;
        border-radius: 100px;
        color: #1D4ED8;
        font-size: 13px;
        font-weight: 500;
        padding: 8px 16px;
        height: auto;
        min-height: 36px;
        white-space: normal;
        line-height: 1.45;
        text-align: left;
        margin-top: 6px;
    }}
    .followup .stButton > button:hover {{
        background: #DBEAFE;
        border-color: #3B82F6;
        color: #1E40AF;
    }}

    /* ── Starter prompt chips ── */
    .starter-prompt .stButton > button {{
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(220,225,238,0.85);
        border-radius: 10px;
        color: {TEXT_SEC};
        font-size: 12.5px;
        font-weight: 400;
        padding: 7px 12px;
        height: auto;
        min-height: 36px;
        white-space: normal;
        line-height: 1.4;
        text-align: left;
        box-shadow: none;
        transition: border-color 0.15s, box-shadow 0.15s, color 0.15s, background 0.15s;
    }}
    .starter-prompt .stButton > button:hover {{
        background: #FFFFFF;
        border-color: rgba(255,68,31,0.40);
        box-shadow: 0 2px 10px rgba(255,68,31,0.10);
        color: {TEXT_PRI};
    }}

    /* ── Inline export pills ── */
    .chat-ex-btn {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 6px 15px;
        background: transparent;
        border: 1px solid {BORDER};
        border-radius: 100px;
        color: {TEXT_SEC};
        font-size: 12px;
        font-weight: 500;
        text-decoration: none;
        transition: border-color 0.15s, color 0.15s, background 0.15s;
        font-family: Inter, -apple-system, sans-serif;
        line-height: 1.4;
    }}
    .chat-ex-btn:hover {{
        border-color: {RAPPI_RED};
        color: {RAPPI_RED};
        background: rgba(255,68,31,0.04);
        text-decoration: none;
    }}

    /* ── COLLAPSE EMPTY WRAPPERS ─────────────────────────────────────────────
       Streamlit emits wrapper divs for every conditional block even when the
       block renders nothing. They carry emotion-cache padding/min-height and
       create visible blank space. Targets below zero them out safely.         */

    /* element-container: wraps every st.xxx() output; empty = nothing rendered */
    .element-container:empty {{
        display: none !important;
    }}

    /* stVerticalBlock: Streamlit's column/row container; empty = unused branch */
    [data-testid="stVerticalBlock"]:empty {{
        display: none !important;
    }}

    /* stVerticalBlockBorderWrapper: from st.container(height=N) if any remain */
    [data-testid="stVerticalBlockBorderWrapper"]:empty,
    [data-testid="stVerticalBlockBorderWrapper"] > div:empty {{
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* stCustomComponent: wrapper around components.html(); height:0 iframe
       but the surrounding div keeps its own padding unless zeroed here        */
    [data-testid="stCustomComponent"] {{
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }}

    /* Catch any leftover emotion-cache div that is truly empty inside scroll area */
    .main .block-container > [data-testid="stVerticalBlock"] > div:empty {{
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* Streamlit sometimes injects stSpacer or stHeading wrappers as placeholders */
    [data-testid="stSpacer"] {{
        display: none !important;
    }}

    /* ── components.html(height=0) iframe wrapper ────────────────────────────
       eqt0gmo8 is the emotion hash for this specific wrapper div. It contains
       a height:0 iframe so :empty doesn't match it. Zero it directly.
       The structural fallback (iframe[height="0"]) handles future version changes. */
    .eqt0gmo8 {{
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }}
    iframe[height="0"],
    iframe[height="0px"] {{
        display: none !important;
    }}
    /* Structural fallback: any direct child of stCustomComponent that only wraps a hidden iframe */
    [data-testid="stCustomComponent"] > div {{
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    if not api_key_set:
        st.warning(f"Configura **{_key_name}** en `.env` y reinicia Streamlit.", icon="⚠️")

    # ── Helper: render one history message ────────────────────────────────────
    def _render_history_msg(msg: dict) -> None:
        if msg["role"] == "user":
            escaped = (msg["content"]
                       .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            st.markdown(
                f'<div class="umsg"><div class="ububble">{escaped}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message("assistant", avatar="🛵"):
                body, suggestion = extract_suggested_question(msg["content"])
                st.markdown(body)
                _hist_fig = None
                if msg.get("user_question"):
                    _hist_intent = detect_question_intent(msg["user_question"], metrics_df)
                    _hist_fig = compute_intent_chart(
                        metrics_df, orders_df, _hist_intent, msg["user_question"]
                    )
                if _hist_fig is None and msg.get("chart_spec"):
                    _hist_fig = render_chart_from_spec(
                        msg["chart_spec"], metrics_df, orders_df, None, compact=True
                    )
                if _hist_fig:
                    st.plotly_chart(_hist_fig, use_container_width=True)
                if suggestion and api_key_set:
                    _fu_key = f"fu_{abs(hash(suggestion + body[:20])) % 999983}"
                    st.markdown('<div class="followup">', unsafe_allow_html=True)
                    st.button(
                        f"💡  {suggestion}",
                        key=_fu_key,
                        on_click=lambda q=suggestion: st.session_state.update(
                            {"_pending_question": q}
                        ),
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── 1. Capture user input (renders as fixed floating bar regardless of DOM pos) ──
    pending    = st.session_state.pop("_pending_question", None)
    user_input = st.chat_input(
        "Pregunta sobre operaciones Rappi...",
        disabled=not api_key_set,
    ) or pending

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # ── 2. Messages rendered DIRECTLY into the outer stVerticalBlock ──────────
    #    That element is the scroll container (CSS + JS both target it).
    #    No st.container() wrapper — messages are direct children.
    if not st.session_state.chat_messages:
        # ── Welcome state ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center;padding:30px 0 20px;user-select:none;">
          <div style="display:inline-flex;align-items:center;gap:7px;
                      margin-bottom:10px;">
            <div style="width:28px;height:28px;background:{RAPPI_RED};border-radius:8px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:14px;box-shadow:0 3px 10px rgba(255,68,31,0.22);">🛵</div>
            <span style="font-size:11px;font-weight:600;color:{TEXT_SEC};
                         letter-spacing:1.4px;text-transform:uppercase;">Copiloto IA</span>
          </div>
          <div style="font-size:22px;font-weight:800;color:{TEXT_PRI};
                      letter-spacing:-0.5px;line-height:1.25;margin-bottom:7px;">
            ¿En qué te puedo ayudar?
          </div>
          <div style="font-size:12.5px;color:{TEXT_SEC};line-height:1.6;max-width:380px;margin:0 auto;">
            Pregúntame sobre métricas, zonas, tendencias o pedidos en LATAM
          </div>
        </div>
        """, unsafe_allow_html=True)

        _starter_prompts = [
            "¿Cuáles son las 10 zonas con mayor Lead Penetration en LATAM esta semana?",
            "Compara Perfect Orders entre zonas Wealthy y Non Wealthy en Colombia",
            "Muéstrame la evolución de Turbo Adoption en México en las últimas 8 semanas",
            "¿Qué zonas High Priority en Brasil tienen mayor caída de Perfect Orders esta semana?",
            "¿Cuáles son los países con mayor crecimiento de pedidos en las últimas 4 semanas?",
            "¿Existe correlación entre Perfect Orders y Turbo Adoption a nivel de zona?",
        ]
        _pad_l, _grid_c, _pad_r = st.columns([0.5, 9, 0.5])
        with _grid_c:
            _sp_col1, _sp_col2 = st.columns(2, gap="small")
            for i, _sp in enumerate(_starter_prompts):
                _col = _sp_col1 if i % 2 == 0 else _sp_col2
                with _col:
                    st.markdown('<div class="starter-prompt">', unsafe_allow_html=True)
                    st.button(
                        _sp,
                        key=f"starter_{i}",
                        use_container_width=True,
                        on_click=lambda q=_sp: st.session_state.update({"_pending_question": q}),
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── Conversation history ───────────────────────────────────────────────
        for msg in st.session_state.chat_messages:
            _render_history_msg(msg)

        # ── Stream response for new input ─────────────────────────────────────
        if user_input:
            _cur_intent   = detect_question_intent(user_input, metrics_df)
            context       = build_dynamic_context(metrics_df, orders_df, user_input)
            system_prompt = load_system_prompt()

            with st.chat_message("assistant", avatar="🛵"):
                response_ph = st.empty()
                response_ph.markdown(
                    f"<span style='color:{TEXT_SEC};font-size:13px;"
                    "font-style:italic;'>Analizando datos operativos…</span>",
                    unsafe_allow_html=True,
                )
                full_text  = ""
                error_text = None

                try:
                    for chunk in stream_response(
                        user_input, context, st.session_state.api_history,
                        system_prompt, provider=ai_provider,
                    ):
                        full_text += chunk
                        display = re.sub(r"```chart.*?```", "", full_text, flags=re.DOTALL)
                        display = re.sub(r"```chart[^`]*$", "", display, flags=re.DOTALL)
                        display = re.sub(
                            r"\*\*Pregunta\s+sugerida:\*\*.*", "", display,
                            flags=re.DOTALL | re.IGNORECASE,
                        ).strip()
                        response_ph.markdown(display + " ▌")

                    clean = strip_chart_block(full_text)
                    body, suggestion = extract_suggested_question(clean)
                    response_ph.markdown(body)

                    chart_spec = parse_chart_spec(full_text)
                    _new_fig = compute_intent_chart(
                        metrics_df, orders_df, _cur_intent, user_input
                    )
                    if _new_fig is None and chart_spec:
                        _new_fig = render_chart_from_spec(
                            chart_spec, metrics_df, orders_df, None, compact=True
                        )
                    if _new_fig:
                        st.plotly_chart(_new_fig, use_container_width=True)

                    if suggestion and api_key_set:
                        _fu_key_new = f"fu_new_{abs(hash(suggestion)) % 999983}"
                        st.markdown('<div class="followup">', unsafe_allow_html=True)
                        st.button(
                            f"💡  {suggestion}",
                            key=_fu_key_new,
                            on_click=lambda q=suggestion: st.session_state.update(
                                {"_pending_question": q}
                            ),
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                except RateLimitError:
                    error_text = "El copiloto no está disponible en este momento. Intenta de nuevo en unos minutos."
                except AuthError:
                    error_text = "Clave API inválida. Verifica la configuración."
                except NetworkError:
                    error_text = "Sin conexión con la API. Verifica tu red."
                except ProviderError as exc:
                    error_text = f"Error del proveedor: {exc}"
                except Exception as exc:
                    error_text = f"Error inesperado: {exc}"

                if error_text:
                    response_ph.error(error_text)

            if not error_text:
                stored   = strip_chart_block(full_text)
                spec_out = parse_chart_spec(full_text)
                st.session_state.chat_messages.append({
                    "role":          "assistant",
                    "content":       stored,
                    "chart_spec":    spec_out,
                    "user_question": user_input,
                })
                st.session_state.api_history.append({"role": "user",      "content": user_input})
                body_only, _ = extract_suggested_question(stored)
                st.session_state.api_history.append({"role": "assistant", "content": body_only})

        st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)

    # ── 3. Inline export bar (scrolls with messages) ──────────────────────────
    _chat_msgs = st.session_state.get("chat_messages", [])
    if any(m["role"] == "assistant" for m in _chat_msgs):
        _ex_csv_b64 = _ex_pdf_b64 = ""
        try:
            _ex_csv_b64 = base64.b64encode(
                generate_chat_csv(_chat_msgs).encode("utf-8")
            ).decode()
        except Exception:
            pass
        try:
            _ex_pdf_b64 = base64.b64encode(
                generate_chat_pdf(_chat_msgs)
            ).decode()
        except Exception:
            pass

        _ex_links = ""
        if _ex_pdf_b64:
            _ex_links += (
                f'<a href="data:application/pdf;base64,{_ex_pdf_b64}" '
                f'download="copiloto_rappi.pdf" class="chat-ex-btn">📄 Exportar PDF</a>'
            )
        if _ex_csv_b64:
            _ex_links += (
                f'<a href="data:text/csv;base64,{_ex_csv_b64}" '
                f'download="copiloto_rappi.csv" class="chat-ex-btn">📊 Exportar CSV</a>'
            )
        if _ex_links:
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end;gap:7px;'
                f'padding:5px 0 10px;align-items:center;">{_ex_links}</div>',
                unsafe_allow_html=True,
            )

    # JS: pin input + set precise height on the outer stVerticalBlock (the scroll
    # container). setProperty with 'important' wins over CSS !important.
    components.html("""<script>
    (function() {
        var doc = window.parent.document;
        var win = window.parent;

        function getScrollEl() {
            // The outer stVerticalBlock is the direct child of .block-container.
            // It is the scroll container — messages render directly inside it.
            var bc = doc.querySelector('[data-testid="stMainBlockContainer"]');
            if (!bc) return null;
            // Walk children until we find a stVerticalBlock
            var kids = bc.children;
            for (var i = 0; i < kids.length; i++) {
                if (kids[i].getAttribute('data-testid') === 'stVerticalBlock') {
                    return kids[i];
                }
            }
            // Fallback: first child of block-container's first div child
            var inner = bc.firstElementChild;
            return inner ? inner.querySelector('[data-testid="stVerticalBlock"]') : null;
        }

        function fixChatLayout() {
            var chatInput = doc.querySelector('.stChatFloatingInputContainer');
            var sidebar   = doc.querySelector('[data-testid="stSidebar"]');

            // 1. Pin input to viewport bottom
            if (chatInput) {
                var sr = sidebar
                    ? Math.max(sidebar.getBoundingClientRect().right, 0) : 0;
                chatInput.style.setProperty('position', 'fixed',  'important');
                chatInput.style.setProperty('bottom',   '0',      'important');
                chatInput.style.setProperty('left',     sr + 'px','important');
                chatInput.style.setProperty('right',    '0',      'important');
                chatInput.style.setProperty('z-index',  '1000',   'important');
            }

            // 2. Set precise height on the scroll container
            var scrollEl = getScrollEl();
            if (scrollEl) {
                var top    = scrollEl.getBoundingClientRect().top;
                var inputH = chatInput
                    ? Math.max(chatInput.getBoundingClientRect().height, 60) : 90;
                var avail  = Math.floor(win.innerHeight - top - inputH - 8);
                if (avail > 80) {
                    scrollEl.style.setProperty('height',     avail + 'px', 'important');
                    scrollEl.style.setProperty('max-height', avail + 'px', 'important');
                    scrollEl.style.setProperty('overflow-y', 'auto',       'important');
                    scrollEl.style.setProperty('overflow-x', 'hidden',     'important');
                }
                // 3. Scroll to newest message
                scrollEl.scrollTop = scrollEl.scrollHeight;
            }
        }

        fixChatLayout();
        [80, 300, 700, 1500].forEach(function(t) { setTimeout(fixChatLayout, t); });

        win.addEventListener('resize', fixChatLayout);
        var sb = doc.querySelector('[data-testid="stSidebar"]');
        if (sb && win.ResizeObserver) new win.ResizeObserver(fixChatLayout).observe(sb);
    })();
    </script>""", height=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE · Datos
# ═══════════════════════════════════════════════════════════════════════════════
if _page == "Datos":
    sheet = st.radio("Hoja", ["RAW_INPUT_METRICS", "RAW_ORDERS"], horizontal=True)
    raw = metrics_df if sheet == "RAW_INPUT_METRICS" else orders_df

    col_info, col_dl = st.columns([4, 1])
    with col_info:
        n_rows = st.slider("Filas a mostrar", 10, min(500, len(raw)), 50, step=10)
    with col_dl:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        st.download_button(
            "Descargar CSV",
            data=raw.to_csv(index=False).encode("utf-8"),
            file_name=f"{sheet.lower()}.csv",
            mime="text/csv",
            width='stretch',
        )

    st.dataframe(raw.head(n_rows), width='stretch')
    st.markdown(
        f"<div style='font-size:11.5px;color:{TEXT_SEC};margin-top:6px;'>"
        f"Dimension completa: <b>{raw.shape[0]:,}</b> filas x <b>{raw.shape[1]}</b> columnas</div>",
        unsafe_allow_html=True,
    )
