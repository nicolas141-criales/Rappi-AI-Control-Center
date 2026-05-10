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
from src.insights import generate_insights, count_by_severity, SEVERITY_CONFIG
from src.chat import (
    build_analytics_context,
    build_dynamic_context,
    build_full_context,
    extract_suggested_question,
    load_system_prompt,
    parse_chart_spec,
    render_chart_from_spec,
    stream_response,
    strip_chart_block,
)
from src.providers import get_provider
from src.providers.base import AuthError, NetworkError, ProviderError, RateLimitError
from src.report import generate_html_report, generate_markdown_report, build_email_body

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rappi Operations Analytics",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
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

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

/* ── Layout ── */
.main .block-container {{
    padding: 0.5rem 2rem 3rem 2rem !important;
    max-width: 1480px !important;
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

/* ── Tab navigation (pill style) ── */
.stTabs [data-baseweb="tab-list"] {{
    background: #ECEEF5;
    border-radius: 12px;
    padding: 5px 5px;
    gap: 2px;
    border-bottom: none !important;
    box-shadow: none !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 9px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: {TEXT_SEC} !important;
    padding: 7px 15px !important;
    transition: all 0.15s !important;
}}
.stTabs [aria-selected="true"][data-baseweb="tab"] {{
    background: {CARD_BG} !important;
    color: {TEXT_PRI} !important;
    box-shadow: 0 1px 5px rgba(28,28,40,0.13) !important;
}}
/* ── Copiloto IA tab (first) — RAPPI RED brand ── */
.stTabs [data-baseweb="tab"]:first-child {{
    color: {RAPPI_RED} !important;
    font-weight: 700 !important;
    font-size: 12.5px !important;
}}
.stTabs [data-baseweb="tab"]:first-child[aria-selected="true"] {{
    background: {RAPPI_RED} !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(255,68,31,0.28) !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div:first-child {{
    background: {DARK_NAV} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {{
    color: rgba(255,255,255,0.78) !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: rgba(255,255,255,0.95) !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stMultiSelect > div {{
    background: rgba(255,255,255,0.07) !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.12) !important;
}}

/* ── Sidebar shortcut buttons ── */
section[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: rgba(255,255,255,0.80) !important;
    border-radius: 10px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 9px 13px !important;
    transition: all 0.15s !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,68,31,0.18) !important;
    border-color: rgba(255,68,31,0.40) !important;
    color: #FFFFFF !important;
}}
/* ── Sidebar expander ── */
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.03) !important;
    margin-top: 4px !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
    color: rgba(255,255,255,0.55) !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
}}
section[data-testid="stSidebar"] .stCaption p {{
    color: rgba(255,255,255,0.35) !important;
    font-size: 10.5px !important;
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
        f"""<div style="padding:20px 4px 18px 4px;border-bottom:1px solid rgba(255,255,255,0.1);
                        margin-bottom:16px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:34px;height:34px;background:{RAPPI_RED};border-radius:9px;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              <span style="font-size:18px;line-height:1;">🛵</span>
            </div>
            <div>
              <div style="font-size:18px;font-weight:900;color:#FFFFFF;letter-spacing:-0.3px;
                          line-height:1.1;">rappi</div>
              <div style="font-size:9.5px;font-weight:600;color:rgba(255,255,255,0.45);
                          letter-spacing:1.8px;text-transform:uppercase;">operations</div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Quick analysis shortcuts ───────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;"
        "color:rgba(255,255,255,0.35);margin-bottom:8px;margin-top:0;'>ANÁLISIS RÁPIDO</p>",
        unsafe_allow_html=True,
    )
    _SHORTCUTS = [
        ("⚠️  Alertas críticas",    "¿Cuáles son las zonas con mayor deterioro SaS esta semana?"),
        ("📈  Mayor crecimiento",    "Top 10 zonas con mayor crecimiento de pedidos en las últimas 4 semanas"),
        ("🌎  Benchmarking países",  "Compara Perfect Orders entre todos los países LATAM"),
        ("💡  Oportunidades",        "Principales zonas de oportunidad en Lead Penetration esta semana"),
        ("🏆  Top Perfect Orders",   "Top 10 zonas por Perfect Orders en la red esta semana"),
        ("💰  Gross Profit UE",      "Ranking de países por Gross Profit UE esta semana"),
    ]
    for _lbl, _q in _SHORTCUTS:
        if st.button(_lbl, key=f"sc_{abs(hash(_lbl)) % 99991}", use_container_width=True):
            st.session_state._pending_question = _q
            st.rerun()

    # Nueva sesión
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("↺  Nueva sesión", key="sidebar_new_session", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.api_history   = []
        st.rerun()

    st.markdown("---")

    # ── Status footer ──────────────────────────────────────────────────────────
    _prov_key  = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(ai_provider.name, "ANTHROPIC_API_KEY")
    api_status = "Activo" if os.environ.get(_prov_key) else "Sin clave API"
    api_color  = "#10B981" if os.environ.get(_prov_key) else "#F59E0B"
    st.markdown(
        f"<div style='font-size:10.5px;color:rgba(255,255,255,0.38);line-height:2.0;'>"
        f"<span style='color:{api_color};'>&#9679;</span> IA Copiloto: "
        f"<span style='color:rgba(255,255,255,0.58);font-weight:600;'>{api_status}</span><br>"
        f"Rappi AI Engineer Assessment · 2025</div>",
        unsafe_allow_html=True,
    )


# ── Global defaults — each tab defines its own filter context ────────────────
selected_country = None
metric_filter    = "Perfect Orders" if "Perfect Orders" in all_metrics else all_metrics[0]
top_n            = 10
filtered_df      = metrics_df



# ── Insight card renderer ─────────────────────────────────────────────────────
def _render_card(insight: dict) -> None:
    cfg   = SEVERITY_CONFIG[insight["severity"]]
    color = cfg["color"]
    label = cfg["label"]
    delta = insight["delta_pct"]
    delta_str  = f"{delta:+.1f}%"
    delta_color = GREEN if delta > 0 else RED_ALERT
    st.markdown(
        f"""
        <div style="border-left:4px solid {color};background:{CARD_BG};border-radius:0 10px 10px 0;
                    padding:14px 18px;margin-bottom:10px;
                    box-shadow:0 1px 4px rgba(28,28,40,0.07);">
          <div style="margin-bottom:8px;display:flex;align-items:center;gap:7px;flex-wrap:wrap;">
            <span style="background:{color};color:white;font-size:9.5px;font-weight:700;
                         letter-spacing:.6px;padding:3px 9px;border-radius:100px;">{label}</span>
            <span style="background:#F1F3F9;color:{delta_color};font-size:10px;font-weight:700;
                         padding:3px 9px;border-radius:100px;">{delta_str}</span>
            <span style="font-weight:700;font-size:14px;color:{TEXT_PRI};">{insight['title']}</span>
          </div>
          <p style="margin:4px 0;color:{TEXT_PRI};font-size:13.5px;line-height:1.5;">
            <b>Hallazgo:</b> {insight['finding']}</p>
          <p style="margin:4px 0;color:#4B5563;font-size:12.5px;line-height:1.5;">
            <b>Por que importa:</b> {insight['explanation']}</p>
          <p style="margin:4px 0 0 0;color:{BLUE};font-size:12.5px;line-height:1.5;">
            <b>Accion recomendada:</b> {insight['action']}</p>
          <p style="margin:8px 0 0 0;color:#9CA3AF;font-size:10.5px;">
            {insight['country']} &nbsp;·&nbsp; {insight['city']}
            &nbsp;·&nbsp; {insight['zone']} &nbsp;·&nbsp; {insight['metric']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Shared chart style helper ─────────────────────────────────────────────────
def _chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font_family="Inter, -apple-system, sans-serif",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=12, r=12, t=40, b=12),
        title_font_size=13,
        title_font_color=TEXT_PRI,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font_size=11,
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont_size=11)
    fig.update_yaxes(gridcolor="#F0F2F8", linecolor="white", tickfont_size=11)
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_insights, tab_cmd, tab_metrics, tab_orders, tab_data = st.tabs([
    "Copiloto IA",
    "Alertas e Insights",
    "Centro de Mando",
    "Analisis de Metricas",
    "Pedidos",
    "Datos",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · Centro de Mando
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cmd:

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
            template="plotly_white",
            labels={"avg_value": "Valor prom. W-0", "COUNTRY": ""},
            text="avg_value",
        )
        fig_bench.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            textfont_size=10,
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
            template="plotly_white",
            labels={"value": metric_filter, "week": ""},
            title=scope_lbl,
            color_discrete_sequence=[RAPPI_RED],
        )
        fig_trend.update_traces(line_width=2.5, marker_size=8, marker_color=RAPPI_RED)
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
            template="plotly_white",
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
# TAB 2 · Analisis de Metricas
# ═══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
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
            template="plotly_white",
            labels={"value": "Valor prom.", "week": ""},
            title=scope_lbl2,
            color_discrete_sequence=[RAPPI_RED],
        )
        fig_t2.update_traces(line_width=2.5, marker_size=8)
        _chart(fig_t2)
        st.plotly_chart(fig_t2, width='stretch')

    with col_type2:
        st.subheader("Por tipo de zona")
        zt2 = get_metric_by_zone_type(filtered_df, metric_filter, country=selected_country)
        fig_zt2 = px.bar(
            zt2, x="ZONE_TYPE", y="avg_value", color="ZONE_TYPE",
            template="plotly_white",
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
            template="plotly_white",
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
# TAB 3 · Pedidos
# ═══════════════════════════════════════════════════════════════════════════════
with tab_orders:
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
            template="plotly_white",
            labels={"orders": "Total pedidos", "week": ""},
            title=scope_ord,
            color_discrete_sequence=[RAPPI_RED],
        )
        fig_ot.update_traces(line_width=2.5, marker_size=8)
        _chart(fig_ot)
        st.plotly_chart(fig_ot, width='stretch')

    with col_b:
        st.subheader("Participacion por pais (ultima semana)")
        obc2 = get_orders_by_country(orders_df)
        fig_pie = px.pie(
            obc2, names="COUNTRY", values="total_orders",
            template="plotly_white", hole=0.42,
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
# TAB 4 · Alertas e Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:

    # Insight filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        ins_metric = st.selectbox(
            "Metrica", ["Todas"] + all_metrics, key="ins_metric",
            index=all_metrics.index(metric_filter) + 1,
        )
    with fc2:
        ins_country = st.selectbox(
            "Pais", ["Todos"] + all_countries, key="ins_country",
            index=(all_countries.index(selected_country) + 1) if selected_country else 0,
        )
    with fc3:
        sev_options = list(SEVERITY_CONFIG.keys())
        sev_filter = st.multiselect(
            "Severidad", sev_options,
            default=sev_options,
            format_func=lambda s: f"{SEVERITY_CONFIG[s]['emoji']} {SEVERITY_CONFIG[s]['label']}",
            key="ins_sev",
        )

    ins_country_arg = None if ins_country == "Todos" else ins_country
    ins_metric_arg  = None if ins_metric == "Todas" else ins_metric
    insights = generate_insights(filtered_df, country=ins_country_arg, metric=ins_metric_arg)
    insights = [i for i in insights if i["severity"] in sev_filter]
    counts   = count_by_severity(insights)

    # Severity summary KPIs
    kc, kw, ko, kp = st.columns(4)
    kc.metric("🔴 Critico",       counts.get("critical", 0))
    kw.metric("🟠 Alerta",        counts.get("warning", 0))
    ko.metric("🔵 Oportunidades", counts.get("opportunity", 0))
    kp.metric("🟢 Positivo",      counts.get("positive", 0))

    st.markdown("---")

    # Alert banner
    n_crit_i = counts.get("critical", 0)
    n_warn_i = counts.get("warning", 0)
    if n_crit_i:
        st.error(
            f"⚠️ {n_crit_i} alerta{'s' if n_crit_i > 1 else ''} critica{'s' if n_crit_i > 1 else ''} "
            f"detectada{'s' if n_crit_i > 1 else ''} — revision operativa inmediata recomendada."
        )
    elif n_warn_i:
        st.warning(
            f"{n_warn_i} alerta{'s' if n_warn_i > 1 else ''} detectada{'s' if n_warn_i > 1 else ''} "
            "— monitorear de cerca y preparar planes de respuesta."
        )
    else:
        st.success("Sin alertas criticas ni advertencias en la seleccion actual.")

    # Insight cards
    if not insights:
        st.info("Ningun insight coincide con los filtros seleccionados.")
    else:
        st.markdown(f"**{len(insights)} insights** · ordenados por severidad y magnitud")
        st.markdown("")
        left_items  = [i for i in insights if i["severity"] in ("critical", "warning")]
        right_items = [i for i in insights if i["severity"] in ("opportunity", "positive")]

        if left_items and right_items:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(
                    f"<div class='section-label' style='color:{RED_ALERT};'>Alertas</div>",
                    unsafe_allow_html=True,
                )
                for ins in left_items:
                    _render_card(ins)
            with col_r:
                st.markdown(
                    f"<div class='section-label' style='color:{GREEN};'>Oportunidades y Logros</div>",
                    unsafe_allow_html=True,
                )
                for ins in right_items:
                    _render_card(ins)
        else:
            for ins in insights:
                _render_card(ins)

    # Anomaly detail tables
    st.markdown("---")
    st.subheader("Tablas de detalle")

    _COLS = ["severity", "country", "city", "zone", "metric", "value", "delta_pct"]

    atab_dec, atab_consec, atab_opp, atab_pos = st.tabs(
        ["Caidas SaS", "Tendencias 3 Semanas", "Oportunidades", "Mejoras"]
    )

    with atab_dec:
        rows = [i for i in insights
                if i["severity"] in ("critical", "warning") and "tendencia" not in i["title"].lower()]
        if rows:
            df_dec = pd.DataFrame(rows)[[c for c in _COLS if c in pd.DataFrame(rows).columns]]
            st.dataframe(df_dec.sort_values("delta_pct"), width='stretch', hide_index=True)
        else:
            st.info("Sin alertas de caida SaS en la seleccion actual.")

    with atab_consec:
        rows = [i for i in insights if "tendencia" in i["title"].lower()]
        if rows:
            df_con = pd.DataFrame(rows)[[c for c in _COLS if c in pd.DataFrame(rows).columns]]
            st.dataframe(df_con.sort_values("delta_pct"), width='stretch', hide_index=True)
        else:
            st.info("Sin tendencias bajistas de 3 semanas en la seleccion actual.")

    with atab_opp:
        rows = [i for i in insights if i["severity"] == "opportunity"]
        if rows:
            df_opp = pd.DataFrame(rows)[[c for c in _COLS if c in pd.DataFrame(rows).columns]]
            st.dataframe(df_opp.sort_values("delta_pct"), width='stretch', hide_index=True)
        else:
            st.info("Sin zonas de oportunidad en la seleccion actual.")

    with atab_pos:
        rows = [i for i in insights if i["severity"] == "positive"]
        if rows:
            df_pos = pd.DataFrame(rows)[[c for c in _COLS if c in pd.DataFrame(rows).columns]]
            st.dataframe(
                df_pos.sort_values("delta_pct", ascending=False),
                width='stretch', hide_index=True,
            )
        else:
            st.info("Sin zonas con mejora positiva en la seleccion actual.")

    # ── Executive Report ───────────────────────────────────────────────────────
    st.markdown("---")

    rep_header, rep_actions = st.columns([3, 2])
    with rep_header:
        st.markdown(
            f"<div style='font-size:15px;font-weight:700;color:{TEXT_PRI};margin-bottom:2px;'>"
            f"Informe Ejecutivo</div>"
            f"<div style='font-size:12.5px;color:{TEXT_SEC};'>"
            f"Genera un informe completo con todos los insights actuales en formato HTML y Markdown."
            f"</div>",
            unsafe_allow_html=True,
        )
    with rep_actions:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        if st.button("Generar Informe Ejecutivo", type="primary", key="btn_gen_report",
                     width='stretch'):
            with st.spinner("Generando informe..."):
                _trend_df = get_weekly_trend(filtered_df, ins_metric_arg or metric_filter,
                                             country=ins_country_arg)
                _avg_df   = get_country_averages(filtered_df, ins_metric_arg or metric_filter)
                st.session_state["_report_html"] = generate_html_report(
                    insights, summary,
                    ins_metric_arg or metric_filter,
                    ins_country_arg,
                    _avg_df, _trend_df,
                )
                st.session_state["_report_md"] = generate_markdown_report(
                    insights, summary,
                    ins_metric_arg or metric_filter,
                    ins_country_arg,
                    _avg_df,
                )
                st.session_state["_report_email"] = build_email_body(
                    insights, summary,
                    ins_metric_arg or metric_filter,
                    ins_country_arg,
                )

    if st.session_state.get("_report_html"):
        html_bytes = st.session_state["_report_html"].encode("utf-8")
        md_bytes   = st.session_state["_report_md"].encode("utf-8")

        # Download buttons
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "Descargar HTML",
                data=html_bytes,
                file_name="informe_ejecutivo_rappi.html",
                mime="text/html",
                width='stretch',
                key="dl_html",
            )
        with dl2:
            st.download_button(
                "Descargar Markdown",
                data=md_bytes,
                file_name="informe_ejecutivo_rappi.md",
                mime="text/markdown",
                width='stretch',
                key="dl_md",
            )
        with dl3:
            if st.button("Limpiar informe", width='stretch', key="clear_report"):
                for k in ("_report_html", "_report_md", "_report_email"):
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
                    width='stretch',
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 · Copiloto IA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "api_history" not in st.session_state:
        st.session_state.api_history = []

    _provider_key_map = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
    _key_name   = _provider_key_map.get(ai_provider.name, "ANTHROPIC_API_KEY")
    api_key_set = bool(os.environ.get(_key_name))

    # ── Messaging-app CSS ─────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    /* User bubble — right-aligned, RAPPI RED */
    .umsg {{
        display: flex;
        justify-content: flex-end;
        margin: 2px 0 2px 15%;
    }}
    .ububble {{
        background: {RAPPI_RED};
        color: #FFFFFF;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 15px;
        font-size: 13.5px;
        line-height: 1.55;
        word-break: break-word;
        box-shadow: 0 2px 6px rgba(255,68,31,0.25);
    }}
    /* Assistant message — tight spacing */
    [data-testid="stChatMessage"] {{
        margin: 2px 15% 2px 0 !important;
        gap: 8px !important;
    }}
    [data-testid="stChatMessageContent"] {{
        background: #FFFFFF !important;
        border: 1px solid #EDF0F8 !important;
        border-radius: 4px 18px 18px 18px !important;
        padding: 10px 14px 8px !important;
        box-shadow: 0 1px 3px rgba(28,28,40,0.08) !important;
    }}
    [data-testid="stChatMessageContent"] p {{
        font-size: 13.5px !important;
        line-height: 1.65 !important;
        margin-bottom: 6px !important;
        color: {TEXT_PRI} !important;
    }}
    [data-testid="stChatMessageContent"] strong {{
        font-weight: 700 !important;
        color: {TEXT_PRI} !important;
    }}
    [data-testid="stChatMessageContent"] li {{
        font-size: 13.5px !important;
        line-height: 1.6 !important;
        color: {TEXT_PRI} !important;
    }}
    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3 {{
        font-size: 14px !important;
        font-weight: 700 !important;
        margin: 10px 0 4px !important;
        color: {TEXT_PRI} !important;
    }}
    /* Chat input bar */
    .stChatFloatingInputContainer {{
        border-top: 1px solid {BORDER} !important;
        background: {PAGE_BG} !important;
        padding-top: 10px !important;
    }}
    .stChatFloatingInputContainer textarea {{
        font-size: 14px !important;
        border-radius: 14px !important;
        border: 1.5px solid {BORDER} !important;
        background: white !important;
    }}
    /* Suggestion chips */
    .chip-row .stButton > button {{
        background: white;
        border: 1.5px solid {BORDER};
        border-radius: 100px;
        color: {TEXT_PRI};
        font-size: 12.5px;
        font-weight: 500;
        padding: 9px 15px;
        text-align: left;
        white-space: normal;
        height: auto;
        min-height: 44px;
        line-height: 1.4;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: all 0.15s ease;
    }}
    .chip-row .stButton > button:hover {{
        background: #FFF0EC;
        border-color: {RAPPI_RED};
        color: {RAPPI_RED};
        box-shadow: 0 2px 8px rgba(255,68,31,0.12);
    }}
    /* Follow-up suggestion chip */
    .followup .stButton > button {{
        background: #EFF6FF;
        border: 1.5px solid #BFDBFE;
        border-radius: 100px;
        color: #1D4ED8;
        font-size: 12px;
        font-weight: 500;
        padding: 7px 15px;
        height: auto;
        min-height: 34px;
        white-space: normal;
        line-height: 1.4;
        text-align: left;
    }}
    .followup .stButton > button:hover {{
        background: #DBEAFE;
        border-color: #3B82F6;
        color: #1E40AF;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── Ultra-minimal status line ─────────────────────────────────────────────
    dot_color  = GREEN if api_key_set else AMBER
    status_txt = "Listo" if api_key_set else "Sin clave API"
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:flex-end;
                padding:4px 0 12px 0;margin-bottom:4px;">
      <span style="width:6px;height:6px;background:{dot_color};border-radius:50%;
                   display:inline-block;margin-right:5px;"></span>
      <span style="font-size:11px;color:{TEXT_SEC};">{status_txt}</span>
    </div>
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
                if msg.get("chart_spec"):
                    fig = render_chart_from_spec(
                        msg["chart_spec"], metrics_df, orders_df, None, compact=True
                    )
                    if fig:
                        col_c, _ = st.columns([3, 1])
                        with col_c:
                            st.plotly_chart(fig, width="stretch")
                if suggestion and api_key_set:
                    st.markdown('<div class="followup">', unsafe_allow_html=True)
                    if st.button(
                        f"💡  {suggestion}",
                        key=f"fu_{abs(hash(suggestion + body[:20])) % 999983}",
                    ):
                        st.session_state._pending_question = suggestion
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── Resolve pending + input BEFORE rendering decisions ───────────────────
    pending    = st.session_state.pop("_pending_question", None)
    user_input = st.chat_input(
        "Pregunta sobre operaciones Rappi...",
        disabled=not api_key_set,
    ) or pending

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.session_state.chat_messages:
        _, clr_col = st.columns([7, 1])
        with clr_col:
            if st.button("Limpiar", key="clear_chat", width="stretch"):
                st.session_state.chat_messages = []
                st.session_state.api_history   = []
                st.rerun()

    # ── Empty state (only when conversation is truly blank) ───────────────────
    if not st.session_state.chat_messages and not user_input:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 0 36px;">
          <div style="width:64px;height:64px;background:{RAPPI_RED};border-radius:18px;
                      display:flex;align-items:center;justify-content:center;
                      font-size:30px;margin:0 auto 20px;
                      box-shadow:0 8px 28px rgba(255,68,31,0.28);">🛵</div>
          <div style="font-size:26px;font-weight:900;color:{TEXT_PRI};
                      letter-spacing:-0.5px;margin-bottom:12px;">
            ¿En qué te puedo ayudar?
          </div>
          <div style="font-size:13.5px;color:{TEXT_SEC};max-width:500px;
                      margin:0 auto;line-height:1.75;">
            Analista operacional con acceso completo a
            <strong style="color:{TEXT_PRI};">964 zonas</strong>,
            <strong style="color:{TEXT_PRI};">13 métricas</strong> y
            <strong style="color:{TEXT_PRI};">9 países LATAM</strong>.<br>
            Escribe cualquier pregunta — sin necesidad de filtros.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Category labels + chips
        _chip_groups = [
            ("Alertas y deterioro", [
                "¿Cuáles son las zonas con mayor deterioro SaS esta semana?",
                "Top zonas con caída en Perfect Orders las últimas 4 semanas",
            ]),
            ("Rankings y crecimiento", [
                "Top 10 zonas con mayor crecimiento de pedidos",
                "¿Qué país lidera en Perfect Orders esta semana?",
            ]),
            ("Comparaciones y análisis", [
                "Compara Gross Profit UE entre todos los países LATAM",
                "Oportunidades de mejora en Lead Penetration esta semana",
            ]),
        ]
        for _cat_label, _cat_qs in _chip_groups:
            st.markdown(
                f"<p style='font-size:10px;font-weight:700;letter-spacing:1.4px;"
                f"text-transform:uppercase;color:{TEXT_SEC};margin:16px 0 8px 0;'>"
                f"{_cat_label}</p>",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chip-row">', unsafe_allow_html=True)
            _cc1, _cc2 = st.columns(2)
            for _ci, _cq in enumerate(_cat_qs):
                with [_cc1, _cc2][_ci % 2]:
                    if st.button(_cq, key=f"chip_{abs(hash(_cq)) % 999983}",
                                 width="stretch", disabled=not api_key_set):
                        st.session_state._pending_question = _cq
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── Append new user message to state BEFORE the container renders ─────
        # This ensures the for loop inside the container includes the new message,
        # so user bubble and streaming both happen inside the same fixed container.
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # ── Fixed-height scrollable container (page never grows) ──────────────
        try:
            _chat_area = st.container(height=670, border=False)
        except TypeError:
            _chat_area = st.container()

        with _chat_area:
            # History + new user bubble (all inside the container)
            for msg in st.session_state.chat_messages:
                _render_history_msg(msg)

            # Stream assistant response INSIDE the same container
            if user_input:
                context       = build_dynamic_context(metrics_df, orders_df, user_input)
                system_prompt = load_system_prompt()

                with st.chat_message("assistant", avatar="🛵"):
                    response_ph = st.empty()
                    response_ph.markdown(
                        f"<span style='color:{TEXT_SEC};font-size:13px;font-style:italic;'>"
                        "Analizando datos operativos…</span>",
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
                        if chart_spec:
                            fig = render_chart_from_spec(
                                chart_spec, metrics_df, orders_df, None, compact=True
                            )
                            if fig:
                                col_c, _ = st.columns([3, 1])
                                with col_c:
                                    st.plotly_chart(fig, width="stretch")

                        if suggestion and api_key_set:
                            st.markdown('<div class="followup">', unsafe_allow_html=True)
                            if st.button(
                                f"💡  {suggestion}",
                                key=f"fu_new_{abs(hash(suggestion)) % 999983}",
                            ):
                                st.session_state._pending_question = suggestion
                            st.markdown('</div>', unsafe_allow_html=True)

                    except AuthError:
                        error_text = "Clave API inválida. Verifica la clave del proveedor."
                    except NetworkError:
                        error_text = "Sin conexión con la API. Verifica tu red."
                    except RateLimitError:
                        error_text = "Límite de tasa. Espera un momento e intenta de nuevo."
                    except ProviderError as exc:
                        error_text = f"Error del proveedor: {exc}"
                    except Exception as exc:
                        error_text = f"Error inesperado: {exc}"

                    if error_text:
                        response_ph.error(error_text)

                # Save assistant to state (user was already saved before the container)
                if not error_text:
                    stored   = strip_chart_block(full_text)
                    spec_out = parse_chart_spec(full_text)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": stored,
                        "chart_spec": spec_out,
                    })
                    st.session_state.api_history.append({"role": "user",      "content": user_input})
                    body_only, _ = extract_suggested_question(stored)
                    st.session_state.api_history.append({"role": "assistant", "content": body_only})

            # Anchor at end of container content — JS scrolls to this
            st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)

    # ── Scroll container to latest message (targets internal anchor) ──────────
    components.html("""<script>
    (function() {
        var a = window.parent.document.getElementById('chat-scroll-anchor');
        if (a) a.scrollIntoView({block: 'end'});
    })();
    </script>""", height=0)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 · Datos
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
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
