"""
Executive report generator.
Produces self-contained HTML and Markdown reports from live insight data.
All visible text is in professional Spanish. No external images required.
"""
from __future__ import annotations

import html as _html_mod
from datetime import datetime

import pandas as pd

from src.insights import SEVERITY_CONFIG, SEVERITY_ORDER, CATEGORY_CONFIG

# ── Brand palette (mirrors app.py constants) ──────────────────────────────────
_RED    = "#FF441F"
_DARK   = "#1C1C28"
_GREEN  = "#10B981"
_AMBER  = "#F59E0B"
_DANGER = "#EF4444"
_BLUE   = "#3B82F6"
_BORDER = "#E4E8F0"
_BG     = "#F5F6FA"
_CARD   = "#FFFFFF"


def _e(v) -> str:
    """HTML-escape a value to a string."""
    return _html_mod.escape(str(v))


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _insight_card(ins: dict) -> str:
    sev_cfg = SEVERITY_CONFIG[ins["severity"]]
    cat_cfg = CATEGORY_CONFIG.get(ins.get("category", "anomaly"), {})
    color   = sev_cfg["color"]
    label   = sev_cfg["label"]
    cat_lbl = cat_cfg.get("emoji", "") + " " + cat_cfg.get("label", "")
    cat_col = cat_cfg.get("color", "#6B7280")
    delta   = ins["delta_pct"]
    score   = ins.get("score", 0)
    delta_color = _GREEN if delta > 0 else _DANGER
    delta_str   = f"{delta:+.1f}%"
    return f"""
    <div style="border-left:4px solid {color};background:#FAFBFF;border-radius:0 10px 10px 0;
                padding:13px 17px;margin-bottom:10px;">
      <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:7px;">
        <span style="background:{color};color:white;font-size:9.5px;font-weight:700;
                     letter-spacing:.5px;padding:3px 9px;border-radius:100px;">{_e(label)}</span>
        <span style="background:{cat_col}18;color:{cat_col};font-size:9.5px;font-weight:700;
                     letter-spacing:.4px;padding:3px 9px;border-radius:100px;">{_e(cat_lbl)}</span>
        <span style="background:#F1F3F9;color:{delta_color};font-size:10px;font-weight:700;
                     padding:3px 9px;border-radius:100px;">{_e(delta_str)}</span>
        <span style="background:#F8F9FB;color:#9CA3AF;font-size:9.5px;font-weight:600;
                     padding:3px 9px;border-radius:100px;">Score: {score:.0f}</span>
        <span style="font-weight:700;font-size:13.5px;color:{_DARK};">{_e(ins['title'])}</span>
      </div>
      <p style="margin:4px 0;font-size:13px;color:{_DARK};">
        <b>Hallazgo:</b> {_e(ins['finding'])}</p>
      <p style="margin:4px 0;font-size:12.5px;color:#4B5563;">
        <b>Por que importa:</b> {_e(ins['explanation'])}</p>
      <p style="margin:4px 0;font-size:12.5px;color:{_BLUE};">
        <b>Accion recomendada:</b> {_e(ins['action'])}</p>
      <p style="margin:8px 0 0 0;font-size:10.5px;color:#9CA3AF;">
        {_e(ins['country'])} &nbsp;·&nbsp; {_e(ins['city'])}
        &nbsp;·&nbsp; {_e(ins['zone'])} &nbsp;·&nbsp; {_e(ins['metric'])}</p>
    </div>"""


def _section(title: str, icon: str, icon_bg: str, items: list[dict], count_label: bool = True) -> str:
    if not items:
        return ""
    cards = "".join(_insight_card(i) for i in items)
    suffix = (
        f'<span style="font-size:11px;font-weight:500;color:#9CA3AF;margin-left:6px;">'
        f'({len(items)})</span>'
        if count_label else ""
    )
    return f"""
    <div style="background:{_CARD};border:1.5px solid {_BORDER};border-radius:14px;
                padding:22px 26px;margin-bottom:18px;box-shadow:0 2px 6px rgba(28,28,40,0.04);">
      <div style="font-size:14px;font-weight:800;color:{_DARK};margin-bottom:16px;
                  padding-bottom:10px;border-bottom:1.5px solid #F1F3F9;
                  display:flex;align-items:center;gap:8px;">
        <span style="width:28px;height:28px;border-radius:7px;background:{icon_bg};
                     display:inline-flex;align-items:center;justify-content:center;
                     font-size:14px;">{icon}</span>
        {_e(title)}{suffix}
      </div>
      {cards}
    </div>"""


def _benchmark_table(avg_df: pd.DataFrame) -> str:
    if avg_df.empty:
        return "<p style='color:#9CA3AF;font-size:13px;'>Sin datos de benchmark disponibles.</p>"
    net_avg = avg_df["avg_value"].mean()
    rows = ""
    for pos, (_, row) in enumerate(avg_df.iterrows(), 1):
        is_leader  = pos == 1
        is_laggard = pos == len(avg_df)
        badge = ""
        if is_leader:
            badge = ('<span style="background:#F0FDF4;color:#10B981;font-size:10px;font-weight:700;'
                     'padding:2px 8px;border-radius:100px;">LIDER</span>')
        elif is_laggard:
            badge = ('<span style="background:#FFF5F5;color:#EF4444;font-size:10px;font-weight:700;'
                     'padding:2px 8px;border-radius:100px;">REZAGADO</span>')
        val_color = _GREEN if row["avg_value"] >= net_avg else _DANGER
        rows += (
            f'<tr><td style="padding:8px 12px;border-bottom:1px solid #F1F3F9;">{_e(row["COUNTRY"])}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #F1F3F9;'
            f'color:{val_color};font-weight:700;">{row["avg_value"]:.3f}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #F1F3F9;">{badge}</td></tr>'
        )
    return f"""
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px 12px;font-size:10.5px;font-weight:700;
                     letter-spacing:.5px;text-transform:uppercase;color:#9CA3AF;
                     background:#F8F9FB;border-bottom:1.5px solid {_BORDER};">Pais</th>
          <th style="text-align:left;padding:8px 12px;font-size:10.5px;font-weight:700;
                     letter-spacing:.5px;text-transform:uppercase;color:#9CA3AF;
                     background:#F8F9FB;border-bottom:1.5px solid {_BORDER};">Valor Prom. W-0</th>
          <th style="text-align:left;padding:8px 12px;font-size:10.5px;font-weight:700;
                     letter-spacing:.5px;text-transform:uppercase;color:#9CA3AF;
                     background:#F8F9FB;border-bottom:1.5px solid {_BORDER};">Posicion</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _recommendations(insights: list[dict]) -> str:
    seen: set[str] = set()
    recs: list[str] = []
    for ins in insights:
        a = ins.get("action", "").strip()
        if a and a not in seen:
            seen.add(a)
            recs.append(a)
        if len(recs) >= 6:
            break
    if not recs:
        return "<p style='color:#9CA3AF;font-size:13px;'>Sin recomendaciones especificas para la seleccion actual.</p>"
    items = ""
    for i, rec in enumerate(recs, 1):
        items += f"""
        <div style="display:flex;gap:10px;align-items:flex-start;padding:10px 0;
                    border-bottom:1px solid #F1F3F9;">
          <div style="width:28px;height:28px;min-width:28px;background:#EFF6FF;color:{_BLUE};
                      border-radius:50%;font-size:11px;font-weight:800;
                      display:flex;align-items:center;justify-content:center;">{i}</div>
          <div style="font-size:13px;color:{_DARK};line-height:1.5;">{_e(rec)}</div>
        </div>"""
    return items


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_html_report(
    insights: list[dict],
    summary: dict,
    metric: str,
    country: str | None,
    avg_df: pd.DataFrame,
    trend_df: pd.DataFrame | None = None,
    generated_by: str = "Rappi Operations Analytics",
) -> str:
    """Return a self-contained UTF-8 HTML string."""
    now   = datetime.now().strftime("%d %b %Y, %H:%M")
    scope = country if country else "LATAM — 9 paises"

    ordered   = sorted(insights, key=lambda x: (SEVERITY_ORDER.get(x["severity"], 9), -x.get("score", 0)))
    critical  = [i for i in ordered if i["severity"] == "critical"]
    warnings  = [i for i in ordered if i["severity"] == "warning"]
    opps      = [i for i in ordered if i["severity"] == "opportunity"]
    positives = [i for i in ordered if i["severity"] == "positive"]

    # Category groups — used for structured sections
    by_cat = {cat: [i for i in ordered if i.get("category") == cat] for cat in CATEGORY_CONFIG}

    top_findings = (critical + warnings)[:5]

    # Trend bar text
    trend_bar = ""
    if trend_df is not None and not trend_df.empty:
        vals = list(zip(trend_df["week"].tolist(), trend_df["value"].tolist()))
        if len(vals) >= 2:
            v0, vN = vals[0][1], vals[-1][1]
            pct    = (vN - v0) / abs(v0) * 100 if v0 else 0
            direc  = "creciente" if vN > v0 else ("decreciente" if vN < v0 else "estable")
            col    = _GREEN if vN >= v0 else _DANGER
            trend_bar = (
                f'<div style="background:#F1F3F9;border-radius:8px;padding:10px 14px;'
                f'font-size:12.5px;color:#4B5563;margin-bottom:14px;">'
                f'Tendencia de 8 semanas: <b style="color:{col};">{direc}</b>'
                f' &nbsp;·&nbsp; inicio {v0:.3f} &rarr; actual <b>{vN:.3f}</b>'
                f' &nbsp;·&nbsp; cambio: <span style="color:{col};font-weight:700;">'
                f'{pct:+.1f}%</span></div>'
            )

    # Top findings bullets
    findings_html = ""
    if top_findings:
        items = "".join(
            f'<li style="display:flex;gap:10px;align-items:flex-start;padding:9px 0;'
            f'border-bottom:1px solid #F1F3F9;">'
            f'<div style="width:24px;height:24px;min-width:24px;background:{_RED};color:white;'
            f'border-radius:50%;font-size:11px;font-weight:800;display:flex;'
            f'align-items:center;justify-content:center;">{i + 1}</div>'
            f'<div><b>{_e(ins["title"])}</b><br>'
            f'<span style="color:#4B5563;font-size:12.5px;">{_e(ins["finding"])}</span></div></li>'
            for i, ins in enumerate(top_findings)
        )
        findings_html = f"<ul style='list-style:none;padding:0;margin:0;'>{items}</ul>"
    else:
        findings_html = "<p style='color:#9CA3AF;font-size:13px;'>Sin alertas criticas en la seleccion actual.</p>"

    critical_count_color = _DANGER if critical else _GREEN

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Informe Ejecutivo — {_e(metric)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: {_BG}; color: {_DARK}; font-size: 13.5px; line-height: 1.6;
    }}
    .container {{ max-width: 860px; margin: 0 auto; padding: 40px 20px; }}
    @media print {{
      body {{ background: white; }}
      .container {{ padding: 20px; }}
    }}
  </style>
</head>
<body><div class="container">

<!-- ── Header ── -->
<div style="background:{_DARK};color:white;padding:28px 36px;border-radius:14px;
            margin-bottom:24px;display:flex;justify-content:space-between;align-items:flex-start;">
  <div>
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
      <div style="width:38px;height:38px;background:{_RED};border-radius:10px;
                  display:flex;align-items:center;justify-content:center;font-size:20px;">🛵</div>
      <div>
        <div style="font-size:22px;font-weight:900;letter-spacing:-0.3px;">rappi</div>
        <div style="font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
                    color:rgba(255,255,255,0.45);">operations analytics</div>
      </div>
    </div>
    <div style="font-size:12px;color:rgba(255,255,255,0.55);">
      Informe Ejecutivo de Operaciones &nbsp;·&nbsp; {_e(scope)}<br>
      Metrica principal: <b style="color:rgba(255,255,255,0.85);">{_e(metric)}</b>
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:11.5px;color:rgba(255,255,255,0.5);margin-bottom:8px;">
      Generado: {now}
    </div>
    <span style="background:{_RED};color:white;font-size:10.5px;font-weight:700;
                 padding:5px 13px;border-radius:100px;">LATAM · 9 PAISES</span>
  </div>
</div>

<!-- ── KPI row ── -->
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">
  {''.join([
    f'<div style="background:{_CARD};border:1.5px solid {_BORDER};border-radius:12px;'
    f'padding:16px 18px;box-shadow:0 2px 6px rgba(28,28,40,0.05);">'
    f'<div style="font-size:10px;font-weight:700;letter-spacing:.7px;text-transform:uppercase;'
    f'color:#9CA3AF;margin-bottom:6px;">{lbl}</div>'
    f'<div style="font-size:22px;font-weight:800;color:{val_color};letter-spacing:-0.3px;">{val}</div>'
    f'<div style="font-size:11.5px;font-weight:600;color:#9CA3AF;margin-top:3px;">{sub}</div></div>'
    for lbl, val, sub, val_color in [
        ("Total Pedidos", f"{summary['total_orders']:,}",
         f"{summary['orders_wow_pct']:+.1f}% SaS",
         _GREEN if summary['orders_wow_pct'] >= 0 else _DANGER),
        ("Paises Activos", str(summary['countries']), "LATAM", _DARK),
        ("Zonas Monitoreadas", f"{summary['zones']:,}", f"{summary['cities']} ciudades", _DARK),
        ("Alertas Criticas", str(len(critical)),
         f"{len(warnings)} alertas", critical_count_color),
    ]
  ])}
</div>

<!-- ── Executive Summary ── -->
<div style="background:{_CARD};border:1.5px solid {_BORDER};border-radius:14px;
            padding:22px 26px;margin-bottom:18px;box-shadow:0 2px 6px rgba(28,28,40,0.04);">
  <div style="font-size:14px;font-weight:800;color:{_DARK};margin-bottom:16px;
              padding-bottom:10px;border-bottom:1.5px solid #F1F3F9;
              display:flex;align-items:center;gap:8px;">
    <span style="width:28px;height:28px;border-radius:7px;background:#FFF0EC;
                 display:inline-flex;align-items:center;justify-content:center;
                 font-size:14px;">📋</span>
    Resumen Ejecutivo
  </div>
  {trend_bar}
  <p style="margin-bottom:14px;color:#4B5563;font-size:13px;">
    La red registra <b>{len(critical)} alertas criticas</b>, <b>{len(warnings)} alertas</b>,
    <b>{len(opps)} oportunidades identificadas</b> y <b>{len(positives)} senales positivas</b>
    para la metrica <b>{_e(metric)}</b>.
  </p>
  <p style="color:#4B5563;font-size:13px;margin-bottom:12px;">Principales hallazgos a priorizar:</p>
  {findings_html}
</div>

<!-- ── Category bar ── -->
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:20px;">
  {''.join([
    f'<div style="background:{cfg["color"]}12;border:1.5px solid {cfg["color"]}30;'
    f'border-radius:10px;padding:10px 12px;text-align:center;">'
    f'<div style="font-size:20px;font-weight:800;color:{cfg["color"]};">'
    f'{len(by_cat[cat])}</div>'
    f'<div style="font-size:9.5px;font-weight:700;color:{cfg["color"]};'
    f'letter-spacing:.5px;text-transform:uppercase;margin-top:2px;">'
    f'{cfg["emoji"]} {cfg["label"]}</div></div>'
    for cat, cfg in CATEGORY_CONFIG.items()
  ])}
</div>

{_section("🔺 Anomalías", "🔺", "#FFF5F5", by_cat["anomaly"])}
{_section("📉 Tendencias Preocupantes", "📉", "#FFFBEB", by_cat["trend"])}
{_section("🌎 Benchmarking", "🌎", "#EFF6FF", by_cat["benchmark"])}

<!-- ── Benchmark de paises ── -->
<div style="background:{_CARD};border:1.5px solid {_BORDER};border-radius:14px;
            padding:22px 26px;margin-bottom:18px;box-shadow:0 2px 6px rgba(28,28,40,0.04);">
  <div style="font-size:14px;font-weight:800;color:{_DARK};margin-bottom:16px;
              padding-bottom:10px;border-bottom:1.5px solid #F1F3F9;
              display:flex;align-items:center;gap:8px;">
    <span style="width:28px;height:28px;border-radius:7px;background:#EFF6FF;
                 display:inline-flex;align-items:center;justify-content:center;
                 font-size:14px;">🌎</span>
    Benchmark de Paises — {_e(metric)}
  </div>
  {_benchmark_table(avg_df)}
</div>

{_section("🔗 Correlaciones", "🔗", "#F5F3FF", by_cat["correlation"])}
{_section("💡 Oportunidades", "💡", "#F0FDF4", by_cat["opportunity"])}
{_section("🟢 Mejoras y Recuperaciones", "🟢", "#F0FDF4", positives)}

<!-- ── Recommendations ── -->
<div style="background:{_CARD};border:1.5px solid {_BORDER};border-radius:14px;
            padding:22px 26px;margin-bottom:18px;box-shadow:0 2px 6px rgba(28,28,40,0.04);">
  <div style="font-size:14px;font-weight:800;color:{_DARK};margin-bottom:16px;
              padding-bottom:10px;border-bottom:1.5px solid #F1F3F9;
              display:flex;align-items:center;gap:8px;">
    <span style="width:28px;height:28px;border-radius:7px;background:#EFF6FF;
                 display:inline-flex;align-items:center;justify-content:center;
                 font-size:14px;">⚡</span>
    Acciones Recomendadas
  </div>
  {_recommendations(critical + warnings + opps)}
</div>

<!-- ── Footer ── -->
<div style="text-align:center;font-size:11px;color:#9CA3AF;margin-top:28px;
            padding-top:16px;border-top:1px solid {_BORDER};">
  Generado por {_e(generated_by)} &nbsp;·&nbsp; {now}<br>
  Datos: dataset RAW_INPUT_METRICS &nbsp;·&nbsp; Uso exclusivo interno Rappi
</div>

</div></body></html>"""


# ── Markdown export ────────────────────────────────────────────────────────────

def _md_insight(ins: dict) -> list[str]:
    delta_str = f"{ins['delta_pct']:+.1f}%"
    return [
        f"### {ins['title']} ({delta_str})",
        f"**Hallazgo:** {ins['finding']}",
        "",
        f"**Por que importa:** {ins['explanation']}",
        "",
        f"**Accion recomendada:** {ins['action']}",
        "",
        f"*{ins['country']} · {ins['city']} · {ins['zone']} · {ins['metric']}*",
        "",
        "---",
        "",
    ]


def generate_markdown_report(
    insights: list[dict],
    summary: dict,
    metric: str,
    country: str | None,
    avg_df: pd.DataFrame,
) -> str:
    """Return a UTF-8 Markdown string."""
    now   = datetime.now().strftime("%d %b %Y, %H:%M")
    scope = country if country else "LATAM — 9 paises"

    ordered   = sorted(insights, key=lambda x: (SEVERITY_ORDER.get(x["severity"], 9), -x.get("score", 0)))
    critical  = [i for i in ordered if i["severity"] == "critical"]
    warnings  = [i for i in ordered if i["severity"] == "warning"]
    opps      = [i for i in ordered if i["severity"] == "opportunity"]
    positives = [i for i in ordered if i["severity"] == "positive"]
    by_cat    = {cat: [i for i in ordered if i.get("category") == cat] for cat in CATEGORY_CONFIG}

    lines: list[str] = [
        f"# Informe Ejecutivo de Operaciones — {metric}",
        "",
        f"**Alcance:** {scope}  |  **Generado:** {now}",
        "",
        "---",
        "",
        "## Resumen Ejecutivo",
        "",
        f"| KPI | Valor |",
        f"|-----|-------|",
        f"| Total pedidos (W-0) | {summary['total_orders']:,} ({summary['orders_wow_pct']:+.1f}% SaS) |",
        f"| Paises activos | {summary['countries']} |",
        f"| Zonas monitoreadas | {summary['zones']:,} ({summary['cities']} ciudades) |",
        f"| Alertas criticas | {len(critical)} |",
        f"| Alertas | {len(warnings)} |",
        f"| Oportunidades | {len(opps)} |",
        f"| Senales positivas | {len(positives)} |",
        "",
    ]

    # Category summary table
    lines += [
        "## Cobertura por Categoría", "",
        "| Categoría | Insights | Descripción |",
        "|-----------|----------|-------------|",
    ]
    _CAT_DESC = {
        "anomaly":     "Cambios WoW drásticos (>10%)",
        "trend":       "Deterioro/mejora 3+ semanas consecutivas",
        "benchmark":   "Comparación zona/país vs pares del segmento",
        "correlation": "Relaciones entre métricas operativas",
        "opportunity": "Zonas con potencial de mejora significativo",
    }
    for cat, cfg in CATEGORY_CONFIG.items():
        n = len(by_cat[cat])
        lines.append(f"| {cfg['emoji']} {cfg['label']} | **{n}** | {_CAT_DESC.get(cat, '')} |")
    lines.append("")

    # Sections by category
    _CAT_HEADERS = {
        "anomaly":     "## 🔺 Anomalías (Cambios WoW Drásticos)",
        "trend":       "## 📉 Tendencias Preocupantes (3+ Semanas)",
        "benchmark":   "## 🌎 Benchmarking (Comparación entre Pares)",
        "correlation": "## 🔗 Correlaciones entre Métricas",
        "opportunity": "## 💡 Oportunidades Operativas",
    }
    for cat, header in _CAT_HEADERS.items():
        cat_ins = by_cat[cat]
        if cat_ins:
            lines += [header, ""]
            for ins in cat_ins:
                lines += _md_insight(ins)

    # Country benchmark table
    if not avg_df.empty:
        lines += ["## 🌎 Benchmark de Países — Tabla Comparativa", "",
                  f"| País | {metric} (W-0) |", "|------|---------|"]
        for _, row in avg_df.iterrows():
            lines.append(f"| {row['COUNTRY']} | {row['avg_value']:.3f} |")
        lines.append("")

    if positives:
        lines += ["## 🟢 Señales Positivas y Recuperaciones", ""]
        for ins in positives:
            lines += _md_insight(ins)

    # Recommendations
    seen: set[str] = set()
    recs: list[str] = []
    for ins in critical + warnings + opps:
        a = ins.get("action", "").strip()
        if a and a not in seen:
            seen.add(a)
            recs.append(a)
        if len(recs) >= 6:
            break

    if recs:
        lines += ["## ⚡ Acciones Recomendadas", ""]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    lines += [
        "---",
        "",
        f"*Generado por Rappi Operations Analytics · {now}*",
        "",
        f"*Datos: dataset RAW_INPUT_METRICS · Uso exclusivo interno Rappi*",
    ]

    return "\n".join(lines)


# ── CSV insights export ───────────────────────────────────────────────────────

def generate_insights_csv(insights: list[dict]) -> str:
    """Return a UTF-8 CSV string with all insight fields, ready for Excel/Sheets."""
    _COLS = [
        "severity", "category", "score",
        "country", "city", "zone",
        "metric", "value", "delta_pct",
        "title", "finding", "explanation", "action",
    ]
    df = pd.DataFrame(insights)
    if df.empty:
        return ",".join(_COLS) + "\n"
    out_cols = [c for c in _COLS if c in df.columns]
    return df[out_cols].to_csv(index=False)


# ── PDF report ─────────────────────────────────────────────────────────────────

def generate_pdf_report(
    insights: list[dict],
    summary: dict,
    metric: str,
    country: str | None,
    avg_df: pd.DataFrame,
) -> bytes:
    """Return a self-contained A4 PDF as bytes. Requires fpdf2>=2.7.0."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError("fpdf2 is required for PDF export: pip install fpdf2") from exc

    def _s(v: object, maxlen: int = 0) -> str:
        """Encode to Latin-1 for fpdf2 core fonts; replace unmappable chars."""
        s = str(v).encode("latin-1", errors="replace").decode("latin-1")
        return s[:maxlen] if maxlen else s

    now   = datetime.now().strftime("%d %b %Y, %H:%M")
    scope = country if country else "LATAM"

    ordered   = sorted(insights, key=lambda x: (SEVERITY_ORDER.get(x["severity"], 9), -x.get("score", 0)))
    critical  = [i for i in ordered if i["severity"] == "critical"]
    warnings  = [i for i in ordered if i["severity"] == "warning"]
    opps      = [i for i in ordered if i["severity"] == "opportunity"]
    pos       = [i for i in ordered if i["severity"] == "positive"]

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_margins(14, 14, 14)
    W = 182  # usable width: 210 - 2*14

    # ── Header band ──────────────────────────────────────────────────────────
    pdf.set_fill_color(28, 28, 40)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_xy(14, 6)
    pdf.cell(W, 7, "rappi  Operations Analytics", ln=False)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(14, 15)
    pdf.cell(W, 5, _s(f"Informe Ejecutivo  .  {scope}  .  {now}"), ln=False)
    pdf.set_text_color(255, 100, 60)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_xy(14, 22)
    pdf.cell(W, 5, _s(f"Metrica principal: {metric}"), ln=False)
    pdf.set_text_color(28, 28, 40)
    pdf.set_y(36)

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpis = [
        ("TOTAL PEDIDOS",    f"{summary['total_orders']:,}",  f"{summary['orders_wow_pct']:+.1f}% SaS"),
        ("ALERTAS CRITICAS", str(len(critical)),              f"{len(warnings)} alertas adicionales"),
        ("OPORTUNIDADES",    str(len(opps)),                  "zonas con potencial"),
        ("MEJORAS",          str(len(pos)),                   "senales positivas"),
    ]
    kpi_w = (W - 6) / 4  # 4 cards, 2mm gap between each
    pdf.set_draw_color(228, 232, 240)
    for i, (lbl, val, sub) in enumerate(kpis):
        xk = 14 + i * (kpi_w + 2)
        pdf.set_fill_color(255, 255, 255)
        pdf.rect(xk, 36, kpi_w, 20, "FD")
        pdf.set_font("Helvetica", "B", 6)
        pdf.set_text_color(156, 163, 175)
        pdf.set_xy(xk + 2, 38.5)
        pdf.cell(kpi_w - 4, 3.5, lbl, ln=True)
        r, g, b = (239, 68, 68) if (lbl == "ALERTAS CRITICAS" and len(critical) > 0) else (28, 28, 40)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(r, g, b)
        pdf.set_xy(xk + 2, 42.5)
        pdf.cell(kpi_w - 4, 7, val, ln=True)
        pdf.set_font("Helvetica", "", 6)
        pdf.set_text_color(156, 163, 175)
        pdf.set_xy(xk + 2, 51)
        pdf.cell(kpi_w - 4, 3.5, _s(sub), ln=True)

    pdf.set_y(62)

    # ── Insights table ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(28, 28, 40)
    pdf.cell(W, 5, f"Insights Operativos  ({len(ordered)} detectados)", ln=True)
    pdf.ln(1.5)

    _SEV_LABEL = {"critical": "CRITICA", "warning": "ALERTA", "opportunity": "OPORT.", "positive": "MEJORA"}
    _SEV_RGB   = {
        "critical":    (239,  68,  68),
        "warning":     (245, 158,  11),
        "opportunity": ( 59, 130, 246),
        "positive":    ( 16, 185, 129),
    }
    # column (label, width)
    _ICOLS = [("SEV", 15), ("METRICA", 43), ("ZONA", 49), ("PAIS", 12),
              ("VALOR", 18), ("SaS%", 16), ("SCORE", 14), ("CAT", 15)]
    # assert sum(w for _, w in _ICOLS) == 182

    pdf.set_fill_color(248, 249, 251)
    pdf.set_draw_color(228, 232, 240)
    pdf.set_font("Helvetica", "B", 6)
    pdf.set_text_color(156, 163, 175)
    for lbl, w in _ICOLS:
        pdf.cell(w, 4, lbl, border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 6.5)
    for ins in ordered:
        sev = ins.get("severity", "warning")
        r, g, b = _SEV_RGB.get(sev, (107, 114, 128))
        delta = ins.get("delta_pct", 0)
        val   = ins.get("value", 0)
        cat_label = CATEGORY_CONFIG.get(ins.get("category", "anomaly"), {}).get("label", "")

        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(15, 3.5, _SEV_LABEL.get(sev, ""), border=1, align="C", fill=True)

        pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(28, 28, 40)
        pdf.cell(43, 3.5, _s(ins.get("metric", ""), 23), border=1)
        pdf.cell(49, 3.5, _s(ins.get("zone",   ""), 27), border=1)
        pdf.cell(12, 3.5, _s(ins.get("country",""),  4), border=1, align="C")
        pdf.cell(18, 3.5, f"{val:.3f}",                 border=1, align="R")
        dr, dg, db = (16, 185, 129) if delta >= 0 else (239, 68, 68)
        pdf.set_text_color(dr, dg, db)
        pdf.cell(16, 3.5, f"{delta:+.1f}%", border=1, align="R")
        pdf.set_text_color(107, 114, 128)
        pdf.cell(14, 3.5, f"{ins.get('score', 0):.0f}", border=1, align="R")
        pdf.cell(15, 3.5, _s(cat_label, 10), border=1)
        pdf.set_text_color(28, 28, 40)
        pdf.ln()

    # ── Country benchmark ─────────────────────────────────────────────────────
    if not avg_df.empty:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(28, 28, 40)
        pdf.cell(W, 5, _s(f"Benchmark de Paises  —  {metric}"), ln=True)
        pdf.ln(1.5)

        net_avg = avg_df["avg_value"].mean()
        _BCOLS  = [("PAIS", 30), ("VALOR W-0", 55), ("DELTA VS RED", 47), ("POSICION", 50)]
        pdf.set_fill_color(248, 249, 251)
        pdf.set_draw_color(228, 232, 240)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(156, 163, 175)
        for hdr, w in _BCOLS:
            pdf.cell(w, 4.5, hdr, border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for pos_i, (_, row) in enumerate(avg_df.iterrows(), 1):
            badge   = "LIDER" if pos_i == 1 else ("REZAGADO" if pos_i == len(avg_df) else "")
            gap_pct = (row["avg_value"] - net_avg) / abs(net_avg) * 100 if net_avg else 0
            vr, vg, vb = (16, 185, 129) if row["avg_value"] >= net_avg else (239, 68, 68)
            pdf.set_text_color(28, 28, 40)
            pdf.cell(30, 4.5, _s(row["COUNTRY"]), border=1, align="C")
            pdf.set_text_color(vr, vg, vb)
            pdf.cell(55, 4.5, f"{row['avg_value']:.4f}", border=1, align="R")
            pdf.cell(47, 4.5, f"{gap_pct:+.1f}%",        border=1, align="R")
            pdf.set_text_color(107, 114, 128)
            pdf.cell(50, 4.5, badge, border=1, align="C")
            pdf.set_text_color(28, 28, 40)
            pdf.ln()

    # ── Recommendations ───────────────────────────────────────────────────────
    seen: set[str] = set()
    recs: list[str] = []
    for ins in critical + warnings + opps:
        a = ins.get("action", "").strip()
        if a and a not in seen:
            seen.add(a)
            recs.append(a)
        if len(recs) >= 6:
            break

    if recs:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(28, 28, 40)
        pdf.cell(W, 5, "Acciones Recomendadas", ln=True)
        pdf.ln(1.5)
        for i, rec in enumerate(recs, 1):
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(28, 28, 40)
            pdf.multi_cell(W, 5, _s(f"{i}. {rec}", 220))

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-13)
    pdf.set_draw_color(228, 232, 240)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(1.5)
    pdf.set_font("Helvetica", "", 6.5)
    pdf.set_text_color(156, 163, 175)
    pdf.cell(W, 4, _s(f"Rappi Operations Analytics  .  {now}  .  Uso exclusivo interno Rappi"), align="C")

    return bytes(pdf.output())


# ── Chat transcript exports ───────────────────────────────────────────────────

def generate_chat_csv(messages: list[dict]) -> str:
    """Return a UTF-8 CSV transcript of a Copiloto IA conversation."""
    rows = []
    turn = 1
    for m in messages:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        rows.append({
            "turno":     turn,
            "rol":       "Usuario" if role == "user" else "Copiloto IA",
            "contenido": m.get("content", ""),
        })
        if role == "assistant":
            turn += 1
    if not rows:
        return "turno,rol,contenido\n"
    return pd.DataFrame(rows).to_csv(index=False)


def generate_chat_pdf(messages: list[dict]) -> bytes:
    """Return an A4 PDF transcript of a Copiloto IA conversation. Requires fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError("fpdf2 is required: pip install fpdf2") from exc

    def _s(v: object, maxlen: int = 0) -> str:
        s = str(v).encode("latin-1", errors="replace").decode("latin-1")
        return s[:maxlen] if maxlen else s

    now = datetime.now().strftime("%d %b %Y, %H:%M")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_margins(14, 14, 14)
    W = 182

    # Header band
    pdf.set_fill_color(28, 28, 40)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(14, 6)
    pdf.cell(W, 7, "rappi  Copiloto IA", ln=False)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(14, 15)
    pdf.cell(W, 5, _s(f"Transcripcion de sesion  .  {now}"), ln=False)
    pdf.set_text_color(28, 28, 40)
    pdf.set_y(36)

    turn = 1
    for m in messages:
        role    = m.get("role", "")
        content = m.get("content", "")
        if role not in ("user", "assistant"):
            continue

        if role == "user":
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(255, 68, 31)
            pdf.cell(W, 4, "TU", align="R", ln=True)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(28, 28, 40)
            pdf.set_fill_color(255, 244, 242)
            pdf.set_draw_color(255, 210, 200)
            pdf.multi_cell(W, 5, _s(content), border=1, fill=True)
            pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(107, 114, 128)
            pdf.cell(W, 4, _s(f"COPILOTO IA  .  Respuesta {turn}"), ln=True)
            turn += 1
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(28, 28, 40)
            pdf.set_fill_color(248, 249, 251)
            pdf.set_draw_color(228, 232, 240)
            pdf.multi_cell(W, 5, _s(content), border=1, fill=True)
            pdf.ln(4)

    # Footer
    pdf.set_y(-13)
    pdf.set_draw_color(228, 232, 240)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(1.5)
    pdf.set_font("Helvetica", "", 6.5)
    pdf.set_text_color(156, 163, 175)
    pdf.cell(W, 4, _s(f"Rappi Operations Analytics  .  {now}  .  Uso exclusivo interno Rappi"), align="C")

    return bytes(pdf.output())


# ── Email body helper ──────────────────────────────────────────────────────────

def build_email_body(
    insights: list[dict],
    summary: dict,
    metric: str,
    country: str | None,
) -> str:
    """
    Return a compact HTML email body (inline styles only, no external resources).
    Suitable for real SMTP delivery when credentials are configured.
    """
    now   = datetime.now().strftime("%d %b %Y, %H:%M")
    scope = country if country else "LATAM"

    ordered  = sorted(insights, key=lambda x: SEVERITY_ORDER.get(x["severity"], 9))
    critical = [i for i in ordered if i["severity"] == "critical"]
    warnings = [i for i in ordered if i["severity"] == "warning"]
    opps     = [i for i in ordered if i["severity"] == "opportunity"]

    top = (critical + warnings)[:5]
    top_rows = "".join(
        f'<tr><td style="padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;">'
        f'{i + 1}. <b>{_e(ins["title"])}</b></td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;'
        f'color:#EF4444;">{ins["delta_pct"]:+.1f}%</td></tr>'
        for i, ins in enumerate(top)
    ) if top else '<tr><td colspan="2" style="padding:8px;color:#9CA3AF;font-size:12px;">Sin alertas criticas.</td></tr>'

    return f"""<!DOCTYPE html>
<html><body style="font-family:Arial,sans-serif;background:#F5F6FA;padding:20px;">
<div style="max-width:580px;margin:0 auto;background:white;border-radius:12px;overflow:hidden;
            box-shadow:0 2px 10px rgba(0,0,0,0.08);">
  <div style="background:#1C1C28;padding:22px 28px;color:white;">
    <div style="font-size:20px;font-weight:900;">rappi <span style="color:#FF441F;">·</span> analytics</div>
    <div style="font-size:12px;color:rgba(255,255,255,0.5);margin-top:4px;">
      Alerta Operativa Automatica &nbsp;·&nbsp; {_e(scope)} &nbsp;·&nbsp; {now}
    </div>
  </div>
  <div style="padding:24px 28px;">
    <p style="font-size:14px;color:#1C1C28;margin-bottom:20px;">
      Se detectaron <b style="color:#EF4444;">{len(critical)} alertas criticas</b> y
      <b style="color:#F59E0B;">{len(warnings)} alertas</b> para <b>{_e(metric)}</b>.
      Se identificaron ademas <b style="color:#3B82F6;">{len(opps)} oportunidades</b>.
    </p>
    <table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
      <thead>
        <tr><th style="text-align:left;padding:8px;background:#F8F9FB;font-size:11px;
                        text-transform:uppercase;letter-spacing:.5px;color:#9CA3AF;">Hallazgo</th>
            <th style="text-align:left;padding:8px;background:#F8F9FB;font-size:11px;
                        text-transform:uppercase;letter-spacing:.5px;color:#9CA3AF;">Cambio</th></tr>
      </thead>
      <tbody>{top_rows}</tbody>
    </table>
    <p style="font-size:12px;color:#9CA3AF;">
      Para ver el informe completo con graficos y analisis detallado, abre el dashboard de operaciones.
    </p>
  </div>
  <div style="background:#F8F9FB;padding:14px 28px;font-size:11px;color:#9CA3AF;
              border-top:1px solid #E4E8F0;">
    Rappi Operations Analytics &nbsp;·&nbsp; Generado automaticamente &nbsp;·&nbsp; {now}
  </div>
</div>
</body></html>"""
