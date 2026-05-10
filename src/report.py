"""
Executive report generator.
Produces self-contained HTML and Markdown reports from live insight data.
All visible text is in professional Spanish. No external images required.
"""
from __future__ import annotations

import html as _html_mod
from datetime import datetime

import pandas as pd

from src.insights import SEVERITY_CONFIG, SEVERITY_ORDER

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
    cfg   = SEVERITY_CONFIG[ins["severity"]]
    color = cfg["color"]
    label = cfg["label"]
    delta = ins["delta_pct"]
    delta_color = _GREEN if delta > 0 else _DANGER
    delta_str   = f"{delta:+.1f}%"
    return f"""
    <div style="border-left:4px solid {color};background:#FAFBFF;border-radius:0 10px 10px 0;
                padding:13px 17px;margin-bottom:10px;">
      <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:7px;">
        <span style="background:{color};color:white;font-size:9.5px;font-weight:700;
                     letter-spacing:.5px;padding:3px 9px;border-radius:100px;">{_e(label)}</span>
        <span style="background:#F1F3F9;color:{delta_color};font-size:10px;font-weight:700;
                     padding:3px 9px;border-radius:100px;">{_e(delta_str)}</span>
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

    ordered   = sorted(insights, key=lambda x: SEVERITY_ORDER.get(x["severity"], 9))
    critical  = [i for i in ordered if i["severity"] == "critical"]
    warnings  = [i for i in ordered if i["severity"] == "warning"]
    opps      = [i for i in ordered if i["severity"] == "opportunity"]
    positives = [i for i in ordered if i["severity"] == "positive"]
    trend_insights = [
        i for i in ordered
        if "tendencia" in i.get("title", "").lower() or "recuperacion" in i.get("title", "").lower()
    ]

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

{_section("Alertas Criticas", "🔴", "#FFF5F5", critical)}
{_section("Alertas de Atencion", "🟠", "#FFFBEB", warnings)}
{_section("Tendencias Preocupantes (3+ semanas)", "📉", "#FFF8E1", trend_insights) if trend_insights else ""}

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

{_section("Oportunidades Identificadas", "🔵", "#EFF6FF", opps)}
{_section("Senales Positivas", "🟢", "#F0FDF4", positives)}

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

    ordered   = sorted(insights, key=lambda x: SEVERITY_ORDER.get(x["severity"], 9))
    critical  = [i for i in ordered if i["severity"] == "critical"]
    warnings  = [i for i in ordered if i["severity"] == "warning"]
    opps      = [i for i in ordered if i["severity"] == "opportunity"]
    positives = [i for i in ordered if i["severity"] == "positive"]

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

    if critical:
        lines += ["## 🔴 Alertas Criticas", ""]
        for ins in critical:
            lines += _md_insight(ins)

    if warnings:
        lines += ["## 🟠 Alertas de Atencion", ""]
        for ins in warnings:
            lines += _md_insight(ins)

    trend_ins = [
        i for i in ordered
        if "tendencia" in i.get("title", "").lower() or "recuperacion" in i.get("title", "").lower()
    ]
    if trend_ins:
        lines += ["## 📉 Tendencias Preocupantes (3+ semanas)", ""]
        for ins in trend_ins:
            lines += _md_insight(ins)

    # Country benchmark
    if not avg_df.empty:
        lines += ["## 🌎 Benchmark de Paises", "", f"| Pais | {metric} (W-0) |", "|------|---------|"]
        for _, row in avg_df.iterrows():
            lines.append(f"| {row['COUNTRY']} | {row['avg_value']:.3f} |")
        lines.append("")

    if opps:
        lines += ["## 🔵 Oportunidades Identificadas", ""]
        for ins in opps:
            lines += _md_insight(ins)

    if positives:
        lines += ["## 🟢 Senales Positivas", ""]
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
