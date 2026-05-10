import pandas as pd

SEVERITY_ORDER = {"critical": 0, "warning": 1, "opportunity": 2, "positive": 3}

SEVERITY_CONFIG = {
    "critical":    {"color": "#EF4444", "emoji": "🔴", "label": "CRITICO"},
    "warning":     {"color": "#F59E0B", "emoji": "🟠", "label": "ALERTA"},
    "opportunity": {"color": "#3B82F6", "emoji": "🔵", "label": "OPORTUNIDAD"},
    "positive":    {"color": "#10B981", "emoji": "🟢", "label": "POSITIVO"},
}

# ── Thresholds ─────────────────────────────────────────────────────────────────
_CRITICAL_DECLINE  = -20.0   # severe WoW drop — true emergencies
_WARNING_DECLINE   = -10.0   # >10 % WoW deterioration (assessment requirement)
_POSITIVE_GAIN     = 10.0    # >10 % WoW improvement
_CONSEC_MIN_DROP   = -5.0    # minimum cumulative drop for a 3-week sustained trend
_SUSTAINED_UP_MIN  = 4.0     # minimum cumulative gain for a 3-week sustained improvement
_OPP_Z_THRESHOLD   = -1.2    # z-score for opportunity detection within country
_BENCH_Z_THRESHOLD = -1.5    # z-score for within-segment benchmarking
_PRIO_DECLINE      = -5.0    # lower bar for high-priority zones
_PRIO_CRITICAL     = -12.0   # critical bar for high-priority zones


# ── Internal helpers ───────────────────────────────────────────────────────────

def _needs(df: pd.DataFrame, cols: set[str]) -> bool:
    return cols.issubset(df.columns)


def _insight(severity, title, finding, explanation, action, row, delta_pct):
    return {
        "severity":    severity,
        "title":       title,
        "finding":     finding,
        "explanation": explanation,
        "action":      action,
        "metric":      row["METRIC"],
        "country":     row["COUNTRY"],
        "city":        row["CITY"],
        "zone":        row["ZONE"],
        "value":       round(float(row.get("L0W_ROLL", 0)), 4),
        "delta_pct":   round(float(delta_pct), 1),
    }


def _free_insight(severity, title, finding, explanation, action,
                  metric, country, city, zone, value, delta_pct):
    return {
        "severity":    severity,
        "title":       title,
        "finding":     finding,
        "explanation": explanation,
        "action":      action,
        "metric":      metric,
        "country":     country,
        "city":        city,
        "zone":        zone,
        "value":       round(float(value), 4),
        "delta_pct":   round(float(delta_pct), 1),
    }


# ── Detector 1 · WoW anomalies (>10 % deterioration / improvement) ─────────────

def detect_wow_changes(df: pd.DataFrame) -> list[dict]:
    """Flag >10 % week-over-week deterioration and improvement (L1W_ROLL -> L0W_ROLL)."""
    if not _needs(df, {"L0W_ROLL", "L1W_ROLL", "METRIC", "ZONE", "COUNTRY", "CITY"}):
        return []

    work = (
        df[["COUNTRY", "CITY", "ZONE", "METRIC", "L1W_ROLL", "L0W_ROLL"]]
        .dropna()
        .query("L1W_ROLL != 0")
        .copy()
    )
    work["rel"] = (work["L0W_ROLL"] - work["L1W_ROLL"]) / work["L1W_ROLL"].abs() * 100
    work = work[work["rel"].between(-500, 500)]

    insights = []

    # Declines — anything worse than -10 %
    for _, r in work[work["rel"] <= _WARNING_DECLINE].sort_values("rel").head(10).iterrows():
        sev  = "critical" if r["rel"] <= _CRITICAL_DECLINE else "warning"
        drop = abs(r["rel"])
        insights.append(_insight(
            sev,
            f"Caida de metrica - {r['ZONE']}",
            f"{r['METRIC']} cayo {drop:.1f}% SaS ({r['L1W_ROLL']:.3f} -> {r['L0W_ROLL']:.3f})",
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) registro una caida "
             f"{'severa' if sev == 'critical' else 'significativa'} semana a semana en "
             f"{r['METRIC']}. Puede indicar escasez de repartidores, problemas de calidad "
             f"en tiendas o disrupcion de demanda que requiere investigacion en campo."),
            ("Revisar cobertura de repartidores y disponibilidad de tiendas en esta zona. "
             "Contrastar con tasas de cancelacion de pedidos y comparar con zonas vecinas."),
            r, r["rel"],
        ))

    # Improvements — anything better than +10 %
    for _, r in work[work["rel"] >= _POSITIVE_GAIN].sort_values("rel", ascending=False).head(8).iterrows():
        insights.append(_insight(
            "positive",
            f"Mejora notable - {r['ZONE']}",
            f"{r['METRIC']} mejoro {r['rel']:.1f}% SaS ({r['L1W_ROLL']:.3f} -> {r['L0W_ROLL']:.3f})",
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) mostro un cambio positivo "
             f"destacado en {r['METRIC']} esta semana. Puede reflejar intervenciones operativas "
             f"recientes, mejores condiciones de oferta o activaciones comerciales exitosas."),
            ("Documentar que cambio en esta zona y replicar el enfoque "
             "en zonas con bajo rendimiento y perfil similar."),
            r, r["rel"],
        ))

    return insights


# ── Detector 2 · Worrying trends (3+ consecutive weeks of decline) ─────────────

def detect_consecutive_decline(df: pd.DataFrame) -> list[dict]:
    """Zones with 3 consecutive weeks of decline above a minimum magnitude."""
    if not _needs(df, {"L2W_ROLL", "L1W_ROLL", "L0W_ROLL", "METRIC", "ZONE", "COUNTRY", "CITY"}):
        return []

    work = df[["COUNTRY", "CITY", "ZONE", "METRIC", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]].dropna().copy()
    work = work[(work["L2W_ROLL"] > work["L1W_ROLL"]) & (work["L1W_ROLL"] > work["L0W_ROLL"])]
    work["total_pct"] = (work["L0W_ROLL"] - work["L2W_ROLL"]) / work["L2W_ROLL"].abs() * 100
    work = work[work["total_pct"] <= _CONSEC_MIN_DROP].sort_values("total_pct").head(8)

    insights = []
    for _, r in work.iterrows():
        sev = "critical" if r["total_pct"] <= -12 else "warning"
        insights.append(_insight(
            sev,
            f"Tendencia bajista 3 semanas - {r['ZONE']}",
            (f"{r['METRIC']}: {r['L2W_ROLL']:.3f} -> {r['L1W_ROLL']:.3f} -> {r['L0W_ROLL']:.3f} "
             f"(-{abs(r['total_pct']):.1f}% en 3 semanas)"),
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) ha declinado cada semana durante "
             f"3 semanas consecutivas en {r['METRIC']}. Las tendencias sostenidas son una senal "
             f"mas fuerte que caidas puntuales y requieren intervencion operativa estructurada."),
            ("Escalar al lider de operaciones para analisis de causa raiz. Revisar tendencias "
             "de densidad de repartidores, calidad de incorporacion de tiendas y patrones "
             "de demanda local en el mismo periodo."),
            r, r["total_pct"],
        ))

    return insights


# ── Detector 3 · Sustained improvement (3+ consecutive weeks up) ───────────────

def detect_sustained_improvement(df: pd.DataFrame) -> list[dict]:
    """Zones with 3 consecutive weeks of improvement — operational wins to replicate."""
    if not _needs(df, {"L2W_ROLL", "L1W_ROLL", "L0W_ROLL", "METRIC", "ZONE", "COUNTRY", "CITY"}):
        return []

    work = df[["COUNTRY", "CITY", "ZONE", "METRIC", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]].dropna().copy()
    improving = work[
        (work["L2W_ROLL"] < work["L1W_ROLL"]) & (work["L1W_ROLL"] < work["L0W_ROLL"])
    ].copy()
    improving["total_pct"] = (
        (improving["L0W_ROLL"] - improving["L2W_ROLL"]) / improving["L2W_ROLL"].abs() * 100
    )
    improving = improving[improving["total_pct"] >= _SUSTAINED_UP_MIN].sort_values(
        "total_pct", ascending=False
    ).head(8)

    insights = []
    for _, r in improving.iterrows():
        insights.append(_insight(
            "positive",
            f"Recuperacion sostenida - {r['ZONE']}",
            (f"{r['METRIC']}: {r['L2W_ROLL']:.3f} -> {r['L1W_ROLL']:.3f} -> {r['L0W_ROLL']:.3f} "
             f"(+{r['total_pct']:.1f}% en 3 semanas)"),
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) ha mejorado de forma consistente "
             f"durante 3 semanas en {r['METRIC']}. Este patron indica una intervencion exitosa "
             f"o una recuperacion organica sostenible que debe capitalizarse."),
            ("Identificar y documentar los factores de exito. "
             "Replicar el modelo en zonas con perfil similar que esten por debajo del promedio."),
            r, r["total_pct"],
        ))

    return insights


# ── Detector 4 · Within-segment benchmarking ───────────────────────────────────

def detect_zone_type_benchmarking(df: pd.DataFrame) -> list[dict]:
    """
    Compare zones to peers of the same ZONE_TYPE within the same country.
    More precise than country-average benchmarking — apples-to-apples.
    """
    required = {"ZONE_TYPE", "L0W_ROLL", "METRIC", "ZONE", "COUNTRY", "CITY"}
    if not _needs(df, required):
        return []

    work = df[["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "METRIC", "L0W_ROLL"]].dropna()
    stats = (
        work.groupby(["COUNTRY", "ZONE_TYPE", "METRIC"])["L0W_ROLL"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats = stats[stats["count"] >= 4]  # meaningful peer group

    merged = work.merge(stats, on=["COUNTRY", "ZONE_TYPE", "METRIC"])
    merged = merged[merged["std"] > 0.005].copy()
    merged["z"] = (merged["L0W_ROLL"] - merged["mean"]) / merged["std"]

    laggards = merged[merged["z"] <= _BENCH_Z_THRESHOLD].nsmallest(10, "z")

    insights = []
    for _, r in laggards.iterrows():
        gap_pct = (r["mean"] - r["L0W_ROLL"]) / abs(r["mean"]) * 100 if r["mean"] else 0
        insights.append(_free_insight(
            "opportunity",
            f"Rezagado en segmento {r['ZONE_TYPE']} - {r['ZONE']}",
            (f"{r['METRIC']}: {r['L0W_ROLL']:.3f} vs promedio {r['ZONE_TYPE']} "
             f"en {r['COUNTRY']}: {r['mean']:.3f} (brecha: {gap_pct:.1f}%)"),
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) tiene bajo rendimiento "
             f"en {r['METRIC']} comparada con sus pares del segmento {r['ZONE_TYPE']} en el "
             f"mismo pais. Este benchmark entre pares es la comparacion mas justa disponible "
             f"porque elimina diferencias estructurales entre tipos de zona."),
            (f"Analizar las mejores zonas {r['ZONE_TYPE']} del mismo pais para identificar "
             "practicas replicables. Priorizar si la zona es de alta priorizacion operativa."),
            r["METRIC"], r["COUNTRY"], r["CITY"], r["ZONE"],
            r["L0W_ROLL"], -gap_pct,
        ))

    return insights


# ── Detector 5 · Cross-metric correlations ─────────────────────────────────────

def detect_metric_correlations(df: pd.DataFrame) -> list[dict]:
    """
    Detect zones with operationally significant divergence between related metrics.
    Example: high Perfect Orders but low Turbo Adoption = adoption gap opportunity.
    """
    if "L0W_ROLL" not in df.columns:
        return []

    # Pivot to zone x metric matrix
    try:
        pivot = df.pivot_table(
            index=["COUNTRY", "CITY", "ZONE"],
            columns="METRIC",
            values="L0W_ROLL",
            aggfunc="mean",
        )
    except Exception:
        return []

    if pivot.empty:
        return []

    available = set(pivot.columns.tolist())

    # Operationally meaningful divergence pairs
    PAIRS = [
        (
            "Perfect Orders", "Turbo Adoption",
            "Calidad operativa alta pero baja adopcion Turbo",
            (
                "La zona tiene buen desempeno en pedidos perfectos pero esta rezagada en "
                "adopcion Turbo. La calidad operativa existente soporta el incremento de "
                "velocidad sin riesgo de deterioro de experiencia del usuario."
            ),
            (
                "Lanzar campana de activacion Turbo focalizada en esta zona. "
                "La calidad operativa del historial es el mejor argumento de adopcion."
            ),
        ),
        (
            "Perfect Orders", "Pro Adoption (Last Week Status)",
            "Calidad alta pero baja penetracion Pro",
            (
                "La zona tiene alta calidad de pedidos pero baja adopcion del programa Pro. "
                "Los usuarios activos con buena experiencia son candidatos naturales de "
                "conversion — el churn risk es bajo."
            ),
            (
                "Activar campana de conversion Pro en esta zona. "
                "El historial de calidad de pedidos es el mejor argumento de venta del programa."
            ),
        ),
        (
            "Gross Profit UE", "Perfect Orders",
            "Alta rentabilidad pero calidad de pedidos por debajo del promedio",
            (
                "La zona genera buena rentabilidad por unidad economica pero tiene calidad "
                "de pedidos por debajo de la media. Mejorar Perfect Orders en esta zona "
                "protegeria la retencion de usuarios de alto valor y aumentaria LTV."
            ),
            (
                "Priorizar mejoras operativas en esta zona. "
                "La combinacion de alta rentabilidad y calidad recuperable representa "
                "el mayor ROI potencial de inversion operativa a corto plazo."
            ),
        ),
        (
            "Lead Penetration", "Perfect Orders",
            "Alta penetracion de leads pero baja calidad de conversion",
            (
                "La zona tiene alta penetracion de leads pero el desempeno de pedidos perfectos "
                "esta por debajo del promedio. Los leads captados no estan convirtiendo a "
                "experiencias de calidad — riesgo de churning de nuevos usuarios."
            ),
            (
                "Revisar el flujo de onboarding y la calidad de tiendas activas para nuevos usuarios. "
                "Considerar seleccion de tiendas de alta calidad en el primer pedido."
            ),
        ),
    ]

    insights = []
    for metric_a, metric_b, label, explanation, action in PAIRS:
        if metric_a not in available or metric_b not in available:
            continue

        sub = pivot[[metric_a, metric_b]].dropna().copy()
        if len(sub) < 6:
            continue

        mean_a, std_a = sub[metric_a].mean(), sub[metric_a].std()
        mean_b, std_b = sub[metric_b].mean(), sub[metric_b].std()
        if std_a < 0.001 or std_b < 0.001:
            continue

        sub["z_a"] = (sub[metric_a] - mean_a) / std_a
        sub["z_b"] = (sub[metric_b] - mean_b) / std_b

        # High metric_a but low metric_b
        divergent = sub[(sub["z_a"] > 0.6) & (sub["z_b"] < -0.6)].nsmallest(4, "z_b")

        for (country, city, zone), row in divergent.iterrows():
            val_a = row[metric_a]
            val_b = row[metric_b]
            gap = (mean_b - val_b) / abs(mean_b) * 100 if mean_b else 0
            insights.append(_free_insight(
                "opportunity",
                f"{label} - {zone}",
                (f"{metric_a}: {val_a:.3f} (alto) vs {metric_b}: {val_b:.3f} "
                 f"(brecha vs prom: {gap:.1f}%)"),
                explanation,
                action,
                f"{metric_a} / {metric_b}", country, city, zone,
                val_a, -gap,
            ))

    return insights


# ── Detector 6 · Opportunity zones (below country average) ────────────────────

def detect_opportunity_zones(df: pd.DataFrame) -> list[dict]:
    """Zones significantly below the country average — high uplift potential."""
    if not _needs(df, {"L0W_ROLL", "METRIC", "ZONE", "COUNTRY", "CITY"}):
        return []

    work = df[["COUNTRY", "CITY", "ZONE", "METRIC", "L0W_ROLL"]].dropna()
    stats = work.groupby(["COUNTRY", "METRIC"])["L0W_ROLL"].agg(["mean", "std"]).reset_index()
    merged = work.merge(stats, on=["COUNTRY", "METRIC"])
    merged = merged[merged["std"] > 0.001].copy()
    merged["z"] = (merged["L0W_ROLL"] - merged["mean"]) / merged["std"]

    laggards = merged[merged["z"] <= _OPP_Z_THRESHOLD].sort_values("z").head(10)

    insights = []
    for _, r in laggards.iterrows():
        gap_pct = (r["mean"] - r["L0W_ROLL"]) / abs(r["mean"]) * 100 if r["mean"] else 0
        insights.append(_insight(
            "opportunity",
            f"Zona de oportunidad - {r['ZONE']}",
            (f"{r['METRIC']} esta {gap_pct:.1f}% por debajo del promedio pais "
             f"({r['L0W_ROLL']:.3f} vs {r['mean']:.3f}, z={r['z']:.1f})"),
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) es un rezagado significativo "
             f"en {r['METRIC']} respecto al benchmark nacional. Cerrar esta brecha "
             f"representa un potencial de GMV medible."),
            ("Comparar con las mejores zonas de perfil similar en riqueza y priorizacion. "
             "Considerar activacion focalizada o incentivo comercial para esta zona."),
            r, -gap_pct,
        ))

    return insights


# ── Detector 7 · High-priority zone risk ──────────────────────────────────────

def detect_priority_zone_risk(df: pd.DataFrame) -> list[dict]:
    """High-priority zones with declining metrics — elevated operational risk."""
    required = {"ZONE_PRIORITIZATION", "L0W_ROLL", "L1W_ROLL", "METRIC", "ZONE", "COUNTRY", "CITY"}
    if not _needs(df, required):
        return []

    prio_mask = df["ZONE_PRIORITIZATION"].astype(str).str.lower().str.contains(
        r"high|alta|1\b|priorit", na=False, regex=True
    )
    prio_df = df[prio_mask]
    if prio_df.empty:
        return []

    work = (
        prio_df[["COUNTRY", "CITY", "ZONE", "METRIC", "ZONE_PRIORITIZATION", "L1W_ROLL", "L0W_ROLL"]]
        .dropna()
        .query("L1W_ROLL != 0")
        .copy()
    )
    work["rel"] = (work["L0W_ROLL"] - work["L1W_ROLL"]) / work["L1W_ROLL"].abs() * 100
    work = work[work["rel"].between(-500, 500)]
    declining = work[work["rel"] <= _PRIO_DECLINE].sort_values("rel").head(5)

    insights = []
    for _, r in declining.iterrows():
        sev  = "critical" if r["rel"] <= _PRIO_CRITICAL else "warning"
        drop = abs(r["rel"])
        insights.append(_insight(
            sev,
            f"Zona prioritaria en riesgo - {r['ZONE']}",
            (f"{r['METRIC']} cayo {drop:.1f}% SaS en zona de alta prioridad "
             f"({r['L1W_ROLL']:.3f} -> {r['L0W_ROLL']:.3f})"),
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) es de alta prioridad operativa "
             f"y registra una caida en {r['METRIC']}. El impacto en GMV es desproporcionadamente "
             f"mayor que en zonas estandar y requiere atencion inmediata del equipo de campo."),
            ("Escalar directamente al gerente de operaciones del pais. "
             "Revisar KPIs de repartidores activos, store uptime y tasa de cancelacion en los ultimos 7 dias."),
            r, r["rel"],
        ))

    return insights


# ── Aggregator ─────────────────────────────────────────────────────────────────

def generate_insights(
    df: pd.DataFrame,
    country: str | None = None,
    metric: str | None = None,
    max_insights: int = 40,
) -> list[dict]:
    """Run all detectors against an optionally filtered slice of the metrics DataFrame."""
    if "L0W_ROLL" not in df.columns:
        return []

    # Country filter applied to all detectors
    country_df = df[df["COUNTRY"] == country] if country else df

    # Metric filter applied to single-metric detectors
    filtered = country_df[country_df["METRIC"] == metric] if metric else country_df

    all_insights = (
        detect_wow_changes(filtered)
        + detect_consecutive_decline(filtered)
        + detect_sustained_improvement(filtered)
        + detect_opportunity_zones(filtered)
        + detect_zone_type_benchmarking(filtered)
        + detect_metric_correlations(country_df)  # needs all metrics for pivot
        + detect_priority_zone_risk(filtered)
    )

    # Deduplicate: keep highest-severity insight per (zone, metric)
    seen: dict[tuple, int] = {}
    deduped: list[dict] = []
    for ins in all_insights:
        key = (ins["zone"], ins["metric"])
        new_rank = SEVERITY_ORDER.get(ins["severity"], 9)
        if key not in seen or new_rank < seen[key]:
            seen[key] = new_rank
            deduped = [i for i in deduped if (i["zone"], i["metric"]) != key]
            deduped.append(ins)

    # Sort by severity, then by absolute delta magnitude within each group
    from itertools import groupby
    deduped.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"], 9))
    sorted_groups: list[dict] = []
    for _, group in groupby(deduped, key=lambda x: x["severity"]):
        sorted_groups.extend(sorted(group, key=lambda x: abs(x.get("delta_pct", 0)), reverse=True))

    # Per-category caps for balanced output
    _CAPS = {"critical": 15, "warning": 10, "opportunity": 12, "positive": 10}
    _counts: dict[str, int] = {k: 0 for k in _CAPS}
    balanced: list[dict] = []
    for ins in sorted_groups:
        sev = ins["severity"]
        if _counts.get(sev, 0) < _CAPS.get(sev, 99):
            balanced.append(ins)
            _counts[sev] = _counts.get(sev, 0) + 1
    return balanced[:max_insights]


# ── Summary counts ─────────────────────────────────────────────────────────────

def count_by_severity(insights: list[dict]) -> dict[str, int]:
    counts = {s: 0 for s in SEVERITY_CONFIG}
    for i in insights:
        counts[i["severity"]] = counts.get(i["severity"], 0) + 1
    return counts
