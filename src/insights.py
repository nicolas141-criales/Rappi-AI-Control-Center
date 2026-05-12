import pandas as pd

SEVERITY_ORDER = {"critical": 0, "warning": 1, "opportunity": 2, "positive": 3}

SEVERITY_CONFIG = {
    "critical":    {"color": "#EF4444", "emoji": "🔴", "label": "CRITICO"},
    "warning":     {"color": "#F59E0B", "emoji": "🟠", "label": "ALERTA"},
    "opportunity": {"color": "#3B82F6", "emoji": "🔵", "label": "OPORTUNIDAD"},
    "positive":    {"color": "#10B981", "emoji": "🟢", "label": "POSITIVO"},
}

CATEGORY_CONFIG = {
    "anomaly":     {"label": "Anomalía",    "emoji": "🔺", "color": "#EF4444"},
    "trend":       {"label": "Tendencia",   "emoji": "📉", "color": "#F59E0B"},
    "benchmark":   {"label": "Benchmark",   "emoji": "🌎", "color": "#3B82F6"},
    "correlation": {"label": "Correlación", "emoji": "🔗", "color": "#8B5CF6"},
    "opportunity": {"label": "Oportunidad", "emoji": "💡", "color": "#10B981"},
}

# ── Thresholds ─────────────────────────────────────────────────────────────────
_CRITICAL_DECLINE      = -20.0   # severe WoW drop — true emergencies
_WARNING_DECLINE       = -10.0   # >10 % WoW deterioration (assessment requirement)
_POSITIVE_GAIN         = 10.0    # >10 % WoW improvement
_CONSEC_MIN_DROP       = -5.0    # minimum cumulative drop for a 3-week sustained trend
_SUSTAINED_UP_MIN      = 4.0     # minimum cumulative gain for a 3-week sustained improvement
_OPP_Z_THRESHOLD       = -1.2    # z-score for opportunity detection within country
_OPP_Z_PRIO_THRESHOLD  = -0.8    # lower bar for high-priority zones in opportunity detection
_BENCH_Z_THRESHOLD     = -1.5    # z-score for within-segment benchmarking
_PRIO_DECLINE          = -5.0    # lower bar for high-priority zones
_PRIO_CRITICAL         = -12.0   # critical bar for high-priority zones
_BENCH_COUNTRY_Z       = -1.0    # z-score for country-level benchmark laggards

# Severity score multipliers — used to compute sortable score across categories
_SEV_SCORE = {"critical": 3.0, "warning": 2.0, "opportunity": 1.5, "positive": 1.0}

# Score / gap normalization
_GAP_PCT_CAP = 200.0   # cap relative gap at ±200 % — prevents explosive values on near-zero means
_SCORE_SCALE = 6.0     # normalize score to 0–100: max_raw = _GAP_PCT_CAP * 3.0 = 600 → 600/6 = 100

# Per-metric operational definitions — used to enrich explanations beyond generic analytics language
_METRIC_CONTEXT = {
    "Perfect Orders":                "pedidos completados sin incidencias (cancelaciones, errores o problemas de entrega) como % del total",
    "Gross Profit UE":               "utilidad bruta por unidad economica — rentabilidad neta por pedido en la zona",
    "Lead Penetration":              "% de leads nuevos captados que completan su primer pedido en la plataforma",
    "Turbo Adoption":                "% de pedidos o usuarios que utilizan el servicio de entrega express Turbo",
    "Pro Adoption (Last Week Status)": "% de usuarios activos suscritos al programa Rappi Pro la semana pasada",
    "% PRO Users Who Breakeven":     "% de suscriptores Pro que recuperan el costo de su suscripcion en pedidos realizados",
    "Non-Pro PTC > OP":              "tasa de conversion de carrito a pedido para usuarios no suscritos a Pro",
    "Restaurants SST > SS CVR":      "tasa de conversion de sesion a busqueda en el vertice de restaurantes",
    "Restaurants SS > ATC CVR":      "tasa de conversion de busqueda a agregar al carrito en restaurantes",
}

# Per-metric likely causes of a decline — replaces generic "escasez de repartidores" fallback
_METRIC_DROP_CAUSE = {
    "Perfect Orders":                "puede reflejar aumento en cancelaciones, errores de preparacion o problemas de entrega que deterioran la experiencia del usuario",
    "Gross Profit UE":               "puede indicar presion en costos operativos, cambio en mix de pedidos hacia tickets mas bajos o reduccion en eficiencia de entrega",
    "Lead Penetration":              "puede reflejar problemas en el flujo de activacion de nuevos usuarios, baja calidad de leads captados o friccion en el primer pedido",
    "Turbo Adoption":                "puede indicar deterioro en disponibilidad del servicio Turbo, aumento de tiempos de entrega express o menor visibilidad del servicio en la plataforma",
    "Pro Adoption (Last Week Status)": "puede indicar churn de suscriptores existentes o falla en activacion de nuevos suscriptores Pro",
    "% PRO Users Who Breakeven":     "indica que los suscriptores no ordenan con frecuencia suficiente para recuperar la inversion — riesgo de churn del programa",
    "Non-Pro PTC > OP":              "puede indicar friccion en el checkout, percepcion de precio no competitivo o experiencia degradada para usuarios no suscritos",
    "Restaurants SST > SS CVR":      "puede reflejar baja relevancia del contenido de entrada o problemas de oferta visible al abrir la app de restaurantes",
    "Restaurants SS > ATC CVR":      "puede indicar que el surtido disponible no satisface la demanda expresada o que los resultados de busqueda no son relevantes",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _needs(df: pd.DataFrame, cols: set[str]) -> bool:
    return cols.issubset(df.columns)


def _priority_weight(val) -> float:
    """Score multiplier based on zone prioritization level."""
    if pd.isna(val):
        return 1.0
    s = str(val).lower()
    if any(k in s for k in ("high", "alta", "1")):
        return 2.0
    if "priorit" in s:
        return 1.5
    return 1.0


def _safe_gap_pct(value: float, mean: float) -> float:
    """
    Relative gap capped at ±_GAP_PCT_CAP.
    Returns 0 when mean is effectively zero to avoid division explosions.
    Gross Profit UE and similar metrics can have means near zero or negative,
    causing raw percentages in the thousands.
    """
    if abs(mean) < 1e-6:
        return 0.0
    raw = (mean - value) / abs(mean) * 100.0
    return max(-_GAP_PCT_CAP, min(_GAP_PCT_CAP, raw))


def _gap_label(gap_pct: float, z: float | None = None) -> str:
    """
    Executive-friendly gap description.
    When the percentage hit the cap (extreme case), falls back to z-score language
    to avoid showing suspicious-looking numbers to stakeholders.
    """
    hit_cap = abs(gap_pct) >= _GAP_PCT_CAP - 5
    if hit_cap:
        if z is not None:
            if z <= -3.0:
                return "extremadamente por debajo del benchmark de pares"
            if z <= -2.0:
                return "muy por debajo del benchmark de pares"
            return f"por debajo del benchmark (desviacion z = {z:.1f})"
        return "significativamente por debajo del benchmark"
    direction = "por debajo" if gap_pct >= 0 else "por encima"
    return f"{abs(gap_pct):.1f}% {direction} del promedio"


def _insight(severity, title, finding, explanation, action, row, delta_pct,
             category: str = "anomaly"):
    eff   = min(_GAP_PCT_CAP, abs(float(delta_pct)))
    score = round(eff * _SEV_SCORE.get(severity, 1.0) / _SCORE_SCALE, 1)
    return {
        "severity":    severity,
        "category":    category,
        "score":       score,
        "title":       title,
        "finding":     finding,
        "explanation": explanation,
        "action":      action,
        "metric":      row["METRIC"],
        "country":     row["COUNTRY"],
        "city":        row["CITY"],
        "zone":        row["ZONE"],
        "value":       round(float(row.get("L0W_ROLL", 0)), 4),
        "delta_pct":   round(max(-_GAP_PCT_CAP, min(_GAP_PCT_CAP, float(delta_pct))), 1),
    }


def _free_insight(severity, title, finding, explanation, action,
                  metric, country, city, zone, value, delta_pct,
                  category: str = "opportunity"):
    eff   = min(_GAP_PCT_CAP, abs(float(delta_pct)))
    score = round(eff * _SEV_SCORE.get(severity, 1.0) / _SCORE_SCALE, 1)
    return {
        "severity":    severity,
        "category":    category,
        "score":       score,
        "title":       title,
        "finding":     finding,
        "explanation": explanation,
        "action":      action,
        "metric":      metric,
        "country":     country,
        "city":        city,
        "zone":        zone,
        "value":       round(float(value), 4),
        "delta_pct":   round(max(-_GAP_PCT_CAP, min(_GAP_PCT_CAP, float(delta_pct))), 1),
    }


# ── Detector 1 · WoW anomalies (>10 % deterioration / improvement) ─────────────

def detect_wow_changes(df: pd.DataFrame, limit: int = 10) -> list[dict]:
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
    for _, r in work[work["rel"] <= _WARNING_DECLINE].sort_values("rel").head(limit).iterrows():
        sev  = "critical" if r["rel"] <= _CRITICAL_DECLINE else "warning"
        drop = abs(r["rel"])
        _drop_cause = _METRIC_DROP_CAUSE.get(
            r["METRIC"],
            "puede indicar problemas operativos o de demanda que requieren investigacion en campo"
        )
        insights.append(_insight(
            sev,
            f"Caida de metrica - {r['ZONE']}",
            f"{r['METRIC']} cayo {drop:.1f}% SaS ({r['L1W_ROLL']:.3f} -> {r['L0W_ROLL']:.3f})",
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) registro una caida "
             f"{'severa' if sev == 'critical' else 'significativa'} en {r['METRIC']}. "
             f"{_drop_cause.capitalize()}."),
            ("Revisar los factores operativos asociados a este indicador en la zona. "
             "Contrastar con semanas previas y comparar con zonas de perfil similar."),
            r, r["rel"], category="anomaly",
        ))

    # Improvements — anything better than +10 %
    for _, r in work[work["rel"] >= _POSITIVE_GAIN].sort_values("rel", ascending=False).head(max(1, limit * 4 // 5)).iterrows():
        insights.append(_insight(
            "positive",
            f"Mejora notable - {r['ZONE']}",
            f"{r['METRIC']} mejoro {r['rel']:.1f}% SaS ({r['L1W_ROLL']:.3f} -> {r['L0W_ROLL']:.3f})",
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) mostro un cambio positivo "
             f"destacado en {r['METRIC']} esta semana. Puede reflejar intervenciones operativas "
             f"recientes, mejores condiciones de oferta o activaciones comerciales exitosas."),
            ("Documentar que cambio en esta zona y replicar el enfoque "
             "en zonas con bajo rendimiento y perfil similar."),
            r, r["rel"], category="anomaly",
        ))

    return insights


# ── Detector 2 · Worrying trends (3+ consecutive weeks of decline) ─────────────

def detect_consecutive_decline(df: pd.DataFrame, limit: int = 8) -> list[dict]:
    """Zones with 3 consecutive weeks of decline above a minimum magnitude."""
    if not _needs(df, {"L2W_ROLL", "L1W_ROLL", "L0W_ROLL", "METRIC", "ZONE", "COUNTRY", "CITY"}):
        return []

    work = df[["COUNTRY", "CITY", "ZONE", "METRIC", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]].dropna().copy()
    work = work[(work["L2W_ROLL"] > work["L1W_ROLL"]) & (work["L1W_ROLL"] > work["L0W_ROLL"])]
    work["total_pct"] = (work["L0W_ROLL"] - work["L2W_ROLL"]) / work["L2W_ROLL"].abs() * 100
    work = work[work["total_pct"] <= _CONSEC_MIN_DROP].sort_values("total_pct").head(limit)

    insights = []
    for _, r in work.iterrows():
        sev = "critical" if r["total_pct"] <= -12 else "warning"
        _ctx = _METRIC_CONTEXT.get(r["METRIC"], "")
        _ctx_note = f" ({_ctx})" if _ctx else ""
        _drop_cause = _METRIC_DROP_CAUSE.get(r["METRIC"], "")
        _cause_note = f" {_drop_cause.capitalize()}." if _drop_cause else ""
        insights.append(_insight(
            sev,
            f"Tendencia bajista 3 semanas - {r['ZONE']}",
            (f"{r['METRIC']}: {r['L2W_ROLL']:.3f} -> {r['L1W_ROLL']:.3f} -> {r['L0W_ROLL']:.3f} "
             f"(-{abs(r['total_pct']):.1f}% en 3 semanas)"),
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) ha declinado cada semana durante "
             f"3 semanas consecutivas en {r['METRIC']}{_ctx_note}. Las tendencias sostenidas "
             f"indican una causa sistematica, no un evento puntual.{_cause_note}"),
            ("Escalar al lider de operaciones para analisis de causa raiz. "
             "Revisar la evolucion de este indicador con datos de campo del mismo periodo."),
            r, r["total_pct"], category="trend",
        ))

    return insights


# ── Detector 3 · Sustained improvement (3+ consecutive weeks up) ───────────────

def detect_sustained_improvement(df: pd.DataFrame, limit: int = 8) -> list[dict]:
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
    ).head(limit)

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
            r, r["total_pct"], category="trend",
        ))

    return insights


# ── Detector 4 · Within-segment benchmarking ───────────────────────────────────

def detect_zone_type_benchmarking(df: pd.DataFrame, limit: int = 10) -> list[dict]:
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

    laggards = merged[merged["z"] <= _BENCH_Z_THRESHOLD].nsmallest(limit, "z")

    insights = []
    for _, r in laggards.iterrows():
        gap_pct  = _safe_gap_pct(r["L0W_ROLL"], r["mean"])
        gap_desc = _gap_label(gap_pct, r["z"])
        insights.append(_free_insight(
            "opportunity",
            f"Rezagado en segmento {r['ZONE_TYPE']} - {r['ZONE']}",
            (f"{r['METRIC']}: {r['L0W_ROLL']:.3f} vs promedio {r['ZONE_TYPE']} "
             f"en {r['COUNTRY']}: {r['mean']:.3f} — {gap_desc}"),
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) tiene bajo rendimiento "
             f"en {r['METRIC']} comparada con sus pares del segmento {r['ZONE_TYPE']} en el "
             f"mismo pais. Este benchmark entre pares es la comparacion mas justa disponible "
             f"porque elimina diferencias estructurales entre tipos de zona."),
            (f"Analizar las mejores zonas {r['ZONE_TYPE']} del mismo pais para identificar "
             "practicas replicables. Priorizar si la zona es de alta priorizacion operativa."),
            r["METRIC"], r["COUNTRY"], r["CITY"], r["ZONE"],
            r["L0W_ROLL"], -gap_pct, category="benchmark",
        ))

    return insights


# ── Detector 5 · Cross-metric correlations ─────────────────────────────────────

def detect_metric_correlations(df: pd.DataFrame, limit: int = 4) -> list[dict]:
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

    # Operationally meaningful divergence pairs — high metric_a, low metric_b
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
        (
            "Lead Penetration", "Non-Pro PTC > OP",
            "Alta captacion de leads pero baja conversion carrito a pedido",
            (
                "La zona tiene buena penetracion de leads pero los usuarios no suscritos "
                "no convierten desde el carrito al pedido. Los leads captados se pierden "
                "en el checkout — posible friccion de precio, experiencia o seleccion de tiendas."
            ),
            (
                "Revisar el flujo de checkout para usuarios no-Pro en esta zona. "
                "Evaluar incentivos de primer pedido y reduccion de friccion en el checkout."
            ),
        ),
        (
            "Pro Adoption (Last Week Status)", "% PRO Users Who Breakeven",
            "Alta adopcion Pro pero baja recuperacion de inversion del suscriptor",
            (
                "La zona tiene buena adopcion del programa Pro pero sus suscriptores "
                "no estan recuperando la inversion. Indica que la frecuencia de pedidos "
                "Pro no es suficiente para justificar la suscripcion — riesgo de churn del programa."
            ),
            (
                "Activar campana de engagement para usuarios Pro con baja frecuencia en esta zona. "
                "Revisar si la oferta de restaurantes satisface las preferencias de los suscriptores."
            ),
        ),
        (
            "Restaurants SST > SS CVR", "Restaurants SS > ATC CVR",
            "Alta intencion de busqueda pero baja conversion a carrito en restaurantes",
            (
                "Los usuarios entran al flujo de restaurantes e inician busquedas, pero no "
                "agregan productos al carrito. El surtido disponible no satisface la demanda "
                "expresada — brecha entre lo que el usuario busca y lo que encuentra."
            ),
            (
                "Analizar los terminos de busqueda con mayor volumen y sin conversion en esta zona. "
                "Activar restaurantes con el surtido faltante o mejorar la relevancia del search."
            ),
        ),
        (
            "Turbo Adoption", "Gross Profit UE",
            "Alta adopcion Turbo pero baja rentabilidad por UE",
            (
                "La zona tiene alta adopcion del servicio Turbo pero la rentabilidad por "
                "unidad economica esta por debajo del promedio. El volumen incremental de "
                "pedidos Turbo no se esta traduciendo en margen — puede reflejar costos "
                "operativos elevados del servicio express o un mix de pedidos de ticket bajo."
            ),
            (
                "Revisar la estructura de costos de los pedidos Turbo en esta zona: "
                "distancias de entrega, tiempos operativos y mix de restaurantes participantes. "
                "Evaluar si el subsidio actual del servicio Turbo esta justificado por el LTV "
                "de los usuarios captados."
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
        divergent = sub[(sub["z_a"] > 0.6) & (sub["z_b"] < -0.6)].nsmallest(limit, "z_b")

        for (country, city, zone), row in divergent.iterrows():
            val_a    = row[metric_a]
            val_b    = row[metric_b]
            z_b      = row["z_b"]
            gap      = _safe_gap_pct(val_b, mean_b)
            gap_desc = _gap_label(gap, z_b)
            insights.append(_free_insight(
                "opportunity",
                f"{label} - {zone}",
                (f"{metric_a}: {val_a:.3f} (alto) | {metric_b}: {val_b:.3f} "
                 f"vs prom {mean_b:.3f} — {gap_desc}"),
                explanation,
                action,
                f"{metric_a} / {metric_b}", country, city, zone,
                val_a, -gap, category="correlation",
            ))

    return insights


# ── Detector 6 · Opportunity zones (below country average) ────────────────────

def detect_opportunity_zones(df: pd.DataFrame, limit: int = 12) -> list[dict]:
    """
    Zones significantly below the country average — weighted by zone prioritization.
    High-priority zones qualify at a lower z-score threshold and receive a score boost,
    ensuring they surface even when the absolute gap is smaller than in non-priority zones.
    """
    if not _needs(df, {"L0W_ROLL", "METRIC", "ZONE", "COUNTRY", "CITY"}):
        return []

    has_prio = "ZONE_PRIORITIZATION" in df.columns
    base_cols = ["COUNTRY", "CITY", "ZONE", "METRIC", "L0W_ROLL"]
    extra     = ["ZONE_PRIORITIZATION"] if has_prio else []
    work = df[base_cols + extra].dropna(subset=base_cols)

    stats  = work.groupby(["COUNTRY", "METRIC"])["L0W_ROLL"].agg(["mean", "std"]).reset_index()
    merged = work.merge(stats, on=["COUNTRY", "METRIC"])
    merged = merged[merged["std"] > 0.001].copy()
    merged["z"]   = (merged["L0W_ROLL"] - merged["mean"]) / merged["std"]
    merged["_pw"] = merged["ZONE_PRIORITIZATION"].apply(_priority_weight) if has_prio else 1.0

    # High-priority zones qualify at a lower threshold; sort by z / priority_weight
    if has_prio:
        mask = (merged["z"] <= _OPP_Z_THRESHOLD) | (
            (merged["_pw"] >= 2.0) & (merged["z"] <= _OPP_Z_PRIO_THRESHOLD)
        )
    else:
        mask = merged["z"] <= _OPP_Z_THRESHOLD

    merged["_sort_key"] = merged["z"] / merged["_pw"]
    laggards = merged[mask].sort_values("_sort_key").head(limit)

    insights = []
    for _, r in laggards.iterrows():
        gap_pct    = _safe_gap_pct(r["L0W_ROLL"], r["mean"])
        gap_desc   = _gap_label(gap_pct, r["z"])
        prio_label = str(r["ZONE_PRIORITIZATION"]) if has_prio and pd.notna(r.get("ZONE_PRIORITIZATION")) else ""
        prio_note  = f" — zona {prio_label}" if prio_label else ""
        metric_ctx = _METRIC_CONTEXT.get(r["METRIC"], "")
        ctx_note   = f" ({metric_ctx})" if metric_ctx else ""

        ins = _insight(
            "opportunity",
            f"Zona de oportunidad - {r['ZONE']}",
            (f"{r['METRIC']}: {r['L0W_ROLL']:.3f} vs promedio pais {r['mean']:.3f} "
             f"— {gap_desc}{prio_note}"),
            (f"La zona {r['ZONE']} ({r['CITY']}, {r['COUNTRY']}) es un rezagado significativo "
             f"en {r['METRIC']}{ctx_note}. Cerrar esta brecha representa potencial de GMV medible."),
            ("Comparar con las mejores zonas de perfil similar en el pais. "
             "Considerar activacion focalizada o incentivo comercial en esta zona."),
            r, -gap_pct, category="opportunity",
        )
        ins["score"] = round(ins["score"] * float(r["_pw"]), 1)
        insights.append(ins)

    return insights


# ── Detector 7 · High-priority zone risk ──────────────────────────────────────

def detect_priority_zone_risk(df: pd.DataFrame, limit: int = 5) -> list[dict]:
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
    declining = work[work["rel"] <= _PRIO_DECLINE].sort_values("rel").head(limit)

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
            r, r["rel"], category="anomaly",
        ))

    return insights


# ── Detector 8 · Country-level benchmarking ───────────────────────────────────

def detect_country_benchmarking(df: pd.DataFrame, limit: int = 8) -> list[dict]:
    """
    Compare each country's metric average against the LATAM network average.
    Surfaces countries with meaningfully divergent performance across metrics.
    """
    if not _needs(df, {"L0W_ROLL", "METRIC", "COUNTRY", "CITY", "ZONE"}):
        return []

    work = df[["COUNTRY", "CITY", "ZONE", "METRIC", "L0W_ROLL"]].dropna()

    country_avg = (
        work.groupby(["COUNTRY", "METRIC"])["L0W_ROLL"]
        .mean()
        .reset_index()
        .rename(columns={"L0W_ROLL": "country_avg"})
    )

    net_stats = (
        country_avg.groupby("METRIC")["country_avg"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    net_stats = net_stats[(net_stats["std"] > 0.003) & (net_stats["count"] >= 3)]

    merged = country_avg.merge(net_stats, on="METRIC")
    merged["z"] = (merged["country_avg"] - merged["mean"]) / merged["std"]

    laggards = merged[merged["z"] <= _BENCH_COUNTRY_Z].nsmallest(limit, "z")

    insights = []
    for _, r in laggards.iterrows():
        gap_pct  = _safe_gap_pct(r["country_avg"], r["mean"])
        gap_desc = _gap_label(gap_pct, r["z"])
        best = merged[merged["METRIC"] == r["METRIC"]].nlargest(1, "country_avg")
        best_country = best.iloc[0]["COUNTRY"] if not best.empty else "N/A"
        best_val     = best.iloc[0]["country_avg"] if not best.empty else r["mean"]

        rep = work[(work["COUNTRY"] == r["COUNTRY"]) & (work["METRIC"] == r["METRIC"])]
        if rep.empty:
            continue
        rep_row = rep.iloc[0]

        insights.append(_free_insight(
            "opportunity",
            f"Pais rezagado en benchmark — {r['COUNTRY']}",
            (f"{r['METRIC']}: {r['COUNTRY']} promedia {r['country_avg']:.3f} "
             f"vs red LATAM {r['mean']:.3f} — {gap_desc}. "
             f"Lider: {best_country} ({best_val:.3f})"),
            (f"{r['COUNTRY']} tiene rendimiento por debajo del promedio LATAM en {r['METRIC']}. "
             f"La diferencia respecto al lider {best_country} ({best_val:.3f}) requiere "
             f"analisis de causas raiz — puede reflejar diferencias en oferta, demanda local "
             f"o ejecucion operativa que son gestionables."),
            (f"Revisar las mejores practicas de {best_country} en {r['METRIC']} y evaluar "
             f"su aplicabilidad en el contexto operativo de {r['COUNTRY']}. "
             f"Priorizar zonas con mayor potencial de mejora rapida dentro del pais."),
            r["METRIC"], r["COUNTRY"],
            rep_row.get("CITY", ""), rep_row.get("ZONE", ""),
            r["country_avg"], -gap_pct,
            category="benchmark",
        ))

    return insights


# ── Aggregator ─────────────────────────────────────────────────────────────────

def generate_insights(
    df: pd.DataFrame,
    country: str | None = None,
    metric: str | None = None,
    max_insights: int | None = None,
) -> list[dict]:
    """
    Run all detectors. Per-detector limits and output caps scale with the number
    of countries in scope so that LATAM-wide produces substantially more insights
    than a single-country view.
      scale factor : 1 for ≤3 countries, 2 for 4-6, 3 for 7-9
      max insights : max(40, n_countries × 10)
    """
    if "L0W_ROLL" not in df.columns:
        return []

    country_df = df[df["COUNTRY"] == country] if country else df
    filtered   = country_df[country_df["METRIC"] == metric] if metric else country_df

    # Scale factor — 1 for single-country scope, up to 3 for full LATAM
    _n_countries = max(1, min(int(country_df["COUNTRY"].nunique()), 9))
    _f           = (_n_countries + 2) // 3   # 1 for ≤3, 2 for 4-6, 3 for 7-9
    _max         = max_insights or max(40, _n_countries * 10)

    all_insights = (
        detect_wow_changes(filtered,               limit=10 * _f)
        + detect_consecutive_decline(filtered,     limit=8  * _f)
        + detect_sustained_improvement(filtered,   limit=8  * _f)
        + detect_opportunity_zones(filtered,       limit=12 * _f)
        + detect_priority_zone_risk(filtered,      limit=5  * _f)
        + detect_zone_type_benchmarking(country_df, limit=10 * _f)
        + detect_country_benchmarking(country_df,   limit=8  * _f)
        + detect_metric_correlations(country_df,    limit=4)        # fixed: 4 per pair × 8 pairs is enough
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

    # Sort by severity, then by absolute delta magnitude within each severity group
    from itertools import groupby
    deduped.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"], 9))
    sorted_groups: list[dict] = []
    for _, group in groupby(deduped, key=lambda x: x["severity"]):
        sorted_groups.extend(sorted(group, key=lambda x: abs(x.get("delta_pct", 0)), reverse=True))

    # Per-category caps — scale with same factor
    _CAPS = {
        "critical":    15 * _f,
        "warning":     10 * _f,
        "opportunity": 12 * _f,
        "positive":    10 * _f,
    }
    _counts: dict[str, int] = {k: 0 for k in _CAPS}
    balanced: list[dict] = []
    for ins in sorted_groups:
        sev = ins["severity"]
        if _counts.get(sev, 0) < _CAPS.get(sev, 99):
            balanced.append(ins)
            _counts[sev] = _counts.get(sev, 0) + 1
    return balanced[:_max]


# ── Executive prioritization ───────────────────────────────────────────────────

# Category bonus applied at display-time: persistent / cross-metric patterns
# carry more strategic weight than isolated single-week anomalies.
_EXEC_CAT_BONUS: dict[str, float] = {
    "trend":       1.40,   # 3-week persistence = systemic cause, highest executive signal
    "correlation": 1.30,   # cross-metric = higher information density, harder to spot manually
    "benchmark":   1.10,   # structural gap vs peers = structural, not random
    "anomaly":     1.00,   # baseline — common, useful but not uniquely insightful
    "opportunity": 0.95,
}

# Diminishing returns for the same metric appearing multiple times.
# First occurrence = full adjusted score; second = 60%; third+ = 30%.
# Prevents one bad metric from filling every executive slot.
_EXEC_REP_FACTOR: list[float] = [1.0, 0.60, 0.30]


def _exec_score(ins: dict, metric_counts: dict[str, int]) -> float:
    base = ins.get("score", 0.0)
    rep  = min(metric_counts.get(ins["metric"], 0), 2)
    return base * _EXEC_CAT_BONUS.get(ins.get("category", "anomaly"), 1.0) * _EXEC_REP_FACTOR[rep]


def prioritize_insights(insights: list[dict], top_n: int = 10) -> list[dict]:
    """
    Select top_n insights for executive display from a larger generated pool.

    Algorithm:
      Pass 1 — diversity guarantee: take the highest-scoring insight from each
               populated category (anomaly → trend → correlation → benchmark →
               opportunity), in that priority order, up to top_n slots.
      Pass 2 — greedy fill: remaining slots go to the highest adjusted-score
               candidate not already seen. The adjusted score applies the category
               bonus and a per-metric diminishing-returns penalty so no single
               metric can dominate the executive view.
      Final  — re-sort by severity then score for presentation order.
    """
    if not insights:
        return []

    pool: list[dict]        = list(insights)
    selected: list[dict]    = []
    seen_keys: set[tuple]   = set()
    metric_counts: dict[str, int] = {}

    def _best_from(candidates: list[dict]) -> "dict | None":
        eligible = [i for i in candidates if (i["zone"], i["metric"]) not in seen_keys]
        return max(eligible, key=lambda x: _exec_score(x, metric_counts)) if eligible else None

    def _commit(ins: dict) -> None:
        selected.append(ins)
        if ins in pool:
            pool.remove(ins)
        seen_keys.add((ins["zone"], ins["metric"]))
        metric_counts[ins["metric"]] = metric_counts.get(ins["metric"], 0) + 1

    # Pass 1: one best insight per populated category (fixed diversity order)
    populated = {i.get("category") for i in insights} & CATEGORY_CONFIG.keys()
    for cat in ("anomaly", "trend", "correlation", "benchmark", "opportunity"):
        if cat not in populated or len(selected) >= top_n:
            continue
        best = _best_from([i for i in pool if i.get("category") == cat])
        if best:
            _commit(best)

    # Pass 2: greedy fill by adjusted score, no duplicate (zone, metric) keys
    while len(selected) < top_n:
        best = _best_from(pool)
        if best is None:
            break
        _commit(best)

    # Re-sort for presentation: severity first, then score descending
    selected.sort(key=lambda x: (SEVERITY_ORDER.get(x["severity"], 9), -x.get("score", 0)))
    return selected


# ── Summary counts ─────────────────────────────────────────────────────────────

def count_by_severity(insights: list[dict]) -> dict[str, int]:
    counts = {s: 0 for s in SEVERITY_CONFIG}
    for i in insights:
        counts[i["severity"]] = counts.get(i["severity"], 0) + 1
    return counts


def count_by_category(insights: list[dict]) -> dict[str, int]:
    counts = {c: 0 for c in CATEGORY_CONFIG}
    for i in insights:
        cat = i.get("category", "anomaly")
        counts[cat] = counts.get(cat, 0) + 1
    return counts
