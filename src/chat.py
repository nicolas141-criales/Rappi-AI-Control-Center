import json
import re
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import (
    WEEK_COLS_METRICS,
    WEEK_COLS_ORDERS,
    WEEK_LABELS,
    get_country_averages,
    get_dataset_summary,
    get_metric_by_zone_type,
    get_orders_by_country,
    get_orders_trend,
    get_top_zones,
    get_weekly_trend,
    get_wow_zones,
)

if TYPE_CHECKING:
    from src.providers.base import LLMProvider

MAX_TOKENS    = 2500
HISTORY_PAIRS = 6

_TEMPORAL_RE = re.compile(
    r"evoluci[oó]n|tendencia|[uú]ltimas?\s*\d*\s*semanas?|hist[oó]rico|"
    r"over\s+time|trend|c[oó]mo\s+ha|ha\s+cambiado|semanal|por\s+semana|"
    r"evolucion|progres|variaci[oó]n|a\s+lo\s+largo",
    re.IGNORECASE,
)

# ── Extended analytical intent patterns ───────────────────────────────────────

_GROWTH_RE = re.compile(
    r"mayor\s+crecimiento|m[aá]s\s+creci(?:eron|[oó]|do|miento)|"
    r"top\s+crecimiento|fastest\s+growing|growth\s+ranking|"
    r"crecimiento\s+(?:de\s+)?(?:[oó]rdenes?|pedidos?|zonas?)|"
    r"(?:[oó]rdenes?|pedidos?)\s+(?:con\s+)?(?:m[aá]s\s+)?crecimiento|"
    r"(?:zonas?|pa[íi]ses?)\s+(?:que\s+)?(?:m[aá]s\s+)?creci|"
    r"que\s+(?:m[aá]s\s+)?creci(?:eron|[oó])|"
    r"\bcrec(?:ieron|i[oó])\b",
    re.IGNORECASE,
)

_DECLINE_RE = re.compile(
    r"mayor\s+(?:ca[íi]da|deterioro|decrecimiento)|"
    r"m[aá]s\s+(?:cayeron|bajaron|deterioraron)|"
    r"peor\s+(?:rendimiento|desempe[nñ]o)|"
    r"(?:zonas?|pa[íi]ses?)\s+(?:que\s+)?(?:m[aá]s\s+)?cay|"
    r"decline\s+ranking|worst\s+performing|declive|"
    r"\b(?:cayeron|cay[oó])\b",
    re.IGNORECASE,
)

_RANKING_RE = re.compile(
    r"\btop\s*\d+\b|\branking\b|"
    r"mejores\s+zonas?|peores\s+zonas?|"
    r"(?:las?\s+)?mejores?\s+(?:zonas?|pa[íi]ses?)|"
    r"(?:las?\s+)?peores?\s+(?:zonas?|pa[íi]ses?)|"
    r"zonas?\s+con\s+(?:mayor|menor|m[aá]s\s+alto|m[aá]s\s+bajo)\b|"
    r"cu[aá]les?\s+(?:son\s+las?\s+)?(?:\d+\s+)?(?:mejores?|peores?|principales?)",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\bcompar[ae]\b|\bversus\b|\bvs\.?\b|"
    r"diferencia\s+entre|contraste\s+entre|"
    r"c[oó]mo\s+(?:se\s+)?compara|mejor\s+que|peor\s+que",
    re.IGNORECASE,
)

_OPPORTUNITY_RE = re.compile(
    r"\boportunidad(?:es)?\b|bajo\s+rendimiento|"
    r"debajo\s+del?\s+promedio|rezagad[ao]s?|"
    r"por\s+mejorar|potencial\s+de\s+mejora",
    re.IGNORECASE,
)

_BOTTOM_RE = re.compile(
    r"\bpeores?\b|\bmenores?\b|\bbottom\b|\bm[aá]s\s+bajos?\b",
    re.IGNORECASE,
)

_ORDERS_RE = re.compile(
    r"\b(?:[oó]rdenes?|pedidos?|orders?)\b",
    re.IGNORECASE,
)

_ZONE_TYPE_RE = re.compile(
    r"\b(?:wealthy|non.?wealthy|zona\s+(?:rica|no\s+rica)|"
    r"tipo\s+de\s+zona|por\s+tipo\s+de\s+zona|segmento\s+de\s+zona|"
    r"zonas?\s+ricas?|zonas?\s+no\s+ricas?)\b",
    re.IGNORECASE,
)

# ── Prioritization filter detection ───────────────────────────────────────────
_PRIO_HIGH_RE = re.compile(
    r"\b(?:high\s*priority|alta\s+prioridad|alta\s+priorizaci[oó]n|prioridad\s+alta)\b",
    re.IGNORECASE,
)
_PRIO_MED_RE = re.compile(
    r"\b(?:medium\s*priority|media\s+prioridad|prioridad\s+media)\b",
    re.IGNORECASE,
)
_PRIO_LOW_RE = re.compile(
    r"\b(?:low\s*priority|baja\s+prioridad|prioridad\s+baja)\b",
    re.IGNORECASE,
)
_PRIO_ANY_RE = re.compile(
    r"\bzonas?\s+prioritarias?\b|\bzonas?\s+priorizadas?\b",
    re.IGNORECASE,
)

_TOP_N_RE = re.compile(
    r"\btop\s*(\d+)\b|(?:las?|los?)\s+(\d+)\s+(?:mejores?|peores?|zonas?|pa[íi]ses?|principales?)",
    re.IGNORECASE,
)

_N_WEEKS_RE = re.compile(
    r"[úu]ltimas?\s+(\d+)\s+semanas?|(\d+)\s+semanas?\s+(?:atr[aá]s?|anteriores?)",
    re.IGNORECASE,
)

# Words that appear frequently in analytics questions but never in zone/city names
_ENTITY_STOP_WORDS = frozenset([
    "pedidos", "ordenes", "metricas", "semanas", "semana", "evolucion",
    "tendencia", "historico", "historica", "promedio", "calidad", "perfect",
    "orders", "turbo", "retail", "gross", "profit", "adoption", "penetration",
    "conversion", "restaurants", "markdowns", "sessions", "optimal",
    "assortment", "breakeven", "ultimas", "ultima", "rappi", "latam",
    "colombia", "argentina", "brasil", "brazil", "chile", "ecuador",
    "mexico", "uruguay", "weekly", "zonas", "paises", "ciudades",
])


def _extract_top_n(question: str, default: int = 10) -> int:
    m = _TOP_N_RE.search(question)
    if m:
        val = m.group(1) or m.group(2)
        if val:
            return min(int(val), 20)
    return default


def _extract_n_weeks(question: str, default: int = 4) -> int:
    # "esta semana" / "SaS" / "semana a semana" → W-1 → W-0 (single-week comparison)
    if re.search(r"\besta\s+semana\b|\bsemana\s+a\s+semana\b|\bSaS\b", question, re.IGNORECASE):
        return 1
    m = _N_WEEKS_RE.search(question)
    if m:
        val = m.group(1) or m.group(2)
        if val:
            return min(int(val), 8)
    return default


# ── System prompt ─────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    p = Path("prompts/analytics_system.txt")
    return p.read_text(encoding="utf-8") if p.exists() else (
        "Eres un copiloto de analitica de operaciones para Rappi. "
        "Responde solo con los datos del [CONTEXT] proporcionado. Nunca inventes datos."
    )


# ── Context builder ───────────────────────────────────────────────────────────

def _fmt_country_row(row) -> str:
    return f"{row.COUNTRY}: {row.avg_value:.3f}"


# ── Intent detection ───────────────────────────────────────────────────────────

def detect_question_intent(question: str, metrics_df: pd.DataFrame) -> dict:
    """
    Parse a natural-language question and return detected temporal intent
    plus the best-matching metric, zone, city, and country.
    Matching uses longest-first substring to prefer specific names over partial ones.
    """
    q_low = question.lower()

    metrics_list = metrics_df["METRIC"].dropna().unique().tolist()
    countries    = metrics_df["COUNTRY"].dropna().unique().tolist()
    cities       = metrics_df["CITY"].dropna().unique().tolist()
    zones        = metrics_df["ZONE"].dropna().unique().tolist()

    temporal = bool(_TEMPORAL_RE.search(question))

    # Metric — longest match first (exact substring)
    matched_metric = None
    for m in sorted(metrics_list, key=len, reverse=True):
        if m.lower() in q_low:
            matched_metric = m
            break

    # Fuzzy fallback: word-stem overlap for pluralization / partial names
    # e.g. "Perfect Order" → matches "Perfect Orders", "Adoption" → "Turbo Adoption"
    if not matched_metric:
        def _wstem(w: str) -> str:
            for sfx in ("tion", "sion", "ing", "ers", "es", "s"):
                if w.endswith(sfx) and len(w) > len(sfx) + 3:
                    return w[: -len(sfx)]
            return w
        q_stems = {_wstem(w) for w in re.findall(r"\b\w{4,}\b", q_low)}
        for m in sorted(metrics_list, key=len, reverse=True):
            m_stems = {_wstem(w) for w in re.findall(r"\b\w{4,}\b", m.lower())}
            # Require ≥2 distinctive stems to avoid false positives on short/abbrev names
            if len(m_stems) >= 2 and m_stems <= q_stems:
                matched_metric = m
                break

    # Country — 2-letter code or full name
    _COUNTRY_NAMES = {
        "argentina": "AR", "brasil": "BR", "brazil": "BR",
        "chile": "CL", "colombia": "CO", "costa rica": "CR",
        "ecuador": "EC", "mexico": "MX", "méxico": "MX",
        "peru": "PE", "perú": "PE", "uruguay": "UY",
    }
    matched_country = None
    for name, code in _COUNTRY_NAMES.items():
        if name in q_low:
            matched_country = code
            break
    if not matched_country:
        for c in sorted(countries, key=len, reverse=True):
            if re.search(r'\b' + re.escape(c.lower()) + r'\b', q_low):
                matched_country = c
                break

    # City — longest match first, min 4 chars to avoid noise
    matched_city = None
    for city in sorted(cities, key=len, reverse=True):
        if len(city) >= 4 and city.lower() in q_low:
            matched_city = city
            break
    # Reverse match: question word contained in city name (e.g. "medellin" ⊂ "MEDELLIN NORTE")
    if not matched_city:
        _q_words = sorted(re.findall(r'\b\w{5,}\b', q_low), key=len, reverse=True)
        for _w in _q_words:
            if _w in _ENTITY_STOP_WORDS:
                continue
            for city in sorted(cities, key=len, reverse=True):
                if _w in city.lower():
                    matched_city = city
                    break
            if matched_city:
                break

    # Zone — longest match first, min 4 chars
    matched_zone = None
    for zone in sorted(zones, key=len, reverse=True):
        if len(zone) >= 4 and zone.lower() in q_low:
            matched_zone = zone
            break
    # Reverse match: question word contained in zone name (e.g. "maschwitz" ⊂ "GRAL MASCHWITZ")
    if not matched_zone:
        _q_words = sorted(re.findall(r'\b\w{5,}\b', q_low), key=len, reverse=True)
        for _w in _q_words:
            if _w in _ENTITY_STOP_WORDS:
                continue
            for zone in sorted(zones, key=len, reverse=True):
                if _w in zone.lower() and len(zone) >= 4:
                    matched_zone = zone
                    break
            if matched_zone:
                break

    # ── Intent type classification ────────────────────────────────────────────
    intent_types: list[str] = []
    if _GROWTH_RE.search(question):
        intent_types.append("growth_ranking")
    if _DECLINE_RE.search(question):
        intent_types.append("decline_ranking")
    if _RANKING_RE.search(question):
        intent_types.append("ranking")
    if _COMPARISON_RE.search(question):
        intent_types.append("comparison")
    if _OPPORTUNITY_RE.search(question):
        intent_types.append("opportunity")
    if _ZONE_TYPE_RE.search(question):
        intent_types.append("zone_type_comparison")
    if temporal:
        intent_types.append("temporal")

    # Prioritization filter — checked in specificity order: high > medium > low > generic
    prioritization: str | None = None
    if _PRIO_HIGH_RE.search(question):
        prioritization = "High Priority"
    elif _PRIO_MED_RE.search(question):
        prioritization = "Prioritized"
    elif _PRIO_LOW_RE.search(question):
        prioritization = "Not Prioritized"
    elif _PRIO_ANY_RE.search(question):
        prioritization = "High Priority"   # "zonas prioritarias" defaults to high

    return {
        "temporal":       temporal,
        "intent_types":   intent_types,
        "metric":         matched_metric,
        "country":        matched_country,
        "city":           matched_city,
        "zone":           matched_zone,
        "prioritization": prioritization,
    }


# ── Entity-specific historical context ────────────────────────────────────────

def build_entity_context(metrics_df: pd.DataFrame, intent: dict) -> str:
    """
    Build a detailed historical section injected when temporal or entity-specific
    intent is detected. Returns an empty string when nothing relevant was found.
    """
    temporal = intent["temporal"]
    metric   = intent["metric"]
    zone     = intent["zone"]
    city     = intent["city"]
    country  = intent["country"]

    # Only inject when there is something specific to look up
    if not (temporal or zone or city) and not metric:
        return ""

    cols   = WEEK_COLS_METRICS   # L8W_ROLL … L0W_ROLL
    labels = WEEK_LABELS          # W-8 … W-0

    parts: list[str] = ["", "=== DATOS HISTORICOS DETALLADOS ==="]

    # ── Zone-level series ─────────────────────────────────────────────────────
    if zone:
        zone_mask = metrics_df["ZONE"].str.lower() == zone.lower()
        if not zone_mask.any():
            zone_mask = metrics_df["ZONE"].str.lower().str.contains(
                re.escape(zone.lower()), na=False
            )
        zone_df = metrics_df[zone_mask]

        if zone_df.empty:
            parts.append(
                f"\nNota: Zona '{zone}' no encontrada en el dataset. "
                "Verifica el nombre exacto (puede diferir en mayúsculas o abreviatura)."
            )
        else:
            # Availability scan: which metrics have non-null weekly series
            _avail: list[str] = []
            for _m in metrics_df["METRIC"].dropna().unique():
                _sub = zone_df[zone_df["METRIC"] == _m]
                if not _sub.empty:
                    _row = _sub.iloc[0]
                    if any(c in _row.index and pd.notna(_row[c]) for c in cols):
                        _avail.append(_m)
            _r0 = zone_df.iloc[0]
            parts.append(
                f"\nZona detectada: {_r0['ZONE']}"
                f" ({_r0.get('CITY', '')}, {_r0.get('COUNTRY', '')})"
            )
            parts.append(
                f"Métricas con datos históricos disponibles ({len(_avail)}): "
                + (", ".join(_avail) if _avail else "ninguna")
            )

        # Show series — all zone-available metrics when no specific metric requested
        _mshow = [metric] if metric else (
            zone_df["METRIC"].dropna().unique().tolist() if not zone_df.empty else []
        )
        for m in _mshow:
            sub = zone_df[zone_df["METRIC"] == m]
            if sub.empty:
                continue
            row = sub.iloc[0]
            week_vals = [
                (c, lbl) for c, lbl in zip(cols, labels)
                if c in row.index and pd.notna(row[c])
            ]
            if not week_vals:
                continue
            parts.append(
                f"\n{m} — Zona: {row['ZONE']} ({row['CITY']}, {row['COUNTRY']})"
            )
            for col, lbl in week_vals:
                parts.append(f"  {lbl}: {float(row[col]):.4f}")

        if len(parts) > 2 and country is None and city is None:
            return "\n".join(parts)

    # ── City-level series ─────────────────────────────────────────────────────
    if city:
        city_mask = metrics_df["CITY"].str.lower() == city.lower()
        if not city_mask.any():
            city_mask = metrics_df["CITY"].str.lower().str.contains(
                re.escape(city.lower()), na=False
            )
        city_df = metrics_df[city_mask]

        if city_df.empty:
            parts.append(
                f"\nNota: Ciudad '{city}' no encontrada en el dataset. "
                "Verifica el nombre exacto."
            )
        else:
            # Availability scan
            _avail_c: list[str] = []
            for _m in metrics_df["METRIC"].dropna().unique():
                _sub = city_df[city_df["METRIC"] == _m]
                if not _sub.empty:
                    if any(
                        c in city_df.columns and pd.notna(_sub[c].mean())
                        for c in cols
                    ):
                        _avail_c.append(_m)
            _c0 = city_df.iloc[0]
            parts.append(f"\nCiudad detectada: {_c0['CITY']} ({_c0.get('COUNTRY', '')})")
            parts.append(
                f"Métricas con datos históricos disponibles ({len(_avail_c)}): "
                + (", ".join(_avail_c) if _avail_c else "ninguna")
            )

        _mshow_c = [metric] if metric else (
            city_df["METRIC"].dropna().unique().tolist() if not city_df.empty else []
        )
        for m in _mshow_c:
            sub = city_df[city_df["METRIC"] == m]
            if sub.empty:
                continue
            city_name = sub["CITY"].iloc[0]
            country_c = sub["COUNTRY"].iloc[0]
            parts.append(f"\n{m} — Ciudad: {city_name} ({country_c})")
            for col, lbl in zip(cols, labels):
                if col in sub.columns:
                    avg = sub[col].mean()
                    if pd.notna(avg):
                        parts.append(f"  {lbl}: {avg:.4f}")

        if len(parts) > 2 and zone is None:
            return "\n".join(parts)

    # ── Country-level series ───────────────────────────────────────────────────
    if country:
        c_df = metrics_df[metrics_df["COUNTRY"] == country.upper()]
        _mshow_cty = [metric] if metric else metrics_df["METRIC"].dropna().unique().tolist()[:3]
        for m in _mshow_cty:
            sub = c_df[c_df["METRIC"] == m]
            if sub.empty:
                continue
            parts.append(f"\n{m} — País: {country.upper()}")
            for col, lbl in zip(cols, labels):
                if col in sub.columns:
                    avg = sub[col].mean()
                    if pd.notna(avg):
                        parts.append(f"  {lbl}: {avg:.4f}")
        return "\n".join(parts)

    # ── Metric-only network trend (no specific entity) ────────────────────────
    if metric and temporal:
        sub = metrics_df[metrics_df["METRIC"] == metric]
        if not sub.empty:
            parts.append(f"\n{metric} — Red LATAM (promedio de todas las zonas)")
            for col, lbl in zip(cols, labels):
                if col in sub.columns:
                    avg = sub[col].mean()
                    if pd.notna(avg):
                        parts.append(f"  {lbl}: {avg:.4f}")

            parts.append(f"\n{metric} — Tendencia por país (W-8 → W-0):")
            for c in sorted(sub["COUNTRY"].unique()):
                c_sub = sub[sub["COUNTRY"] == c]
                trend = []
                for col, lbl in zip(cols, labels):
                    if col in c_sub.columns:
                        avg = c_sub[col].mean()
                        if pd.notna(avg):
                            trend.append(f"{lbl}:{avg:.3f}")
                if trend:
                    parts.append(f"  {c}: {' | '.join(trend)}")

    return "\n".join(parts) if len(parts) > 2 else ""


# ── Query-aware analytics computation ─────────────────────────────────────────

def _apply_priority_filter(df: pd.DataFrame, prioritization: str | None) -> pd.DataFrame:
    """
    Filter DataFrame rows by ZONE_PRIORITIZATION.
    Falls back to the full df when the column is absent or no rows match,
    so the caller always gets a non-empty result if the data exists at all.
    """
    if not prioritization or "ZONE_PRIORITIZATION" not in df.columns:
        return df
    col = df["ZONE_PRIORITIZATION"].fillna("").str.strip()
    exact = df[col == prioritization]
    if not exact.empty:
        return exact
    # Partial match: "High" matches "High Priority", etc.
    key = prioritization.split()[0]
    partial = df[col.str.contains(re.escape(key), case=False, na=False)]
    return partial if not partial.empty else df


def _md_table(headers: list[str], rows: list[list]) -> list[str]:
    """Return lines forming a markdown table ready to embed in the context string."""
    sep   = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return lines


def compute_order_growth_ranking(
    orders_df: pd.DataFrame,
    n: int = 10,
    n_weeks: int = 4,
    country: str | None = None,
    ascending: bool = False,
    prioritization: str | None = None,
) -> pd.DataFrame:
    """Rank zones by order volume % growth from W-{n_weeks} to W-0."""
    mask = orders_df["METRIC"] == "Orders"
    if country:
        mask &= orders_df["COUNTRY"] == country

    recent_col = "L0W"
    base_col   = f"L{n_weeks}W"
    if base_col not in orders_df.columns:
        available = [c for c in WEEK_COLS_ORDERS if c in orders_df.columns]
        base_col  = available[0] if available else recent_col

    if recent_col not in orders_df.columns:
        return pd.DataFrame()

    select_cols = [c for c in ["COUNTRY", "CITY", "ZONE", base_col, recent_col]
                   if c in orders_df.columns]
    work = orders_df.loc[mask, select_cols].dropna(subset=[base_col, recent_col]).copy()
    work = work[work[base_col] > 0]
    work = _apply_priority_filter(work, prioritization)
    work["growth_pct"]     = (work[recent_col] - work[base_col]) / work[base_col] * 100
    work["base_orders"]    = work[base_col]
    work["current_orders"] = work[recent_col]
    return work.sort_values("growth_pct", ascending=ascending).head(n).reset_index(drop=True)


def compute_country_order_growth(
    orders_df: pd.DataFrame,
    n_weeks: int = 4,
) -> pd.DataFrame:
    """Country-level aggregated order growth from W-{n_weeks} to W-0."""
    mask = orders_df["METRIC"] == "Orders"
    recent_col = "L0W"
    base_col   = f"L{n_weeks}W"
    if base_col not in orders_df.columns:
        available = [c for c in WEEK_COLS_ORDERS if c in orders_df.columns]
        base_col  = available[0] if available else recent_col

    if recent_col not in orders_df.columns:
        return pd.DataFrame()

    work = orders_df.loc[mask, ["COUNTRY", base_col, recent_col]].dropna().copy()
    grouped = work.groupby("COUNTRY")[[base_col, recent_col]].sum().reset_index()
    grouped = grouped[grouped[base_col] > 0].copy()
    grouped["growth_pct"]     = (grouped[recent_col] - grouped[base_col]) / grouped[base_col] * 100
    grouped["base_orders"]    = grouped[base_col]
    grouped["current_orders"] = grouped[recent_col]
    return grouped.sort_values("growth_pct", ascending=False).reset_index(drop=True)


def compute_metric_growth_ranking(
    metrics_df: pd.DataFrame,
    metric: str,
    n: int = 10,
    n_weeks: int = 4,
    country: str | None = None,
    ascending: bool = False,
    prioritization: str | None = None,
) -> pd.DataFrame:
    """Rank zones by metric value % growth from W-{n_weeks} to W-0."""
    mask = metrics_df["METRIC"] == metric
    if country:
        mask &= metrics_df["COUNTRY"] == country

    recent_col = "L0W_ROLL"
    base_col   = f"L{n_weeks}W_ROLL"
    if base_col not in metrics_df.columns:
        available = [c for c in WEEK_COLS_METRICS if c in metrics_df.columns]
        base_col  = available[0] if available else recent_col

    if recent_col not in metrics_df.columns:
        return pd.DataFrame()

    select_cols = [c for c in
                   ["COUNTRY", "CITY", "ZONE", "ZONE_PRIORITIZATION", base_col, recent_col]
                   if c in metrics_df.columns]
    work = metrics_df.loc[mask, select_cols].dropna(subset=[base_col, recent_col]).copy()
    work = work[work[base_col].abs() > 1e-6]
    work = _apply_priority_filter(work, prioritization)
    work["growth_pct"]  = (work[recent_col] - work[base_col]) / work[base_col].abs() * 100
    work["base_val"]    = work[base_col]
    work["current_val"] = work[recent_col]
    return work.sort_values("growth_pct", ascending=ascending).head(n).reset_index(drop=True)


def compute_extended_metric_ranking(
    metrics_df: pd.DataFrame,
    metric: str,
    n: int = 10,
    country: str | None = None,
    ascending: bool = False,
    prioritization: str | None = None,
) -> pd.DataFrame:
    """Top-N zones by current metric value (W-0) including WoW change."""
    mask = metrics_df["METRIC"] == metric
    if country:
        mask &= metrics_df["COUNTRY"] == country

    select_cols = [c for c in
                   ["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION",
                    "L0W_ROLL", "L1W_ROLL"]
                   if c in metrics_df.columns]
    work = metrics_df.loc[mask, select_cols].dropna(subset=["L0W_ROLL"]).copy()
    work = _apply_priority_filter(work, prioritization)
    if "L1W_ROLL" in work.columns:
        valid = work["L1W_ROLL"].abs() > 1e-6
        work.loc[valid, "wow_pct"] = (
            (work.loc[valid, "L0W_ROLL"] - work.loc[valid, "L1W_ROLL"])
            / work.loc[valid, "L1W_ROLL"].abs() * 100
        )
    return work.sort_values("L0W_ROLL", ascending=ascending).head(n).reset_index(drop=True)


def compute_extended_order_ranking(
    orders_df: pd.DataFrame,
    n: int = 10,
    country: str | None = None,
    ascending: bool = False,
    prioritization: str | None = None,
) -> pd.DataFrame:
    """Top-N zones by current order volume (W-0) including WoW change."""
    mask = orders_df["METRIC"] == "Orders"
    if country:
        mask &= orders_df["COUNTRY"] == country

    week_cols   = [c for c in ["L0W", "L1W"] if c in orders_df.columns]
    select_cols = [c for c in ["COUNTRY", "CITY", "ZONE"] + week_cols
                   if c in orders_df.columns]
    work = orders_df.loc[mask, select_cols].dropna(subset=["L0W"]).copy()
    work = _apply_priority_filter(work, prioritization)
    if "L1W" in work.columns:
        valid = work["L1W"] > 0
        work.loc[valid, "wow_pct"] = (
            (work.loc[valid, "L0W"] - work.loc[valid, "L1W"])
            / work.loc[valid, "L1W"] * 100
        )
    return work.sort_values("L0W", ascending=ascending).head(n).reset_index(drop=True)


def compute_opportunity_analysis(
    metrics_df: pd.DataFrame,
    metric: str,
    country: str | None = None,
    n: int = 10,
) -> pd.DataFrame:
    """Zones with highest improvement potential (significantly below country average)."""
    if "L0W_ROLL" not in metrics_df.columns:
        return pd.DataFrame()

    mask = metrics_df["METRIC"] == metric
    if country:
        mask &= metrics_df["COUNTRY"] == country

    select_cols = [c for c in ["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "L0W_ROLL"]
                   if c in metrics_df.columns]
    work = metrics_df.loc[mask, select_cols].dropna(subset=["L0W_ROLL"]).copy()

    country_avg = (
        work.groupby("COUNTRY")["L0W_ROLL"].mean().reset_index()
        .rename(columns={"L0W_ROLL": "country_avg"})
    )
    work = work.merge(country_avg, on="COUNTRY")
    work = work[work["country_avg"].abs() > 1e-6].copy()
    work["gap_from_avg"] = work["country_avg"] - work["L0W_ROLL"]
    work["gap_pct"]      = work["gap_from_avg"] / work["country_avg"].abs() * 100
    work = work[work["gap_pct"] > 5]
    return work.sort_values("gap_pct", ascending=False).head(n).reset_index(drop=True)


def compute_intent_chart(
    metrics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    intent: dict,
    question: str,
) -> "go.Figure | None":
    """
    Build a Plotly chart from the EXACT same filtered data as compute_query_analytics.
    Returns None for non-ranking intents so callers fall back to AI-generated chart specs.
    """
    intent_types  = intent.get("intent_types", [])
    ranking_types = {"growth_ranking", "decline_ranking", "ranking"}
    if not (ranking_types & set(intent_types)):
        return None

    metric          = intent.get("metric")
    country         = intent.get("country")
    prioritization  = intent.get("prioritization")
    top_n           = _extract_top_n(question)
    n_weeks         = _extract_n_weeks(question)
    mentions_orders = bool(_ORDERS_RE.search(question))
    wants_bottom    = bool(_BOTTOM_RE.search(question))
    prio_label      = f" [{prioritization}]" if prioritization else ""
    scope           = f" — {country}" if country else " — LATAM"

    def _zone_labels(df: pd.DataFrame) -> list[str]:
        return [
            f"{z[:22]}… ({c})" if len(z) > 22 else f"{z} ({c})"
            for z, c in zip(df["ZONE"].tolist(), df["COUNTRY"].tolist())
        ]

    try:
        # ── Growth ranking ────────────────────────────────────────────────────
        if "growth_ranking" in intent_types:
            if metric:
                df = compute_metric_growth_ranking(
                    metrics_df, metric, n=top_n, n_weeks=n_weeks, country=country,
                    prioritization=prioritization,
                )
                if not df.empty:
                    df = df.sort_values("growth_pct")  # best growth last → top bar highlighted
                    title = f"Crecimiento {metric} · W-{n_weeks}→W-0{scope}{prio_label}"
                    return _chart_hbar(_zone_labels(df), df["growth_pct"].tolist(), title, fmt="+.1f")
            else:
                df = compute_order_growth_ranking(
                    orders_df, n=top_n, n_weeks=n_weeks, country=country,
                    prioritization=prioritization,
                )
                if not df.empty:
                    df = df.sort_values("growth_pct")
                    title = f"Crecimiento Órdenes · W-{n_weeks}→W-0{scope}{prio_label}"
                    return _chart_hbar(_zone_labels(df), df["growth_pct"].tolist(), title, fmt="+.1f")

        # ── Decline ranking ───────────────────────────────────────────────────
        if "decline_ranking" in intent_types:
            if metric:
                df = compute_metric_growth_ranking(
                    metrics_df, metric, n=top_n, n_weeks=n_weeks, country=country,
                    ascending=True, prioritization=prioritization,
                )
                if not df.empty:
                    df = df.sort_values("growth_pct", ascending=False)  # worst decline last → top bar
                    title = f"Mayor Caída {metric} · W-{n_weeks}→W-0{scope}{prio_label}"
                    return _chart_hbar(_zone_labels(df), df["growth_pct"].tolist(), title, fmt="+.1f")
            else:
                df = compute_order_growth_ranking(
                    orders_df, n=top_n, n_weeks=n_weeks, country=country, ascending=True,
                    prioritization=prioritization,
                )
                if not df.empty:
                    df = df.sort_values("growth_pct", ascending=False)  # worst decline last → top bar
                    title = f"Mayor Caída de Órdenes · W-{n_weeks}→W-0{scope}{prio_label}"
                    return _chart_hbar(_zone_labels(df), df["growth_pct"].tolist(), title, fmt="+.1f")

        # ── Extended ranking (static W-0 snapshot) ────────────────────────────
        if "ranking" in intent_types and not {"growth_ranking", "decline_ranking"} & set(intent_types):
            if metric:
                df = compute_extended_metric_ranking(
                    metrics_df, metric, n=top_n, country=country, ascending=wants_bottom,
                    prioritization=prioritization,
                )
                if not df.empty and "L0W_ROLL" in df.columns:
                    # sort so the "most extreme" value is last → appears at top of hbar
                    df_s = df.sort_values("L0W_ROLL", ascending=not wants_bottom)
                    direction = "Menor" if wants_bottom else "Mayor"
                    title = f"{metric} W-0 · {direction}{scope}{prio_label}"
                    return _chart_hbar(_zone_labels(df_s), df_s["L0W_ROLL"].tolist(), title, fmt=".3f")
            elif mentions_orders:
                df = compute_extended_order_ranking(
                    orders_df, n=top_n, country=country, ascending=wants_bottom,
                    prioritization=prioritization,
                )
                if not df.empty and "L0W" in df.columns:
                    df_s = df.sort_values("L0W", ascending=not wants_bottom)
                    direction = "Menor" if wants_bottom else "Mayor"
                    title = f"Órdenes W-0 · {direction}{scope}{prio_label}"
                    return _chart_hbar(_zone_labels(df_s), df_s["L0W"].tolist(), title, fmt=",.0f")

    except Exception:
        pass

    return None


def compute_query_analytics(
    metrics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    intent: dict,
    question: str,
) -> str:
    """
    Route detected intent to computation functions.
    Returns a pre-formatted RANKING OPERATIVO block for the LLM to present verbatim.
    """
    intent_types = intent.get("intent_types", [])
    # temporal-only is handled by build_entity_context
    active_types = [t for t in intent_types if t != "temporal"]
    if not active_types:
        return ""

    metric          = intent.get("metric")
    country         = intent.get("country")
    prioritization  = intent.get("prioritization")
    top_n           = _extract_top_n(question)
    n_weeks         = _extract_n_weeks(question)
    mentions_orders = bool(_ORDERS_RE.search(question))
    wants_bottom    = bool(_BOTTOM_RE.search(question))
    prio_label      = f" [{prioritization}]" if prioritization else ""

    parts: list[str] = ["", "=== RANKING OPERATIVO ==="]
    added = False

    # ── Growth ranking ────────────────────────────────────────────────────────
    if "growth_ranking" in intent_types:
        if mentions_orders or metric is None:
            df = compute_order_growth_ranking(
                orders_df, n=top_n, n_weeks=n_weeks, country=country,
                prioritization=prioritization,
            )
            if not df.empty:
                scope = f" — {country}" if country else " — Red LATAM"
                parts.append(
                    f"\nTOP {len(df)} ZONAS: CRECIMIENTO DE ÓRDENES"
                    f" (W-{n_weeks} → W-0{scope}{prio_label}):"
                )
                rows = [
                    [i, r.ZONE, r.CITY, r.COUNTRY,
                     f"{r.growth_pct:+.1f}%",
                     f"{int(r.base_orders):,}",
                     f"{int(r.current_orders):,}"]
                    for i, r in enumerate(df.itertuples(), 1)
                ]
                parts += _md_table(
                    ["#", "Zona", "Ciudad", "País", f"Δ%", f"W-{n_weeks}", "W-0"],
                    rows,
                )
                c_df = compute_country_order_growth(orders_df, n_weeks=n_weeks)
                if not c_df.empty:
                    parts.append(f"\nCrecimiento de órdenes por país (W-{n_weeks} → W-0):")
                    c_rows = [
                        [r.COUNTRY, f"{r.growth_pct:+.1f}%",
                         f"{int(r.base_orders):,}", f"{int(r.current_orders):,}"]
                        for r in c_df.itertuples()
                    ]
                    parts += _md_table(["País", "Δ%", f"W-{n_weeks}", "W-0"], c_rows)
                added = True

        if metric:
            df = compute_metric_growth_ranking(
                metrics_df, metric, n=top_n, n_weeks=n_weeks, country=country,
                prioritization=prioritization,
            )
            if not df.empty:
                scope = f" — {country}" if country else " — Red LATAM"
                parts.append(
                    f"\nTOP {len(df)} ZONAS: CRECIMIENTO DE {metric.upper()}"
                    f" (W-{n_weeks} → W-0{scope}{prio_label}):"
                )
                rows = [
                    [i, r.ZONE, r.CITY, r.COUNTRY,
                     f"{r.growth_pct:+.1f}%",
                     f"{r.base_val:.3f}",
                     f"{r.current_val:.3f}"]
                    for i, r in enumerate(df.itertuples(), 1)
                ]
                parts += _md_table(
                    ["#", "Zona", "Ciudad", "País", "Δ%", f"W-{n_weeks}", "W-0"],
                    rows,
                )
                added = True

    # ── Decline ranking ───────────────────────────────────────────────────────
    if "decline_ranking" in intent_types:
        if mentions_orders or metric is None:
            df = compute_order_growth_ranking(
                orders_df, n=top_n, n_weeks=n_weeks, country=country, ascending=True,
                prioritization=prioritization,
            )
            if not df.empty:
                scope = f" — {country}" if country else " — Red LATAM"
                parts.append(
                    f"\nTOP {len(df)} ZONAS: MAYOR CAÍDA DE ÓRDENES"
                    f" (W-{n_weeks} → W-0{scope}{prio_label}):"
                )
                rows = [
                    [i, r.ZONE, r.CITY, r.COUNTRY,
                     f"{r.growth_pct:+.1f}%",
                     f"{int(r.base_orders):,}",
                     f"{int(r.current_orders):,}"]
                    for i, r in enumerate(df.itertuples(), 1)
                ]
                parts += _md_table(
                    ["#", "Zona", "Ciudad", "País", "Δ%", f"W-{n_weeks}", "W-0"],
                    rows,
                )
                added = True

        if metric:
            df = compute_metric_growth_ranking(
                metrics_df, metric, n=top_n, n_weeks=n_weeks, country=country, ascending=True,
                prioritization=prioritization,
            )
            if not df.empty:
                scope = f" — {country}" if country else " — Red LATAM"
                parts.append(
                    f"\nTOP {len(df)} ZONAS: MAYOR CAÍDA DE {metric.upper()}"
                    f" (W-{n_weeks} → W-0{scope}{prio_label}):"
                )
                rows = [
                    [i, r.ZONE, r.CITY, r.COUNTRY,
                     f"{r.growth_pct:+.1f}%",
                     f"{r.base_val:.3f}",
                     f"{r.current_val:.3f}"]
                    for i, r in enumerate(df.itertuples(), 1)
                ]
                parts += _md_table(
                    ["#", "Zona", "Ciudad", "País", "Δ%", f"W-{n_weeks}", "W-0"],
                    rows,
                )
                added = True

    # ── Extended ranking (top/bottom N) ───────────────────────────────────────
    if "ranking" in intent_types and not {"growth_ranking", "decline_ranking"} & set(intent_types):
        if metric:
            df = compute_extended_metric_ranking(
                metrics_df, metric, n=top_n, country=country, ascending=wants_bottom,
                prioritization=prioritization,
            )
            if not df.empty:
                scope     = f" — {country}" if country else ""
                direction = "MENOR" if wants_bottom else "MAYOR"
                parts.append(
                    f"\nTOP {len(df)} ZONAS POR {metric.upper()}"
                    f" (W-0, {direction}{scope}{prio_label}):"
                )
                rows = []
                for i, r in enumerate(df.itertuples(), 1):
                    wow_val = getattr(r, "wow_pct", None)
                    wow_str = (
                        f"{wow_val:+.1f}%"
                        if wow_val is not None and pd.notna(wow_val)
                        else "—"
                    )
                    rows.append([i, r.ZONE, r.CITY, r.COUNTRY, f"{r.L0W_ROLL:.3f}", wow_str])
                parts += _md_table(["#", "Zona", "Ciudad", "País", "W-0", "SaS"], rows)
                added = True
        elif mentions_orders:
            df = compute_extended_order_ranking(
                orders_df, n=top_n, country=country, ascending=wants_bottom,
                prioritization=prioritization,
            )
            if not df.empty:
                scope     = f" — {country}" if country else ""
                direction = "MENOR" if wants_bottom else "MAYOR"
                parts.append(
                    f"\nTOP {len(df)} ZONAS POR ÓRDENES (W-0, {direction}{scope}{prio_label}):"
                )
                rows = []
                for i, r in enumerate(df.itertuples(), 1):
                    wow_val = getattr(r, "wow_pct", None)
                    wow_str = (
                        f"{wow_val:+.1f}%"
                        if wow_val is not None and pd.notna(wow_val)
                        else "—"
                    )
                    rows.append([i, r.ZONE, r.CITY, r.COUNTRY, f"{int(r.L0W):,}", wow_str])
                parts += _md_table(["#", "Zona", "Ciudad", "País", "Pedidos W-0", "SaS"], rows)
                added = True

    # ── Opportunity analysis ──────────────────────────────────────────────────
    if "opportunity" in intent_types and metric:
        df = compute_opportunity_analysis(
            metrics_df, metric, country=country, n=top_n
        )
        if not df.empty:
            scope = f" — {country}" if country else ""
            parts.append(
                f"\nZONAS DE OPORTUNIDAD — {metric.upper()}"
                f" (debajo del promedio país{scope}):"
            )
            rows = [
                [i, r.ZONE, r.CITY, r.COUNTRY,
                 f"{r.L0W_ROLL:.3f}",
                 f"{r.country_avg:.3f}",
                 f"-{r.gap_pct:.1f}%"]
                for i, r in enumerate(df.itertuples(), 1)
            ]
            parts += _md_table(
                ["#", "Zona", "Ciudad", "País", "W-0", "Prom. País", "Brecha"],
                rows,
            )
            added = True

    # ── Zone-type comparison (Wealthy vs Non Wealthy) ─────────────────────────
    if "zone_type_comparison" in intent_types:
        _available_metrics = set(metrics_df["METRIC"].dropna().unique())
        _KEY_ZT_METRICS    = [
            "Perfect Orders", "Gross Profit UE", "Turbo Adoption", "Lead Penetration",
        ]
        _zt_show = [metric] if metric else [
            m for m in _KEY_ZT_METRICS if m in _available_metrics
        ]
        scope = f" — {country}" if country else ""
        _zt_added = False
        for _zt_m in _zt_show:
            df = get_metric_by_zone_type(metrics_df, _zt_m, country=country)
            if not df.empty:
                if not _zt_added:
                    parts.append(f"\nCOMPARACIÓN POR TIPO DE ZONA{scope}:")
                    _zt_added = True
                net = metrics_df.loc[metrics_df["METRIC"] == _zt_m, "L0W_ROLL"].mean()
                parts.append(f"\n  {_zt_m.upper()}:")
                rows = []
                for r in df.sort_values("avg_value", ascending=False).itertuples():
                    gap = (r.avg_value - net) / abs(net) * 100 if net else 0
                    rows.append([r.ZONE_TYPE, f"{r.avg_value:.3f}", f"{net:.3f}", f"{gap:+.1f}%"])
                parts += _md_table(["Tipo de Zona", "Promedio", "Red", "Brecha"], rows)
        if _zt_added:
            added = True

    if not added:
        return ""

    return "\n".join(parts)


# ── Dynamic context (base + entity injection) ─────────────────────────────────

def build_dynamic_context(
    metrics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    question: str,
) -> str:
    """
    Entry point for the chat pipeline.
    Builds the full base context and appends entity-specific historical data
    when temporal or entity intent is detected in the question.
    """
    base     = build_full_context(metrics_df, orders_df)
    intent   = detect_question_intent(question, metrics_df)
    extra    = build_entity_context(metrics_df, intent)
    computed = compute_query_analytics(metrics_df, orders_df, intent, question)
    return base + extra + computed


def build_analytics_context(
    metrics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    selected_metric: str,
    selected_country: str | None = None,
) -> str:
    parts: list[str] = []
    summary = get_dataset_summary(metrics_df, orders_df)
    countries = sorted(metrics_df["COUNTRY"].dropna().unique().tolist())
    metrics_list = sorted(metrics_df["METRIC"].dropna().unique().tolist())
    scope = selected_country if selected_country else "todos los paises"

    # ── 1. Dataset context (brief) ─────────────────────────────────────────────
    parts += [
        "=== CONTEXTO DEL DATASET ===",
        f"Paises: {', '.join(countries)}",
        f"Metricas disponibles: {', '.join(metrics_list)}",
        f"Red: {summary['zones']} zonas en {summary['cities']} ciudades",
        "Etiquetas de semana: W-0 = mas reciente, W-8 = hace 8 semanas",
        "",
    ]

    # ── 2. Orders network summary ─────────────────────────────────────────────
    obc = get_orders_by_country(orders_df)
    orders_line = " | ".join(
        f"{r.COUNTRY}: {int(r.total_orders):,}" for r in obc.itertuples()
    )
    parts += [
        "=== PEDIDOS (W-0, ultima semana) ===",
        f"Total red: {summary['total_orders']:,} pedidos  |  Cambio SaS: {summary['orders_wow_pct']:+.1f}%",
        orders_line,
        "",
    ]

    # ── 3. Executive network summary for selected metric ───────────────────────
    avg = get_country_averages(metrics_df, selected_metric)
    mask_scope = metrics_df["METRIC"] == selected_metric
    if selected_country:
        mask_scope &= metrics_df["COUNTRY"] == selected_country

    net_l0 = metrics_df.loc[mask_scope, "L0W_ROLL"].mean() if "L0W_ROLL" in metrics_df.columns else None
    net_l1 = metrics_df.loc[mask_scope, "L1W_ROLL"].mean() if "L1W_ROLL" in metrics_df.columns else None
    net_wow = (net_l0 - net_l1) / abs(net_l1) * 100 if net_l1 and net_l1 != 0 else 0

    best = avg.iloc[0] if not avg.empty else None
    worst = avg.iloc[-1] if not avg.empty else None
    gap = (
        (best.avg_value - worst.avg_value) / abs(worst.avg_value) * 100
        if best is not None and worst is not None and worst.avg_value != 0
        else 0
    )

    parts += [
        f"=== {selected_metric.upper()} — INTELIGENCIA OPERATIVA ({scope}) ===",
        "",
        "RESUMEN EJECUTIVO DE RED (W-0):",
        (f"  Promedio red: {net_l0:.3f}  |  Cambio SaS: {net_wow:+.1f}%"
         if net_l0 is not None else "  (datos insuficientes)"),
    ]
    if best is not None and worst is not None:
        parts.append(
            f"  Lider: {best.COUNTRY} {best.avg_value:.3f}  |  "
            f"Rezagado: {worst.COUNTRY} {worst.avg_value:.3f}  |  Brecha: {gap:.1f}%"
        )
    parts.append("")

    # ── 4. Country benchmark ───────────────────────────────────────────────────
    benchmark_line = " | ".join(_fmt_country_row(r) for r in avg.itertuples())
    parts += [
        "BENCHMARK DE PAISES (W-0, mayor a menor):",
        f"  {benchmark_line}",
        "",
    ]

    # ── 5. Top deteriorating zones (WoW) ──────────────────────────────────────
    worst_zones = get_wow_zones(metrics_df, selected_metric, country=selected_country, n=6, ascending=True)
    if not worst_zones.empty:
        parts.append("SENALES DE DETERIORO — top zonas con mayor caida SaS:")
        for i, r in enumerate(worst_zones.itertuples(), 1):
            parts.append(
                f"  {i}. {r.ZONE} ({r.CITY}, {r.COUNTRY}): "
                f"{r.wow_pct:+.1f}% SaS  ({r.L1W_ROLL:.3f} -> {r.L0W_ROLL:.3f})"
            )
        parts.append("")

    # ── 6. Top improving zones (WoW) ──────────────────────────────────────────
    best_zones = get_wow_zones(metrics_df, selected_metric, country=selected_country, n=6, ascending=False)
    if not best_zones.empty:
        parts.append("SENALES DE MEJORA — top zonas con mayor ganancia SaS:")
        for i, r in enumerate(best_zones.itertuples(), 1):
            parts.append(
                f"  {i}. {r.ZONE} ({r.CITY}, {r.COUNTRY}): "
                f"{r.wow_pct:+.1f}% SaS  ({r.L1W_ROLL:.3f} -> {r.L0W_ROLL:.3f})"
            )
        parts.append("")

    # ── 7. Sustained trends (3 consecutive weeks) ─────────────────────────────
    if all(c in metrics_df.columns for c in ["L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]):
        mask_t = metrics_df["METRIC"] == selected_metric
        if selected_country:
            mask_t &= metrics_df["COUNTRY"] == selected_country
        trend_src = (
            metrics_df.loc[mask_t, ["COUNTRY", "CITY", "ZONE", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]]
            .dropna()
            .copy()
        )
        if not trend_src.empty:
            # 3-week declines
            down3 = trend_src[
                (trend_src["L2W_ROLL"] > trend_src["L1W_ROLL"]) &
                (trend_src["L1W_ROLL"] > trend_src["L0W_ROLL"])
            ].copy()
            down3["total_pct"] = (
                (down3["L0W_ROLL"] - down3["L2W_ROLL"]) / down3["L2W_ROLL"].abs() * 100
            )
            down3 = down3[down3["total_pct"] <= -4].nsmallest(5, "total_pct")

            # 3-week improvements
            up3 = trend_src[
                (trend_src["L2W_ROLL"] < trend_src["L1W_ROLL"]) &
                (trend_src["L1W_ROLL"] < trend_src["L0W_ROLL"])
            ].copy()
            up3["total_pct"] = (
                (up3["L0W_ROLL"] - up3["L2W_ROLL"]) / up3["L2W_ROLL"].abs() * 100
            )
            up3 = up3[up3["total_pct"] >= 3].nlargest(5, "total_pct")

            if not down3.empty or not up3.empty:
                parts.append("TENDENCIAS SOSTENIDAS (3 semanas consecutivas):")
                if not down3.empty:
                    parts.append("  Tendencias negativas persistentes:")
                    for r in down3.itertuples():
                        parts.append(
                            f"    - {r.ZONE} ({r.CITY}, {r.COUNTRY}): "
                            f"{r.total_pct:+.1f}% acumulado  "
                            f"({r.L2W_ROLL:.3f} -> {r.L1W_ROLL:.3f} -> {r.L0W_ROLL:.3f})"
                        )
                if not up3.empty:
                    parts.append("  Tendencias positivas sostenidas:")
                    for r in up3.itertuples():
                        parts.append(
                            f"    - {r.ZONE} ({r.CITY}, {r.COUNTRY}): "
                            f"+{r.total_pct:.1f}% acumulado  "
                            f"({r.L2W_ROLL:.3f} -> {r.L1W_ROLL:.3f} -> {r.L0W_ROLL:.3f})"
                        )
                parts.append("")

    # ── 8. Opportunity zones (below z-score threshold) ────────────────────────
    if "L0W_ROLL" in metrics_df.columns:
        mask_o = metrics_df["METRIC"] == selected_metric
        if selected_country:
            mask_o &= metrics_df["COUNTRY"] == selected_country
        opp_src = metrics_df.loc[mask_o, ["COUNTRY", "CITY", "ZONE", "L0W_ROLL"]].dropna()
        stats = opp_src.groupby("COUNTRY")["L0W_ROLL"].agg(["mean", "std"]).reset_index()
        merged = opp_src.merge(stats, on="COUNTRY")
        merged = merged[merged["std"] > 0.001].copy()
        merged["z"] = (merged["L0W_ROLL"] - merged["mean"]) / merged["std"]
        laggards = merged[merged["z"] <= -1.2].nsmallest(5, "z")
        if not laggards.empty:
            parts.append("ZONAS DE OPORTUNIDAD (>1.2 desv. bajo promedio pais):")
            for r in laggards.itertuples():
                gap_pct = (r.mean - r.L0W_ROLL) / abs(r.mean) * 100 if r.mean else 0
                parts.append(
                    f"  - {r.ZONE} ({r.CITY}, {r.COUNTRY}): "
                    f"{r.L0W_ROLL:.3f} vs promedio {r.COUNTRY} {r.mean:.3f}  "
                    f"(brecha: {gap_pct:.1f}%)"
                )
            parts.append("")

    # ── 9. Zone type distribution ─────────────────────────────────────────────
    zt = get_metric_by_zone_type(metrics_df, selected_metric, country=selected_country)
    if not zt.empty:
        zt_line = " | ".join(f"{r.ZONE_TYPE}: {r.avg_value:.3f}" for r in zt.itertuples())
        parts += [f"DISTRIBUCION POR TIPO DE ZONA: {zt_line}", ""]

    # ── 10. Historical trend (8 weeks, compact) ───────────────────────────────
    trend = get_weekly_trend(metrics_df, selected_metric, country=selected_country)
    trend_str = " -> ".join(
        f"{lbl}: {val:.3f}" for lbl, val in zip(trend["week"], trend["value"])
    )
    parts += [f"TENDENCIA HISTORICA (8 semanas): {trend_str}", ""]

    # ── 11. Top and bottom zones (reference) ─────────────────────────────────
    top = get_top_zones(metrics_df, selected_metric, country=selected_country, n=8)
    bot = get_top_zones(metrics_df, selected_metric, country=selected_country, n=8, ascending=True)
    parts += [
        f"TOP 8 ZONAS — {selected_metric} (W-0, {scope}):",
        top.to_string(index=False),
        "",
        f"BOTTOM 8 ZONAS — {selected_metric} (W-0, {scope}):",
        bot.to_string(index=False),
    ]

    return "\n".join(parts)


# ── Full multi-metric context (for chat — not bound to any filter) ────────────

def build_full_context(
    metrics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> str:
    """
    Build a comprehensive context covering ALL 13 metrics and all countries.
    Used by the AI copilot so it can answer questions about any metric freely.
    """
    parts: list[str] = []
    summary  = get_dataset_summary(metrics_df, orders_df)
    countries = sorted(metrics_df["COUNTRY"].dropna().unique().tolist())
    metrics_list = sorted(metrics_df["METRIC"].dropna().unique().tolist())

    # ── 1. Dataset overview ───────────────────────────────────────────────────
    parts += [
        "=== CONTEXTO OPERATIVO COMPLETO — RAPPI LATAM ===",
        f"Paises: {', '.join(countries)}",
        f"Red: {summary['zones']} zonas en {summary['cities']} ciudades",
        f"Metricas disponibles ({len(metrics_list)}): {', '.join(metrics_list)}",
        "Semanas: W-0 = mas reciente, W-1..W-8 = semanas anteriores",
        "",
    ]

    # ── 2. Orders summary ─────────────────────────────────────────────────────
    obc = get_orders_by_country(orders_df)
    orders_line = " | ".join(f"{r.COUNTRY}: {int(r.total_orders):,}" for r in obc.itertuples())
    parts += [
        "=== PEDIDOS (W-0) ===",
        f"Total red: {summary['total_orders']:,}  |  Cambio SaS: {summary['orders_wow_pct']:+.1f}%",
        orders_line,
        "",
    ]

    # ── 3. Per-metric intelligence snapshot ───────────────────────────────────
    parts.append("=== INTELIGENCIA POR METRICA (W-0) ===")
    for metric in metrics_list:
        avg = get_country_averages(metrics_df, metric)
        mask = metrics_df["METRIC"] == metric
        net_l0 = metrics_df.loc[mask, "L0W_ROLL"].mean() if "L0W_ROLL" in metrics_df.columns else None
        net_l1 = metrics_df.loc[mask, "L1W_ROLL"].mean() if "L1W_ROLL" in metrics_df.columns else None
        net_wow = (net_l0 - net_l1) / abs(net_l1) * 100 if net_l1 and net_l1 != 0 else 0

        best  = avg.iloc[0]  if not avg.empty else None
        worst = avg.iloc[-1] if not avg.empty else None

        parts.append(f"--- {metric} ---")
        if net_l0 is not None:
            parts.append(f"  Red: {net_l0:.3f}  ({net_wow:+.1f}% SaS)")
        if best is not None and worst is not None:
            parts.append(f"  Lider: {best.COUNTRY} {best.avg_value:.3f}  |  Rezagado: {worst.COUNTRY} {worst.avg_value:.3f}")

        # Top 5 zones by current value (W-0)
        top_zones = get_top_zones(metrics_df, metric, n=5)
        if not top_zones.empty and "L0W_ROLL" in top_zones.columns:
            top_line = " | ".join(
                f"{r.ZONE} ({r.COUNTRY}) {r.L0W_ROLL:.3f}"
                for r in top_zones.itertuples()
            )
            parts.append(f"  Top 5 zonas W-0: {top_line}")

        # Bottom 5 zones by current value
        bot_zones = get_top_zones(metrics_df, metric, n=5, ascending=True)
        if not bot_zones.empty and "L0W_ROLL" in bot_zones.columns:
            bot_line = " | ".join(
                f"{r.ZONE} ({r.COUNTRY}) {r.L0W_ROLL:.3f}"
                for r in bot_zones.itertuples()
            )
            parts.append(f"  Bottom 5 zonas W-0: {bot_line}")

        # Top 3 declining zones (WoW)
        worst_zones = get_wow_zones(metrics_df, metric, n=3, ascending=True)
        if not worst_zones.empty:
            declines = []
            for r in worst_zones.itertuples():
                declines.append(f"{r.ZONE} ({r.COUNTRY}) {r.wow_pct:+.1f}%")
            parts.append(f"  Mayores caidas SaS: {' | '.join(declines)}")

        # Top 3 improving zones (WoW)
        best_zones = get_wow_zones(metrics_df, metric, n=3, ascending=False)
        if not best_zones.empty:
            gains = []
            for r in best_zones.itertuples():
                gains.append(f"{r.ZONE} ({r.COUNTRY}) {r.wow_pct:+.1f}%")
            parts.append(f"  Mayores ganancias SaS: {' | '.join(gains)}")
        parts.append("")

    # ── 4. Historical trend for key metric ────────────────────────────────────
    if "Perfect Orders" in metrics_list:
        trend = get_weekly_trend(metrics_df, "Perfect Orders")
        trend_str = " -> ".join(f"{lbl}: {val:.3f}" for lbl, val in zip(trend["week"], trend["value"]))
        parts += [f"=== TENDENCIA HISTORICA — Perfect Orders: {trend_str} ===", ""]

    return "\n".join(parts)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def parse_chart_spec(text: str) -> dict | None:
    match = re.search(r"```chart\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        spec = json.loads(match.group(1))
        return spec if "query" in spec else None
    except (json.JSONDecodeError, KeyError):
        return None


def strip_chart_block(text: str) -> str:
    return re.sub(r"```chart.*?```", "", text, flags=re.DOTALL).strip()


def extract_suggested_question(text: str) -> tuple[str, str | None]:
    """
    Split response into (body, suggested_question).
    Extracts '**Pregunta sugerida:**' section so it can be rendered as a chip.
    """
    m = re.search(r"\*\*Pregunta\s+sugerida:\*\*\s*", text, re.IGNORECASE)
    if m:
        rest = text[m.end():].strip()
        suggestion = rest.split("\n")[0].strip()
        body = text[:m.start()].rstrip()
        return body, suggestion if suggestion else None
    return text, None


_C_RED   = "#FF441F"
_C_DARK  = "#1C1C28"
_C_MID   = "#6B7280"
_C_MUTED = "#9CA3AF"
_C_GRID  = "rgba(228,232,240,0.55)"
_C_BAR   = "#D1D5DB"
_C_FONT  = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"


def _ai_layout(title_str: str, height: int, hbar: bool = False) -> dict:
    """Minimal, premium layout shared across all AI chat charts."""
    _axis_common = dict(zeroline=False, showline=False, tickfont=dict(size=10, color=_C_MUTED))
    x_cfg = dict(**_axis_common, showgrid=False)
    y_cfg = dict(**_axis_common, showgrid=True, gridcolor=_C_GRID, gridwidth=0.5)
    if hbar:
        x_cfg, y_cfg = (
            dict(**_axis_common, showgrid=True, gridcolor=_C_GRID, gridwidth=0.5),
            dict(**_axis_common, showgrid=False),
        )
    return dict(
        font=dict(family=_C_FONT, size=11, color=_C_MID),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=6, r=16, t=34, b=6),
        height=height,
        showlegend=False,
        title=dict(
            text=title_str,
            font=dict(size=12, color=_C_DARK, family=_C_FONT),
            x=0, xanchor="left", pad=dict(l=4, t=0),
        ),
        xaxis=x_cfg,
        yaxis=y_cfg,
        hoverlabel=dict(
            bgcolor="white", bordercolor="#E4E8F0",
            font=dict(size=11, family=_C_FONT),
        ),
    )


def _chart_hbar(
    labels: list,
    values: list,
    title: str,
    fmt: str = ".3f",
) -> go.Figure:
    """Horizontal bar — top item highlighted red, rest neutral gray."""
    n = len(labels)
    colors = [_C_RED if i == n - 1 else _C_BAR for i in range(n)]
    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"{v:{fmt}}" for v in values],
        textposition="outside",
        textfont=dict(size=9.5, color=_C_MUTED),
        hovertemplate="%{y}: %{x}<extra></extra>",
        cliponaxis=False,
    ))
    height = max(180, 28 * n + 40)
    fig.update_layout(**_ai_layout(title, height=height, hbar=True))
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="", ticklabelposition="outside left")
    return fig


def _chart_line(
    x_vals: list,
    y_vals: list,
    title: str,
) -> go.Figure:
    """Sparkline-style trend — clean red line, subtle markers, no fill noise."""
    fig = go.Figure(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines+markers",
        line=dict(color=_C_RED, width=2, shape="spline", smoothing=0.4),
        marker=dict(
            size=5, color=_C_RED,
            line=dict(color="white", width=1.5),
        ),
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_ai_layout(title, height=175))
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="", tickformat=".3f")
    return fig


def _render_chart_impl(
    spec: dict,
    metrics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    fallback_country: str | None = None,
) -> go.Figure | None:
    query   = spec.get("query", "")
    title   = spec.get("title", "")
    metric  = spec.get("metric") or ""
    raw_c   = spec.get("country")
    country = (raw_c if raw_c and raw_c != "null" else None) or fallback_country

    try:
        if query == "country_averages" and metric:
            df = get_country_averages(metrics_df, metric).sort_values("avg_value")
            return _chart_hbar(df["COUNTRY"].tolist(), df["avg_value"].tolist(), title)

        if query == "weekly_trend" and metric:
            df = get_weekly_trend(metrics_df, metric, country=country)
            return _chart_line(df["week"].tolist(), df["value"].tolist(), title)

        if query == "top_zones" and metric:
            df = get_top_zones(metrics_df, metric, country=country, n=8)
            if "L0W_ROLL" in df.columns:
                df = df.sort_values("L0W_ROLL")
                labels = [z[:28] + "…" if len(z) > 28 else z for z in df["ZONE"].tolist()]
                return _chart_hbar(labels, df["L0W_ROLL"].tolist(), title)

        if query == "orders_by_country":
            df = get_orders_by_country(orders_df).sort_values("total_orders")
            return _chart_hbar(
                df["COUNTRY"].tolist(), df["total_orders"].tolist(), title, fmt=",.0f"
            )

        if query == "orders_trend":
            df = get_orders_trend(orders_df, country=country)
            fig = _chart_line(df["week"].tolist(), df["orders"].tolist(), title)
            fig.update_yaxes(tickformat=",.0f")
            return fig

        if query == "zone_trend" and metric:
            zone_name = spec.get("zone", "")
            mask = metrics_df["ZONE"].str.lower() == zone_name.lower()
            if not mask.any():
                mask = metrics_df["ZONE"].str.lower().str.contains(
                    re.escape(zone_name.lower()), na=False
                )
            sub = metrics_df[mask & (metrics_df["METRIC"] == metric)]
            if not sub.empty:
                row = sub.iloc[0]
                weeks, vals = [], []
                for col, lbl in zip(WEEK_COLS_METRICS, WEEK_LABELS):
                    if col in row.index and pd.notna(row[col]):
                        weeks.append(lbl)
                        vals.append(float(row[col]))
                if weeks:
                    return _chart_line(weeks, vals, title or f"{metric} · {row['ZONE']}")

        if query == "city_trend" and metric:
            city_name = spec.get("city", "")
            mask = metrics_df["CITY"].str.lower() == city_name.lower()
            if not mask.any():
                mask = metrics_df["CITY"].str.lower().str.contains(
                    re.escape(city_name.lower()), na=False
                )
            sub = metrics_df[mask & (metrics_df["METRIC"] == metric)]
            if not sub.empty:
                weeks, vals = [], []
                for col, lbl in zip(WEEK_COLS_METRICS, WEEK_LABELS):
                    if col in sub.columns:
                        avg = sub[col].mean()
                        if pd.notna(avg):
                            weeks.append(lbl)
                            vals.append(float(avg))
                if weeks:
                    return _chart_line(weeks, vals, title or f"{metric} · {sub['CITY'].iloc[0]}")

        if query == "wow_ranking" and metric:
            df = get_wow_zones(metrics_df, metric, country=country, n=8, ascending=True)
            if not df.empty and "wow_pct" in df.columns:
                df_sorted = df.sort_values("wow_pct", ascending=False)
                labels = [
                    f"{z[:20]}… ({c})" if len(z) > 20 else f"{z} ({c})"
                    for z, c in zip(df_sorted["ZONE"].tolist(), df_sorted["COUNTRY"].tolist())
                ]
                return _chart_hbar(labels, df_sorted["wow_pct"].tolist(), title, fmt="+.1f")

        if query == "zone_type_comparison" and metric:
            df = get_metric_by_zone_type(metrics_df, metric, country=country)
            if not df.empty:
                df = df.sort_values("avg_value")
                return _chart_hbar(df["ZONE_TYPE"].tolist(), df["avg_value"].tolist(), title)

    except Exception:
        pass

    return None


def render_chart_from_spec(
    spec: dict,
    metrics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    fallback_country: str | None = None,
    compact: bool = False,
) -> go.Figure | None:
    """Public wrapper — builds chart and optionally tightens for inline chat use."""
    fig = _render_chart_impl(spec, metrics_df, orders_df, fallback_country)
    if fig is not None and compact:
        cur = fig.layout.height or 200
        # Bar charts need more vertical space; line/small charts can stay compact
        cap = 280 if cur > 220 else 195
        fig.update_layout(
            height=min(cur, cap),
            margin=dict(l=4, r=12, t=28, b=4),
            title=dict(font=dict(size=11)),
        )
    return fig


# ── Streaming ─────────────────────────────────────────────────────────────────

def stream_response(
    user_question: str,
    context: str,
    api_history: list[dict],
    system_prompt: str,
    provider: "LLMProvider | None" = None,
) -> Generator[str, None, None]:
    """
    Yield response text chunks via the given provider.

    If no provider is passed, the default is resolved from
    the LLM_PROVIDER env var (falls back to Claude).
    """
    if provider is None:
        from src.providers import get_provider
        provider = get_provider()

    messages = list(api_history[-(HISTORY_PAIRS * 2):])
    messages.append({
        "role": "user",
        "content": f"[CONTEXT]\n{context}\n\n[QUESTION]\n{user_question}",
    })

    yield from provider.stream(system_prompt, messages, MAX_TOKENS)
