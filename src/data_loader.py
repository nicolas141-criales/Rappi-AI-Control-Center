import pandas as pd
import streamlit as st

WEEK_COLS_METRICS = ["L8W_ROLL", "L7W_ROLL", "L6W_ROLL", "L5W_ROLL",
                     "L4W_ROLL", "L3W_ROLL", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]
WEEK_COLS_ORDERS  = ["L8W", "L7W", "L6W", "L5W", "L4W", "L3W", "L2W", "L1W", "L0W"]
WEEK_LABELS       = ["W-8", "W-7", "W-6", "W-5", "W-4", "W-3", "W-2", "W-1", "W-0"]


# ── Loading ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(file_path: str = "data/Data1.xlsx") -> tuple[pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(file_path)
    metrics = xl.parse("RAW_INPUT_METRICS")
    orders  = xl.parse("RAW_ORDERS")
    metrics.columns = metrics.columns.str.strip().str.upper()
    orders.columns  = orders.columns.str.strip().str.upper()
    return metrics, orders


# ── Internal utility ──────────────────────────────────────────────────────────

def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return only the columns from `cols` that actually exist in `df`."""
    return [c for c in cols if c in df.columns]


# ── Dimension helpers ──────────────────────────────────────────────────────────

def get_countries(df: pd.DataFrame) -> list[str]:
    return sorted(df["COUNTRY"].dropna().unique().tolist())


def get_metrics(df: pd.DataFrame) -> list[str]:
    return sorted(df["METRIC"].dropna().unique().tolist())


def get_zone_types(df: pd.DataFrame) -> list[str]:
    if "ZONE_TYPE" not in df.columns:
        return []
    return sorted(df["ZONE_TYPE"].dropna().unique().tolist())


def get_prioritizations(df: pd.DataFrame) -> list[str]:
    if "ZONE_PRIORITIZATION" not in df.columns:
        return []
    return sorted(df["ZONE_PRIORITIZATION"].dropna().unique().tolist())


# ── Analytical helpers ─────────────────────────────────────────────────────────

def get_top_zones(
    df: pd.DataFrame,
    metric: str,
    country: str | None = None,
    week_col: str = "L0W_ROLL",
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    mask = df["METRIC"] == metric
    if country:
        mask &= df["COUNTRY"] == country
    select_cols = _present(df, ["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION", week_col])
    return (
        df.loc[mask, select_cols]
        .dropna(subset=[week_col])
        .sort_values(week_col, ascending=ascending)
        .head(n)
        .reset_index(drop=True)
    )


def get_weekly_trend(
    df: pd.DataFrame,
    metric: str,
    country: str | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    mask = df["METRIC"] == metric
    if country:
        mask &= df["COUNTRY"] == country
    cols = [c for c in WEEK_COLS_METRICS if c in df.columns]
    series = df.loc[mask, cols].agg(agg)
    return pd.DataFrame({"week": WEEK_LABELS[-len(cols):], "value": series.values})


def get_country_averages(
    df: pd.DataFrame,
    metric: str,
    week_col: str = "L0W_ROLL",
) -> pd.DataFrame:
    mask = df["METRIC"] == metric
    return (
        df.loc[mask, ["COUNTRY", week_col]]
        .dropna()
        .groupby("COUNTRY")[week_col]
        .mean()
        .reset_index()
        .rename(columns={week_col: "avg_value"})
        .sort_values("avg_value", ascending=False)
    )


def get_zone_comparison(
    df: pd.DataFrame,
    metric: str,
    country: str,
    week_col: str = "L0W_ROLL",
    n: int = 20,
) -> pd.DataFrame:
    mask = (df["METRIC"] == metric) & (df["COUNTRY"] == country)
    select_cols = _present(df, ["CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION", week_col])
    return (
        df.loc[mask, select_cols]
        .dropna(subset=[week_col])
        .sort_values(week_col, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def get_orders_trend(
    orders_df: pd.DataFrame,
    country: str | None = None,
) -> pd.DataFrame:
    mask = orders_df["METRIC"] == "Orders"
    if country:
        mask &= orders_df["COUNTRY"] == country
    cols = [c for c in WEEK_COLS_ORDERS if c in orders_df.columns]
    series = orders_df.loc[mask, cols].sum()
    return pd.DataFrame({"week": WEEK_LABELS[-len(cols):], "orders": series.values})


def get_orders_by_country(
    orders_df: pd.DataFrame,
    week_col: str = "L0W",
) -> pd.DataFrame:
    mask = orders_df["METRIC"] == "Orders"
    return (
        orders_df.loc[mask, ["COUNTRY", week_col]]
        .dropna()
        .groupby("COUNTRY")[week_col]
        .sum()
        .reset_index()
        .rename(columns={week_col: "total_orders"})
        .sort_values("total_orders", ascending=False)
    )


def get_metric_by_zone_type(
    df: pd.DataFrame,
    metric: str,
    country: str | None = None,
    week_col: str = "L0W_ROLL",
) -> pd.DataFrame:
    if "ZONE_TYPE" not in df.columns or week_col not in df.columns:
        return pd.DataFrame(columns=["ZONE_TYPE", "avg_value"])
    mask = df["METRIC"] == metric
    if country:
        mask &= df["COUNTRY"] == country
    return (
        df.loc[mask, ["ZONE_TYPE", week_col]]
        .dropna()
        .groupby("ZONE_TYPE")[week_col]
        .mean()
        .reset_index()
        .rename(columns={week_col: "avg_value"})
    )


# ── Summary for KPI cards ──────────────────────────────────────────────────────

def get_dataset_summary(metrics_df: pd.DataFrame, orders_df: pd.DataFrame) -> dict:
    orders_mask = orders_df["METRIC"] == "Orders"
    l0 = orders_df.loc[orders_mask, "L0W"].sum()
    l1 = orders_df.loc[orders_mask, "L1W"].sum()
    wow = round((l0 - l1) / l1 * 100, 1) if l1 else 0
    return {
        "countries":      metrics_df["COUNTRY"].nunique(),
        "cities":         metrics_df["CITY"].nunique(),
        "zones":          metrics_df["ZONE"].nunique(),
        "metrics_count":  metrics_df["METRIC"].nunique(),
        "total_orders":   int(l0),
        "orders_wow_pct": wow,
    }


# ── WoW zone analysis ─────────────────────────────────────────────────────────

def get_wow_zones(
    df: pd.DataFrame,
    metric: str,
    country: str | None = None,
    n: int = 5,
    ascending: bool = True,
) -> pd.DataFrame:
    """Return top-n zones sorted by WoW relative change. ascending=True → worst first."""
    if "L0W_ROLL" not in df.columns or "L1W_ROLL" not in df.columns:
        return pd.DataFrame()
    mask = df["METRIC"] == metric
    if country:
        mask &= df["COUNTRY"] == country
    select_cols = _present(
        df, ["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION", "L1W_ROLL", "L0W_ROLL"]
    )
    work = df.loc[mask, select_cols].dropna(subset=["L0W_ROLL", "L1W_ROLL"]).copy()
    work = work[work["L1W_ROLL"] != 0].copy()
    work["wow_pct"] = (work["L0W_ROLL"] - work["L1W_ROLL"]) / work["L1W_ROLL"].abs() * 100
    work = work[work["wow_pct"].between(-500, 500)]
    return work.sort_values("wow_pct", ascending=ascending).head(n).reset_index(drop=True)


# ── AI context serializer ──────────────────────────────────────────────────────

def df_to_context_string(df: pd.DataFrame, max_rows: int = 30) -> str:
    return (
        f"Shape: {len(df)} rows × {len(df.columns)} cols\n"
        f"Columns: {df.columns.tolist()}\n\n"
        f"Sample ({max_rows} rows):\n{df.head(max_rows).to_csv(index=False)}"
    )
