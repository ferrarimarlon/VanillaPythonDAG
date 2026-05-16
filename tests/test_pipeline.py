"""Tests for pipeline.py — each node function tested in isolation with synthetic data."""

import pytest
import pandas as pd
from unittest.mock import patch
from config import PipelineConfig
import pipeline as pl


# ── synthetic dataset fixtures ─────────────────────────────────────────────────

@pytest.fixture
def customers_df():
    return pd.DataFrame({
        "customer_id":           ["c1", "c2", "c3", None],
        "customer_unique_id":    ["u1", "u2", "u3", "u4"],
        "customer_zip_code_prefix": ["01001", "20001", "30001", "40001"],
        "customer_city":         ["sao paulo", "rio de janeiro", "belo horizonte", "salvador"],
        "customer_state":        ["SP", "RJ", "MG", None],
    })

@pytest.fixture
def orders_df():
    # o1, o2, o4 on time — o3 late — o5 not delivered
    return pd.DataFrame({
        "order_id":                       ["o1",          "o2",          "o3",          "o4",          "o5"],
        "customer_id":                    ["c1",          "c2",          "c3",          "c1",          "c2"],
        "order_status":                   ["delivered",   "delivered",   "delivered",   "delivered",   "shipped"],
        "order_purchase_timestamp":       ["2024-01-01"] * 5,
        "order_approved_at":              ["2024-01-01"] * 5,
        "order_delivered_carrier_date":   ["2024-01-05"] * 5,
        "order_delivered_customer_date":  ["2024-01-10", "2024-01-10", "2024-01-25", "2024-01-10", None],
        "order_estimated_delivery_date":  ["2024-01-15", "2024-01-15", "2024-01-20", "2024-01-15", "2024-01-20"],
    })
    # delivered: o1, o2, o3, o4 (4 orders) — late: o3 → rate = 25%

@pytest.fixture
def order_items_df():
    return pd.DataFrame({
        "order_id":           ["o1",  "o2",  "o3",  "o4"],
        "order_item_id":      [1,     1,     1,     1],
        "product_id":         ["p1",  "p2",  "p3",  "p1"],
        "seller_id":          ["s1",  "s2",  "s3",  "s1"],
        "shipping_limit_date":["2024-01-05"] * 4,
        "price":              [100.0, 200.0, 150.0, 50.0],
        "freight_value":      [10.0,  20.0,  15.0,  5.0],
    })
    # revenues: o1=110, o2=220, o3=165, o4=55
    # SP (c1): o1+o4 = 165 | RJ (c2): o2 = 220 | MG (c3): o3 = 165

@pytest.fixture
def cfg():
    return PipelineConfig(
        min_customers=1,
        min_orders=1,
        min_order_items=1,
    )

@pytest.fixture
def base_ctx(customers_df, orders_df, order_items_df, cfg):
    """Context pre-loaded with all datasets and a lenient config."""
    valid = customers_df.dropna(subset=["customer_id", "customer_state"])
    dim   = (
        valid[["customer_id", "customer_state", "customer_city"]]
        .drop_duplicates("customer_id")
        .set_index("customer_id")
    )
    return {
        "config":          cfg,
        "customers":       customers_df,
        "valid_customers": valid,
        "orders":          orders_df,
        "order_items":     order_items_df,
        "dim_customers":   dim,
    }


# ── extract_customers ──────────────────────────────────────────────────────────

def test_extract_customers_loads_dataframe(customers_df, cfg):
    ctx = {"config": cfg}
    with patch("pipeline._load", return_value=customers_df):
        pl.extract_customers(ctx)
    assert isinstance(ctx["customers"], pd.DataFrame)
    assert len(ctx["customers"]) == 4


def test_extract_customers_fails_on_missing_column(cfg):
    bad = pd.DataFrame({"customer_id": ["c1"], "customer_city": ["sp"]})  # missing customer_state
    ctx = {"config": cfg}
    with patch("pipeline._load", return_value=bad):
        with pytest.raises(ValueError, match="missing required columns"):
            pl.extract_customers(ctx)


def test_extract_customers_fails_below_min_rows(customers_df):
    ctx = {"config": PipelineConfig(min_customers=100_000)}
    with patch("pipeline._load", return_value=customers_df):
        with pytest.raises(ValueError, match="sanity check failed"):
            pl.extract_customers(ctx)


# ── validate_customers ─────────────────────────────────────────────────────────

def test_validate_customers_drops_nulls(customers_df, cfg):
    ctx = {"config": cfg, "customers": customers_df}
    pl.validate_customers(ctx)
    assert len(ctx["valid_customers"]) == 3  # 1 row dropped (null customer_id and state on same row)
    assert ctx["valid_customers"]["customer_id"].notna().all()
    assert ctx["valid_customers"]["customer_state"].notna().all()


def test_validate_customers_all_valid(cfg):
    clean = pd.DataFrame({
        "customer_id":    ["c1", "c2"],
        "customer_state": ["SP", "RJ"],
    })
    ctx = {"config": cfg, "customers": clean}
    pl.validate_customers(ctx)
    assert len(ctx["valid_customers"]) == 2


# ── extract_orders ─────────────────────────────────────────────────────────────

def test_extract_orders_loads_dataframe(orders_df, cfg):
    ctx = {"config": cfg}
    with patch("pipeline._load", return_value=orders_df):
        pl.extract_orders(ctx)
    assert isinstance(ctx["orders"], pd.DataFrame)
    assert len(ctx["orders"]) == 5


def test_extract_orders_fails_on_missing_column(cfg):
    bad = pd.DataFrame({"order_id": ["o1"], "customer_id": ["c1"]})  # missing order_status etc.
    ctx = {"config": cfg}
    with patch("pipeline._load", return_value=bad):
        with pytest.raises(ValueError, match="missing required columns"):
            pl.extract_orders(ctx)


def test_extract_orders_fails_below_min_rows(orders_df):
    ctx = {"config": PipelineConfig(min_orders=100_000)}
    with patch("pipeline._load", return_value=orders_df):
        with pytest.raises(ValueError, match="sanity check failed"):
            pl.extract_orders(ctx)


# ── extract_order_items ────────────────────────────────────────────────────────

def test_extract_order_items_loads_dataframe(order_items_df, cfg):
    ctx = {"config": cfg}
    with patch("pipeline._load", return_value=order_items_df):
        pl.extract_order_items(ctx)
    assert isinstance(ctx["order_items"], pd.DataFrame)
    assert len(ctx["order_items"]) == 4


def test_extract_order_items_fails_on_missing_column(cfg):
    bad = pd.DataFrame({"order_id": ["o1"], "price": [10.0]})  # missing freight_value
    ctx = {"config": cfg}
    with patch("pipeline._load", return_value=bad):
        with pytest.raises(ValueError, match="missing required columns"):
            pl.extract_order_items(ctx)


def test_extract_order_items_fails_below_min_rows(order_items_df):
    ctx = {"config": PipelineConfig(min_order_items=100_000)}
    with patch("pipeline._load", return_value=order_items_df):
        with pytest.raises(ValueError, match="sanity check failed"):
            pl.extract_order_items(ctx)


# ── dim_customers ──────────────────────────────────────────────────────────────

def test_dim_customers_builds_index(base_ctx):
    pl.dim_customers(base_ctx)
    dim = base_ctx["dim_customers"]
    assert "customer_state" in dim.columns
    assert dim.index.name == "customer_id"
    assert dim.loc["c1", "customer_state"] == "SP"
    assert dim.loc["c2", "customer_state"] == "RJ"


def test_dim_customers_deduplicates(cfg):
    dupes = pd.DataFrame({
        "customer_id":    ["c1", "c1", "c2"],
        "customer_state": ["SP", "SP", "RJ"],
        "customer_city":  ["sao paulo", "sao paulo", "rio"],
    })
    ctx = {"config": cfg, "valid_customers": dupes}
    pl.dim_customers(ctx)
    assert len(ctx["dim_customers"]) == 2


# ── quality_check ──────────────────────────────────────────────────────────────

def test_quality_check_fails_above_threshold(base_ctx):
    # fixture has 25% late — default threshold is 5%
    base_ctx["config"] = PipelineConfig(max_late_delivery_rate=0.05, min_orders=1)
    with pytest.raises(ValueError, match="SLA check failed"):
        pl.quality_check(base_ctx)


def test_quality_check_passes_below_threshold(base_ctx):
    base_ctx["config"] = PipelineConfig(max_late_delivery_rate=0.30, min_orders=1)
    pl.quality_check(base_ctx)  # should not raise
    assert "on_time_orders" in base_ctx


def test_quality_check_on_time_orders_excludes_late(base_ctx):
    base_ctx["config"] = PipelineConfig(max_late_delivery_rate=0.30, min_orders=1)
    pl.quality_check(base_ctx)
    on_time = base_ctx["on_time_orders"]
    # o3 was late — must not appear in on_time_orders
    assert "o3" not in on_time["order_id"].values


# ── fact_sales ─────────────────────────────────────────────────────────────────

def test_fact_sales_produces_one_row_per_state(base_ctx):
    pl.fact_sales(base_ctx)
    states = set(base_ctx["fact_sales"]["state"])
    assert states == {"SP", "RJ", "MG"}


def test_fact_sales_revenue_per_state(base_ctx):
    pl.fact_sales(base_ctx)
    fact = base_ctx["fact_sales"].set_index("state")
    # SP: c1 → o1(110) + o4(55) = 165
    assert abs(fact.loc["SP", "revenue"] - 165.0) < 0.01
    # RJ: c2 → o2(220)
    assert abs(fact.loc["RJ", "revenue"] - 220.0) < 0.01
    # MG: c3 → o3(165)
    assert abs(fact.loc["MG", "revenue"] - 165.0) < 0.01


def test_fact_sales_total_orders(base_ctx):
    pl.fact_sales(base_ctx)
    fact = base_ctx["fact_sales"]
    assert fact["orders"].sum() == 5  # all 5 orders in fixture


def test_fact_sales_late_flag(base_ctx):
    pl.fact_sales(base_ctx)
    fact = base_ctx["fact_sales"].set_index("state")
    # o3 (MG, c3) is the only late delivered order
    assert fact.loc["MG", "late"] == 1
    assert fact.loc["SP", "late"] == 0
    assert fact.loc["RJ", "late"] == 0


# ── publish_kpis ───────────────────────────────────────────────────────────────

def test_publish_kpis_runs_without_error(base_ctx, capsys):
    pl.fact_sales(base_ctx)
    pl.publish_kpis(base_ctx)
    out = capsys.readouterr().out
    assert "total revenue" in out
    assert "total orders" in out
    assert "late delivery rate" in out


def test_publish_kpis_correct_totals(base_ctx, capsys):
    pl.fact_sales(base_ctx)
    pl.publish_kpis(base_ctx)
    out = capsys.readouterr().out
    assert "550.00" in out   # total revenue: 165 + 220 + 165
    assert "5" in out        # total orders


# ── config ─────────────────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = PipelineConfig()
    assert cfg.max_late_delivery_rate == 0.05
    assert cfg.min_customers == 50_000
    assert cfg.min_orders == 50_000
    assert cfg.min_order_items == 50_000
    assert "customer_id" in cfg.required_customer_columns
    assert "order_id" in cfg.required_order_columns
    assert "price" in cfg.required_item_columns


def test_config_override():
    cfg = PipelineConfig(max_late_delivery_rate=0.10, min_orders=1_000)
    assert cfg.max_late_delivery_rate == 0.10
    assert cfg.min_orders == 1_000
    assert cfg.min_customers == 50_000  # unchanged
