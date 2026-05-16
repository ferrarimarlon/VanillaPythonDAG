"""
Sales pipeline using real Olist Brazilian e-commerce data.

    extract_customers
    ├── validate_customers → dim_customers ──────────────────────┐
    │                                                             ├─→ fact_sales → publish_kpis
    ├── extract_orders ───────────────────────────────────────────┘
    │         └──────────→ quality_check (FAILS) → downstream_blocked
    └── extract_order_items ───────────────────────────────────────┘
"""

import os
import pandas as pd
from config import PipelineConfig
from vanilla_dag import DAG

# ── urls & local cache ─────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

URLS = {
    "customers":   "https://raw.githubusercontent.com/RG2021/E-Commerce-Data-Analytics/master/olist_customers_dataset.csv",
    "orders":      "https://raw.githubusercontent.com/RG2021/E-Commerce-Data-Analytics/master/olist_orders_dataset.csv",
    "order_items": "https://raw.githubusercontent.com/RG2021/E-Commerce-Data-Analytics/master/olist_order_items_dataset.csv",
}

def _load(key: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"olist_{key}_dataset.csv")
    if not os.path.exists(path):
        print(f"      downloading {os.path.basename(path)}...")
        df = pd.read_csv(URLS[key])
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)

# ── pipeline nodes ─────────────────────────────────────────────────────────────

def extract_customers(ctx: dict):
    cfg = ctx["config"]
    ctx["customers"] = _load("customers")
    df = ctx["customers"]

    missing = [c for c in cfg.required_customer_columns if c not in df.columns]
    if missing:
        raise ValueError(f"customers dataset missing required columns: {missing}")
    if len(df) < cfg.min_customers:
        raise ValueError(f"customers sanity check failed: {len(df):,} rows < minimum {cfg.min_customers:,}")

    print(f"      {len(df):,} records  |  columns: {list(df.columns)}")


def validate_customers(ctx: dict):
    df      = ctx["customers"]
    valid   = df.dropna(subset=["customer_id", "customer_state"])
    dropped = len(df) - len(valid)
    ctx["valid_customers"] = valid
    print(f"      {len(valid):,} valid  |  {dropped} dropped (null id or state)")


def extract_orders(ctx: dict):
    cfg = ctx["config"]
    ctx["orders"] = _load("orders")
    df = ctx["orders"]

    missing = [c for c in cfg.required_order_columns if c not in df.columns]
    if missing:
        raise ValueError(f"orders dataset missing required columns: {missing}")
    if len(df) < cfg.min_orders:
        raise ValueError(f"orders sanity check failed: {len(df):,} rows < minimum {cfg.min_orders:,}")

    print(f"      {len(df):,} orders  |  status breakdown: {df['order_status'].value_counts().to_dict()}")


def extract_order_items(ctx: dict):
    cfg = ctx["config"]
    ctx["order_items"] = _load("order_items")
    df = ctx["order_items"]

    missing = [c for c in cfg.required_item_columns if c not in df.columns]
    if missing:
        raise ValueError(f"order_items dataset missing required columns: {missing}")
    if len(df) < cfg.min_order_items:
        raise ValueError(f"order_items sanity check failed: {len(df):,} rows < minimum {cfg.min_order_items:,}")

    print(f"      {len(df):,} items  |  gross revenue: R$ {(df['price'] + df['freight_value']).sum():,.2f}")


def dim_customers(ctx: dict):
    ctx["dim_customers"] = (
        ctx["valid_customers"][["customer_id", "customer_state", "customer_city"]]
        .drop_duplicates("customer_id")
        .set_index("customer_id")
    )
    states = ctx["dim_customers"]["customer_state"].nunique()
    print(f"      {len(ctx['dim_customers']):,} customers indexed across {states} states")


def quality_check(ctx: dict):
    df = ctx["orders"].copy()
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")

    delivered = df[df["order_status"] == "delivered"].dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    )
    late = delivered[
        delivered["order_delivered_customer_date"] > delivered["order_estimated_delivery_date"]
    ]
    rate      = len(late) / len(delivered)
    threshold = ctx["config"].max_late_delivery_rate

    print(f"      {len(late):,}/{len(delivered):,} delivered orders were late ({rate:.1%})  [threshold: {threshold:.0%}]")
    if rate > threshold:
        raise ValueError(f"SLA check failed: {rate:.1%} late delivery rate exceeds {threshold:.0%} threshold")

    ctx["on_time_orders"] = delivered[
        delivered["order_delivered_customer_date"] <= delivered["order_estimated_delivery_date"]
    ]


def downstream_blocked(ctx: dict):
    print("      detailed SLA analysis — blocked by quality gate")


def fact_sales(ctx: dict):
    dim    = ctx["dim_customers"].reset_index()   # customer_id, customer_state, customer_city
    orders = ctx["orders"].copy()
    items  = ctx["order_items"].copy()

    # revenue per order (sum of price + freight across all items)
    revenue_per_order = (
        items.assign(revenue=items["price"] + items["freight_value"])
        .groupby("order_id", as_index=False)["revenue"].sum()
    )

    # join: orders × dim_customers → state per order
    orders_with_state = orders.merge(
        dim[["customer_id", "customer_state"]], on="customer_id", how="left"
    )

    # join: orders × revenue
    orders_full = orders_with_state.merge(revenue_per_order, on="order_id", how="left")
    orders_full["revenue"] = orders_full["revenue"].fillna(0)

    # late delivery flag
    orders_full["order_delivered_customer_date"] = pd.to_datetime(
        orders_full["order_delivered_customer_date"], errors="coerce"
    )
    orders_full["order_estimated_delivery_date"] = pd.to_datetime(
        orders_full["order_estimated_delivery_date"], errors="coerce"
    )
    delivered = orders_full["order_status"] == "delivered"
    orders_full["late"] = (
        delivered
        & (orders_full["order_delivered_customer_date"] > orders_full["order_estimated_delivery_date"])
    )

    fact = (
        orders_full.groupby("customer_state").agg(
            orders    =("order_id",      "count"),
            delivered =("order_status",  lambda s: (s == "delivered").sum()),
            late      =("late",          "sum"),
            revenue   =("revenue",       "sum"),
        )
        .reset_index()
        .rename(columns={"customer_state": "state"})
        .sort_values("revenue", ascending=False)
    )

    ctx["fact_sales"] = fact
    print(f"      fact materialized — {len(fact)} states, {fact['orders'].sum():,} orders, R$ {fact['revenue'].sum():,.2f}")


def publish_kpis(ctx: dict):
    fact = ctx["fact_sales"]

    total_revenue   = fact["revenue"].sum()
    total_orders    = fact["orders"].sum()
    total_delivered = fact["delivered"].sum()
    total_late      = fact["late"].sum()

    print(f"      total revenue     : R$ {total_revenue:>14,.2f}")
    print(f"      total orders      : {total_orders:,}")
    print(f"      delivered         : {total_delivered:,} ({total_delivered/total_orders:.1%})")
    print(f"      late delivery rate: {total_late/total_delivered:.1%}")
    print()
    print(f"      {'STATE':<8} {'REVENUE (R$)':>14} {'ORDERS':>8} {'LATE':>8}")
    print(f"      {'─'*8} {'─'*14} {'─'*8} {'─'*8}")
    for _, r in fact.head(10).iterrows():
        late_rate = r["late"] / r["delivered"] if r["delivered"] else 0
        print(f"      {r['state']:<8} {r['revenue']:>14,.2f} {r['orders']:>8,} {late_rate:>7.1%}")
    if len(fact) > 10:
        print(f"      ... ({len(fact) - 10} more states)")


# ── graph factory ──────────────────────────────────────────────────────────────

def build_dag(config: PipelineConfig) -> DAG:
    dag = DAG(name="olist_sales_pipeline")

    dag.add_node("extract_customers",   extract_customers)
    dag.add_node("validate_customers",  validate_customers)
    dag.add_node("extract_orders",      extract_orders)
    dag.add_node("extract_order_items", extract_order_items)
    dag.add_node("dim_customers",       dim_customers)
    dag.add_node("quality_check",       quality_check)
    dag.add_node("downstream_blocked",  downstream_blocked)
    dag.add_node("fact_sales",          fact_sales)
    dag.add_node("publish_kpis",        publish_kpis)

    dag.add_edge("extract_customers",   "validate_customers")
    dag.add_edge("extract_customers",   "extract_orders")
    dag.add_edge("extract_customers",   "extract_order_items")
    dag.add_edge("validate_customers",  "dim_customers")
    dag.add_edge("extract_orders",      "quality_check")
    dag.add_edge("extract_orders",      "fact_sales")
    dag.add_edge("extract_order_items", "fact_sales")
    dag.add_edge("quality_check",       "downstream_blocked")
    dag.add_edge("dim_customers",       "fact_sales")
    dag.add_edge("fact_sales",          "publish_kpis")

    dag.context["config"] = config
    return dag


if __name__ == "__main__":
    dag = build_dag(PipelineConfig())
    dag.run()
