from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # ── quality thresholds ─────────────────────────────────────────────────────
    max_late_delivery_rate: float = 0.05   # fail if late deliveries exceed this share

    # ── sanity checks ──────────────────────────────────────────────────────────
    min_customers:   int = 50_000
    min_orders:      int = 50_000
    min_order_items: int = 50_000

    required_customer_columns: list[str] = field(default_factory=lambda: [
        "customer_id", "customer_state", "customer_city",
    ])
    required_order_columns: list[str] = field(default_factory=lambda: [
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp", "order_estimated_delivery_date",
    ])
    required_item_columns: list[str] = field(default_factory=lambda: [
        "order_id", "price", "freight_value",
    ])
