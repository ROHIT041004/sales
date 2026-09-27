"""
reorder.py
----------
Inventory math shared by the dashboard:
  - a deterministic SIMULATED on-hand stock level per SKU (the raw dataset
    has no real stock-on-hand column, so we derive a stable pseudo-random
    figure from the Item_Identifier so numbers don't jump around on rerun)
  - classic reorder-point / safety-stock / EOQ formulas
  - financial impact (revenue at risk, holding cost, reorder spend)

Swap `simulate_current_stock()` for a real inventory feed (DB/API) when
you have one -- everything downstream (reorder point, EOQ, $ impact)
already works off a plain "current_stock" number.
"""

import hashlib
import numpy as np
import pandas as pd

LEAD_TIME_DAYS = 7          # supplier lead time assumption
SERVICE_Z = 1.65            # ~95% service level
SALES_WINDOW_DAYS = 90      # Item_Outlet_Sales treated as trailing-90-day revenue
HOLDING_COST_RATE = 0.02    # % of item value per month to hold stock
ORDER_COST_FLAT = 250.0     # ₹ flat cost per purchase order (admin/shipping)
STOCKOUT_DAYS_ASSUMED = 5   # if nothing is done, assume 5 days of lost sales


def _stable_fraction(key: str) -> float:
    """Deterministic 0..1 float from a string, so simulated stock is stable."""
    h = hashlib.md5(str(key).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def simulate_current_stock(row) -> float:
    """Simulated on-hand units: bigger/slower-selling items tend to carry
    more buffer stock; this is illustrative only -- replace with real data."""
    daily_demand = estimate_daily_demand(row)
    frac = _stable_fraction(row.get("Item_Identifier", str(row.get("Item_MRP", 0))))
    days_of_cover = 2 + frac * 18  # between 2 and 20 days of cover on hand
    return max(0.0, round(daily_demand * days_of_cover, 1))


def estimate_daily_demand(row) -> float:
    price = max(float(row.get("Item_MRP", 1)), 1.0)
    sales = float(row.get("Item_Outlet_Sales", price * 5))
    units_sold_in_window = sales / price
    return max(units_sold_in_window / SALES_WINDOW_DAYS, 0.05)


def reorder_point(daily_demand: float, lead_time_days: float = LEAD_TIME_DAYS,
                   demand_std_ratio: float = 0.4) -> float:
    """ROP = lead-time demand + safety stock (safety stock scales with
    demand volatility, approximated as a fraction of mean demand)."""
    demand_std = daily_demand * demand_std_ratio
    safety_stock = SERVICE_Z * demand_std * np.sqrt(lead_time_days)
    return daily_demand * lead_time_days + safety_stock


def economic_order_qty(daily_demand: float, price: float,
                        order_cost: float = ORDER_COST_FLAT,
                        holding_cost_rate: float = HOLDING_COST_RATE) -> float:
    annual_demand = max(daily_demand * 365, 1.0)
    holding_cost_per_unit = max(price * holding_cost_rate * 12, 0.01)  # annualized
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
    return max(eoq, 1.0)


def build_reorder_table(df: pd.DataFrame, risk_proba: np.ndarray,
                         risk_threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out["Risk_Probability"] = risk_proba
    out["Daily_Demand_Est"] = out.apply(estimate_daily_demand, axis=1)
    out["Current_Stock_Sim"] = out.apply(simulate_current_stock, axis=1)
    out["Reorder_Point"] = out["Daily_Demand_Est"].apply(reorder_point)
    out["EOQ_Suggested_Qty"] = out.apply(
        lambda r: economic_order_qty(r["Daily_Demand_Est"], max(r["Item_MRP"], 1)), axis=1
    )
    out["Days_Of_Cover"] = (out["Current_Stock_Sim"] / out["Daily_Demand_Est"]).round(1)
    out["Needs_Reorder"] = (out["Current_Stock_Sim"] < out["Reorder_Point"]) | (
        out["Risk_Probability"] >= risk_threshold
    )
    out["Suggested_Order_Qty"] = np.where(
        out["Needs_Reorder"],
        np.ceil(out["EOQ_Suggested_Qty"]),
        0,
    )
    out["Revenue_At_Risk"] = np.where(
        out["Needs_Reorder"],
        out["Daily_Demand_Est"] * out["Item_MRP"] * STOCKOUT_DAYS_ASSUMED * out["Risk_Probability"],
        0.0,
    )
    out["Holding_Cost_Monthly"] = out["Current_Stock_Sim"] * out["Item_MRP"] * HOLDING_COST_RATE
    out["Reorder_Spend"] = out["Suggested_Order_Qty"] * out["Item_MRP"]
    return out


def financial_summary(reorder_df: pd.DataFrame) -> dict:
    critical = reorder_df[reorder_df["Needs_Reorder"]]
    return {
        "items_at_risk": int(len(critical)),
        "total_revenue_at_risk": round(critical["Revenue_At_Risk"].sum(), 2),
        "total_reorder_spend": round(critical["Reorder_Spend"].sum(), 2),
        "total_monthly_holding_cost": round(reorder_df["Holding_Cost_Monthly"].sum(), 2),
        "avg_days_of_cover_critical": round(critical["Days_Of_Cover"].mean(), 1) if len(critical) else 0,
    }
