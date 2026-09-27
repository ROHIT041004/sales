"""
StockSense AI -- Item Risk Scoring & Reorder Dashboard
Run with:  streamlit run app.py
"""

import json
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from alerts import send_email_alert, send_sms_alert, send_slack_alert, format_risk_alert
from reorder import build_reorder_table, financial_summary
from train_model import load_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, MODEL_PATH

DB_PATH = "stocksense.db"

# ---------------------------------------------------------------- page/style
st.set_page_config(page_title="StockSense AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b0f1a; color: #e6e8ef; }
section[data-testid="stSidebar"] { background-color: #0e1320; }
div[data-testid="stMetricValue"] { color: #7c8cf8; }
.risk-badge-high { background:#3a1420; color:#ff6b81; padding:6px 14px; border-radius:8px; font-weight:600; }
.risk-badge-low { background:#0f2e28; color:#3ddc97; padding:6px 14px; border-radius:8px; font-weight:600; }
.card { background:#111729; border:1px solid #1f2740; border-radius:12px; padding:18px 20px; }
h1, h2, h3 { color:#f3f4fb; }
</style>
""", unsafe_allow_html=True)

st.markdown("### ⚡ StockSense AI &nbsp;·&nbsp; <span style='color:#8a8fa3;font-size:0.6em'>SMART INVENTORY, SMARTER DECISIONS</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------- data / model loading
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS alert_log (
        ts TEXT, channel TEXT, item_id TEXT, message TEXT, success INTEGER)""")
    conn.commit()
    return conn


def save_uploaded_csv_to_db(df: pd.DataFrame):
    conn = init_db()
    df.to_sql("inventory_snapshot", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def load_inventory_from_db():
    conn = init_db()
    try:
        df = pd.read_sql("SELECT * FROM inventory_snapshot", conn)
    except Exception:
        df = None
    conn.close()
    return df


def log_alert(channel, item_id, message, success):
    conn = init_db()
    conn.execute(
        "INSERT INTO alert_log VALUES (?,?,?,?,?)",
        (datetime.utcnow().isoformat(), channel, item_id, message, int(success)),
    )
    conn.commit()
    conn.close()


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error("risk_model.pkl not found. Run `python train_model.py` first (see README).")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_feature_importance():
    if os.path.exists("feature_importance.json"):
        with open("feature_importance.json") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_default_data():
    return load_data("bigmart_with_subtypes.csv")


if "df" not in st.session_state:
    st.session_state.df = get_default_data()
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

model = get_model()
feat_importance = get_feature_importance()

# ---------------------------------------------------------------- sidebar: data source & alert config
with st.sidebar:
    st.markdown("#### 📥 Data Source")
    src = st.radio("Choose data", ["Bundled dataset", "Upload CSV", "Load from local DB"], label_visibility="collapsed")

    if src == "Upload CSV":
        up = st.file_uploader("Upload inventory CSV", type=["csv"])
        if up is not None:
            new_df = pd.read_csv(up, encoding="utf-8-sig")
            missing = [c for c in ALL_FEATURES if c not in new_df.columns]
            if missing:
                st.error(f"CSV is missing required columns: {missing}")
            else:
                if "Item_Outlet_Sales" not in new_df.columns:
                    new_df["Item_Outlet_Sales"] = new_df["Item_MRP"] * 20  # rough fallback
                st.session_state.df = new_df
                st.success(f"Loaded {len(new_df)} rows")
                if st.button("💾 Save as live database snapshot"):
                    save_uploaded_csv_to_db(new_df)
                    st.success("Saved to stocksense.db")
    elif src == "Load from local DB":
        db_df = load_inventory_from_db()
        if db_df is not None:
            st.session_state.df = db_df
            st.success(f"Loaded {len(db_df)} rows from stocksense.db")
        else:
            st.warning("No snapshot saved yet. Upload a CSV and click 'Save as live database snapshot' first.")
    else:
        st.session_state.df = get_default_data()

    st.caption(f"Active dataset: **{len(st.session_state.df)}** SKUs")

    st.markdown("---")
    st.markdown("#### 🔔 Alert Channels")
    with st.expander("Email (SMTP)"):
        st.session_state.smtp_host = st.text_input("SMTP host", value=st.session_state.get("smtp_host", "smtp.gmail.com"))
        st.session_state.smtp_port = st.text_input("SMTP port", value=st.session_state.get("smtp_port", "587"))
        st.session_state.smtp_user = st.text_input("SMTP username / from-address", value=st.session_state.get("smtp_user", ""))
        st.session_state.smtp_pass = st.text_input("SMTP password / app-password", type="password", value=st.session_state.get("smtp_pass", ""))
        st.session_state.email_to = st.text_input("Send alerts to", value=st.session_state.get("email_to", ""))
    with st.expander("SMS (Twilio)"):
        st.session_state.tw_sid = st.text_input("Account SID", value=st.session_state.get("tw_sid", ""))
        st.session_state.tw_token = st.text_input("Auth token", type="password", value=st.session_state.get("tw_token", ""))
        st.session_state.tw_from = st.text_input("From number", value=st.session_state.get("tw_from", ""))
        st.session_state.tw_to = st.text_input("To number", value=st.session_state.get("tw_to", ""))
    with st.expander("Slack"):
        st.session_state.slack_webhook = st.text_input("Incoming Webhook URL", value=st.session_state.get("slack_webhook", ""))

    st.caption("Credentials stay in this session only. For repeat use, put them in `.streamlit/secrets.toml` instead (see README).")


def dispatch_alert(channel, item_id, message):
    if channel == "Email":
        ok, msg = send_email_alert(
            st.session_state.get("smtp_host"), st.session_state.get("smtp_port"),
            st.session_state.get("smtp_user"), st.session_state.get("smtp_pass"),
            st.session_state.get("email_to"), f"StockSense Alert: {item_id}", message,
        )
    elif channel == "SMS":
        ok, msg = send_sms_alert(
            st.session_state.get("tw_sid"), st.session_state.get("tw_token"),
            st.session_state.get("tw_from"), st.session_state.get("tw_to"), message,
        )
    else:
        ok, msg = send_slack_alert(st.session_state.get("slack_webhook"), message)
    log_alert(channel, item_id, msg, ok)
    st.session_state.alert_log.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "channel": channel,
                                           "item": item_id, "status": "✅" if ok else "❌", "detail": msg})
    return ok, msg


# ---------------------------------------------------------------- tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Item Risk Scoring", "📦 Reorder Suggestions", "🧪 What-If Simulator",
    "🔔 Alerts", "💰 Financial Impact",
])

df = st.session_state.df
cat_options = sorted(df["Item_Type"].unique())
subtype_map = df.groupby("Item_Type")["Item_Sub_Type"].unique().apply(sorted).to_dict()
outlet_ids = sorted(df["Outlet_Identifier"].unique()) if "Outlet_Identifier" in df.columns else []


def predict_risk(row_dict):
    row_df = pd.DataFrame([row_dict])[ALL_FEATURES]
    proba = model.predict_proba(row_df)[0, 1]
    return proba


# ================================================================ TAB 1: Item Risk Scoring
with tab1:
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("##### 📋 SKU INGESTION — Item Attributes & Outlet Profile")
        c1, c2 = st.columns(2)
        weight = c1.number_input("Item Weight (kg)", 0.1, 50.0, 12.5, step=0.1)
        mrp = c2.number_input("Maximum Retail Price (₹)", 1.0, 5000.0, 150.0, step=1.0)

        c3, c4 = st.columns(2)
        item_type = c3.selectbox("Item Category", cat_options, index=cat_options.index("Fruits and Vegetables") if "Fruits and Vegetables" in cat_options else 0)
        sub_options = subtype_map.get(item_type, ["Other"])
        sub_type = c4.selectbox("Subcategory", sub_options)

        fat = st.radio("Fat Content Classification", ["Low Fat", "Regular", "High Fat"], horizontal=True)
        if fat == "High Fat":
            st.caption("⚠️ 'High Fat' isn't in the training data — the model treats it as an unknown category.")

        visibility = st.slider("Shelf Visibility", 0.0, 0.30, 0.070, step=0.001)

        st.markdown("##### 🏬 Outlet Profile")
        c5, c6 = st.columns(2)
        outlet_size = c5.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_loc = c6.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        c7, c8 = st.columns(2)
        outlet_type = c7.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
        outlet_year = c8.number_input("Outlet Establishment Year", 1980, 2026, 2007)

        score_clicked = st.button("⚡ Score This Item", type="primary", use_container_width=True)

    with right:
        st.markdown("##### 🎯 LIVE RISK SCORE")
        card = st.container()
        if score_clicked:
            row = {
                "Item_Weight": weight, "Item_MRP": mrp, "Item_Visibility": visibility,
                "Item_Fat_Content": fat, "Item_Type": item_type, "Item_Sub_Type": sub_type,
                "Outlet_Size": outlet_size, "Outlet_Location_Type": outlet_loc, "Outlet_Type": outlet_type,
            }
            proba = predict_risk(row)
            st.session_state.last_scored_row = row
            st.session_state.last_proba = proba

            badge = "risk-badge-high" if proba >= 0.5 else "risk-badge-low"
            label = "HIGH RISK" if proba >= 0.5 else "LOW RISK"
            card.markdown(f"""
            <div class="card" style="text-align:center;">
              <div style="font-size:2.4em;font-weight:700;">{proba:.0%}</div>
              <div style="color:#8a8fa3;">RISK PROBABILITY</div>
              <div style="margin-top:10px;"><span class="{badge}">{label}</span></div>
            </div>
            """, unsafe_allow_html=True)

            from reorder import estimate_daily_demand, simulate_current_stock, reorder_point, economic_order_qty
            demand = estimate_daily_demand(row | {"Item_Outlet_Sales": mrp * 20})
            stock = simulate_current_stock(row | {"Item_Identifier": f"{item_type}-{sub_type}", "Item_Outlet_Sales": mrp * 20})
            rop = reorder_point(demand)
            eoq = economic_order_qty(demand, mrp)
            st.markdown("###### 📦 Order Suggestion")
            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Simulated Stock", f"{stock:.0f} u")
            oc2.metric("Reorder Point", f"{rop:.0f} u")
            oc3.metric("Suggested Order Qty", f"{eoq:.0f} u" if stock < rop or proba >= 0.5 else "0 u")

            revenue_at_risk = demand * mrp * 5 * proba
            st.markdown("###### 💰 Financial Impact")
            fc1, fc2 = st.columns(2)
            fc1.metric("Revenue at risk (5-day stockout)", f"₹{revenue_at_risk:,.0f}")
            fc2.metric("Est. reorder spend", f"₹{eoq * mrp:,.0f}")

            if proba >= 0.5:
                if st.button("🔔 Send alert for this item"):
                    msg = format_risk_alert(f"{item_type}/{sub_type}", item_type, proba, eoq, revenue_at_risk)
                    ok, detail = dispatch_alert("Slack", f"{item_type}/{sub_type}", msg)
                    (st.success if ok else st.error)(detail)
        else:
            card.markdown("""
            <div class="card" style="text-align:center;color:#8a8fa3;">
              <div style="font-size:2em;">—</div>
              RISK PROBABILITY<br><br>
              <em>Fill in the form and click "Score This Item"</em>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("###### TOP MODEL SIGNALS (global feature importance)")
        if feat_importance:
            top3 = list(feat_importance.items())[:3]
            cols = st.columns(len(top3))
            for c, (name, pct) in zip(cols, top3):
                c.metric(name, f"{pct}%")
        st.caption("These are the model's overall most-relied-on fields, not a per-item explanation.")


# ================================================================ TAB 2: Reorder Suggestions
with tab2:
    st.markdown("##### 📦 Automated Stock Reordering & Order Suggestions")
    st.caption("Runs the risk model across every SKU in the active dataset, simulates current stock, and flags items that need a purchase order.")

    threshold = st.slider("Risk threshold to flag for reorder", 0.0, 1.0, 0.5, 0.05)

    run = st.button("▶️ Run reorder scan on active dataset", type="primary")
    if run or "reorder_table" in st.session_state:
        if run:
            proba_all = model.predict_proba(df[ALL_FEATURES])[:, 1]
            st.session_state.reorder_table = build_reorder_table(df, proba_all, threshold)
        rt = st.session_state.reorder_table
        critical = rt[rt["Needs_Reorder"]].sort_values("Risk_Probability", ascending=False)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SKUs scanned", len(rt))
        m2.metric("Flagged for reorder", len(critical))
        m3.metric("Total suggested spend", f"₹{critical['Reorder_Spend'].sum():,.0f}")
        m4.metric("Total revenue at risk", f"₹{critical['Revenue_At_Risk'].sum():,.0f}")

        show_cols = ["Item_Identifier", "Item_Type", "Item_Sub_Type", "Outlet_Identifier",
                     "Risk_Probability", "Current_Stock_Sim", "Reorder_Point", "Days_Of_Cover",
                     "Suggested_Order_Qty", "Revenue_At_Risk"]
        show_cols = [c for c in show_cols if c in critical.columns]
        st.dataframe(
            critical[show_cols].head(200).style.format({
                "Risk_Probability": "{:.0%}", "Current_Stock_Sim": "{:.0f}",
                "Reorder_Point": "{:.0f}", "Suggested_Order_Qty": "{:.0f}",
                "Revenue_At_Risk": "₹{:,.0f}",
            }),
            use_container_width=True, height=420,
        )

        colA, colB = st.columns(2)
        with colA:
            channel = st.selectbox("Alert channel for bulk send", ["Slack", "Email", "SMS"], key="bulk_channel")
        with colB:
            top_n = st.number_input("Send alerts for top N critical items", 1, 50, 5)
        if st.button("🔔 Send bulk alerts"):
            sent, failed = 0, 0
            for _, r in critical.head(int(top_n)).iterrows():
                msg = format_risk_alert(r["Item_Identifier"], r["Item_Type"], r["Risk_Probability"],
                                         r["Suggested_Order_Qty"], r["Revenue_At_Risk"])
                ok, _ = dispatch_alert(channel, r["Item_Identifier"], msg)
                sent += ok
                failed += (not ok)
            st.success(f"Sent {sent} alert(s), {failed} failed. See the Alerts tab for details.")

        st.download_button("⬇️ Download full reorder plan (CSV)", rt.to_csv(index=False), "reorder_plan.csv")


# ================================================================ TAB 3: What-If Simulator
with tab3:
    st.markdown("##### 🧪 What-If Scenario Simulation")
    st.caption("Take the item scored in Tab 1 (or a quick default) and see how risk & financial impact respond to changes.")

    base = st.session_state.get("last_scored_row", {
        "Item_Weight": 12.5, "Item_MRP": 150.0, "Item_Visibility": 0.07,
        "Item_Fat_Content": "Low Fat", "Item_Type": "Fruits and Vegetables",
        "Item_Sub_Type": subtype_map.get("Fruits and Vegetables", ["Exotic Produce"])[0],
        "Outlet_Size": "Medium", "Outlet_Location_Type": "Tier 2", "Outlet_Type": "Supermarket Type1",
    })
    st.info(f"Base item: {base['Item_Type']} / {base['Item_Sub_Type']}  ·  score it in Tab 1 to change the base item.")

    s1, s2, s3 = st.columns(3)
    sim_price = s1.slider("What if Price (₹) changes to…", 1.0, 2000.0, float(base["Item_MRP"]))
    sim_vis = s2.slider("What if Shelf Visibility changes to…", 0.0, 0.30, float(base["Item_Visibility"]), step=0.001)
    sim_promo = s3.selectbox("Promotional placement?", ["No change", "Boost visibility +0.05 (end-cap/promo)"])
    if sim_promo.startswith("Boost"):
        sim_vis = min(0.30, sim_vis + 0.05)

    sim_row = dict(base)
    sim_row["Item_MRP"] = sim_price
    sim_row["Item_Visibility"] = sim_vis
    sim_proba = predict_risk(sim_row)
    base_proba = predict_risk(base)

    d1, d2, d3 = st.columns(3)
    d1.metric("Baseline risk", f"{base_proba:.0%}")
    d2.metric("Scenario risk", f"{sim_proba:.0%}", delta=f"{(sim_proba - base_proba)*100:+.1f} pp")
    from reorder import estimate_daily_demand
    demand = estimate_daily_demand(sim_row | {"Item_Outlet_Sales": sim_price * 20})
    revenue_at_risk_sim = demand * sim_price * 5 * sim_proba
    revenue_at_risk_base = demand * base["Item_MRP"] * 5 * base_proba
    d3.metric("Revenue-at-risk change", f"₹{revenue_at_risk_sim:,.0f}", delta=f"₹{revenue_at_risk_sim - revenue_at_risk_base:+,.0f}")

    st.markdown("###### Risk sensitivity to shelf visibility")
    vis_range = np.linspace(0, 0.30, 30)
    risk_curve = [predict_risk({**sim_row, "Item_Visibility": v}) for v in vis_range]
    chart_df = pd.DataFrame({"Shelf Visibility": vis_range, "Risk Probability": risk_curve})
    st.line_chart(chart_df.set_index("Shelf Visibility"))


# ================================================================ TAB 4: Alerts
with tab4:
    st.markdown("##### 🔔 Real-Time Alert System")
    st.caption("Test each channel independently. Configure credentials in the sidebar first.")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        if st.button("✉️ Send test Email"):
            ok, msg = dispatch_alert("Email", "TEST-SKU", "This is a test alert from StockSense AI.")
            (st.success if ok else st.error)(msg)
    with tc2:
        if st.button("📱 Send test SMS"):
            ok, msg = dispatch_alert("SMS", "TEST-SKU", "This is a test alert from StockSense AI.")
            (st.success if ok else st.error)(msg)
    with tc3:
        if st.button("💬 Send test Slack message"):
            ok, msg = dispatch_alert("Slack", "TEST-SKU", "This is a test alert from StockSense AI.")
            (st.success if ok else st.error)(msg)

    st.markdown("###### Alert history (this session)")
    if st.session_state.alert_log:
        st.dataframe(pd.DataFrame(st.session_state.alert_log), use_container_width=True, height=300)
    else:
        st.caption("No alerts sent yet.")


# ================================================================ TAB 5: Financial Impact
with tab5:
    st.markdown("##### 💰 Financial Impact & Cost Estimation")
    if "reorder_table" not in st.session_state:
        st.info("Run the reorder scan in the 'Reorder Suggestions' tab first to populate this dashboard.")
    else:
        rt = st.session_state.reorder_table
        summary = financial_summary(rt)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Items at risk", summary["items_at_risk"])
        c2.metric("Revenue at risk", f"₹{summary['total_revenue_at_risk']:,.0f}")
        c3.metric("Reorder spend needed", f"₹{summary['total_reorder_spend']:,.0f}")
        c4.metric("Monthly holding cost (all SKUs)", f"₹{summary['total_monthly_holding_cost']:,.0f}")
        c5.metric("Avg days of cover (critical)", summary["avg_days_of_cover_critical"])

        st.markdown("###### Revenue at risk by category")
        by_cat = rt[rt["Needs_Reorder"]].groupby("Item_Type")["Revenue_At_Risk"].sum().sort_values(ascending=False)
        st.bar_chart(by_cat)

        st.markdown("###### Reorder spend vs. revenue protected, by outlet")
        by_outlet = rt[rt["Needs_Reorder"]].groupby("Outlet_Identifier")[["Reorder_Spend", "Revenue_At_Risk"]].sum()
        st.bar_chart(by_outlet)

        st.caption(
            "Methodology: Revenue-at-risk = daily demand × price × assumed 5-day stockout × model risk probability. "
            "Holding cost = simulated on-hand units × price × 2%/month. Reorder spend = EOQ-based suggested order × price. "
            "See reorder.py for the exact formulas — tune the constants there for your business."
        )
