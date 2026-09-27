# StockSense AI — Item Risk Scoring & Reorder Dashboard

A Streamlit dashboard that scores each SKU's stockout risk, suggests reorder
quantities, sends real-time alerts (Email/SMS/Slack), runs what-if scenarios,
and estimates the financial impact — built on top of `bigmart_with_subtypes_.csv`.

## What's inside

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard (5 tabs, matches the mock UI) |
| `train_model.py` | Trains the RandomForest risk-scoring model from the CSV |
| `reorder.py` | Reorder-point / EOQ / financial-impact formulas |
| `alerts.py` | Email (SMTP), SMS (Twilio API), Slack (webhook) senders |
| `bigmart_with_subtypes.csv` | Your dataset (10,000 SKUs) |
| `requirements.txt` | Python dependencies |
| `secrets.toml.example` | Template for storing alert credentials outside the UI |

## 1. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the risk model (one-time, ~10 seconds)

```bash
python train_model.py
```

This reads `bigmart_with_subtypes.csv`, engineers the risk label (see the
big comment at the top of `train_model.py` for the exact rule — high demand
+ low shelf visibility = high risk), trains a RandomForestClassifier, and
saves `risk_model.pkl` + `feature_importance.json` + `model_metrics.json`
in the same folder. Re-run this any time you get new/updated data.

## 3. Run the dashboard

```bash
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser.

## 4. Using each tab

- **Item Risk Scoring** — fill in one item's attributes (weight, price,
  category, visibility, outlet profile) and click "Score This Item" to see
  its risk probability, top model signals, reorder suggestion, and
  financial impact — same layout as your mock.
- **Reorder Suggestions** — runs the model across the *entire* active
  dataset, flags every SKU that needs reordering, shows suggested order
  quantities, and lets you download the full plan as CSV or send bulk
  alerts for the top N critical items.
- **What-If Simulator** — take the last scored item (or a default one) and
  drag price/visibility sliders to see risk and revenue-at-risk update live,
  plus a risk-vs-visibility sensitivity curve.
- **Alerts** — test each channel (Email/SMS/Slack) independently and see a
  running log of everything sent this session (also written to
  `stocksense.db` in the `alert_log` table).
- **Financial Impact** — aggregate $ metrics: total revenue at risk, total
  suggested reorder spend, monthly holding cost, broken down by category
  and by outlet. Run a scan in "Reorder Suggestions" first to populate it.

## 5. Connecting your own data (live CSV / DB)

In the sidebar:
- **Upload CSV** — drop in a new export any time; the dashboard re-scores
  everything on the new data immediately. Click **"Save as live database
  snapshot"** to persist it into `stocksense.db` (SQLite) so it survives a
  restart.
- **Load from local DB** — reloads the last saved snapshot from
  `stocksense.db`. This is your lightweight "live database" — swap
  `load_inventory_from_db()` / `save_uploaded_csv_to_db()` in `app.py` for
  calls to your real warehouse DB (Postgres/MySQL/etc.) when you're ready;
  everything downstream (model, reorder logic, alerts) works unchanged as
  long as the dataframe has the same columns.

Required columns for any uploaded CSV: `Item_Weight, Item_MRP,
Item_Visibility, Item_Fat_Content, Item_Type, Item_Sub_Type, Outlet_Size,
Outlet_Location_Type, Outlet_Type` (plus `Item_Outlet_Sales` if you have it
— used to estimate demand; a rough fallback is used if it's missing).

## 6. Setting up real alerts

Fill in the sidebar's Email / SMS / Slack expanders with real credentials:

- **Email**: any SMTP provider. For Gmail, create an
  [App Password](https://myaccount.google.com/apppasswords) (2FA required)
  — don't use your normal password.
- **SMS**: a [Twilio](https://www.twilio.com/) trial account gives you a
  free number, Account SID, and Auth Token.
- **Slack**: create an
  [Incoming Webhook](https://api.slack.com/messaging/webhooks) in your
  workspace and paste the URL.

Credentials typed into the sidebar live only in that browser session. To
avoid retyping them, copy `secrets.toml.example` to `.streamlit/secrets.toml`
and read values with `st.secrets["smtp"]["host"]` etc. instead — Streamlit
loads that file automatically and keeps it out of your CSV/model exports.

## Notes & assumptions

- **Risk label** is engineered (no ground-truth stockout column exists in
  the CSV) — see the docstring in `train_model.py`. Swap in a real
  historical stockout/backorder field the moment you have one.
- **Current stock levels** are simulated deterministically per SKU (no
  stock-on-hand column exists either) — see `simulate_current_stock()` in
  `reorder.py`. Replace with a real inventory feed for production use.
- Reorder math uses standard formulas: reorder point = lead-time demand +
  safety stock (95% service level), order quantity via Economic Order
  Quantity (EOQ). Tune `LEAD_TIME_DAYS`, `HOLDING_COST_RATE`,
  `ORDER_COST_FLAT` in `reorder.py` for your business.
