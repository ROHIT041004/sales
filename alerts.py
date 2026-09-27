"""
alerts.py
---------
Real-time alert dispatch: Email (SMTP), SMS (Twilio), Slack (Incoming Webhook).
Each function returns (success: bool, message: str) and never raises, so the
Streamlit UI can display a clean success/error toast either way.

Credentials are read from Streamlit's session state / sidebar inputs at
call-time -- nothing is hard-coded or written to disk. For production use,
put real credentials in a local `.streamlit/secrets.toml` (see README) and
read them with st.secrets instead of typing them into the sidebar each run.
"""

import json
import smtplib
import ssl
from email.mime.text import MIMEText
from urllib import request, error, parse


def send_email_alert(smtp_host, smtp_port, smtp_user, smtp_password,
                      to_address, subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_address

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, int(smtp_port), timeout=10) as server:
            server.starttls(context=context)
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_address], msg.as_string())
        return True, f"Email sent to {to_address}"
    except Exception as e:
        return False, f"Email failed: {e}"


def send_sms_alert(twilio_account_sid, twilio_auth_token, from_number, to_number, body):
    """Uses Twilio's REST API directly (no twilio SDK dependency required)."""
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_account_sid}/Messages.json"
        data = parse.urlencode({"From": from_number, "To": to_number, "Body": body}).encode()
        req = request.Request(url, data=data)
        import base64
        auth = base64.b64encode(f"{twilio_account_sid}:{twilio_auth_token}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")
        with request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 201):
                return True, f"SMS sent to {to_number}"
            return False, f"SMS failed: HTTP {resp.status}"
    except error.HTTPError as e:
        return False, f"SMS failed: {e.read().decode()}"
    except Exception as e:
        return False, f"SMS failed: {e}"


def send_slack_alert(webhook_url, message):
    try:
        payload = json.dumps({"text": message}).encode("utf-8")
        req = request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Slack message sent"
            return False, f"Slack failed: HTTP {resp.status}"
    except Exception as e:
        return False, f"Slack failed: {e}"


def format_risk_alert(item_id, item_type, risk_proba, suggested_qty, revenue_at_risk):
    return (
        f"[StockSense AI] HIGH RISK ITEM\n"
        f"SKU: {item_id} ({item_type})\n"
        f"Risk probability: {risk_proba:.0%}\n"
        f"Suggested reorder qty: {suggested_qty:.0f} units\n"
        f"Estimated revenue at risk: Rs.{revenue_at_risk:,.0f}"
    )
