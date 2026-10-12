"""
Async email delivery for digest newsletters.

Tries Resend first (simple REST API, 3k free emails/month),
then falls back to SMTP (Gmail, SendGrid, Mailgun, etc.)

No third-party email library required — uses httpx (already a dependency)
and Python's built-in smtplib wrapped in asyncio.to_thread.
"""

import asyncio
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


async def send_email(
    to: str,
    subject: str,
    html: str,
    text: Optional[str] = None,
    *,
    resend_api_key: str = "",
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_password: str = "",
    from_address: str = "digest@example.com",
    from_name: str = "Parakeet Digest",
) -> bool:
    """
    Send an email via Resend (primary) or SMTP (fallback).

    Returns True on success, False on failure. Never raises.
    """
    if resend_api_key:
        ok = await _send_via_resend(
            to=to,
            subject=subject,
            html=html,
            text=text,
            api_key=resend_api_key,
            from_address=from_address,
            from_name=from_name,
        )
        if ok:
            return True
        logger.warning(f"Resend failed for {to}, trying SMTP fallback")

    if smtp_host and smtp_user and smtp_password:
        return await _send_via_smtp(
            to=to,
            subject=subject,
            html=html,
            text=text,
            host=smtp_host,
            port=smtp_port,
            user=smtp_user,
            password=smtp_password,
            from_address=from_address,
            from_name=from_name,
        )

    logger.error(f"No email provider configured — cannot send to {to}")
    return False


async def _send_via_resend(
    to: str,
    subject: str,
    html: str,
    text: Optional[str],
    api_key: str,
    from_address: str,
    from_name: str,
) -> bool:
    """Send via Resend REST API (https://resend.com)."""
    payload = {
        "from": f"{from_name} <{from_address}>",
        "to": [to],
        "subject": subject,
        "html": html,
    }
    if text:
        payload["text"] = text

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if response.status_code in (200, 201):
            logger.info(f"Resend: sent to {to} (id={response.json().get('id')})")
            return True

        logger.error(f"Resend: HTTP {response.status_code} — {response.text[:200]}")
        return False

    except Exception as e:
        logger.error(f"Resend: exception sending to {to}: {e}")
        return False


async def _send_via_smtp(
    to: str,
    subject: str,
    html: str,
    text: Optional[str],
    host: str,
    port: int,
    user: str,
    password: str,
    from_address: str,
    from_name: str,
) -> bool:
    """Send via SMTP (runs blocking smtplib in a thread)."""
    def _blocking_send():
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{from_name} <{from_address}>"
        msg["To"] = to

        if text:
            msg.attach(MIMEText(text, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))

        ctx = ssl.create_default_context()

        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ctx) as server:
                server.login(user, password)
                server.sendmail(from_address, [to], msg.as_string())
        else:
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.login(user, password)
                server.sendmail(from_address, [to], msg.as_string())

    try:
        await asyncio.to_thread(_blocking_send)
        logger.info(f"SMTP: sent to {to} via {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"SMTP: exception sending to {to}: {e}")
        return False


def render_digest_html(episodes: list, digest_date: str, title: str = "") -> tuple[str, str]:
    """
    Render a list of episodes/articles into (html, plain_text) for email.

    Handles both podcast episodes and newsletter articles in the same digest.
    Returns (html_string, plain_text_string).
    """
    display_title = title or f"Your Daily Digest — {digest_date}"
    item_count = len(episodes)
    item_word = "item" if item_count == 1 else "items"

    # ── HTML ──────────────────────────────────────────────────────────────────
    html_items = []
    for ep in episodes:
        ep_title = ep.get("title", "Untitled")
        source = (ep.get("podcasts") or {}).get("title", "")
        feed_type = (ep.get("podcasts") or {}).get("feed_type", "podcast")
        published = (ep.get("published_date") or "")[:10]

        dur = ep.get("duration_seconds")
        dur_str = ""
        if dur:
            h, m = divmod(dur // 60, 60)
            dur_str = f" &middot; {h}h&nbsp;{m}m" if h else f" &middot; {m}m"

        type_badge = (
            '<span style="background:#e8f4f8;color:#0369a1;border-radius:3px;'
            'padding:2px 6px;font-size:11px;text-transform:uppercase;'
            'letter-spacing:.5px;font-weight:600;">newsletter</span>'
            if feed_type == "newsletter"
            else '<span style="background:#f0fdf4;color:#15803d;border-radius:3px;'
            'padding:2px 6px;font-size:11px;text-transform:uppercase;'
            'letter-spacing:.5px;font-weight:600;">podcast</span>'
        )

        summaries = ep.get("summaries") or []
        if isinstance(summaries, dict):
            summaries = [summaries]
        s = summaries[0] if summaries else {}

        full_summary = s.get("full_summary") or s.get("summary", "")
        key_insights = s.get("key_insights") or []
        takeaways = s.get("actionable_takeaways") or []
        key_topics = s.get("key_topics") or []
        quotes = (s.get("quotes") or [])[:2]

        # Summary block
        summary_html = ""
        if full_summary:
            summary_html = (
                f'<p style="color:#374151;line-height:1.7;margin:12px 0 16px;">'
                f"{full_summary}</p>"
            )

        # Insights
        insights_html = ""
        if key_insights:
            items = "".join(
                f'<li style="margin-bottom:6px;color:#374151;">{i}</li>'
                for i in key_insights
            )
            insights_html = (
                f'<p style="font-weight:600;color:#111827;margin:16px 0 6px;">'
                f"Key Insights</p>"
                f'<ul style="margin:0 0 12px;padding-left:20px;">{items}</ul>'
            )

        # Takeaways
        takeaways_html = ""
        if takeaways:
            items = "".join(
                f'<li style="margin-bottom:6px;color:#374151;">{t}</li>'
                for t in takeaways
            )
            takeaways_html = (
                f'<p style="font-weight:600;color:#111827;margin:16px 0 6px;">'
                f"Takeaways</p>"
                f'<ul style="margin:0 0 12px;padding-left:20px;">{items}</ul>'
            )

        # Quotes
        quotes_html = ""
        if quotes:
            qs = "".join(
                f'<blockquote style="border-left:3px solid #d1d5db;margin:8px 0;'
                f'padding:6px 12px;color:#6b7280;font-style:italic;">{q}</blockquote>'
                for q in quotes
            )
            quotes_html = f'<div style="margin:12px 0;">{qs}</div>'

        # Topics pill row
        topics_html = ""
        if key_topics:
            pills = "".join(
                f'<span style="display:inline-block;background:#f3f4f6;color:#6b7280;'
                f'border-radius:12px;padding:2px 10px;font-size:12px;margin:2px 4px 2px 0;">'
                f"{t}</span>"
                for t in key_topics
            )
            topics_html = f'<div style="margin:12px 0 0;">{pills}</div>'

        meta_parts = []
        if published:
            meta_parts.append(published)
        if dur_str:
            meta_parts.append(dur_str.replace("&middot;", "·").replace("&nbsp;", " ").strip())
        meta_str = " · ".join(meta_parts)

        html_items.append(f"""
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;
                    padding:24px;margin-bottom:20px;">
          <div style="margin-bottom:8px;">{type_badge}</div>
          <div style="color:#6b7280;font-size:13px;margin-bottom:4px;">{source}</div>
          <h2 style="margin:0 0 6px;font-size:18px;color:#111827;line-height:1.4;">{ep_title}</h2>
          <div style="color:#9ca3af;font-size:12px;margin-bottom:12px;">{meta_str}</div>
          {summary_html}
          {insights_html}
          {takeaways_html}
          {quotes_html}
          {topics_html}
        </div>""")

    items_joined = "\n".join(html_items)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{display_title}</title>
</head>
<body style="margin:0;padding:0;background:#f9fafb;font-family:-apple-system,BlinkMacSystemFont,
             'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:640px;margin:0 auto;padding:24px 16px 48px;">

    <!-- Header -->
    <div style="text-align:center;padding:32px 0 24px;">
      <div style="font-size:24px;font-weight:700;color:#111827;letter-spacing:-0.5px;">
        🦜 Parakeet
      </div>
      <div style="font-size:13px;color:#9ca3af;margin-top:4px;">Daily Digest</div>
    </div>

    <!-- Title card -->
    <div style="background:#111827;border-radius:10px;padding:28px 24px;
                margin-bottom:28px;color:#ffffff;">
      <div style="font-size:22px;font-weight:700;line-height:1.3;">{display_title}</div>
      <div style="color:#9ca3af;margin-top:8px;font-size:14px;">
        {item_count} {item_word} curated for you
      </div>
    </div>

    <!-- Items -->
    {items_joined}

    <!-- Footer -->
    <div style="text-align:center;padding:32px 0 0;color:#9ca3af;font-size:12px;
                border-top:1px solid #e5e7eb;margin-top:8px;">
      <p style="margin:0 0 6px;">You're receiving this because you enabled digest emails.</p>
      <p style="margin:0;">Manage your preferences at your Parakeet settings page.</p>
    </div>
  </div>
</body>
</html>"""

    # ── Plain text ────────────────────────────────────────────────────────────
    text_lines = [display_title, f"{item_count} {item_word} curated for you", "", "=" * 60]
    for ep in episodes:
        ep_title = ep.get("title", "Untitled")
        source = (ep.get("podcasts") or {}).get("title", "")
        summaries = ep.get("summaries") or []
        if isinstance(summaries, dict):
            summaries = [summaries]
        s = summaries[0] if summaries else {}
        full_summary = s.get("full_summary") or s.get("summary", "")
        key_insights = s.get("key_insights") or []
        takeaways = s.get("actionable_takeaways") or []

        text_lines += ["", f"{source} — {ep_title}", "-" * 40]
        if full_summary:
            text_lines += ["", full_summary]
        if key_insights:
            text_lines += ["", "Key Insights:"] + [f"  • {i}" for i in key_insights]
        if takeaways:
            text_lines += ["", "Takeaways:"] + [f"  • {t}" for t in takeaways]
        text_lines += ["", "=" * 60]

    return html, "\n".join(text_lines)

