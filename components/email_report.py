"""Send via Email — share one chatbot answer as a professional HTML email.

The email is assembled entirely from the answer that is already on screen:

* **AI Insight Findings**   → ``message["content"]``
* **Charts**                → ``message["figs"]`` (re-rendered on a white background as PNG)
* **Evidence at a Glance**  → the same tiles the insight panel shows (``insight_panel._tiles``)

Nothing here calls the supervisor, the LLM or SQL. The only extra work is rendering the
chart images (needs ``kaleido``) and the SMTP hand-over, both triggered by the user.

Flow
----
``render_email_button`` draws one "📧 Send via Email" button per answer. Clicking it opens a
dialog with recipient / subject / optional note fields, a live preview of the exact email
and a **Send Email** button with success and error messages. No mail is sent until the user
presses that button.

Configuration (secrets are never hard-coded, logged or shown)
-------------------------------------------------------------
Either environment variables / ``.env`` (read by ``core.settings.Settings``)::

    SMTP_HOST, SMTP_PORT, SMTP_SECURITY (starttls | ssl | none), SMTP_USERNAME,
    SMTP_PASSWORD, SMTP_FROM_ADDRESS, SMTP_FROM_NAME,
    EMAIL_ALLOWED_DOMAINS (optional, comma separated), EMAIL_MAX_RECIPIENTS

or a ``[smtp]`` table in ``.streamlit/secrets.toml`` (keys: host, port, security, username,
password, from_address, from_name, allowed_domains, max_recipients). When both exist the
Streamlit secrets win. Charts are embedded as inline CID attachments, which Outlook, Gmail
and Apple Mail display reliably (``data:`` image URLs are blocked by most clients).
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import logging
import re
import smtplib
import socket
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pydantic import SecretStr

from components.insight_panel import _tiles
from components.outlook_mail import esc, short, spell_pp
from components.ui import FOOTER_TEXT

log = logging.getLogger("datatalk.email_report")

MAX_CHARTS = 6                       # charts embedded per email
MAX_IMAGE_BYTES = 8 * 1024 * 1024    # total embedded image size before falling back to 1x resolution
CHART_WIDTH = 900
MAX_NOTE_CHARS = 600
COOLDOWN_SECONDS = 5                 # blocks accidental double clicks
_REPORT_CACHE_SIZE = 5
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,}$")
_FONT = "Segoe UI, Arial, Helvetica, sans-serif"


# ═══════════════════════════════════════════════════════════════════════════════
# configuration
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SmtpConfig:
    host: str = ""
    port: int = 587
    security: str = "starttls"        # starttls | ssl | none
    username: str = ""
    password: str = field(default="", repr=False)   # repr=False: never printed by accident
    from_address: str = ""
    from_name: str = "DataTalk.AI"
    timeout: float = 30.0
    allowed_domains: tuple[str, ...] = ()
    max_recipients: int = 10

    @property
    def configured(self) -> bool:
        return bool(self.host and self.from_address and _EMAIL_RE.match(self.from_address))


def _secrets_block() -> dict:
    try:
        block = st.secrets.get("smtp")
        return dict(block) if block else {}
    except Exception:  # no secrets file / not a table
        return {}


def load_config() -> SmtpConfig:
    """Streamlit secrets ``[smtp]`` win; otherwise ``core.settings`` (env / .env); otherwise defaults."""
    sec = _secrets_block()
    try:
        from core.settings import get_settings
        settings = get_settings()
    except Exception:
        settings = None

    def pick(secret_key: str, setting_attr: str, default: Any = "") -> Any:
        if sec.get(secret_key) not in (None, ""):
            return sec[secret_key]
        value = getattr(settings, setting_attr, None) if settings is not None else None
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        return default if value in (None, "") else value

    try:
        domains_raw = pick("allowed_domains", "email_allowed_domains", "")
        if isinstance(domains_raw, (list, tuple)):
            domains_raw = ",".join(str(d) for d in domains_raw)
        domains = tuple(d.strip().lower().lstrip("@") for d in str(domains_raw).split(",") if d.strip())
        security = str(pick("security", "smtp_security", "starttls")).strip().lower()
        return SmtpConfig(
            host=str(pick("host", "smtp_host")).strip(),
            port=int(pick("port", "smtp_port", 587)),
            security=security if security in {"starttls", "ssl", "none"} else "starttls",
            username=str(pick("username", "smtp_username")).strip(),
            password=str(pick("password", "smtp_password")),
            from_address=str(pick("from_address", "smtp_from_address")).strip(),
            from_name=str(pick("from_name", "smtp_from_name", "DataTalk.AI")).strip() or "DataTalk.AI",
            timeout=float(pick("timeout_seconds", "smtp_timeout_seconds", 30.0)),
            allowed_domains=domains,
            max_recipients=max(1, int(pick("max_recipients", "email_max_recipients", 10))),
        )
    except (TypeError, ValueError):
        log.exception("invalid SMTP configuration; email sending disabled")
        return SmtpConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# input validation
# ═══════════════════════════════════════════════════════════════════════════════
def parse_recipients(raw: str, cfg: SmtpConfig) -> tuple[list[str], list[str]]:
    """Split ``raw`` on commas / semicolons / whitespace and validate. Returns ``(valid, errors)``."""
    candidates = [p for p in re.split(r"[;,\s]+", raw or "") if p]
    if not candidates:
        return [], ["Enter at least one recipient email address."]
    valid: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for addr in candidates:
        low = addr.lower()
        if low in seen:
            continue
        seen.add(low)
        if len(addr) > 254 or not _EMAIL_RE.match(addr):
            errors.append(f"“{addr[:60]}” is not a valid email address.")
            continue
        domain = low.rsplit("@", 1)[1]
        if cfg.allowed_domains and not any(domain == d or domain.endswith("." + d) for d in cfg.allowed_domains):
            errors.append(f"{addr}: only these domains are allowed — {', '.join(cfg.allowed_domains)}.")
            continue
        valid.append(addr)
    if len(valid) > cfg.max_recipients:
        errors.append(f"At most {cfg.max_recipients} recipients are allowed per email.")
    return valid, errors


def clean_subject(subject: str) -> str:
    """One line, no header-injection characters, bounded length."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(subject or "")).strip()[:200]


def clean_note(note: str) -> str:
    text = re.sub(r"[^\S\n]+", " ", str(note or ""))
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()[:MAX_NOTE_CHARS]


# ═══════════════════════════════════════════════════════════════════════════════
# report model (built once per answer from data that already exists)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ChartImage:
    cid: str            # Content-ID including angle brackets, e.g. "<abc@datatalk.ai>"
    data: bytes
    alt: str


@dataclass
class EmailReport:
    subject: str
    question: str
    answer_text: str
    sections: list[dict]                 # [{"title", "finding", "tiles": [{"label","value","evidence"}]}]
    images: list[ChartImage] = field(default_factory=list)
    chart_notice: str = ""               # shown in the dialog only
    generated_at: str = ""


def _sections(insight: Any) -> list[dict]:
    """Evidence-at-a-glance tiles, one section per sub-question for multi-question answers."""
    if not isinstance(insight, dict):
        return []
    questions = [q for q in (insight.get("questions") or []) if isinstance(q, dict)]
    out: list[dict] = []
    if len(questions) > 1:
        for i, q in enumerate(questions, 1):
            kf = q.get("key_finding")
            finding = str(kf.get("text") or "") if isinstance(kf, dict) else ""
            tiles = _tiles(q)
            if tiles or finding:
                out.append({"title": f"Q{i}: {short(q.get('question', ''), 140)}", "finding": finding, "tiles": tiles})
        return out
    tiles = _tiles(insight)
    return [{"title": "", "finding": "", "tiles": tiles}] if tiles else []


def _fig_title(fig: Any, index: int) -> str:
    try:
        text = fig.layout.title.text
    except AttributeError:
        text = None
    return str(text) if text else f"Chart {index}"


def _fig_png(fig: Any, scale: int) -> bytes:
    """PNG of a copy of ``fig`` on a white background (the app figures are theme-transparent)."""
    export = go.Figure(fig)
    height = max(320, min(int(fig.layout.height or 460), 640))
    export.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                         font=dict(color="#1f2937"), width=CHART_WIDTH, height=height)
    return export.to_image(format="png", width=CHART_WIDTH, height=height, scale=scale)


def _render_images(figs: list) -> tuple[list[ChartImage], str]:
    figs = [f for f in (figs or []) if hasattr(f, "to_image")][:MAX_CHARTS]
    if not figs:
        return [], ""
    if importlib.util.find_spec("kaleido") is None:
        return [], ("Charts cannot be added because the 'kaleido' package is not installed on the server. "
                    "The email will contain the findings and KPI cards only.")
    images: list[ChartImage] = []
    for scale in (2, 1):
        images = []
        try:
            for i, fig in enumerate(figs, 1):
                images.append(ChartImage(cid=make_msgid(domain="datatalk.ai"), data=_fig_png(fig, scale),
                                         alt=_fig_title(fig, i)))
        except Exception:
            log.exception("chart rendering for email failed")
            return [], ("The charts could not be rendered as images on this server. "
                        "The email will contain the findings and KPI cards only.")
        if sum(len(i.data) for i in images) <= MAX_IMAGE_BYTES:
            break
    return images, ""


def build_report(message: dict, question: str = "") -> EmailReport:
    """Collect everything for the email from a stored chat answer. No LLM / SQL calls."""
    insight = message.get("insight") if isinstance(message.get("insight"), dict) else None
    question = (question or (insight or {}).get("question") or "").strip()
    answer_text = str(message.get("content") or "").strip()
    images, notice = _render_images(list(message.get("figs") or []))
    subject_seed = question or re.sub(r"[*_`#>]", "", answer_text)
    return EmailReport(
        subject=clean_subject(f"[DataTalk.AI] {short(spell_pp(subject_seed), 90) or 'Analysis report'}"),
        question=question, answer_text=answer_text, sections=_sections(insight),
        images=images, chart_notice=notice, generated_at=datetime.now().strftime("%d %b %Y, %H:%M"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# email content (HTML uses tables + inline CSS so Outlook renders it correctly)
# ═══════════════════════════════════════════════════════════════════════════════
def _inline(text: str) -> str:
    t = esc(spell_pp(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def md_to_html(text: str) -> str:
    """Small markdown subset (paragraphs, bullets, bold, code) → email-safe HTML."""
    p_style = f"margin:0 0 10px;font:14px/1.6 {_FONT};color:#1f2937;"
    out: list[str] = []
    in_list = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        item = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
        if item:
            if not in_list:
                out.append(f"<ul style='margin:0 0 10px;padding-left:20px;font:14px/1.6 {_FONT};color:#1f2937;'>")
                in_list = True
            out.append(f"<li style='margin:0 0 4px;'>{_inline(item.group(1))}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if not line:
            continue
        head = re.match(r"^#{1,6}\s+(.*)$", line)
        out.append(f"<p style='{p_style}{'font-weight:700;' if head else ''}'>{_inline(head.group(1) if head else line)}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def _plain(text: str) -> str:
    return re.sub(r"`(.+?)`", r"\1", re.sub(r"\*\*(.+?)\*\*", r"\1", spell_pp(text)))


def _row(inner: str, pad: str = "0 24px 8px") -> str:
    return f"<tr><td style='padding:{pad};'>{inner}</td></tr>"


def _heading(text: str) -> str:
    return _row(f"<div style='font:700 11px {_FONT};letter-spacing:1.2px;text-transform:uppercase;color:#1d4ed8;"
                f"border-left:3px solid #1d4ed8;padding-left:8px;'>{esc(text)}</div>", "20px 24px 10px")


def _tiles_html(tiles: list[dict]) -> str:
    rows = []
    for i in range(0, len(tiles), 3):
        chunk = tiles[i:i + 3]
        cells = "".join(
            f"<td width='33%' valign='top' style='padding:4px;'>"
            "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' bgcolor='#f8fafc' "
            "style='background:#f8fafc;border:1px solid #dbeafe;border-radius:8px;'><tr><td style='padding:10px 12px;'>"
            f"<div style='font:700 10px/1.3 {_FONT};letter-spacing:.6px;text-transform:uppercase;color:#64748b;'>"
            f"{esc(t['label'])}</div>"
            f"<div style='font:800 18px/1.3 {_FONT};color:#0f172a;padding-top:4px;'>{esc(t['value'])}</div>"
            "</td></tr></table></td>"
            for t in chunk
        ) + "<td width='33%' style='padding:4px;'></td>" * (3 - len(chunk))
        rows.append(f"<tr>{cells}</tr>")
    return ("<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'>"
            + "".join(rows) + "</table>")


def compose_html(report: EmailReport, note: str = "", preview: bool = False) -> str:
    """Full HTML email. ``preview=True`` inlines images as data URIs so the in-app preview shows them;
    the real email references them by ``cid:`` (attached inline)."""
    note = clean_note(note)
    rows: list[str] = [
        f"<tr><td bgcolor='#1e3a8a' style='background:#1e3a8a;padding:20px 24px;border-radius:10px 10px 0 0;'>"
        f"<div style='font:800 22px/1.2 {_FONT};color:#ffffff;'>DataTalk.AI</div>"
        f"<div style='font:12px/1.5 {_FONT};color:#bfdbfe;padding-top:2px;'>Healthcare Appeals Intelligence Platform</div>"
        "</td></tr>"
    ]
    if note:
        rows.append(_row(f"<div style='font:14px/1.6 {_FONT};color:#1f2937;'>{esc(note).replace(chr(10), '<br>')}</div>",
                         "18px 24px 4px"))
    if report.question:
        rows.append(_heading("Question"))
        rows.append(_row(f"<div style='font:600 15px/1.5 {_FONT};color:#0f172a;'>{esc(report.question)}</div>"))
    rows.append(_heading("AI Insight Findings"))
    rows.append(_row(md_to_html(report.answer_text) or f"<p style='font:14px {_FONT};color:#64748b;'>No narrative.</p>"))

    if report.images:
        rows.append(_heading("Charts"))
        for img in report.images:
            if preview:
                src = "data:image/png;base64," + base64.b64encode(img.data).decode("ascii")
            else:
                src = "cid:" + img.cid.strip("<>")
            rows.append(_row(
                f"<img src='{src}' width='592' alt='{esc(img.alt)}' "
                "style='display:block;width:100%;max-width:592px;height:auto;border:1px solid #e2e8f0;border-radius:6px;'>",
                "0 24px 14px"))

    if report.sections:
        rows.append(_heading("Evidence at a Glance"))
        for sec in report.sections:
            if sec["title"]:
                rows.append(_row(f"<div style='font:700 13px/1.5 {_FONT};color:#0f172a;'>{esc(sec['title'])}</div>", "4px 24px 4px"))
            if sec["finding"]:
                rows.append(_row(md_to_html(sec["finding"]), "0 24px 4px"))
            if sec["tiles"]:
                rows.append(_row(_tiles_html(sec["tiles"]), "0 20px 10px"))

    rows.append(
        f"<tr><td style='padding:16px 24px 20px;border-top:1px solid #e2e8f0;font:11px/1.6 {_FONT};color:#64748b;'>"
        f"Generated by DataTalk.AI on {esc(report.generated_at)} from governed appeals data. "
        "This email contains appeals analytics — share it only with authorised recipients.<br>"
        f"<strong>{esc(FOOTER_TEXT)}</strong></td></tr>"
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(report.subject)}</title></head>"
        "<body style='margin:0;padding:0;background:#f1f5f9;'>"
        "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' bgcolor='#f1f5f9' "
        "style='background:#f1f5f9;'><tr><td align='center' style='padding:24px 12px;'>"
        "<table role='presentation' width='640' cellpadding='0' cellspacing='0' border='0' bgcolor='#ffffff' "
        "style='width:100%;max-width:640px;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;'>"
        + "".join(rows) + "</table></td></tr></table></body></html>"
    )


def compose_text(report: EmailReport, note: str = "") -> str:
    """Plain-text alternative for clients that do not render HTML."""
    note = clean_note(note)
    lines = ["DataTalk.AI — Healthcare Appeals Intelligence", ""]
    if note:
        lines += [note, ""]
    if report.question:
        lines += ["QUESTION", report.question, ""]
    lines += ["AI INSIGHT FINDINGS", _plain(report.answer_text), ""]
    if report.images:
        lines += [f"{len(report.images)} chart(s) are included in the HTML version of this email.", ""]
    for sec in report.sections:
        lines.append("EVIDENCE AT A GLANCE" + (f" — {sec['title']}" if sec["title"] else ""))
        if sec["finding"]:
            lines.append(_plain(sec["finding"]))
        lines += [f"- {t['label']}: {t['value']}" for t in sec["tiles"]]
        lines.append("")
    lines += [f"Generated by DataTalk.AI on {report.generated_at}.", FOOTER_TEXT]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# sending
# ═══════════════════════════════════════════════════════════════════════════════
def build_message(cfg: SmtpConfig, report: EmailReport, recipients: list[str], subject: str,
                  note: str = "") -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = clean_subject(subject) or report.subject
    msg["From"] = formataddr((cfg.from_name, cfg.from_address))
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=cfg.from_address.rsplit("@", 1)[-1] or "datatalk.ai")
    msg.set_content(compose_text(report, note))
    msg.add_alternative(compose_html(report, note), subtype="html")
    html_part = msg.get_payload()[1]
    for i, img in enumerate(report.images, 1):
        html_part.add_related(img.data, "image", "png", cid=img.cid, disposition="inline",
                              filename=f"chart_{i}.png")
    return msg


def send_report(cfg: SmtpConfig, report: EmailReport, recipients: list[str], subject: str,
                note: str = "") -> tuple[bool, str]:
    """Send the email. Returns ``(ok, user-facing message)``; never raises, never logs credentials."""
    if not cfg.configured:
        return False, "Email is not configured on this server."
    try:
        msg = build_message(cfg, report, recipients, subject, note)
        context = ssl.create_default_context()
        server: smtplib.SMTP = (smtplib.SMTP_SSL(cfg.host, cfg.port, timeout=cfg.timeout, context=context)
                                if cfg.security == "ssl" else smtplib.SMTP(cfg.host, cfg.port, timeout=cfg.timeout))
        with server:
            server.ehlo()
            if cfg.security == "starttls":
                server.starttls(context=context)
                server.ehlo()
            if cfg.username:
                server.login(cfg.username, cfg.password)
            refused = server.send_message(msg, from_addr=cfg.from_address, to_addrs=recipients)
        if refused:
            failed = ", ".join(sorted(refused))
            log.warning("email partially refused: %d recipient(s)", len(refused))
            return False, f"The mail server rejected: {failed}. The other recipients received the email."
        log.info("email sent: recipients=%d charts=%d", len(recipients), len(report.images))
        return True, f"Email sent to {', '.join(recipients)}."
    except smtplib.SMTPAuthenticationError:
        log.error("SMTP authentication failed")
        return False, "The mail server rejected the sign-in. Check SMTP_USERNAME / SMTP_PASSWORD."
    except smtplib.SMTPRecipientsRefused:
        return False, "The mail server refused every recipient address. Please check them."
    except smtplib.SMTPSenderRefused:
        return False, "The mail server refused the sender address. Check SMTP_FROM_ADDRESS."
    except smtplib.SMTPConnectError:
        return False, "Could not connect to the mail server. Check SMTP_HOST / SMTP_PORT."
    except (socket.timeout, TimeoutError):
        return False, "The mail server did not respond in time. Please try again."
    except (smtplib.SMTPException, ssl.SSLError, OSError) as exc:
        log.error("email send failed: %s", type(exc).__name__)
        return False, f"The email could not be sent ({type(exc).__name__}). Please try again or contact support."
    except Exception:  # keep the chat usable
        log.exception("unexpected email failure")
        return False, "The email could not be sent because of an unexpected error. Details are in the server log."


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════
def _get_report(message: dict, key: str, question: str) -> EmailReport:
    """Build once per answer and reuse while the dialog is used (rendering PNGs is the slow part).

    The cache is keyed by a content signature, not just the card key, because card keys are
    positional and can point at a different answer after the history is trimmed."""
    sig = hashlib.sha1("|".join([question, str(message.get("content") or ""),
                                 str(len(message.get("figs") or []))]).encode("utf-8")).hexdigest()
    store: dict = st.session_state.setdefault("_email_reports", {})
    hit = store.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    report = build_report(message, question)
    store.pop(key, None)
    store[key] = (sig, report)
    while len(store) > _REPORT_CACHE_SIZE:
        store.pop(next(iter(store)))
    return report


@st.dialog("📧 Send report via email", width="large")
def _email_dialog(message: dict, key: str, question: str) -> None:
    cfg = load_config()
    with st.spinner("Preparing email preview…"):
        report = _get_report(message, key, question)

    if not cfg.configured:
        st.warning("Email sending is not set up on this server yet. An administrator must configure the SMTP "
                   "settings (see the README / secrets.toml). You can still review the preview below.")

    to_raw = st.text_input("To", key=f"{key}_mail_to", placeholder="name@company.com — separate several with commas")
    subject_in = st.text_input("Subject", value=report.subject, key=f"{key}_mail_subject", max_chars=200)
    note = st.text_area("Message (optional)", key=f"{key}_mail_note", max_chars=MAX_NOTE_CHARS, height=80,
                        placeholder="Add a short note that appears above the report.")

    recipients, errors = parse_recipients(to_raw, cfg) if to_raw.strip() else ([], [])
    for err in errors:
        st.error(err)
    subject = clean_subject(subject_in) or report.subject
    if report.chart_notice:
        st.warning(report.chart_notice)

    st.caption(f"**From:** {cfg.from_address or '—'}  ·  **To:** {', '.join(recipients) or '—'}  ·  "
               f"**Subject:** {subject}")
    components.html(compose_html(report, note, preview=True), height=560, scrolling=True)

    can_send = cfg.configured and bool(recipients) and not errors
    send_col, close_col, _ = st.columns([1.3, 1, 3])
    if not to_raw.strip():
        st.caption("Enter a recipient to enable sending.")
    if send_col.button("📨 Send Email", type="primary", width="stretch", disabled=not can_send,
                       key=f"{key}_mail_send"):
        if time.time() - st.session_state.get("_email_last_sent", 0.0) < COOLDOWN_SECONDS:
            st.warning("Please wait a few seconds before sending another email.")
        else:
            with st.spinner("Sending…"):
                ok, text = send_report(cfg, report, recipients, subject, note)
            if ok:
                st.session_state["_email_last_sent"] = time.time()
                st.success(f"✅ {text}")
            else:
                st.error(f"❌ {text}")
    if close_col.button("Close", width="stretch", key=f"{key}_mail_close"):
        st.rerun()


def render_email_button(message: dict, key: str, question: str = "") -> None:
    """Draw the "Send via Email" button for one answer. Never raises into the page."""
    try:
        meta = message.get("meta") or {}
        if meta.get("route") == "error" or not (message.get("content") or message.get("figs")):
            return
        _, right = st.columns([5, 1.4])  # same right-aligned slot as the chart Export menu
        with right:
            if st.button("📧 Send via Email", key=f"{key}_mail_btn", width="stretch",
                         help="Email this answer — findings, charts and KPI cards — to a colleague."):
                _email_dialog(message, key, question)
    except Exception:
        log.exception("email button unavailable")


__all__ = ["render_email_button", "build_report", "compose_html", "compose_text", "send_report",
           "parse_recipients", "load_config", "SmtpConfig"]
