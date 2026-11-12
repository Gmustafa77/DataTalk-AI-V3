# """Draft in Outlook for a whole chatbot answer (findings + charts + KPI cards).

# Same idea as the "Draft in Outlook" buttons in Risk & Findings and Recommended Actions: **nothing is
# sent**, the To field is left empty and the user types the recipient in Outlook. Nothing here calls the
# supervisor, the LLM or SQL — the draft is assembled from the answer already on screen:

# * **AI Insight Findings**   → ``message["content"]``
# * **Evidence at a Glance**  → the same tiles the insight panel shows (``insight_panel._tiles``)
# * **Charts**                → ``message["figs"]``, re-drawn on white and embedded as images

# Two ways to open the draft (both under the answer's charts):

# 1. **Draft in Outlook** — a compose link (subject + body filled in, To empty), built with
#    ``components.outlook_mail`` exactly like the existing draft buttons. A web link can only carry text, so
#    this draft has the question, findings and KPI values but no images.
# 2. **Draft with charts (.eml)** — prepared on click (rendering chart images is slow, so it never runs on
#    a normal page refresh). The downloaded file opens in Outlook as an *unsent, editable draft* with the
#    full HTML layout and the charts embedded (``X-Unsent: 1``, no To / From).

# 3. **Copy & open Outlook** — no download: one click copies the full email (findings, KPI cards and charts)
#    to the clipboard and opens the Outlook compose window with the subject filled in and To empty; the
#    user presses Ctrl+V in the message body. If the browser refuses clipboard access, the compose window
#    opens with the text draft instead.

# Chart images need ``kaleido``; without it the .eml still contains findings and KPI cards.
# """
# from __future__ import annotations

# import base64
# import hashlib
# import importlib.util
# import json
# import logging
# import re
# from dataclasses import dataclass, field
# from datetime import date, datetime
# from email.message import EmailMessage
# from email.utils import make_msgid
# from typing import Any

# import plotly.graph_objects as go
# import streamlit as st
# import streamlit.components.v1 as components

# from components.insight_panel import _tiles
# from components.outlook_mail import MAIL_CSS, esc, mail_button, mail_url, short, spell_pp
# from components.ui import FOOTER_TEXT, theme_type

# log = logging.getLogger("datatalk.email_draft")

# MAX_CHARTS = 6                       # charts embedded per draft
# MAX_IMAGE_BYTES = 8 * 1024 * 1024    # total image size before falling back to 1x resolution
# CHART_WIDTH = 900
# _DRAFT_CACHE_SIZE = 5
# _FONT = "Segoe UI, Arial, Helvetica, sans-serif"


# def clean_subject(subject: str) -> str:
#     """One line, no control characters, bounded length."""
#     return re.sub(r"[\x00-\x1f\x7f]+", " ", str(subject or "")).strip()[:200]


# # ═══════════════════════════════════════════════════════════════════════════════
# # report model (built once per answer from data that already exists)
# # ═══════════════════════════════════════════════════════════════════════════════
# @dataclass
# class ChartImage:
#     cid: str            # Content-ID including angle brackets, e.g. "<abc@datatalk.ai>"
#     data: bytes
#     alt: str


# @dataclass
# class EmailReport:
#     subject: str
#     question: str
#     answer_text: str
#     sections: list[dict]                 # [{"title", "finding", "tiles": [{"label","value","evidence"}]}]
#     images: list[ChartImage] = field(default_factory=list)
#     chart_notice: str = ""               # shown in the dialog only
#     generated_at: str = ""


# def _sections(insight: Any) -> list[dict]:
#     """Evidence-at-a-glance tiles, one section per sub-question for multi-question answers."""
#     if not isinstance(insight, dict):
#         return []
#     questions = [q for q in (insight.get("questions") or []) if isinstance(q, dict)]
#     out: list[dict] = []
#     if len(questions) > 1:
#         for i, q in enumerate(questions, 1):
#             kf = q.get("key_finding")
#             finding = str(kf.get("text") or "") if isinstance(kf, dict) else ""
#             tiles = _tiles(q)
#             if tiles or finding:
#                 out.append({"title": f"Q{i}: {short(q.get('question', ''), 140)}", "finding": finding, "tiles": tiles})
#         return out
#     tiles = _tiles(insight)
#     return [{"title": "", "finding": "", "tiles": tiles}] if tiles else []


# def _fig_title(fig: Any, index: int) -> str:
#     try:
#         text = fig.layout.title.text
#     except AttributeError:
#         text = None
#     return str(text) if text else f"Chart {index}"


# def _fig_png(fig: Any, scale: int) -> bytes:
#     """PNG of a copy of ``fig`` on a white background (the app figures are theme-transparent)."""
#     export = go.Figure(fig)
#     height = max(320, min(int(fig.layout.height or 460), 640))
#     export.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
#                          font=dict(color="#1f2937"), width=CHART_WIDTH, height=height)
#     return export.to_image(format="png", width=CHART_WIDTH, height=height, scale=scale)


# def _render_images(figs: list) -> tuple[list[ChartImage], str]:
#     figs = [f for f in (figs or []) if hasattr(f, "to_image")][:MAX_CHARTS]
#     if not figs:
#         return [], ""
#     if importlib.util.find_spec("kaleido") is None:
#         return [], ("Charts cannot be added because the 'kaleido' package is not installed on the server. "
#                     "The draft will contain the findings and KPI cards only.")
#     images: list[ChartImage] = []
#     for scale in (2, 1):
#         images = []
#         try:
#             for i, fig in enumerate(figs, 1):
#                 images.append(ChartImage(cid=make_msgid(domain="datatalk.ai"), data=_fig_png(fig, scale),
#                                          alt=_fig_title(fig, i)))
#         except Exception:
#             log.exception("chart rendering for email failed")
#             return [], ("The charts could not be rendered as images on this server. "
#                         "The draft will contain the findings and KPI cards only.")
#         if sum(len(i.data) for i in images) <= MAX_IMAGE_BYTES:
#             break
#     return images, ""


# def build_report(message: dict, question: str = "") -> EmailReport:
#     """Collect everything for the email from a stored chat answer. No LLM / SQL calls."""
#     insight = message.get("insight") if isinstance(message.get("insight"), dict) else None
#     question = (question or (insight or {}).get("question") or "").strip()
#     answer_text = str(message.get("content") or "").strip()
#     images, notice = _render_images(list(message.get("figs") or []))
#     subject_seed = question or re.sub(r"[*_`#>]", "", answer_text)
#     return EmailReport(
#         subject=clean_subject(f"[DataTalk.AI] {short(spell_pp(subject_seed), 90) or 'Analysis report'}"),
#         question=question, answer_text=answer_text, sections=_sections(insight),
#         images=images, chart_notice=notice, generated_at=datetime.now().strftime("%d %b %Y, %H:%M"),
#     )


# # ═══════════════════════════════════════════════════════════════════════════════
# # draft content (HTML uses tables + inline CSS so Outlook renders it correctly)
# # ═══════════════════════════════════════════════════════════════════════════════
# def _inline(text: str) -> str:
#     t = esc(spell_pp(text))
#     t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
#     return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


# def md_to_html(text: str) -> str:
#     """Small markdown subset (paragraphs, bullets, bold, code) → email-safe HTML."""
#     p_style = f"margin:0 0 10px;font:14px/1.6 {_FONT};color:#1f2937;"
#     out: list[str] = []
#     in_list = False
#     for raw in str(text or "").splitlines():
#         line = raw.strip()
#         item = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
#         if item:
#             if not in_list:
#                 out.append(f"<ul style='margin:0 0 10px;padding-left:20px;font:14px/1.6 {_FONT};color:#1f2937;'>")
#                 in_list = True
#             out.append(f"<li style='margin:0 0 4px;'>{_inline(item.group(1))}</li>")
#             continue
#         if in_list:
#             out.append("</ul>")
#             in_list = False
#         if not line:
#             continue
#         head = re.match(r"^#{1,6}\s+(.*)$", line)
#         out.append(f"<p style='{p_style}{'font-weight:700;' if head else ''}'>{_inline(head.group(1) if head else line)}</p>")
#     if in_list:
#         out.append("</ul>")
#     return "".join(out)


# def _plain(text: str) -> str:
#     return re.sub(r"`(.+?)`", r"\1", re.sub(r"\*\*(.+?)\*\*", r"\1", spell_pp(text)))


# def _row(inner: str, pad: str = "0 24px 8px") -> str:
#     return f"<tr><td style='padding:{pad};'>{inner}</td></tr>"


# def _heading(text: str) -> str:
#     return _row(f"<div style='font:700 11px {_FONT};letter-spacing:1.2px;text-transform:uppercase;color:#1d4ed8;"
#                 f"border-left:3px solid #1d4ed8;padding-left:8px;'>{esc(text)}</div>", "20px 24px 10px")


# def _tiles_html(tiles: list[dict]) -> str:
#     rows = []
#     for i in range(0, len(tiles), 3):
#         chunk = tiles[i:i + 3]
#         cells = "".join(
#             f"<td width='33%' valign='top' style='padding:4px;'>"
#             "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' bgcolor='#f8fafc' "
#             "style='background:#f8fafc;border:1px solid #dbeafe;border-radius:8px;'><tr><td style='padding:10px 12px;'>"
#             f"<div style='font:700 10px/1.3 {_FONT};letter-spacing:.6px;text-transform:uppercase;color:#64748b;'>"
#             f"{esc(t['label'])}</div>"
#             f"<div style='font:800 18px/1.3 {_FONT};color:#0f172a;padding-top:4px;'>{esc(t['value'])}</div>"
#             "</td></tr></table></td>"
#             for t in chunk
#         ) + "<td width='33%' style='padding:4px;'></td>" * (3 - len(chunk))
#         rows.append(f"<tr>{cells}</tr>")
#     return ("<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'>"
#             + "".join(rows) + "</table>")


# def compose_html(report: EmailReport, preview: bool = False) -> str:
#     """Full HTML email. ``preview=True`` inlines images as data URIs so the in-app preview shows them;
#     the real email references them by ``cid:`` (attached inline)."""
#     rows: list[str] = [
#         f"<tr><td bgcolor='#1e3a8a' style='background:#1e3a8a;padding:20px 24px;border-radius:10px 10px 0 0;'>"
#         f"<div style='font:800 22px/1.2 {_FONT};color:#ffffff;'>DataTalk.AI</div>"
#         f"<div style='font:12px/1.5 {_FONT};color:#bfdbfe;padding-top:2px;'>Healthcare Appeals Intelligence Platform</div>"
#         "</td></tr>"
#     ]
#     if report.question:
#         rows.append(_heading("Question"))
#         rows.append(_row(f"<div style='font:600 15px/1.5 {_FONT};color:#0f172a;'>{esc(report.question)}</div>"))
#     rows.append(_heading("AI Insight Findings"))
#     rows.append(_row(md_to_html(report.answer_text) or f"<p style='font:14px {_FONT};color:#64748b;'>No narrative.</p>"))

#     if report.images:
#         rows.append(_heading("Charts"))
#         for img in report.images:
#             if preview:
#                 src = "data:image/png;base64," + base64.b64encode(img.data).decode("ascii")
#             else:
#                 src = "cid:" + img.cid.strip("<>")
#             rows.append(_row(
#                 f"<img src='{src}' width='592' alt='{esc(img.alt)}' "
#                 "style='display:block;width:100%;max-width:592px;height:auto;border:1px solid #e2e8f0;border-radius:6px;'>",
#                 "0 24px 14px"))

#     if report.sections:
#         rows.append(_heading("Evidence at a Glance"))
#         for sec in report.sections:
#             if sec["title"]:
#                 rows.append(_row(f"<div style='font:700 13px/1.5 {_FONT};color:#0f172a;'>{esc(sec['title'])}</div>", "4px 24px 4px"))
#             if sec["finding"]:
#                 rows.append(_row(md_to_html(sec["finding"]), "0 24px 4px"))
#             if sec["tiles"]:
#                 rows.append(_row(_tiles_html(sec["tiles"]), "0 20px 10px"))

#     rows.append(
#         f"<tr><td style='padding:16px 24px 20px;border-top:1px solid #e2e8f0;font:11px/1.6 {_FONT};color:#64748b;'>"
#         f"Generated by DataTalk.AI on {esc(report.generated_at)} from governed appeals data. "
#         "Contains appeals analytics — share only with authorised recipients.<br>"
#         f"<strong>{esc(FOOTER_TEXT)}</strong></td></tr>"
#     )
#     return (
#         "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
#         "<meta name='viewport' content='width=device-width,initial-scale=1'>"
#         f"<title>{esc(report.subject)}</title></head>"
#         "<body style='margin:0;padding:0;background:#f1f5f9;'>"
#         "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' bgcolor='#f1f5f9' "
#         "style='background:#f1f5f9;'><tr><td align='center' style='padding:24px 12px;'>"
#         "<table role='presentation' width='640' cellpadding='0' cellspacing='0' border='0' bgcolor='#ffffff' "
#         "style='width:100%;max-width:640px;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;'>"
#         + "".join(rows) + "</table></td></tr></table></body></html>"
#     )


# def compose_text(report: EmailReport) -> str:
#     """Plain-text alternative for clients that do not render HTML."""
#     lines = ["DataTalk.AI — Healthcare Appeals Intelligence", ""]
#     if report.question:
#         lines += ["QUESTION", report.question, ""]
#     lines += ["AI INSIGHT FINDINGS", _plain(report.answer_text), ""]
#     if report.images:
#         lines += [f"{len(report.images)} chart(s) are included in the HTML version of this email.", ""]
#     for sec in report.sections:
#         lines.append("EVIDENCE AT A GLANCE" + (f" — {sec['title']}" if sec["title"] else ""))
#         if sec["finding"]:
#             lines.append(_plain(sec["finding"]))
#         lines += [f"- {t['label']}: {t['value']}" for t in sec["tiles"]]
#         lines.append("")
#     lines += [f"Generated by DataTalk.AI on {report.generated_at}.", FOOTER_TEXT]
#     return "\n".join(lines)


# # ═══════════════════════════════════════════════════════════════════════════════
# # link draft (text only) — same helpers as the existing "Draft in Outlook" buttons
# # ═══════════════════════════════════════════════════════════════════════════════
# def link_draft(message: dict, question: str = "") -> tuple[str, str]:
#     """``(subject, body)`` for ``outlook_mail.mail_url``. Cheap: text only, no images."""
#     insight = message.get("insight") if isinstance(message.get("insight"), dict) else None
#     question = (question or (insight or {}).get("question") or "").strip()
#     answer = str(message.get("content") or "").strip()
#     subject = clean_subject(f"[DataTalk.AI] {short(spell_pp(question or _plain(answer)), 90) or 'Analysis report'}")
#     lines = ["Hi,", "", "Please find the DataTalk.AI analysis below.", ""]
#     if question:
#         lines += ["QUESTION", short(question, 220), ""]
#     lines += ["AI INSIGHT FINDINGS", short(_plain(answer), 600), ""]
#     sections = _sections(insight)
#     if sections:
#         lines.append("EVIDENCE AT A GLANCE")
#         for sec in sections:
#             if sec["title"]:
#                 lines.append(sec["title"])
#             lines += [f"- {t['label']}: {t['value']}" for t in sec["tiles"]]
#         lines.append("")
#     lines += ["Charts are not part of this link draft; use \"Draft with charts (.eml)\" in DataTalk.AI to include them.",
#               "", "Thanks,"]
#     return subject, "\n".join(lines)


# # ═══════════════════════════════════════════════════════════════════════════════
# # .eml draft (full HTML + charts), opens in Outlook as an unsent message
# # ═══════════════════════════════════════════════════════════════════════════════
# def build_eml(report: EmailReport) -> bytes:
#     """RFC 822 file with ``X-Unsent: 1`` and no To / From: Outlook opens it as an editable draft."""
#     msg = EmailMessage()
#     msg["Subject"] = report.subject
#     msg["X-Unsent"] = "1"
#     msg.set_content(compose_text(report))
#     msg.add_alternative(compose_html(report), subtype="html")
#     html_part = msg.get_payload()[1]
#     for i, img in enumerate(report.images, 1):
#         html_part.add_related(img.data, "image", "png", cid=img.cid, disposition="inline",
#                               filename=f"chart_{i}.png")
#     return msg.as_bytes()


# def _slug(text: str) -> str:
#     return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "report"


# # ═══════════════════════════════════════════════════════════════════════════════
# # Streamlit UI
# # ═══════════════════════════════════════════════════════════════════════════════
# def _js(value: str) -> str:
#     """JSON string literal that is safe inside an inline <script>."""
#     return json.dumps(value).replace("</", "<\\/")


# def _copy_and_open_button(report: EmailReport, link_subject: str, link_body: str) -> None:
#     """One-click: copy the rich email to the clipboard, then open the Outlook compose window.

#     Runs in the browser (a click inside the component is the user gesture the clipboard needs).
#     Compose opens with an empty body after a successful copy so the paste is clean; if copying fails
#     it opens with the text draft so the user is never left with nothing.
#     """
#     dark = theme_type() == "dark"
#     color, border, hover = ("#fafafa", "rgba(250,250,250,.2)", "#60a5fa") if dark else ("#31333f", "rgba(49,51,63,.2)", "#2563eb")
#     html_doc = (
#         "<style>html,body{margin:0;background:transparent;}"
#         f"button{{width:100%;height:46px;border-radius:8px;border:1px solid {border};background:transparent;color:{color};"
#         "font:400 14px 'Source Sans Pro','Segoe UI',Arial,sans-serif;cursor:pointer;}"
#         f"button:hover{{border-color:{hover};color:{hover};}}"
#         "#s{font:12px 'Segoe UI',Arial,sans-serif;color:#94a3b8;margin-top:6px;min-height:16px;}</style>"
#         "<button id='b' type='button'>📋 Copy email &amp; open Outlook</button><div id='s' role='status'></div>"
#         "<div id='src' contenteditable='true' aria-hidden='true' style='position:fixed;left:-99999px;top:0;'></div>"
#         "<script>"
#         f"const HTML={_js(compose_html(report, preview=True))};const PLAIN={_js(compose_text(report))};"
#         f"const URL_EMPTY={_js(mail_url(report.subject, ''))};const URL_TEXT={_js(mail_url(link_subject, link_body))};"
#         "async function copyRich(){"
#         "try{if(navigator.clipboard&&window.ClipboardItem){await navigator.clipboard.write([new ClipboardItem({"
#         "'text/html':new Blob([HTML],{type:'text/html'}),'text/plain':new Blob([PLAIN],{type:'text/plain'})})]);return true;}}catch(e){}"
#         "try{const el=document.getElementById('src');el.innerHTML=HTML;const r=document.createRange();r.selectNodeContents(el);"
#         "const sel=window.getSelection();sel.removeAllRanges();sel.addRange(r);const ok=document.execCommand('copy');"
#         "sel.removeAllRanges();el.innerHTML='';return ok;}catch(e){return false;}}"
#         "document.getElementById('b').onclick=async()=>{const s=document.getElementById('s');const ok=await copyRich();"
#         "window.open(ok?URL_EMPTY:URL_TEXT,'_blank','noopener');"
#         "s.textContent=ok?'Copied. In the Outlook message body press Ctrl+V (Cmd+V on Mac).':"
#         "'Clipboard blocked by the browser, opened the text draft instead.';};"
#         "</script>"
#     )
#     components.html(html_doc, height=78)


# def _signature(message: dict, question: str) -> str:
#     return hashlib.sha1("|".join([question, str(message.get("content") or ""),
#                                   str(len(message.get("figs") or []))]).encode("utf-8")).hexdigest()


# def render_email_draft(message: dict, key: str, question: str = "") -> None:
#     """Draw the draft controls for one answer. Never raises into the page."""
#     try:
#         meta = message.get("meta") or {}
#         if meta.get("route") == "error" or not (message.get("content") or message.get("figs")):
#             return
#         sig = _signature(message, question)
#         subject, body = link_draft(message, question)
#         st.markdown(MAIL_CSS, unsafe_allow_html=True)
#         left, right = st.columns([3.2, 2], vertical_alignment="center")
#         with left:
#             st.markdown(
#                 "<div class='dt-plan-bar'><div><div class='dt-plan-title'>Share this answer</div>"
#                 "<div class='dt-plan-sub'>Outlook draft with findings and KPI values · add the recipient in To</div></div>"
#                 + mail_button(mail_url(subject, body), "Draft in Outlook",
#                               "Opens an Outlook draft with the question, findings and KPI values filled in",
#                               variant="secondary")
#                 + "</div>", unsafe_allow_html=True)
#         store: dict = st.session_state.setdefault("_email_drafts", {})
#         with right:
#             if st.button("🖼️ Draft with charts (.eml)", key=f"{key}_eml_btn", width="stretch",
#                          help="Builds a draft that includes the charts. Opens in Outlook as an unsent message."):
#                 with st.spinner("Preparing draft…"):
#                     report = build_report(message, question)
#                     store.pop(key, None)
#                     store[key] = (sig, report, build_eml(report))
#                     while len(store) > _DRAFT_CACHE_SIZE:
#                         store.pop(next(iter(store)))
#             ready = store.get(key)
#             if ready and ready[0] == sig:
#                 st.download_button("⬇️ Download draft", ready[2], file_name=f"DataTalk_draft_{_slug(question or 'answer')}_{date.today().isoformat()}.eml",
#                                    mime="message/rfc822", key=f"{key}_eml_dl", on_click="ignore", width="stretch")
#                 _copy_and_open_button(ready[1], subject, body)
#         ready = store.get(key)
#         if ready and ready[0] == sig:
#             report = ready[1]
#             if report.chart_notice:
#                 st.warning(report.chart_notice)
#             st.caption("Download the draft and double-click it to open in Outlook — or use Copy email & open Outlook, then press "
#                        "Ctrl+V in the message body. Add the recipient in To either way.")
#             with st.expander("👁️ Preview draft", expanded=False):
#                 components.html(compose_html(report, preview=True), height=560, scrolling=True)
#     except Exception:
#         log.exception("email draft controls unavailable")


# __all__ = ["render_email_draft", "link_draft", "build_report", "build_eml", "compose_html", "compose_text"]







"""Draft in Outlook for a whole chatbot answer (findings + charts + KPI cards).

Same idea as the "Draft in Outlook" buttons in Risk & Findings and Recommended Actions: **nothing is
sent**, the To field is left empty and the user types the recipient in Outlook. Nothing here calls the
supervisor, the LLM or SQL — the draft is assembled from the answer already on screen:

* **AI Insight Findings**   → ``message["content"]``
* **Evidence at a Glance**  → the same tiles the insight panel shows (``insight_panel._tiles``)
* **Charts**                → ``message["figs"]``, re-drawn on white and embedded as images

Two ways to open the draft (both under the answer's charts):

1. **Draft in Outlook** — a compose link (subject + body filled in, To empty), built with
   ``components.outlook_mail`` exactly like the existing draft buttons. A web link can only carry text, so
   this draft has the question, findings and KPI values but no images.
2. **Draft with charts (.eml)** — prepared on click (rendering chart images is slow, so it never runs on
   a normal page refresh). The downloaded file opens in Outlook as an *unsent, editable draft* with the
   full HTML layout and the charts embedded (``X-Unsent: 1``, no To / From).

3. **Copy & open Outlook** — no download: one click copies the full email (findings, KPI cards and charts)
   to the clipboard and opens the Outlook compose window with the subject filled in and To empty; the
   user presses Ctrl+V in the message body. If the browser refuses clipboard access, the compose window
   opens with the text draft instead.

The **Preview draft** panel is interactive (Email / Plain-text tabs, desktop / mobile width, jump-to-section chips,
click-to-zoom charts); it is display-only and does not change the email itself.

Chart images need ``kaleido``; without it the .eml still contains findings and KPI cards.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from components.insight_panel import _tiles
from components.outlook_mail import MAIL_CSS, esc, mail_button, mail_url, short, spell_pp
from components.ui import FOOTER_TEXT, theme_type

log = logging.getLogger("datatalk.email_draft")

MAX_CHARTS = 6                       # charts embedded per draft
MAX_IMAGE_BYTES = 8 * 1024 * 1024    # total image size before falling back to 1x resolution
CHART_WIDTH = 900
_DRAFT_CACHE_SIZE = 5
_FONT = "Segoe UI, Arial, Helvetica, sans-serif"


def clean_subject(subject: str) -> str:
    """One line, no control characters, bounded length."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(subject or "")).strip()[:200]


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
                    "The draft will contain the findings and KPI cards only.")
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
                        "The draft will contain the findings and KPI cards only.")
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
# draft content (HTML uses tables + inline CSS so Outlook renders it correctly)
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


_KPI_COLORS = ("#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4")

# Progressive enhancement: clients that read <style> (Apple Mail, Gmail, Outlook.com / Outlook on the web,
# iOS and Android apps) get hover, a phone layout and dark mode. Clients that ignore it (classic Outlook)
# still show the complete inline-styled email.
_EMAIL_STYLE = """
.dt-kpi{transition:box-shadow .15s ease,transform .15s ease}
.dt-kpi:hover{box-shadow:0 6px 16px rgba(37,99,235,.25)!important;transform:translateY(-2px)}
.dt-img{transition:box-shadow .15s ease}
.dt-img:hover{box-shadow:0 6px 20px rgba(15,23,42,.22)}
@media only screen and (max-width:480px){
  .dt-outer{padding:0!important}
  .dt-card{border-radius:0!important;border-left:0!important;border-right:0!important}
  .dt-pad{padding-left:16px!important;padding-right:16px!important}
  .dt-kpi-cell{display:block!important;width:100%!important;box-sizing:border-box!important}
  .dt-empty{display:none!important}
  .dt-title{font-size:20px!important}
  .dt-kv{font-size:22px!important}
  .dt-stack{display:block!important;width:100%!important;text-align:left!important;padding-top:8px!important}
}
"""
_EMAIL_DARK = """
@media (prefers-color-scheme:dark){
  .dt-bg{background:#0b1220!important}
  .dt-card{background:#111827!important;border-color:#1f2937!important}
  .dt-card p,.dt-card li,.dt-t{color:#e5e7eb!important}
  .dt-h{color:#93c5fd!important;border-color:#60a5fa!important}
  .dt-muted{color:#94a3b8!important}
  .dt-callout{background:#0f1d3a!important}
  .dt-kpi{background:#0f172a!important;border-color:#1f2937!important}
  .dt-kpi .dt-kv{color:#f8fafc!important}
  .dt-kpi .dt-kl{color:#94a3b8!important}
  .dt-imgbox{background:#ffffff!important;border-color:#1f2937!important}
  .dt-foot{border-color:#1f2937!important;color:#94a3b8!important}
}
"""


def _row(inner: str, pad: str = "0 24px 8px", rid: str = "") -> str:
    rid_attr = f" id='{rid}'" if rid else ""
    return f"<tr{rid_attr}><td class='dt-pad' style='padding:{pad};'>{inner}</td></tr>"


def _heading(text: str, anchor: str = "") -> str:
    return _row(f"<div class='dt-h' style='font:700 11px {_FONT};letter-spacing:1.2px;text-transform:uppercase;"
                f"color:#1d4ed8;border-left:3px solid #1d4ed8;padding-left:8px;'>{esc(text)}</div>",
                "22px 24px 10px", anchor)


def _callout(inner_html: str) -> str:
    """Highlighted box for the AI Insight Findings text."""
    return ("<table class='dt-callout' role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' "
            "bgcolor='#eff6ff' style='background:#eff6ff;border-left:4px solid #2563eb;border-radius:0 8px 8px 0;'>"
            f"<tr><td style='padding:12px 16px 4px;'>{inner_html}</td></tr></table>")


def _tiles_html(tiles: list[dict]) -> str:
    """KPI cards, three per row (stacked on phones), each with a coloured top edge."""
    rows = []
    for i in range(0, len(tiles), 3):
        chunk = tiles[i:i + 3]
        cells = "".join(
            "<td class='dt-kpi-cell' width='33%' valign='top' style='padding:4px;'>"
            "<table class='dt-kpi' role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' "
            f"bgcolor='#f8fafc' style='background:#f8fafc;border:1px solid #e2e8f0;"
            f"border-top:3px solid {_KPI_COLORS[(i + j) % len(_KPI_COLORS)]};border-radius:8px;'>"
            "<tr><td style='padding:10px 12px;'>"
            f"<div class='dt-kl' style='font:700 10px/1.3 {_FONT};letter-spacing:.6px;text-transform:uppercase;color:#64748b;'>"
            f"{esc(t['label'])}</div>"
            f"<div class='dt-kv' style='font:800 20px/1.3 {_FONT};color:#0f172a;padding-top:4px;'>{esc(t['value'])}</div>"
            "</td></tr></table></td>"
            for j, t in enumerate(chunk)
        ) + "<td class='dt-kpi-cell dt-empty' width='33%' style='padding:4px;'></td>" * (3 - len(chunk))
        rows.append(f"<tr>{cells}</tr>")
    return ("<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'>"
            + "".join(rows) + "</table>")


def compose_html(report: EmailReport, preview: bool = False) -> str:
    """Full HTML email, used by the .eml download, the clipboard copy and the in-app preview.

    ``preview=True`` inlines images as data URIs (needed for the clipboard copy and the preview); the real
    .eml references them by ``cid:`` (attached inline). ``preview=True`` also leaves out the inbox preheader
    and dark-mode rules, which only make sense in a real mail client. Pure HTML + CSS: no scripts, because
    mail clients remove them.
    """
    rows: list[str] = [
        f"<tr><td class='dt-pad' bgcolor='#1e3a8a' style='background:#1e3a8a;padding:20px 24px;"
        "border-radius:10px 10px 0 0;border-bottom:4px solid #3b82f6;'>"
        "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'><tr>"
        "<td valign='middle'>"
        f"<div class='dt-title' style='font:800 22px/1.2 {_FONT};color:#ffffff;'>DataTalk.AI</div>"
        f"<div style='font:12px/1.5 {_FONT};color:#bfdbfe;padding-top:2px;'>Healthcare Appeals Intelligence Platform</div>"
        "</td><td class='dt-stack' align='right' valign='middle'>"
        f"<div style='font:700 10px {_FONT};letter-spacing:1.2px;color:#93c5fd;'>ANALYSIS REPORT</div>"
        f"<div style='font:12px/1.5 {_FONT};color:#ffffff;'>{esc(report.generated_at)}</div>"
        "</td></tr></table></td></tr>"
    ]
    if report.question:
        rows.append(_heading("Question"))
        rows.append(_row(f"<div class='dt-t' style='font:600 15px/1.5 {_FONT};color:#0f172a;'>{esc(report.question)}</div>"))
    rows.append(_heading("AI Insight Findings", "sec-findings"))
    rows.append(_row(_callout(md_to_html(report.answer_text)
                              or f"<p style='font:14px {_FONT};color:#64748b;'>No narrative.</p>")))

    if report.images:
        rows.append(_heading("Charts", "sec-charts"))
        total = len(report.images)
        for n, img in enumerate(report.images, 1):
            src = ("data:image/png;base64," + base64.b64encode(img.data).decode("ascii")) if preview \
                else "cid:" + img.cid.strip("<>")
            label = f"Chart {n} of {total}" if total > 1 else "Chart"
            rows.append(_row(
                f"<div class='dt-muted' style='font:700 10.5px {_FONT};letter-spacing:.8px;text-transform:uppercase;"
                f"color:#64748b;padding-bottom:6px;'>{label}</div>"
                "<table class='dt-imgbox' role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' "
                "bgcolor='#ffffff' style='background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;'><tr>"
                "<td style='padding:6px;'>"
                f"<img class='dt-img' src='{src}' width='580' alt='{esc(img.alt)}' "
                "style='display:block;width:100%;max-width:580px;height:auto;border:0;border-radius:4px;'>"
                "</td></tr></table>", "0 24px 16px"))

    if report.sections:
        rows.append(_heading("Evidence at a Glance", "sec-kpis"))
        for sec in report.sections:
            if sec["title"]:
                rows.append(_row(f"<div class='dt-t' style='font:700 13px/1.5 {_FONT};color:#0f172a;'>{esc(sec['title'])}</div>",
                                 "4px 24px 4px"))
            if sec["finding"]:
                rows.append(_row(md_to_html(sec["finding"]), "0 24px 4px"))
            if sec["tiles"]:
                rows.append(_row(_tiles_html(sec["tiles"]), "0 20px 10px"))

    rows.append(
        f"<tr><td class='dt-pad dt-foot dt-muted' style='padding:16px 24px 20px;border-top:1px solid #e2e8f0;"
        f"font:11px/1.6 {_FONT};color:#64748b;'>"
        f"Generated by DataTalk.AI on {esc(report.generated_at)} from governed appeals data. "
        "Contains appeals analytics — share only with authorised recipients.<br>"
        f"<strong>{esc(FOOTER_TEXT)}</strong></td></tr>"
    )
    preheader = "" if preview else (
        "<div style='display:none;font-size:1px;line-height:1px;max-height:0;max-width:0;opacity:0;overflow:hidden;"
        f"color:#f1f5f9;'>{esc(short(_plain(report.answer_text), 120))}"
        + "&nbsp;&zwnj;" * 40 + "</div>")
    styles = _EMAIL_STYLE + ("" if preview else _EMAIL_DARK)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        + ("" if preview else "<meta name='color-scheme' content='light dark'><meta name='supported-color-schemes' content='light dark'>")
        + f"<title>{esc(report.subject)}</title><style>{styles}</style></head>"
        "<body class='dt-bg' style='margin:0;padding:0;background:#f1f5f9;'>" + preheader
        + "<table class='dt-bg' role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' bgcolor='#f1f5f9' "
        "style='background:#f1f5f9;'><tr><td class='dt-outer' align='center' style='padding:24px 12px;'>"
        "<table class='dt-card' role='presentation' width='640' cellpadding='0' cellspacing='0' border='0' bgcolor='#ffffff' "
        "style='width:100%;max-width:640px;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;'>"
        + "".join(rows) + "</table></td></tr></table></body></html>"
    )


def compose_text(report: EmailReport) -> str:
    """Plain-text alternative for clients that do not render HTML."""
    lines = ["DataTalk.AI — Healthcare Appeals Intelligence", ""]
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
# link draft (text only) — same helpers as the existing "Draft in Outlook" buttons
# ═══════════════════════════════════════════════════════════════════════════════
def link_draft(message: dict, question: str = "") -> tuple[str, str]:
    """``(subject, body)`` for ``outlook_mail.mail_url``. Cheap: text only, no images."""
    insight = message.get("insight") if isinstance(message.get("insight"), dict) else None
    question = (question or (insight or {}).get("question") or "").strip()
    answer = str(message.get("content") or "").strip()
    subject = clean_subject(f"[DataTalk.AI] {short(spell_pp(question or _plain(answer)), 90) or 'Analysis report'}")
    lines = ["Hi,", "", "Please find the DataTalk.AI analysis below.", ""]
    if question:
        lines += ["QUESTION", short(question, 220), ""]
    lines += ["AI INSIGHT FINDINGS", short(_plain(answer), 600), ""]
    sections = _sections(insight)
    if sections:
        lines.append("EVIDENCE AT A GLANCE")
        for sec in sections:
            if sec["title"]:
                lines.append(sec["title"])
            lines += [f"- {t['label']}: {t['value']}" for t in sec["tiles"]]
        lines.append("")
    lines += ["Charts are not part of this link draft; use \"Draft with charts (.eml)\" in DataTalk.AI to include them.",
              "", "Thanks,"]
    return subject, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# .eml draft (full HTML + charts), opens in Outlook as an unsent message
# ═══════════════════════════════════════════════════════════════════════════════
def build_eml(report: EmailReport) -> bytes:
    """RFC 822 file with ``X-Unsent: 1`` and no To / From: Outlook opens it as an editable draft."""
    msg = EmailMessage()
    msg["Subject"] = report.subject
    msg["X-Unsent"] = "1"
    msg.set_content(compose_text(report))
    msg.add_alternative(compose_html(report), subtype="html")
    html_part = msg.get_payload()[1]
    for i, img in enumerate(report.images, 1):
        html_part.add_related(img.data, "image", "png", cid=img.cid, disposition="inline",
                              filename=f"chart_{i}.png")
    return msg.as_bytes()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "report"


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════
def _js(value: str) -> str:
    """JSON string literal that is safe inside an inline <script>."""
    return json.dumps(value).replace("</", "<\\/")


def _copy_and_open_button(report: EmailReport, link_subject: str, link_body: str) -> None:
    """One-click: copy the rich email to the clipboard, then open the Outlook compose window.

    Runs in the browser (a click inside the component is the user gesture the clipboard needs).
    Compose opens with an empty body after a successful copy so the paste is clean; if copying fails
    it opens with the text draft so the user is never left with nothing.
    """
    dark = theme_type() == "dark"
    color, border, hover = ("#fafafa", "rgba(250,250,250,.2)", "#60a5fa") if dark else ("#31333f", "rgba(49,51,63,.2)", "#2563eb")
    html_doc = (
        "<style>html,body{margin:0;background:transparent;}"
        f"button{{width:100%;height:46px;border-radius:8px;border:1px solid {border};background:transparent;color:{color};"
        "font:400 14px 'Source Sans Pro','Segoe UI',Arial,sans-serif;cursor:pointer;}"
        f"button:hover{{border-color:{hover};color:{hover};}}"
        "#s{font:12px 'Segoe UI',Arial,sans-serif;color:#94a3b8;margin-top:6px;min-height:16px;}</style>"
        "<button id='b' type='button'>📋 Copy email &amp; open Outlook</button><div id='s' role='status'></div>"
        "<div id='src' contenteditable='true' aria-hidden='true' style='position:fixed;left:-99999px;top:0;'></div>"
        "<script>"
        f"const HTML={_js(compose_html(report, preview=True))};const PLAIN={_js(compose_text(report))};"
        f"const URL_EMPTY={_js(mail_url(report.subject, ''))};const URL_TEXT={_js(mail_url(link_subject, link_body))};"
        "async function copyRich(){"
        "try{if(navigator.clipboard&&window.ClipboardItem){await navigator.clipboard.write([new ClipboardItem({"
        "'text/html':new Blob([HTML],{type:'text/html'}),'text/plain':new Blob([PLAIN],{type:'text/plain'})})]);return true;}}catch(e){}"
        "try{const el=document.getElementById('src');el.innerHTML=HTML;const r=document.createRange();r.selectNodeContents(el);"
        "const sel=window.getSelection();sel.removeAllRanges();sel.addRange(r);const ok=document.execCommand('copy');"
        "sel.removeAllRanges();el.innerHTML='';return ok;}catch(e){return false;}}"
        "document.getElementById('b').onclick=async()=>{const s=document.getElementById('s');const ok=await copyRich();"
        "window.open(ok?URL_EMPTY:URL_TEXT,'_blank','noopener');"
        "s.textContent=ok?'Copied. In the Outlook message body press Ctrl+V (Cmd+V on Mac).':"
        "'Clipboard blocked by the browser, opened the text draft instead.';};"
        "</script>"
    )
    components.html(html_doc, height=78)


_PREVIEW_TEMPLATE = """
<style>
*{box-sizing:border-box}html,body{margin:0;background:transparent;font-family:'Source Sans Pro','Segoe UI',Arial,sans-serif;color:__FG__}
.wrap{border:1px solid __BORDER__;border-radius:12px;overflow:hidden;background:__PANEL__}
.hdr{padding:10px 14px;border-bottom:1px solid __BORDER__}
.row{display:flex;gap:10px;align-items:baseline;font-size:13px;line-height:1.7}
.k{width:62px;flex:0 0 62px;color:__MUTED__;font-size:12px}.v{flex:1;min-width:0;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ph{font-weight:400;color:__MUTED__;font-style:italic}
.att{flex-wrap:wrap;gap:6px;padding-top:4px}.pill{display:inline-flex;gap:6px;align-items:center;border:1px solid __BORDER__;border-radius:999px;padding:2px 10px;font-size:11.5px;cursor:pointer;background:transparent;color:inherit}
.pill:hover{border-color:__ACCENT__;color:__ACCENT__}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;padding:8px 14px;border-bottom:1px solid __BORDER__}
.seg{display:inline-flex;border:1px solid __BORDER__;border-radius:8px;overflow:hidden}
.seg button,.chip{font-weight:600;font-size:12.5px;font-family:inherit;border:0;background:transparent;color:inherit;padding:6px 12px;cursor:pointer}
.seg button.on{background:__ACCENT__;color:#fff}.seg button:not(.on):hover{background:rgba(127,127,127,.15)}
.chip{border:1px solid __BORDER__;border-radius:999px;padding:4px 11px;margin-left:6px}.chip:hover{border-color:__ACCENT__;color:__ACCENT__}
.stage{height:470px;background:__STAGE__;display:flex;justify-content:center;overflow:hidden}
iframe{height:100%;width:100%;border:0;background:#fff;transition:width .25s ease;box-shadow:0 0 0 1px rgba(127,127,127,.2)}
pre{margin:0;height:100%;width:100%;overflow:auto;padding:16px 18px;font:13px/1.65 Consolas,Menlo,monospace;white-space:pre-wrap;color:__FG__}
#lb{position:fixed;inset:0;background:rgba(2,6,23,.88);display:none;align-items:center;justify-content:center;flex-direction:column;z-index:9;gap:10px}
#lb img{max-width:94%;max-height:80%;background:#fff;border-radius:8px}
#lb .cap{color:#e2e8f0;font-size:13px}#lb button{position:absolute;top:50%;transform:translateY(-50%);border:0;border-radius:50%;width:38px;height:38px;font-size:18px;cursor:pointer;background:rgba(255,255,255,.18);color:#fff}
#lb .x{top:14px;right:16px;transform:none}#lb .pv{left:14px}#lb .nx{right:14px}
@media (prefers-reduced-motion:reduce){iframe{transition:none}}
</style>
<div class="wrap">
 <div class="hdr">
  <div class="row"><span class="k">Subject</span><span class="v">__SUBJECT__</span></div>
  <div class="row"><span class="k">To</span><span class="v ph">add the recipient in Outlook</span></div>
  <div class="row att" id="att">__ATTACH__</div>
 </div>
 <div class="bar">
  <div class="seg" id="views"><button class="on" data-v="email">✉️ Email</button><button data-v="text">📝 Plain text</button></div>
  <div id="jump">__JUMP__</div>
  <div class="seg" id="devs"><button class="on" data-d="100%">💻 Desktop</button><button data-d="390px">📱 Mobile</button></div>
 </div>
 <div class="stage"><iframe id="f" title="Email preview"></iframe><pre id="t" hidden></pre></div>
</div>
<div id="lb"><button class="x" aria-label="Close">✕</button><button class="pv" aria-label="Previous">‹</button><img alt=""><div class="cap"></div><button class="nx" aria-label="Next">›</button></div>
<script>
const HTML=__HTML__,PLAIN=__PLAIN__;
const f=document.getElementById('f'),t=document.getElementById('t'),lb=document.getElementById('lb');
t.textContent=PLAIN;let imgs=[],cur=0;
function show(i){if(!imgs.length)return;cur=(i+imgs.length)%imgs.length;const im=imgs[cur];lb.querySelector('img').src=im.src;
 lb.querySelector('.cap').textContent=(im.alt||'Chart')+'  ('+(cur+1)+' / '+imgs.length+')';lb.style.display='flex';}
function hide(){lb.style.display='none';}
f.onload=()=>{try{const d=f.contentDocument;const st=d.createElement('style');
 st.textContent="table.dt-kpi{transition:transform .15s,box-shadow .15s}table.dt-kpi:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(37,99,235,.28)}img{cursor:zoom-in;transition:box-shadow .15s}img:hover{box-shadow:0 6px 20px rgba(0,0,0,.25)}html{scroll-behavior:smooth}";
 d.head.appendChild(st);imgs=[...d.querySelectorAll('img')];imgs.forEach((im,i)=>im.onclick=()=>show(i));}catch(e){}};
f.srcdoc=HTML;
document.getElementById('views').onclick=e=>{const b=e.target.closest('button');if(!b)return;
 for(const x of document.querySelectorAll('#views button'))x.classList.toggle('on',x===b);
 const text=b.dataset.v==='text';f.hidden=text;t.hidden=!text;};
document.getElementById('devs').onclick=e=>{const b=e.target.closest('button');if(!b)return;
 for(const x of document.querySelectorAll('#devs button'))x.classList.toggle('on',x===b);f.style.width=b.dataset.d;};
document.getElementById('jump').onclick=e=>{const b=e.target.closest('[data-go]');if(!b)return;
 document.querySelector('#views button[data-v=email]').click();try{f.contentDocument.getElementById(b.dataset.go).scrollIntoView({behavior:'smooth',block:'start'});}catch(x){}};
document.getElementById('att').onclick=e=>{const b=e.target.closest('[data-img]');if(b)show(+b.dataset.img);};
lb.onclick=e=>{if(e.target===lb||e.target.classList.contains('x'))hide();else if(e.target.classList.contains('pv'))show(cur-1);else if(e.target.classList.contains('nx'))show(cur+1);};
document.addEventListener('keydown',e=>{if(lb.style.display!=='flex')return;if(e.key==='Escape')hide();if(e.key==='ArrowLeft')show(cur-1);if(e.key==='ArrowRight')show(cur+1);});
</script>
"""


def render_interactive_preview(report: EmailReport) -> None:
    """In-app preview that behaves like a mail client: header fields, Email / Plain-text tabs, desktop and
    mobile widths, jump-to-section chips, click-to-zoom charts (arrow keys / Esc) and KPI hover. It is
    display-only: the real email (``compose_html``) is unchanged and contains no scripts."""
    dark = theme_type() == "dark"
    colors = {
        "__FG__": "#fafafa" if dark else "#31333f", "__MUTED__": "#94a3b8" if dark else "#64748b",
        "__PANEL__": "rgba(255,255,255,.03)" if dark else "rgba(49,51,63,.02)",
        "__BORDER__": "rgba(250,250,250,.16)" if dark else "rgba(49,51,63,.18)",
        "__ACCENT__": "#3b82f6", "__STAGE__": "#0b1220" if dark else "#e2e8f0",
    }
    attach = "".join(
        f"<button class='pill' data-img='{i - 1}' title='Click to zoom'>🖼️ chart_{i}.png <span class='ph'>{len(img.data) // 1024} KB</span></button>"
        for i, img in enumerate(report.images, 1)) or "<span class='ph'>No attachments</span>"
    jump = "<span class='chip' data-go='sec-findings'>Findings</span>"
    if report.images:
        jump += "<span class='chip' data-go='sec-charts'>Charts</span>"
    if report.sections:
        jump += "<span class='chip' data-go='sec-kpis'>KPIs</span>"
    values = {**colors, "__SUBJECT__": esc(report.subject), "__ATTACH__": attach, "__JUMP__": jump,
              "__HTML__": _js(compose_html(report, preview=True)), "__PLAIN__": _js(compose_text(report))}
    # single pass, so a placeholder-like string inside the content can never be substituted again
    page = re.sub(r"__[A-Z]+__", lambda m: values.get(m.group(0), m.group(0)), _PREVIEW_TEMPLATE)
    components.html(page, height=640)


def _signature(message: dict, question: str) -> str:
    return hashlib.sha1("|".join([question, str(message.get("content") or ""),
                                  str(len(message.get("figs") or []))]).encode("utf-8")).hexdigest()


def render_email_draft(message: dict, key: str, question: str = "") -> None:
    """Draw the draft controls for one answer. Never raises into the page."""
    try:
        meta = message.get("meta") or {}
        if meta.get("route") == "error" or not (message.get("content") or message.get("figs")):
            return
        sig = _signature(message, question)
        subject, body = link_draft(message, question)
        st.markdown(MAIL_CSS, unsafe_allow_html=True)
        left, right = st.columns([3.2, 2], vertical_alignment="center")
        with left:
            st.markdown(
                "<div class='dt-plan-bar'><div><div class='dt-plan-title'>Share this answer</div>"
                "<div class='dt-plan-sub'>Outlook draft with findings and KPI values · add the recipient in To</div></div>"
                + mail_button(mail_url(subject, body), "Draft in Outlook",
                              "Opens an Outlook draft with the question, findings and KPI values filled in",
                              variant="secondary")
                + "</div>", unsafe_allow_html=True)
        store: dict = st.session_state.setdefault("_email_drafts", {})
        with right:
            if st.button("🖼️ Draft with charts (.eml)", key=f"{key}_eml_btn", width="stretch",
                         help="Builds a draft that includes the charts. Opens in Outlook as an unsent message."):
                with st.spinner("Preparing draft…"):
                    report = build_report(message, question)
                    store.pop(key, None)
                    store[key] = (sig, report, build_eml(report))
                    while len(store) > _DRAFT_CACHE_SIZE:
                        store.pop(next(iter(store)))
            ready = store.get(key)
            if ready and ready[0] == sig:
                st.download_button("⬇️ Download draft", ready[2], file_name=f"DataTalk_draft_{_slug(question or 'answer')}_{date.today().isoformat()}.eml",
                                   mime="message/rfc822", key=f"{key}_eml_dl", on_click="ignore", width="stretch")
                _copy_and_open_button(ready[1], subject, body)
        ready = store.get(key)
        if ready and ready[0] == sig:
            report = ready[1]
            if report.chart_notice:
                st.warning(report.chart_notice)
            st.caption("Download the draft and double-click it to open in Outlook — or use Copy email & open Outlook, then press "
                       "Ctrl+V in the message body. Add the recipient in To either way.")
            with st.expander("👁️ Preview draft", expanded=False):
                render_interactive_preview(report)
    except Exception:
        log.exception("email draft controls unavailable")


__all__ = ["render_email_draft", "render_interactive_preview", "link_draft", "build_report", "build_eml", "compose_html", "compose_text"]