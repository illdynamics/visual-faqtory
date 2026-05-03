#!/usr/bin/env python3
"""
server.py — Crowd Control FastAPI Server
═══════════════════════════════════════════════════════════════════════════════

Runs on the Visual FaQtory / visuals generation machine.
Serves the crowd prompt page, QR code, queue API, overlay, and status.

Routes (under configurable prefix, default /visuals):
  GET  {prefix}/              → HTML form + live queue stats
  POST {prefix}/api/submit    → Submit a prompt
  GET  {prefix}/api/next      → Claim next prompt (token-protected)
  POST {prefix}/api/ack       → Ack claimed prompt as served (token-protected)
  POST {prefix}/api/requeue   → Requeue claimed prompt after failure (token-protected)
  GET  {prefix}/api/health    → Health check + queue length
  GET  {prefix}/api/status    → Public queue preview + counters (overlay data)
  GET  {prefix}/overlay       → OBS browser source overlay
  GET  {prefix}/qr.png        → QR code pointing to public URL

Part of Visual FaQtory v0.9.3-beta
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Query, Header
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .db import CrowdDB
from .filtering import PromptFilter
from .models import CrowdControlConfig

logger = logging.getLogger(__name__)

from ..version import __version__ as _VERSION


def _extract_client_ip(request: Request) -> str:
    """Extract the real client IP respecting reverse proxies."""
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client:
        return request.client.host
    return "unknown"


def _build_html_page(config: CrowdControlConfig) -> str:
    """Build the crowd prompt submission page with live queue stats."""
    prefix = config.prefix.rstrip("/")
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Visual FaQtory — Crowd Control</title>\n'
        '<style>\n'
        '  * { margin: 0; padding: 0; box-sizing: border-box; }\n'
        '  body {\n'
        "    background: #0a0a0a; color: #e0e0e0;\n"
        "    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;\n"
        '    min-height: 100vh; display: flex; align-items: center; justify-content: center;\n'
        '  }\n'
        '  .container { width: 90%; max-width: 520px; padding: 2rem; }\n'
        '  h1 { font-size: 1.4rem; margin-bottom: 0.3rem; color: #00e5ff; text-transform: uppercase; letter-spacing: 0.05em; }\n'
        '  .subtitle { font-size: 0.8rem; color: #555; margin-bottom: 1.5rem; }\n'
        '  textarea {\n'
        '    width: 100%; min-height: 100px; padding: 0.75rem;\n'
        '    background: #151515; color: #e0e0e0; border: 1px solid #333;\n'
        '    border-radius: 6px; font-size: 1rem; resize: vertical; font-family: inherit;\n'
        '  }\n'
        '  textarea:focus { outline: none; border-color: #00e5ff; box-shadow: 0 0 0 2px rgba(0,229,255,0.15); }\n'
        '  .char-count { text-align: right; font-size: 0.75rem; color: #555; margin: 0.25rem 0 0.75rem 0; }\n'
        '  .char-count.over { color: #ff4444; }\n'
        '  button {\n'
        '    width: 100%; padding: 0.75rem; background: #00e5ff; color: #000;\n'
        '    border: none; border-radius: 6px; font-size: 1rem; font-weight: 600;\n'
        '    cursor: pointer; text-transform: uppercase; letter-spacing: 0.05em; transition: opacity 0.2s;\n'
        '  }\n'
        '  button:hover { opacity: 0.9; }\n'
        '  button:disabled { opacity: 0.4; cursor: not-allowed; }\n'
        '  #result { margin-top: 1rem; padding: 0.75rem; border-radius: 6px; font-size: 0.9rem; display: none; }\n'
        '  #result.ok { background: rgba(0,229,255,0.1); border: 1px solid #00e5ff; color: #00e5ff; display: block; }\n'
        '  #result.err { background: rgba(255,68,68,0.1); border: 1px solid #ff4444; color: #ff4444; display: block; }\n'
        '  .stats-bar { display: flex; gap: 1rem; margin: 1.2rem 0; padding: 0.75rem; background: rgba(255,255,255,0.03); border: 1px solid #222; border-radius: 6px; font-size: 0.8rem; flex-wrap: wrap; }\n'
        '  .stat { text-align: center; flex: 1; min-width: 60px; }\n'
        '  .stat .val { font-size: 1.3rem; font-weight: 700; color: #00e5ff; display: block; }\n'
        '  .stat .lbl { font-size: 0.65rem; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }\n'
        '  .upcoming { margin-top: 1rem; padding: 0.75rem; background: rgba(255,255,255,0.02); border: 1px solid #1a1a1a; border-radius: 6px; }\n'
        '  .upcoming h3 { font-size: 0.75rem; color: #555; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }\n'
        '  .upcoming-item { font-size: 0.8rem; color: #aaa; padding: 0.3rem 0; border-bottom: 1px solid #1a1a1a; word-break: break-word; }\n'
        '  .upcoming-item:last-child { border-bottom: none; }\n'
        '  .upcoming-empty { font-size: 0.75rem; color: #444; font-style: italic; }\n'
        '  .links { margin-top: 1.2rem; font-size: 0.7rem; color: #444; }\n'
        '  .links a { color: #00e5ff; text-decoration: none; opacity: 0.6; transition: opacity 0.2s; }\n'
        '  .links a:hover { opacity: 1; }\n'
        '  .branding { text-align: center; margin-top: 2rem; font-size: 0.65rem; color: #333; }\n'
        '  .branding a { color: #444; text-decoration: none; }\n'
        '</style>\n</head>\n<body>\n<div class="container">\n'
        '  <h1>Mutate the visuals:</h1>\n'
        '  <p class="subtitle">Your prompt will be injected into the next visual cycle</p>\n'
        '  <div class="stats-bar" id="statsBar">\n'
        '    <div class="stat"><span class="val" id="sQueued">\u2014</span><span class="lbl">Queued</span></div>\n'
        '    <div class="stat"><span class="val" id="sAccepted">\u2014</span><span class="lbl">Accepted</span></div>\n'
        '    <div class="stat"><span class="val" id="sServed">\u2014</span><span class="lbl">Served</span></div>\n'
        '  </div>\n'
        f'  <textarea id="prompt" maxlength="{config.max_chars}" placeholder="Describe what you want to see..."></textarea>\n'
        f'  <div class="char-count" id="charCount">0 / {config.max_chars}</div>\n'
        '  <button id="submitBtn" onclick="submitPrompt()">Send mutation</button>\n'
        '  <div id="result"></div>\n'
        '  <div class="upcoming" id="upcomingBox">\n'
        '    <h3>Up next</h3>\n'
        '    <div id="upcomingList"><span class="upcoming-empty">Queue empty \u2014 be the first!</span></div>\n'
        '  </div>\n'
        '  <div class="links">\n'
        f'    <a href="{prefix}/qr.png" target="_blank">QR&nbsp;Code</a> &middot;\n'
        f'    <a href="{prefix}/overlay" target="_blank">OBS&nbsp;Overlay</a> &middot;\n'
        f'    <a href="{prefix}/api/status" target="_blank">Status&nbsp;JSON</a> &middot;\n'
        f'    <a href="{prefix}/api/health" target="_blank">Health</a>\n'
        '  </div>\n'
        '  <div class="branding">\n'
        '    Powered by <a href="https://wonq.tv" target="_blank">Visual FaQtory</a> &mdash; Ill Dynamics / WoNQ\n'
        '  </div>\n'
        '</div>\n'
        '<script>\n'
        f'const MAX = {config.max_chars};\n'
        f"const PREFIX = '{prefix}';\n"
        "const textarea = document.getElementById('prompt');\n"
        "const charCount = document.getElementById('charCount');\n"
        "const btn = document.getElementById('submitBtn');\n"
        "const result = document.getElementById('result');\n"
        "textarea.addEventListener('input', () => {\n"
        "  const len = textarea.value.length;\n"
        "  charCount.textContent = len + ' / ' + MAX;\n"
        "  charCount.className = len > MAX ? 'char-count over' : 'char-count';\n"
        '});\n'
        'async function submitPrompt() {\n'
        '  const prompt = textarea.value.trim();\n'
        '  if (!prompt) return;\n'
        '  btn.disabled = true;\n'
        "  btn.textContent = 'Sending...';\n"
        "  result.className = '';\n"
        "  result.style.display = 'none';\n"
        '  try {\n'
        "    const resp = await fetch(PREFIX + '/api/submit', {\n"
        "      method: 'POST',\n"
        "      headers: {'Content-Type': 'application/json'},\n"
        '      body: JSON.stringify({prompt: prompt})\n'
        '    });\n'
        '    const data = await resp.json();\n'
        '    if (data.ok) {\n'
        "      result.textContent = data.message || 'Mutation queued!';\n"
        "      result.className = 'ok';\n"
        "      textarea.value = '';\n"
        "      charCount.textContent = '0 / ' + MAX;\n"
        '      refreshStatus();\n'
        '    } else {\n'
        "      result.textContent = data.message || 'Rejected';\n"
        "      result.className = 'err';\n"
        '    }\n'
        '  } catch (e) {\n'
        "    result.textContent = 'Network error \\u2014 try again';\n"
        "    result.className = 'err';\n"
        '  }\n'
        '  btn.disabled = false;\n'
        "  btn.textContent = 'Send mutation';\n"
        '}\n'
        "textarea.addEventListener('keydown', (e) => {\n"
        "  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitPrompt(); }\n"
        '});\n'
        'function esc(s) {\n'
        "  const d = document.createElement('div');\n"
        '  d.textContent = s;\n'
        '  return d.innerHTML;\n'
        '}\n'
        'async function refreshStatus() {\n'
        '  try {\n'
        "    const r = await fetch(PREFIX + '/api/status?limit=5');\n"
        '    const d = await r.json();\n'
        '    if (!d.ok) return;\n'
        "    document.getElementById('sQueued').textContent = d.queue_length;\n"
        "    document.getElementById('sAccepted').textContent = d.accepted_total;\n"
        "    document.getElementById('sServed').textContent = d.served_total;\n"
        "    const list = document.getElementById('upcomingList');\n"
        '    if (d.next_prompts && d.next_prompts.length > 0) {\n'
        "      list.innerHTML = d.next_prompts.map(p => '<div class=\"upcoming-item\">' + esc(p.prompt) + '</div>').join('');\n"
        '    } else {\n'
        "      list.innerHTML = '<span class=\"upcoming-empty\">Queue empty \\u2014 be the first!</span>';\n"
        '    }\n'
        '  } catch(e) { /* fail silent */ }\n'
        '}\n'
        'refreshStatus();\n'
        'setInterval(refreshStatus, 5000);\n'
        '</script>\n</body>\n</html>'
    )


def _build_overlay_page(config: CrowdControlConfig) -> str:
    """Build the OBS browser source overlay — vertical sidebar layout."""
    prefix = config.prefix.rstrip("/")
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<title>Visual FaQtory Overlay</title>\n'
        '<style>\n'
        '  * { margin: 0; padding: 0; box-sizing: border-box; }\n'
        '  body { background: transparent; font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif; color: #e0e0e0; overflow: hidden; }\n'
        '  .sidebar {\n'
        '    position: fixed; top: 20px; right: 20px; width: 280px;\n'
        '    background: rgba(8,8,8,0.85); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);\n'
        '    border: 1px solid rgba(0,229,255,0.2); border-radius: 12px; padding: 20px;\n'
        '    box-shadow: 0 6px 40px rgba(0,0,0,0.7);\n'
        '    display: flex; flex-direction: column; align-items: center;\n'
        '  }\n'
        '  .qr-wrap { margin-bottom: 14px; }\n'
        '  .qr {\n'
        '    width: 180px; height: 180px; border-radius: 10px;\n'
        '    border: 2px solid rgba(0,229,255,0.3); image-rendering: pixelated;\n'
        '    display: block;\n'
        '  }\n'
        '  .heading {\n'
        '    font-size: 0.8rem; color: #00e5ff; text-transform: uppercase;\n'
        '    letter-spacing: 0.07em; font-weight: 700; text-align: center; margin-bottom: 4px;\n'
        '  }\n'
        '  .url {\n'
        '    font-size: 0.85rem; color: #e0e0e0; text-align: center;\n'
        '    letter-spacing: 0.02em; margin-bottom: 16px; font-weight: 600;\n'
        '  }\n'
        '  .stats { display: flex; gap: 8px; width: 100%; margin-bottom: 14px; }\n'
        '  .stat {\n'
        '    flex: 1; text-align: center; padding: 8px 0;\n'
        '    background: rgba(255,255,255,0.04); border-radius: 6px;\n'
        '    border: 1px solid rgba(255,255,255,0.06);\n'
        '  }\n'
        '  .stat .val { display: block; font-size: 1.2rem; font-weight: 700; color: #00e5ff; }\n'
        '  .stat .lbl { font-size: 0.55rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.04em; }\n'
        '  .divider {\n'
        '    width: 100%; height: 1px; background: rgba(0,229,255,0.15);\n'
        '    margin-bottom: 12px;\n'
        '  }\n'
        '  .queue-title {\n'
        '    font-size: 0.65rem; color: #bbb; text-transform: uppercase;\n'
        '    letter-spacing: 0.06em; margin-bottom: 8px; width: 100%;\n'
        '  }\n'
        '  .queue-list { width: 100%; }\n'
        '  .queue-item {\n'
        '    font-size: 0.78rem; color: #f0f0f0; padding: 7px 10px; margin-bottom: 5px;\n'
        '    background: rgba(255,255,255,0.04); border-radius: 5px;\n'
        '    border-left: 3px solid rgba(0,229,255,0.5);\n'
        '    word-break: break-word; overflow: hidden; text-overflow: ellipsis;\n'
        '    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;\n'
        '  }\n'
        '  .queue-empty { font-size: 0.75rem; color: #999; font-style: italic; padding: 6px 0; }\n'
        '  .brand {\n'
        '    margin-top: 12px; text-align: center; font-size: 0.5rem; color: #777;\n'
        '    width: 100%;\n'
        '  }\n'
        '  .brand a { color: #999; text-decoration: none; }\n'
        '  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }\n'
        '  .sidebar { animation: fadeIn 0.4s ease-out; }\n'
        '</style>\n</head>\n<body>\n'
        '<div class="sidebar" id="overlay">\n'
        '  <div class="qr-wrap">\n'
        f'    <img class="qr" id="qrImg" src="{prefix}/qr.png" alt="QR">\n'
        '  </div>\n'
        '  <div class="heading">Scan QR to control visuals</div>\n'
        f'  <div class="url">{config.public_url}</div>\n'
        '  <div class="stats">\n'
        '    <div class="stat"><span class="val" id="sQueued">\u2014</span><span class="lbl">Queued</span></div>\n'
        '    <div class="stat"><span class="val" id="sAccepted">\u2014</span><span class="lbl">Accepted</span></div>\n'
        '    <div class="stat"><span class="val" id="sServed">\u2014</span><span class="lbl">Served</span></div>\n'
        '  </div>\n'
        '  <div class="divider"></div>\n'
        '  <div class="queue-title">Up next</div>\n'
        '  <div class="queue-list" id="queueList"><span class="queue-empty">Waiting for prompts\u2026</span></div>\n'
        '  <div class="brand">Powered by <a href="https://wonq.tv">Visual FaQtory</a></div>\n'
        '</div>\n'
        '<script>\n'
        f"const PREFIX = '{prefix}';\n"
        "const params = new URLSearchParams(window.location.search);\n"
        "const LIMIT = Math.min(Math.max(parseInt(params.get('limit')) || 3, 1), 10);\n"
        'function esc(s) {\n'
        "  const d = document.createElement('div');\n"
        '  d.textContent = s;\n'
        '  return d.innerHTML;\n'
        '}\n'
        'async function refresh() {\n'
        '  try {\n'
        "    const r = await fetch(PREFIX + '/api/status?limit=' + LIMIT);\n"
        '    const d = await r.json();\n'
        '    if (!d.ok) return;\n'
        "    document.getElementById('sQueued').textContent = d.queue_length;\n"
        "    document.getElementById('sAccepted').textContent = d.accepted_total;\n"
        "    document.getElementById('sServed').textContent = d.served_total;\n"
        "    const list = document.getElementById('queueList');\n"
        '    if (d.next_prompts && d.next_prompts.length > 0) {\n'
        "      list.innerHTML = d.next_prompts.map(p => '<div class=\"queue-item\">' + esc(p.prompt) + '</div>').join('');\n"
        '    } else {\n'
        "      list.innerHTML = '<span class=\"queue-empty\">Waiting for prompts\\u2026</span>';\n"
        '    }\n'
        '  } catch(e) { /* fail silent */ }\n'
        '}\n'
        'refresh();\n'
        'setInterval(refresh, 3000);\n'
        '</script>\n</body>\n</html>'
    )


def create_crowd_app(config: CrowdControlConfig) -> FastAPI:
    """Build and return the Crowd Control FastAPI application."""
    prefix = config.prefix.rstrip("/")

    app = FastAPI(
        title="Visual FaQtory Crowd Control",
        version=_VERSION,
        docs_url=None,
        redoc_url=None,
    )

    db = CrowdDB(config.db_path)
    prompt_filter = PromptFilter(
        badwords_path=config.badwords_path,
        max_chars=config.max_chars,
    )

    # ── Routes ───────────────────────────────────────────────────────────

    @app.get(f"{prefix}/", response_class=HTMLResponse)
    async def crowd_page():
        """Serve the crowd prompt page with live stats."""
        return HTMLResponse(content=_build_html_page(config))

    @app.get(f"{prefix}/overlay", response_class=HTMLResponse)
    async def overlay_page():
        """Serve the OBS browser source overlay."""
        return HTMLResponse(content=_build_overlay_page(config))

    @app.post(f"{prefix}/api/submit")
    async def submit_prompt(request: Request):
        """Validate, filter, rate-limit, and enqueue a crowd prompt."""
        ip = _extract_client_ip(request)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"ok": False, "message": "Invalid JSON body"},
                status_code=400,
            )

        raw_prompt = body.get("prompt", "")
        if not isinstance(raw_prompt, str):
            return JSONResponse(
                {"ok": False, "message": "Prompt must be a string"},
                status_code=400,
            )

        cleaned = prompt_filter.sanitize(raw_prompt)

        ok, reason = prompt_filter.validate(cleaned)
        if not ok:
            db.reject(ip, cleaned[:config.max_chars], reason)
            if reason == "empty_prompt":
                msg = "Prompt cannot be empty"
            elif reason.startswith("too_long"):
                msg = f"Prompt too long (max {config.max_chars} characters)"
            elif reason == "bad_word_detected":
                msg = "Prompt contains prohibited content"
            else:
                msg = "Prompt rejected"
            return JSONResponse({"ok": False, "message": msg}, status_code=400)

        allowed, remaining = db.check_rate_limit(ip, config.rate_limit_seconds)
        if not allowed:
            db.reject(ip, cleaned, f"rate_limited:{remaining}s")
            minutes = remaining // 60
            seconds = remaining % 60
            if minutes > 0:
                wait_str = f"{minutes}m {seconds}s" if seconds else f"{minutes}m"
            else:
                wait_str = f"{seconds}s"
            return JSONResponse(
                {"ok": False, "message": f"Rate limited — wait {wait_str}"},
                status_code=429,
            )

        q_len = db.queue_length()
        if q_len >= config.max_queue:
            db.reject(ip, cleaned, "queue_full")
            return JSONResponse(
                {"ok": False, "message": "Queue is full — try again later"},
                status_code=429,
            )

        sub_id = db.enqueue(ip, cleaned)
        db.update_rate_limit(ip)
        logger.info(f"[CrowdServer] Accepted submission #{sub_id} from {ip}: {cleaned[:60]}...")
        return JSONResponse(
            {"ok": True, "message": "Mutation queued! It will appear in the next visual cycle.", "id": sub_id}
        )

    def _require_token(token: Optional[str], authorization: Optional[str]) -> None:
        provided_token = None
        if authorization and authorization.lower().startswith("bearer "):
            provided_token = authorization[7:].strip()
        elif token:
            provided_token = token
        if not provided_token or provided_token != config.pop_token:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get(f"{prefix}/api/next")
    async def pop_next(
        request: Request,
        token: Optional[str] = Query(None),
        authorization: Optional[str] = Header(None),
    ):
        """Claim the next queued prompt. Token-protected."""
        _require_token(token, authorization)
        claim = db.claim_next(claim_timeout_seconds=config.claim_timeout_seconds)
        if not claim:
            return JSONResponse({"prompt": None, "id": None})
        return JSONResponse({
            "prompt": claim["prompt"],
            "id": int(claim["id"]),
            "claim_id": claim.get("claim_id"),
        })

    @app.post(f"{prefix}/api/ack")
    async def ack_claim(
        request: Request,
        token: Optional[str] = Query(None),
        authorization: Optional[str] = Header(None),
    ):
        """Ack a claimed prompt as served after successful generation."""
        _require_token(token, authorization)
        try:
            body = await request.json()
        except Exception:
            body = {}
        sub_id = body.get("id")
        claim_id = body.get("claim_id")
        if sub_id is None:
            return JSONResponse({"ok": False, "message": "Missing id"}, status_code=400)
        ok = db.ack_served(int(sub_id), claim_id=claim_id)
        if not ok:
            return JSONResponse({"ok": False, "message": "Claim ack rejected"}, status_code=409)
        return JSONResponse({"ok": True, "id": int(sub_id)})

    @app.post(f"{prefix}/api/requeue")
    async def requeue_claim(
        request: Request,
        token: Optional[str] = Query(None),
        authorization: Optional[str] = Header(None),
    ):
        """Requeue a claimed prompt (typically after generation failure)."""
        _require_token(token, authorization)
        try:
            body = await request.json()
        except Exception:
            body = {}
        sub_id = body.get("id")
        claim_id = body.get("claim_id")
        reason = str(body.get("reason", "") or "")
        if sub_id is None:
            return JSONResponse({"ok": False, "message": "Missing id"}, status_code=400)
        ok = db.requeue_claimed(int(sub_id), reason=reason, claim_id=claim_id)
        if not ok:
            return JSONResponse({"ok": False, "message": "Claim requeue rejected"}, status_code=409)
        return JSONResponse({"ok": True, "id": int(sub_id)})

    @app.get(f"{prefix}/api/status")
    async def queue_status(
        limit: int = Query(default=3, ge=1, le=10),
    ):
        """Public queue preview + aggregate counters for overlays.

        No authentication required. Does not expose IP addresses.
        """
        counts = db.status_counts()
        preview = db.queue_preview(limit=limit)
        return JSONResponse({
            "ok": True,
            "version": _VERSION,
            "queue_length": counts["queued"],
            "claimed_length": counts.get("claimed", 0),
            "accepted_total": counts["queued"] + counts.get("claimed", 0) + counts["served"],
            "served_total": counts["served"],
            "rejected_total": counts["rejected"],
            "total_submissions": counts["total"],
            "next_prompts": preview,
            "preview_limit": limit,
        })

    @app.get(f"{prefix}/api/health")
    async def health():
        """Health check with queue stats."""
        return JSONResponse({
            "ok": True,
            "queue_length": db.queue_length(),
            "version": _VERSION,
        })

    @app.get(f"{prefix}/qr.png")
    async def qr_code():
        """Generate and serve a QR code pointing to the public URL."""
        import qrcode
        from PIL import Image

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(config.public_url)
        qr.make(fit=True)

        img: Image.Image = qr.make_image(fill_color="white", back_color="black")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=300",
                "Content-Disposition": 'inline; filename="qr.png"',
            },
        )

    return app
