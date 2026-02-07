#!/usr/bin/env python3
"""
crowd_server.py - Crowd Prompt Web Server
═══════════════════════════════════════════════════════════════════════════════

Lightweight web server for live crowd prompt submission:
  - GET  /          → HTML submit form
  - POST /submit    → Submit prompt to queue
  - GET  /status    → Queue status JSON
  - GET  /queue     → Top N queued items
  - POST /pop       → Internal: pop next item (TurboEngine only)

Runs in a background thread, never blocks TURBO engine.
Uses Flask (stdlib-friendly) with optional auth token.

Part of QonQrete Visual FaQtory v0.3.5-beta
"""
import json
import time
import socket
import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# HTML template for the submit page
SUBMIT_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>🎛️ FEED THE MACHINE — Visual FaQtory</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0a; color: #e0e0e0;
    font-family: 'Courier New', monospace;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 20px;
  }
  h1 { color: #00ff88; font-size: 1.8em; margin-bottom: 5px; text-align: center; }
  .subtitle { color: #666; font-size: 0.8em; margin-bottom: 20px; text-align: center; }
  .form-box {
    background: #151515; border: 1px solid #333; border-radius: 8px;
    padding: 20px; width: 100%; max-width: 400px;
  }
  label { display: block; color: #888; font-size: 0.8em; margin-bottom: 4px; }
  input, textarea {
    width: 100%; padding: 10px; margin-bottom: 12px;
    background: #0a0a0a; border: 1px solid #444; border-radius: 4px;
    color: #fff; font-family: inherit; font-size: 1em;
  }
  input:focus, textarea:focus { border-color: #00ff88; outline: none; }
  textarea { resize: vertical; min-height: 60px; }
  button {
    width: 100%; padding: 12px; background: #00ff88; color: #000;
    border: none; border-radius: 4px; font-weight: bold;
    font-size: 1.1em; cursor: pointer; font-family: inherit;
  }
  button:hover { background: #00cc6a; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
  .status { margin-top: 15px; padding: 10px; border-radius: 4px; text-align: center; display: none; }
  .status.ok { background: #002211; border: 1px solid #00ff88; color: #00ff88; display: block; }
  .status.err { background: #220000; border: 1px solid #ff4444; color: #ff4444; display: block; }
  .rules { color: #555; font-size: 0.7em; margin-top: 12px; text-align: center; }
  .queue-info { color: #888; font-size: 0.8em; margin-top: 10px; text-align: center; }
</style>
</head>
<body>
<h1>🎛️ FEED THE MACHINE</h1>
<div class="subtitle">Visual FaQtory — Crowd Queue</div>
<div class="form-box">
  <label for="name">Your Name (optional)</label>
  <input type="text" id="name" placeholder="anon" maxlength="30">
  <label for="prompt">Your Visual Prompt</label>
  <textarea id="prompt" placeholder="neon skull melting into bass waves..." maxlength="MAX_LEN"></textarea>
  <button id="btn" onclick="submitPrompt()">SEND IT 🚀</button>
  <div id="status" class="status"></div>
  <div id="queue-info" class="queue-info"></div>
</div>
<div class="rules">
  Max MAX_LEN chars · MIN_LEN min · RATE_LIMIT submissions per RATE_WINDOW s · Keep it stage-safe 🎶
</div>
<script>
async function submitPrompt() {
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.className = 'status'; status.style.display = 'none';
  try {
    const resp = await fetch('/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        name: document.getElementById('name').value || 'anon',
        prompt: document.getElementById('prompt').value
      })
    });
    const data = await resp.json();
    if (data.ok) {
      status.className = 'status ok';
      status.textContent = '✅ Queued! Position: ' + data.queue_depth;
      document.getElementById('prompt').value = '';
    } else {
      status.className = 'status err';
      status.textContent = '❌ ' + (data.reason || 'Rejected');
    }
  } catch(e) {
    status.className = 'status err';
    status.textContent = '❌ Network error';
  }
  status.style.display = 'block';
  setTimeout(() => { btn.disabled = false; }, 3000);
}
async function updateQueue() {
  try {
    const resp = await fetch('/status');
    const data = await resp.json();
    document.getElementById('queue-info').textContent =
      'Queue: ' + data.queue_depth + ' prompts waiting';
  } catch(e) {}
}
setInterval(updateQueue, 5000);
updateQueue();
</script>
</body>
</html>"""


def get_local_ip() -> str:
    """Detect local network IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class CrowdPromptServer:
    """
    Flask-based crowd prompt server.
    Runs in a background thread, shares PromptQueue instance with TurboEngine.
    """

    def __init__(self, queue, config: Optional[Dict[str, Any]] = None):
        self.queue = queue
        cfg = config or {}
        server_cfg = cfg.get('server', {})
        self.bind = server_cfg.get('bind', '0.0.0.0')
        self.port = server_cfg.get('port', 7777)
        self.public_url = server_cfg.get('public_base_url')
        self.auth_enabled = server_cfg.get('auth', {}).get('enabled', False)
        self.auth_token = server_cfg.get('auth', {}).get('token')

        mod_cfg = cfg.get('moderation', {})
        self.max_len = mod_cfg.get('max_len', 120)
        self.min_len = mod_cfg.get('min_len', 3)

        queue_cfg = cfg.get('queue', {})
        ip_rl = queue_cfg.get('per_ip_rate_limit', {})
        self.rate_limit = ip_rl.get('max_requests', 3)
        self.rate_window = ip_rl.get('window_seconds', 30)

        self._thread = None
        self._app = None

    def _check_auth(self, request_obj) -> bool:
        """Check auth token if enabled."""
        if not self.auth_enabled:
            return True
        token = (
            request_obj.headers.get('X-CROWD-TOKEN') or
            request_obj.args.get('token')
        )
        return token == self.auth_token

    def _build_app(self):
        """Build Flask app with routes."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.error("[CrowdServer] Flask not installed. pip install flask")
            raise

        app = Flask(__name__)
        app.logger.setLevel(logging.WARNING)  # Quiet Flask logs
        queue = self.queue
        server = self

        @app.route('/')
        def index():
            html = SUBMIT_PAGE_HTML
            html = html.replace('MAX_LEN', str(server.max_len))
            html = html.replace('MIN_LEN', str(server.min_len))
            html = html.replace('RATE_LIMIT', str(server.rate_limit))
            html = html.replace('RATE_WINDOW', str(server.rate_window))
            return html

        @app.route('/submit', methods=['POST'])
        def submit():
            if not server._check_auth(request):
                return jsonify({'ok': False, 'reason': 'auth required'}), 401

            data = request.get_json(silent=True) or {}
            prompt = data.get('prompt', '')
            name = data.get('name', 'anon')
            ip = request.remote_addr or ''

            ok, reason = queue.submit(prompt, name, ip)
            return jsonify({
                'ok': ok,
                'reason': reason,
                'queue_depth': queue.depth(),
            })

        @app.route('/status')
        def status():
            peek = queue.peek_next()
            return jsonify({
                'queue_depth': queue.depth(),
                'next_prompt_preview': peek.prompt[:40] if peek else None,
                'stats': queue.stats(),
                'server_time': time.time(),
            })

        @app.route('/queue')
        def queue_list():
            items = queue.list_top(10)
            return jsonify([i.to_dict() for i in items])

        @app.route('/pop', methods=['POST'])
        def pop():
            item = queue.pop_next()
            if item:
                return jsonify(item.to_dict())
            return jsonify(None)

        self._app = app
        return app

    def start(self):
        """Start server in background thread."""
        self._build_app()

        url = self.public_url or f"http://{get_local_ip()}:{self.port}"

        def run():
            try:
                import werkzeug.serving
                werkzeug.serving.run_simple(
                    self.bind, self.port, self._app,
                    use_reloader=False, use_debugger=False,
                    threaded=True,
                )
            except Exception as e:
                logger.error(f"[CrowdServer] Server died: {e}")

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

        logger.info("=" * 50)
        logger.info(f"🎛️  CROWD SERVER LIVE")
        logger.info(f"   URL: {url}")
        logger.info(f"   Port: {self.port}")
        logger.info(f"   Auth: {'enabled' if self.auth_enabled else 'disabled'}")
        logger.info("=" * 50)

        return url

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


__all__ = ['CrowdPromptServer', 'get_local_ip']
