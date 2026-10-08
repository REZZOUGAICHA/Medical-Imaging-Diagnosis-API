import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import src.api as api

BODY = {"class_name": "Moderate DR", "confidence": 0.49, "probabilities": {"Moderate DR": 0.49, "Mild DR": 0.43}}


@pytest.fixture
def fake_openai_server():
    """Minimal OpenAI-compatible /chat/completions endpoint (stands in for Groq etc.)."""
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["payload"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = {
                "id": "x", "object": "chat.completion", "created": 0, "model": seen["payload"]["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": " A short note. "}}],
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/openai/v1", seen
    server.shutdown()


def test_explain_uses_openai_compatible_endpoint(client, fake_openai_server, monkeypatch):
    base_url, seen = fake_openai_server
    monkeypatch.setattr(api, "EXPLAIN_BASE_URL", base_url)
    monkeypatch.setattr(api, "EXPLAIN_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(api, "EXPLAIN_REASONING_EFFORT", "low")
    monkeypatch.setenv("EXPLAIN_API_KEY", "gsk_test")
    client.app.state.hf_client = api.make_explain_client()

    resp = client.post("/explain", json=BODY)

    assert resp.status_code == 200
    assert resp.json() == {"explanation": "A short note.", "model": "openai/gpt-oss-20b"}
    assert seen["path"] == "/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk_test"
    assert seen["payload"]["model"] == "openai/gpt-oss-20b"
    assert seen["payload"]["reasoning_effort"] == "low"


def test_explain_base_url_without_key_disables_service(monkeypatch):
    monkeypatch.setattr(api, "EXPLAIN_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.delenv("EXPLAIN_API_KEY", raising=False)
    assert api.make_explain_client() is None
