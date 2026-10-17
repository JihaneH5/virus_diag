"""Tiny HTTP API (standard library only).

Start:  python -m app.api
Test :  curl -X POST http://localhost:8000/diagnose -H "Content-Type: application/json" ^
             -d "{\"age\":45,\"temperature\":38.9,\"cough\":1,\"fatigue\":1,\"loss_smell\":1,\"contact\":1,\"oxygen_saturation\":92,\"heart_rate\":105,\"wbc_count\":11.2,\"crp\":55}"
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.clinical_predictor import ClinicalPredictor  # noqa: E402
from pipeline.trainer import Trainer  # noqa: E402

MODEL_PATH = os.path.join("models", "neural_network.pkl")


def load_predictor(path=MODEL_PATH):
    model, preprocessor = Trainer.load(path)
    return ClinicalPredictor(model, preprocessor)


def make_handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/diagnose":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                prob = predictor.probability(data)
                self._send(200, {"diagnosis": predictor.diagnose(data),
                                 "probability": round(prob, 4)})
            except Exception as exc:  # invalid JSON, bad values...
                self._send(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    return Handler


def run(port=8000):
    predictor = load_predictor()
    server = HTTPServer(("127.0.0.1", port), make_handler(predictor))
    print(f"API running on http://127.0.0.1:{port}  (POST /diagnose, GET /health)")
    server.serve_forever()


if __name__ == "__main__":
    run()
