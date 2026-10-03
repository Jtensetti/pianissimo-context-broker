import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest

from pianissimo_context import AdaptiveBroker, LiveSession, Ollama


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_local_http_live_bootstrap_and_patch(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                data = json.loads(request["messages"][1]["content"])
                if data.get("mode") == "initial":
                    result = {"topic": "Pianissimo i offentlig sektor", "terms": ["Pianissimo"], "evidence": []}
                else:
                    result = {"patches": [{"start": 0, "end": 9, "source": "pianisimo",
                                          "replacement": "Pianissimo", "confidence": .99}], "terms": []}
                body = json.dumps({"message": {"content": json.dumps(result)}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            events = []
            broker = AdaptiveBroker(initial_context="Intervju om Pianissimo i offentlig sektor",
                                    controller=Ollama("test-double", f"http://127.0.0.1:{server.server_port}"),
                                    emit=events.append)
            async with LiveSession(broker) as session:
                segment = session.push("pianisimo")
                self.assertEqual(events[0]["type"], "raw")
            self.assertEqual(segment.text, "Pianissimo")
            self.assertEqual(len(requests), 2)
            self.assertIn("Track the current topic", requests[0]["messages"][0]["content"])
            self.assertIn("ASR context controller", requests[1]["messages"][0]["content"])
            self.assertEqual(events[-1]["type"], "commit")
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(timeout=1)
