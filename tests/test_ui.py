import asyncio
import importlib.util
from threading import Event
import unittest

from pianissimo_context.app_runtime import AppRuntime
from pianissimo_context.live import LiveSession


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Install app extra to test UI")
class UITests(unittest.TestCase):
    def test_ui_handlers_edit_protection_save_and_short_labels(self):
        from pianissimo_context.app import build_app
        entered = Event()
        async def capture(broker, **kwargs):
            broker.controller = None
            async with LiveSession(broker) as session:
                session.push("Första segmentet.")
                broker.emit({"type": "audio_ready"})
                entered.set()
                await kwargs["stop_event"].wait()
        runtime = AppRuntime(capture)
        try:
            app, _ = build_app(runtime, devices=[("Microphone", 0)], models=["model"])
            handlers = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
            handlers["start_recording"](0, "model", "Initial kontext")
            self.assertTrue(entered.wait(3))
            result = handlers["begin_edit"]()
            self.assertTrue(result[0])
            self.assertFalse(result[1]["visible"])
            self.assertEqual(result[4]["value"], "Initial kontext")
            self.assertTrue(result[4]["visible"])
            poll_outputs = next(f.outputs for f in app.fns.values() if f.fn and f.fn.__name__ == "poll")
            self.assertNotIn("context-draft", [output.elem_id for output in poll_outputs])
            # A timer response never sends a context value during editing.
            poll = handlers["poll"](True, -1)
            self.assertNotIn("value", poll[2])
            self.assertEqual(poll[3], -1)
            saved = handlers["save_edit"]("Rättad kontext")
            self.assertFalse(saved[0])
            self.assertEqual(saved[1]["value"], "Rättad kontext")
            self.assertTrue(saved[1]["visible"])
            self.assertFalse(saved[6]["visible"])
            labels = [c["props"].get("label", "") for c in app.config["components"]]
            self.assertIn("Ljudkälla", labels)
            self.assertIn("Kontext", labels)
            self.assertTrue(all("kommunfullmäktige" not in label.casefold() for label in labels))
            self.assertTrue(all(not c["props"].get("info") for c in app.config["components"]))
            runtime.stop()
            runtime._future.result(timeout=3)
        finally:
            runtime.shutdown()
