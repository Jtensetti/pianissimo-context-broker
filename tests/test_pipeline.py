import asyncio
from dataclasses import asdict
from threading import Event
import unittest

from pianissimo_context import AdaptiveBroker, LiveSession, Patch
from pianissimo_context.app_runtime import AppRuntime


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_unique_source_alignment_without_model_counting_offsets(self):
        class Model:
            async def propose(self, payload):
                return {"patches": [{"source": "aj sveden", "replacement": "AI Sweden",
                                     "confidence": .97, "reason": "Phonetic error", "kind": "asr_error"}]}
        b = AdaptiveBroker(controller=Model())
        unique = b.add("Vi pratar med aj sveden.")
        await b.repair(unique.id)
        self.assertEqual(unique.text, "Vi pratar med AI Sweden.")
        repeated = b.add("aj sveden eller aj sveden")
        await b.repair(repeated.id)
        self.assertEqual(repeated.text, repeated.raw)

    async def test_raw_waits_for_review_and_phonetic_repair_is_first_display(self):
        entered, release = asyncio.Event(), asyncio.Event()
        events = []
        class Model:
            async def propose(self, payload):
                entered.set()
                await release.wait()
                return {"patches": [asdict(Patch(0, 9, "aj sveden", "AI Sweden", .97,
                                                "Phonetic error; organisation in background", "asr_error"))]}
        broker = AdaptiveBroker(initial_context="Intervju med AI Sweden", controller=Model(), emit=events.append)
        async with LiveSession(broker) as session:
            segment = session.push("aj sveden")
            await entered.wait()
            self.assertEqual(events[0]["type"], "raw")
            self.assertFalse(any(e["type"] == "display" for e in events))
            release.set()
        displays = [e for e in events if e["type"] == "display"]
        self.assertEqual(len(displays), 1)
        self.assertEqual(displays[0]["text"], "AI Sweden")
        self.assertEqual(displays[0]["review_status"], "reviewed")
        self.assertEqual(segment.raw, "aj sveden")

    async def test_controller_failure_explicitly_falls_back_without_losing_text(self):
        class Model:
            async def propose(self, payload):
                raise TimeoutError("private payload")
        events = []
        async with LiveSession(AdaptiveBroker(controller=Model(), emit=events.append)) as session:
            session.push("Behåll originalet.")
        display = next(e for e in events if e["type"] == "display")
        self.assertEqual(display["text"], "Behåll originalet.")
        self.assertEqual(display["review_status"], "error")
        self.assertNotIn("private payload", str(events))

    async def test_expired_pending_segment_displays_raw_and_rejects_late_reply(self):
        now, events = [0], []
        entered, release = asyncio.Event(), asyncio.Event()
        class Model:
            async def propose(self, payload):
                entered.set()
                await release.wait()
                return {"patches": [asdict(Patch(0, 3, "hej", "Hej", .99))]}
        broker = AdaptiveBroker(controller=Model(), clock=lambda: now[0], emit=events.append)
        async with LiveSession(broker) as session:
            session.push("hej")
            await entered.wait()
            now[0] = 16
            session.tick()
            display = next(e for e in events if e["type"] == "display")
            self.assertEqual((display["text"], display["review_status"]), ("hej", "expired"))
            release.set()
        self.assertFalse(any(e["type"] == "patch" for e in events))

    async def test_semantic_repairs_keep_numbers_negations_and_opt_out(self):
        for source, replacement in [("inte godkänt", "godkänt"), ("-5 %", "5 %"), ("tolv", "elva")]:
            b = AdaptiveBroker()
            s = b.add(source)
            self.assertFalse(b.apply(s.id, 0, [Patch(0, len(source), source, replacement, .99,
                                                    "A model may be wrong", "asr_error")]))
            self.assertEqual(s.text, source)
        b = AdaptiveBroker(allow_context_repairs=False)
        s = b.add("aj sveden")
        self.assertFalse(b.apply(s.id, 0, [Patch(0, 9, s.raw, "AI Sweden", .99, "Phonetic", "asr_error")]))

    async def test_stack_edit_is_atomic_and_preserves_original_briefing(self):
        b = AdaptiveBroker(initial_context="Original")
        b.set_context_stack("Rättad bakgrund", "Aktuellt", [["Svea", "Rättat minne"]])
        self.assertEqual(b.initial_context, "Original")
        self.assertEqual(b.manual_context, "Rättad bakgrund")
        self.assertEqual(b.memory_payload()[0]["summary"], "Rättat minne")
        previous = b.context_snapshot()
        with self.assertRaises(ValueError):
            b.set_context_stack("Fel bakgrund", "Fel ämne", [["dubblett", "a"], ["dubblett", "b"]])
        self.assertEqual(b.context_snapshot(), previous)

    async def test_full_stack_edit_invalidates_inflight_model_and_retries(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        class Model:
            async def propose(self, payload):
                calls.append(payload)
                if len(calls) == 1:
                    entered.set()
                    await release.wait()
                return {"patches": [asdict(Patch(0, 3, "hej", "Hej", .99))]}
        b = AdaptiveBroker(controller=Model())
        async with LiveSession(b) as session:
            s = session.push("hej")
            await entered.wait()
            b.set_context_stack("Manuell", "Nytt ämne", [["Tidigare", "Korrigerat"]])
            release.set()
        self.assertEqual(s.text, "Hej")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["manual_context"], "Manuell")
        self.assertEqual(calls[1]["remembered_topics"][0]["summary"], "Korrigerat")


class RuntimeStackTests(unittest.TestCase):
    def test_frontend_ignores_raw_and_patch_until_display_and_rejects_stale_stack(self):
        entered = Event()
        async def capture(broker, **kwargs):
            broker.controller = None
            entered.set()
            await kwargs["stop_event"].wait()
        r = AppRuntime(capture)
        try:
            r.start(0, "model", "Initial")
            self.assertTrue(entered.wait(2))
            r._event({"type": "raw", "segment": {"id": 1, "text": "rå"}})
            r._event({"type": "patch", "segment_id": 1, "text": "rättad"})
            self.assertEqual(r.snapshot().transcript, "")
            r._event({"type": "display", "segment_id": 1, "text": "rättad", "review_status": "reviewed"})
            self.assertEqual(r.snapshot().transcript, "rättad")
            revision = r.snapshot().revision
            r.update_stack("Bakgrund", "Nu", [["Förr", "Minne"]], revision)
            self.assertEqual(r.snapshot().memory, [["Förr", "Minne"]])
            with self.assertRaises(ValueError):
                r.update_stack("Gammalt utkast", "Nu", [], revision)
            self.assertEqual(r.snapshot().background, "Bakgrund")
        finally:
            r.shutdown()
