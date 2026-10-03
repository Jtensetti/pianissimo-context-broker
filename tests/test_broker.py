import asyncio
import unittest

from pianissimo_context import Broker, Context, Ollama, Patch


class Stub:
    def __init__(self, result):
        self.result = result

    async def propose(self, payload):
        return self.result


class BrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_first_alias_and_original_preserved(self):
        events = []
        broker = Broker(Context(aliases={"aj sveden": "AI Sweden"}), emit=events.append)
        segment = broker.add("vi mötte aj sveden")
        self.assertEqual([e["type"] for e in events], ["raw"])
        await broker.repair(segment.id)
        self.assertEqual(segment.raw, "vi mötte aj sveden")
        self.assertEqual(segment.text, "vi mötte AI Sweden")
        self.assertEqual(events[-1]["base_revision"], 0)

    async def test_commit_blocks_late_llm_response(self):
        now = [0]
        class Slow:
            async def propose(self, payload):
                now[0] = 16
                return {"patches": [{"start": 0, "end": 4, "source": "svea",
                         "replacement": "Svea", "confidence": .99}], "terms": ["svea"]}
        broker = Broker(controller=Slow(), clock=lambda: now[0])
        segment = broker.add("svea")
        self.assertFalse(await broker.repair(segment.id))
        self.assertTrue(segment.committed)
        self.assertEqual(segment.text, "svea")
        self.assertEqual(broker.dynamic_terms, [])

    async def test_semantic_invention_is_rejected_even_in_glossary(self):
        broker = Broker(Context(terms=("Socialstyrelsen",)))
        s = broker.add("socialen")
        self.assertFalse(broker.apply(s.id, 0, [Patch(0, 8, "socialen", "Socialstyrelsen", .999)]))

    async def test_formatting_numeric_and_negation_guards(self):
        for source, replacement, valid in [("hej", "Hej.", True), ("social tjänsten", "socialtjänsten", True),
                                            ("1,5", "15", False), ("1,5", "1.5", False),
                                            ("inte", "Inte", True), ("inte", "int e", False),
                                            ("är inte", "är", False)]:
            broker = Broker()
            s = broker.add(source)
            self.assertEqual(broker.apply(s.id, 0, [Patch(0, len(source), source, replacement, .99)]), valid)

    async def test_overlap_stale_revision_and_wrong_offsets(self):
        broker = Broker()
        s = broker.add("hej du")
        self.assertFalse(broker.apply(s.id, 0, [Patch(0, 3, "hej", "Hej", .99),
                                              Patch(0, 6, "hej du", "Hej du", .99)]))
        self.assertFalse(broker.apply(s.id, 0, [Patch(1, 3, "hej", "Hej", .99)]))
        self.assertTrue(broker.apply(s.id, 0, [Patch(0, 3, "hej", "Hej", .99)]))
        self.assertFalse(broker.apply(s.id, 0, [Patch(4, 6, "du", "Du", .99)]))

    async def test_dynamic_terms_require_raw_word_boundaries(self):
        controller = Stub({"terms": ["BBIC", "Socialstyrelsen", "social", "socialtjänsten"]})
        broker = Broker(controller=controller)
        s = broker.add("BBIC används i socialtjänsten")
        await broker.repair(s.id)
        self.assertEqual(broker.glossary(), ["BBIC", "socialtjänsten"])

    async def test_repaired_text_is_not_dynamic_evidence(self):
        broker = Broker(Context(aliases={"aj sveden": "AI Sweden"}), Stub({"terms": ["AI Sweden"]}))
        await broker.repair(broker.add("aj sveden").id)
        self.assertEqual(broker.dynamic_terms, [])

    async def test_failure_keeps_raw_and_does_not_log_sensitive_error(self):
        class Broken:
            async def propose(self, payload):
                raise RuntimeError("sensitive patient text")
        events = []
        broker = Broker(controller=Broken(), emit=events.append)
        s = broker.add("hej")
        await broker.repair(s.id)
        self.assertEqual(s.text, s.raw)
        self.assertEqual(events[-1], {"type": "controller_error", "error": "RuntimeError"})

    async def test_bad_json_structure_and_nonfinite_confidence(self):
        for result in [{"patches": "wrong"}, {"patches": [{"start": "0"}]}, []]:
            broker = Broker(controller=Stub(result))
            self.assertFalse(await broker.repair(broker.add("hej").id))
        broker = Broker()
        s = broker.add("hej")
        for confidence in [float("nan"), float("inf"), True, .5]:
            self.assertFalse(broker.apply(s.id, 0, [Patch(0, 3, "hej", "Hej", confidence)]))

    async def test_finish_and_release_reject_inflight_reply(self):
        entered, release = asyncio.Event(), asyncio.Event()
        class Slow:
            async def propose(self, payload):
                entered.set()
                await release.wait()
                return {"terms": ["hej"]}
        broker = Broker(controller=Slow())
        s = broker.add("hej")
        task = asyncio.create_task(broker.repair(s.id))
        await entered.wait()
        broker.finish()
        broker.release_committed()
        release.set()
        self.assertFalse(await task)
        self.assertEqual(broker.segments, {})

    def test_loopback_only(self):
        for url in ["https://example.org", "http://127.0.0.1.evil.com", "http://localhost@evil.com", "http://localhost/path"]:
            with self.assertRaises(ValueError):
                Ollama("local", url)
        Ollama("local")

    async def test_sign_percentage_currency_and_unit_changes_are_rejected(self):
        for source, replacement in [("-5", "5"), ("−5", "5"), ("5%", "5"),
                                    ("5 mg", "5 g"), ("5 mW", "5 MW"), ("12 €", "12 $")]:
            broker = Broker(Context(terms=(replacement,)), allow_context_repairs=True)
            s = broker.add(source)
            self.assertFalse(broker.apply(s.id, 0, [Patch(0, len(source), source, replacement, .99)]), source)
            self.assertEqual(s.text, source)

    async def test_malformed_terms_cannot_partially_apply_a_reply(self):
        broker = Broker(controller=Stub({"patches": [{"start": 0, "end": 3, "source": "hej",
                                    "replacement": "Hej", "confidence": .99}], "terms": "invalid"}))
        s = broker.add("hej")
        self.assertFalse(await broker.repair(s.id))
        self.assertEqual(s.text, "hej")


if __name__ == "__main__":
    unittest.main()
