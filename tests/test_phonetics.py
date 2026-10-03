import unittest

from pianissimo_context import AdaptiveBroker, Broker, Context, LiveSession, Patch
from pianissimo_context.phonetics import phonetically_close


class ProximityTests(unittest.TestCase):
    def test_sound_near_recognition_errors_and_compounds(self):
        for raw, candidate in [("aj sveden", "AI Sweden"), ("AI sveden", "AI Sweden"),
                               ("pianisimo", "Pianissimo"), ("psedonymisering", "pseudonymisering"),
                               ("informations säkerhet", "informationssäkerhet"),
                               ("bevatning", "bevattning")]:
            with self.subTest(raw=raw):
                self.assertTrue(phonetically_close(raw, candidate))

    def test_assessments_paraphrases_certainty_and_unchanged_padding_rejected(self):
        for raw, candidate in [("ledsen", "deprimerad"), ("oro", "ångest"),
                               ("vill", "måste"), ("kanske", "säkert"),
                               ("jag är ledsen", "jag är deprimerad"),
                               ("jag är ledsen idag", "jag är deprimerad idag"),
                               ("idag ledsen", "idag"), ("hej", "hej igen"),
                               ("猫", "犬"), ("AI", "IT")]:
            with self.subTest(raw=raw):
                self.assertFalse(phonetically_close(raw, candidate))

    def test_glossary_and_maximum_model_confidence_cannot_bypass_gate(self):
        for source, replacement in [("ledsen", "deprimerad"),
                                    ("Jag är ledsen", "Jag är deprimerad")]:
            events = []
            b = AdaptiveBroker(Context(terms=(replacement,)), emit=events.append)
            segment = b.add(source)
            patch = Patch(0, len(source), source, replacement, 1.0,
                          "Model claims this fits clinical context", "asr_error")
            self.assertFalse(b.apply(segment.id, 0, [patch]))
            self.assertEqual(segment.text, source)
            self.assertEqual(events[-1]["type"], "suggestion")

    def test_explicit_aliases_do_not_bypass_sound_proximity(self):
        b = Broker(Context(aliases={"ledsen": "deprimerad"}))
        s = b.add("ledsen")
        self.assertFalse(b.apply(s.id, 0, [Patch(0, 6, s.raw, "deprimerad", 1)]))
        self.assertEqual(s.text, "ledsen")

    def test_repeated_small_repairs_cannot_drift_from_original_asr(self):
        b = AdaptiveBroker()
        s = b.add("transkribering")
        for target in ["transkriberinga", "transkriberingar"]:
            self.assertTrue(b.apply(s.id, s.revision, [Patch(0, len(s.text), s.text, target, 1,
                                                          "Small sound-near change", "asr_error")]))
        self.assertFalse(b.apply(s.id, s.revision, [Patch(0, len(s.text), s.text, "transkriberingarna", 1,
                                                        "Another small change", "asr_error")]))
        self.assertEqual(s.raw, "transkribering")
        self.assertEqual(s.text, "transkriberingar")


class PhoneticPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_display_has_phonetic_repair_but_keeps_original_emotion(self):
        events = []
        class Model:
            async def propose(self, payload):
                return {"patches": [
                    {"source": "aj sveden", "replacement": "AI Sweden", "confidence": .99,
                     "reason": "Organisation in context, sounds similar", "kind": "asr_error"},
                    {"source": "ledsen", "replacement": "deprimerad", "confidence": 1,
                     "reason": "Model wrongly claims a clinical ASR error", "kind": "asr_error"}]}
        broker = AdaptiveBroker(initial_context="AI Sweden; samtal om psykisk hälsa och depression.",
                                controller=Model(), emit=events.append)
        async with LiveSession(broker) as session:
            segment = session.push("Jag pratade med aj sveden och var ledsen.")
        displays = [e for e in events if e["type"] == "display"]
        self.assertEqual(displays[0]["text"], "Jag pratade med AI Sweden och var ledsen.")
        self.assertEqual(segment.raw, "Jag pratade med aj sveden och var ledsen.")
        self.assertTrue(any(e["type"] == "suggestion" and e["patch"]["source"] == "ledsen" for e in events))
