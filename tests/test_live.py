import asyncio
import unittest

from pianissimo_context import AdaptiveBroker, Context, LiveSession, Patch


INITIAL = "Intervju om Pianissimo i Svea, en AI-chattbot för offentlig sektor."


class Controller:
    def __init__(self):
        self.reviews = []
        self.repairs = []
        self.response = {"topic": "Pianissimo och Svea", "summary": "Transkribering i offentlig sektor",
                         "terms": ["Pianissimo", "Svea", "offentlig sektor", "hittat på"], "evidence": []}

    async def review_context(self, payload):
        self.reviews.append(payload)
        return self.response

    async def propose(self, payload):
        self.repairs.append(payload)
        return {"patches": [], "terms": []}


class AdaptiveTests(unittest.IsolatedAsyncioTestCase):
    def make_broker(self, **kwargs):
        self.now = 0
        self.events = []
        self.controller = Controller()
        return AdaptiveBroker(controller=self.controller, initial_context=INITIAL,
                              clock=lambda: self.now, emit=self.events.append, **kwargs)

    async def test_bootstrap_free_text_grounding_and_interval(self):
        b = self.make_broker()
        self.assertTrue(await b.review_context())
        self.assertEqual(b.glossary(), ["Pianissimo", "Svea", "offentlig sektor"])
        self.assertEqual(self.controller.reviews[0]["initial_context"], INITIAL)
        self.now = 24
        b.add("Vi pratar vidare om Svea.")
        self.assertFalse(b.review_due())
        self.now = 25
        self.assertTrue(b.review_due())
        self.controller.response = {"topic": "Svea", "terms": ["Svea"], "evidence": ["Vi pratar vidare om Svea."]}
        self.assertTrue(await b.review_context())
        self.assertEqual(len(self.controller.reviews), 2)

    async def test_new_topic_replaces_bias_but_keeps_prior_background(self):
        b = self.make_broker()
        await b.review_context()
        s = b.add("Pianissimo kan användas i Svea.")
        self.now = 16
        b.tick()
        b.release_committed()
        self.assertNotIn(s.id, b.segments)
        self.now = 25
        b.add("Nu pratar vi om trädgården och kompostering.")
        self.controller.response = {"topic": "Trädgård", "summary": "Samtalet handlar nu om kompostering",
                                   "terms": ["kompostering", "Pianissimo"],
                                   "evidence": ["Nu pratar vi om trädgården och kompostering."]}
        await b.review_context()
        # Recent raw may include old words; the model is instructed to select
        # active vocabulary. Only current evidence validation can be deterministic.
        self.assertEqual(b.active.topic, "Trädgård")
        self.assertEqual(b.remembered_topics[0]["topic"], "Pianissimo och Svea")
        self.assertEqual(b.initial_context, INITIAL)
        self.assertIn("Pianissimo kan användas", self.controller.reviews[-1]["recent_raw"])
        self.now = 51
        b.add("Vi fortsätter med kompostering.")
        self.controller.response = {"topic": "Trädgård", "terms": ["kompostering", "Svea"],
                                   "evidence": ["Vi fortsätter med kompostering."]}
        await b.review_context()
        self.assertEqual(b.glossary(), ["kompostering"])
        await b.repair(b.add("kompostering").id)
        self.assertIn("remembered_topics", self.controller.repairs[-1])

    async def test_context_change_requires_exact_raw_evidence(self):
        b = self.make_broker()
        await b.review_context()
        self.now = 25
        b.add("Vi fortsätter intervjun.")
        self.controller.response = {"topic": "Fotboll", "terms": ["fotboll"], "evidence": ["hittad på mening"]}
        self.assertFalse(await b.review_context())
        self.assertEqual(b.active.topic, "Pianissimo och Svea")

    async def test_silence_does_not_generate_repeated_reviews(self):
        b = self.make_broker()
        await b.review_context()
        self.now = 100
        self.assertFalse(b.review_due())
        self.assertFalse(await b.review_context())
        self.assertEqual(len(self.controller.reviews), 1)

    async def test_memory_is_bounded_and_raw_history_expires(self):
        b = self.make_broker()
        await b.review_context()
        for i in range(12):
            self.now += 25
            phrase = f"ämne{i}"
            b.add(phrase)
            self.controller.response = {"topic": phrase, "terms": [phrase], "evidence": [phrase]}
            await b.review_context()
        self.assertEqual(len(b.remembered_topics), 8)
        self.now += 31
        self.assertEqual(b.recent_raw(), "")
        self.assertEqual(b.initial_context, INITIAL)

    async def test_domain_word_repairs_and_unverified_semantic_suggestions(self):
        b = self.make_broker(context=Context(terms=("transkriberingsmodellen",)))
        s = b.add("transkriberingsmodulen")
        self.assertTrue(b.apply(s.id, 0, [Patch(0, len(s.raw), s.raw, "transkriberingsmodellen", .99)]))
        s = b.add("vädret är fint")
        self.assertFalse(b.apply(s.id, 0, [Patch(0, len(s.raw), s.raw, "modellen är snabb", .99)]))
        self.assertEqual(s.text, "vädret är fint")
        self.assertEqual(self.events[-1]["type"], "suggestion")

    async def test_suggestions_only_disables_near_term_autocorrection(self):
        b = self.make_broker(context=Context(terms=("Pianissimo",)), allow_context_repairs=False)
        s = b.add("pianisimo")
        self.assertFalse(b.apply(s.id, 0, [Patch(0, 9, s.raw, "Pianissimo", .99)]))
        self.assertEqual(self.events[-1]["type"], "suggestion")

    async def test_unrelated_topic_is_unchanged_and_negations_stay(self):
        b = self.make_broker(context=Context(terms=("pianissimo",)))
        s = b.add("Vi ska inte prata om AI utan om vädret.")
        self.assertFalse(b.apply(s.id, 0, [Patch(0, len(s.raw), s.raw, "Vi ska prata om AI.", .999)]))
        self.assertEqual(s.text, s.raw)
        self.assertFalse(any(e["type"] == "suggestion" for e in self.events))

    async def test_late_review_cannot_change_finished_conversation(self):
        b = self.make_broker()
        original = self.controller.review_context
        async def close_first(payload):
            b.finish()
            return await original(payload)
        self.controller.review_context = close_first
        self.assertFalse(await b.review_context())
        self.assertEqual(b.active.revision, 0)

    async def test_raw_ingestion_and_commit_continue_while_context_waits(self):
        b = self.make_broker()
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.controller.review_context
        async def blocked(payload):
            entered.set()
            await release.wait()
            return await original(payload)
        self.controller.review_context = blocked
        async with LiveSession(b) as session:
            await entered.wait()
            s = session.push("Hej")
            self.assertEqual(self.events[-1]["type"], "raw")
            self.now = 16
            session.tick()
            self.assertTrue(s.committed)
            release.set()
        self.assertTrue(b._finished)

    async def test_closed_session_rejects_new_input(self):
        b = self.make_broker()
        async with LiveSession(b) as session:
            session.push("Hej")
        with self.assertRaises(RuntimeError):
            session.push("Sent")

    async def test_controller_error_preserves_context(self):
        b = self.make_broker()
        await b.review_context()
        self.now = 25
        b.add("Annat ämne.")
        self.controller.response = {"topic": []}
        self.assertFalse(await b.review_context())
        self.assertEqual(b.active.revision, 1)
        self.assertEqual(self.events[-1]["operation"], "context")

    async def test_initial_raw_is_reviewed_once_even_if_speech_stops(self):
        b = self.make_broker()
        b.add("Vi pratar redan om kompostering.")
        await b.review_context()
        self.now = 25
        self.assertTrue(b.review_due())
        self.controller.response = {"topic": "Kompostering", "terms": ["kompostering"],
                                   "evidence": ["Vi pratar redan om kompostering."]}
        await b.review_context()
        self.assertEqual(b.active.topic, "Kompostering")
        self.now = 50
        self.assertFalse(b.review_due())

    async def test_resumed_topic_gets_prior_memory(self):
        b = self.make_broker()
        await b.review_context()
        self.now = 25
        b.add("Nu är det kompostering.")
        self.controller.response = {"topic": "Kompostering", "terms": ["kompostering"],
                                   "evidence": ["Nu är det kompostering."]}
        await b.review_context()
        self.now = 50
        b.add("Tillbaka till Pianissimo och Svea.")
        self.controller.response = {"topic": "Pianissimo och Svea", "terms": ["Pianissimo", "Svea"],
                                   "evidence": ["Tillbaka till Pianissimo och Svea."]}
        await b.review_context()
        self.assertEqual(self.controller.reviews[-1]["remembered_topics"][0]["topic"], "Pianissimo och Svea")
        self.assertEqual(b.active.topic, "Pianissimo och Svea")
        self.assertEqual(b.glossary(), ["Pianissimo", "Svea"])

    async def test_slow_context_response_is_discarded(self):
        b = self.make_broker()
        async def too_slow(payload):
            self.now = 26
            return self.controller.response
        self.controller.review_context = too_slow
        self.assertFalse(await b.review_context())
        self.assertEqual(b.active.revision, 0)

    async def test_manual_context_wins_over_inflight_context_response(self):
        b = self.make_broker()
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.controller.review_context
        async def blocked(payload):
            entered.set()
            await release.wait()
            return await original(payload)
        self.controller.review_context = blocked
        task = asyncio.create_task(b.review_context())
        await entered.wait()
        b.set_manual_context("Rättad bakgrund om Kompostering")
        release.set()
        self.assertFalse(await task)
        self.assertEqual(b.manual_context, "Rättad bakgrund om Kompostering")
        self.assertEqual(b.active.summary, b.manual_context)
        self.assertTrue(b.review_due())
        self.controller.review_context = original
        self.controller.response = {"topic": "Kompostering", "terms": ["Kompostering", "Svea"], "evidence": []}
        await b.review_context()
        self.assertEqual(self.controller.reviews[-1]["mode"], "manual")
        self.assertEqual(b.glossary(), ["Kompostering"])
        self.assertEqual(b.initial_context, INITIAL)

    async def test_manual_context_invalidates_inflight_transcript_patch(self):
        b = self.make_broker()
        entered, release = asyncio.Event(), asyncio.Event()
        async def blocked(payload):
            entered.set()
            await release.wait()
            return {"patches": [{"start": 0, "end": 3, "source": "hej", "replacement": "Hej", "confidence": .99}]}
        self.controller.propose = blocked
        s = b.add("hej")
        task = asyncio.create_task(b.repair(s.id))
        await entered.wait()
        b.set_manual_context("Annan bakgrund")
        release.set()
        self.assertFalse(await task)
        self.assertEqual(s.text, "hej")
