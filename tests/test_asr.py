from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pianissimo_context.asr import NemoASR, PhraseBoostingUnavailable


class FakeModel:
    def __init__(self):
        self.cfg = SimpleNamespace(decoding=SimpleNamespace(strategy="original", greedy=SimpleNamespace()))
        self.calls = []
        self.fail = False

    def change_decoding_strategy(self, config):
        self.calls.append(config)
        if self.fail and config.strategy == "greedy_batch":
            raise ValueError("unsupported version")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.module_patch = patch.dict("sys.modules", {"omegaconf": SimpleNamespace(open_dict=lambda c: nullcontext(c))})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_boost_cache_and_empty_list_restore(self):
        model = FakeModel()
        asr = NemoASR(model)
        asr.update_glossary(["AI Sweden", "AI Sweden"])
        self.assertEqual(model.calls[-1].greedy.boosting_tree["key_phrases_list"], ["AI Sweden"])
        self.assertFalse(model.calls[-1].greedy.boosting_tree["use_triton"])
        self.assertEqual(model.calls[-1].greedy.boosting_tree_alpha, .5)
        asr.update_glossary(["AI Sweden"])
        self.assertEqual(len(model.calls), 1)
        asr.update_glossary([])
        self.assertEqual(model.calls[-1].strategy, "original")

    def test_failure_restores_and_is_not_cached(self):
        model = FakeModel()
        model.fail = True
        asr = NemoASR(model)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                asr.update_glossary(["Svea"])
            self.assertEqual(model.calls[-1].strategy, "original")
        self.assertEqual(len(model.calls), 4)

    def test_invalid_weight_or_glossary(self):
        with self.assertRaises(ValueError):
            NemoASR(FakeModel(), boost_alpha=2)
        asr = NemoASR(FakeModel())
        with self.assertRaises(ValueError):
            asr.update_glossary([""])

    def test_failed_restore_is_fatal_not_a_recoverable_boost_warning(self):
        class BrokenModel(FakeModel):
            def change_decoding_strategy(self, config):
                raise RuntimeError("Decoder failure")
        asr = NemoASR(BrokenModel())
        with self.assertRaises(RuntimeError) as raised:
            asr.update_glossary(["Svea"])
        self.assertNotIsInstance(raised.exception, PhraseBoostingUnavailable)
