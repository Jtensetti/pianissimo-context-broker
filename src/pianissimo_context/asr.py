"""Optional NeMo adapter. A dedicated model instance per conversation is required."""
from __future__ import annotations

import copy
from threading import RLock


class PhraseBoostingUnavailable(RuntimeError):
    """Boosting failed, but the original decoder was successfully restored."""


class NemoASR:
    def __init__(self, model, boost_alpha: float = .5):
        if not 0 <= boost_alpha <= 1:
            raise ValueError("Use a conservative boost_alpha between 0 and 1")
        self.model, self.boost_alpha = model, boost_alpha
        self._original = copy.deepcopy(model.cfg.decoding)
        self._phrases: tuple[str, ...] | None = None
        self._lock = RLock()

    @classmethod
    def load(cls, model_name: str = "KlangAI/pianissimo-sv", **kwargs):
        from nemo.collections.asr.models import ASRModel
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(ASRModel.from_pretrained(model_name=model_name, map_location=device).eval(), **kwargs)

    def update_glossary(self, phrases: list[str]) -> None:
        from omegaconf import open_dict
        phrases_tuple = tuple(dict.fromkeys(phrases))
        if len(phrases_tuple) > 100 or any(not isinstance(p, str) or not p.strip() or len(p) > 120 for p in phrases_tuple):
            raise ValueError("Invalid glossary")
        with self._lock:
            if self._phrases == phrases_tuple:
                return
            decoding = copy.deepcopy(self._original)
            if phrases_tuple:
                with open_dict(decoding):
                    decoding.strategy = "greedy_batch"
                    decoding.greedy.boosting_tree = {
                        "key_phrases_list": list(phrases_tuple), "context_score": 1.0,
                        "depth_scaling": 2.0, "use_triton": False}
                    decoding.greedy.boosting_tree_alpha = self.boost_alpha
            try:
                self.model.change_decoding_strategy(decoding)
            except Exception as exc:
                self._phrases = None
                self.model.change_decoding_strategy(copy.deepcopy(self._original))
                raise PhraseBoostingUnavailable("Phrase boosting unavailable for this NeMo/model combination") from exc
            self._phrases = phrases_tuple

    def transcribe(self, audio_path: str) -> str:
        import torch
        with self._lock, torch.inference_mode():
            hypotheses = self.model.transcribe(audio=[audio_path], batch_size=1, return_hypotheses=True)
            return hypotheses[0].text
