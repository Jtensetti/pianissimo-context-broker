"""Raw-first transcript events with evidence-limited, revision-checked repair."""
from __future__ import annotations

import asyncio
import math
import re
import time
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from .phonetics import changes_stay_close, phonetically_close


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def contains_phrase(text: str, phrase: str) -> bool:
    needle, haystack = words(phrase), words(text)
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                                for i in range(len(haystack) - len(needle) + 1))


@dataclass(frozen=True)
class Context:
    topic: str = ""
    people: tuple[str, ...] = ()
    organisations: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()  # User-supplied; never inferred by the LLM.
    aliases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        phrases = (*self.people, *self.organisations, *self.terms, *self.aliases.values())
        if len(phrases) > 100 or any(not isinstance(p, str) or not p.strip() or len(p) > 120 for p in phrases):
            raise ValueError("Context supports at most 100 nonempty phrases of 120 characters")
        if any(not isinstance(a, str) or not a.strip() or len(a) > 120 for a in self.aliases):
            raise ValueError("Invalid alias")
        if len(self.topic) > 1000 or len(self.facts) > 20 or any(len(f) > 500 for f in self.facts):
            raise ValueError("Context exceeds prompt limits")

    def glossary(self) -> list[str]:
        return list(dict.fromkeys((*self.people, *self.organisations, *self.terms,
                                   *self.aliases.values())))


@dataclass
class Segment:
    id: int
    raw: str
    text: str
    received_at: float
    revision: int = 0
    committed: bool = False
    start: float | None = None
    end: float | None = None
    review_status: str = "pending"


@dataclass(frozen=True)
class Patch:
    start: int
    end: int
    source: str
    replacement: str
    confidence: float
    reason: str = ""
    kind: str = ""


class Controller(Protocol):
    async def propose(self, payload: dict) -> dict: ...


class Broker:
    """One broker per conversation. Call tick regularly, even during silence.

    emit is synchronous and must not block. All public methods run on the same
    asyncio event loop. LLM work is bounded to one request per broker.
    """
    def __init__(self, context: Context | None = None, controller: Controller | None = None,
                 emit: Callable[[dict], None] | None = None, mutable_seconds: float = 15,
                 confidence_threshold: float = .95, clock: Callable[[], float] = time.monotonic,
                 allow_context_repairs: bool = False, emit_suggestions: bool = False):
        if not math.isfinite(mutable_seconds) or mutable_seconds <= 0:
            raise ValueError("mutable_seconds must be positive")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.context = context or Context()
        self.controller, self.emit = controller, emit or (lambda event: None)
        self.mutable_seconds, self.threshold, self.clock = mutable_seconds, confidence_threshold, clock
        self.segments: dict[int, Segment] = {}
        self.dynamic_terms: list[str] = []
        self._next_id = 0
        self._busy = False
        self.allow_context_repairs = allow_context_repairs
        self.emit_suggestions = emit_suggestions
        self.allow_semantic_repairs = False

    def repair_payload(self, segment: Segment, revision: int, original: str) -> dict:
        return {"context": asdict(self.context), "glossary": self.glossary(),
                "segment": {"id": segment.id, "revision": revision, "text": original},
                "previous_text": " ".join(s.raw for s in list(self.segments.values())[-8:]
                                           if s.id < segment.id)[-4000:]}

    def context_version(self) -> int:
        return 0

    def glossary(self) -> list[str]:
        return list(dict.fromkeys(self.context.glossary() + self.dynamic_terms))[:100]

    def add(self, raw: str, *, start: float | None = None, end: float | None = None) -> Segment:
        if not isinstance(raw, str) or len(raw) > 16000:
            raise ValueError("Expected text of at most 16000 characters")
        self.tick()
        segment = Segment(self._next_id, raw, raw, self.clock(), start=start, end=end)
        self._next_id += 1
        self.segments[segment.id] = segment
        self.emit({"type": "raw", "segment": asdict(segment)})
        return segment

    def tick(self) -> None:
        now = self.clock()
        for segment in self.segments.values():
            if not segment.committed and now - segment.received_at >= self.mutable_seconds:
                self._commit(segment)

    def _commit(self, segment: Segment) -> None:
        segment.committed = True
        self.emit({"type": "commit", "segment_id": segment.id,
                   "revision": segment.revision, "text": segment.text})

    def finish(self) -> None:
        """Lock all text; any in-flight response is rejected afterwards."""
        for segment in self.segments.values():
            if not segment.committed:
                self._commit(segment)

    def release_committed(self) -> None:
        """After persisting commit events, release old transcript from RAM."""
        self.segments = {i: s for i, s in self.segments.items() if not s.committed}

    async def repair(self, segment_id: int) -> bool:
        self.tick()
        segment = self.segments.get(segment_id)
        if segment is None or segment.committed or self._busy:
            return False
        revision, original = segment.revision, segment.text
        # Explicit aliases work offline. Live mode additionally permits tightly
        # bounded active-term substitutions; model confidence is not ASR evidence.
        patches = []
        for alias, canonical in self.context.aliases.items():
            for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", original, re.I):
                patches.append(Patch(match.start(), match.end(), match.group(), canonical, 1.0, "explicit alias"))
        if patches:
            self.apply(segment_id, revision, patches)
            revision, original = segment.revision, segment.text
        if self.controller is None:
            segment.review_status = "unavailable"
            return bool(patches)
        payload = self.repair_payload(segment, revision, original)
        context_version = self.context_version()
        self._busy = True
        try:
            result = await self.controller.propose(payload)
            self.tick()
            if (segment.committed or segment.revision != revision or segment.id not in self.segments
                or self.context_version() != context_version):
                return False
            if not isinstance(result, dict):
                raise ValueError("Expected an object")
            proposals = result.get("patches", [])
            if not isinstance(proposals, list) or len(proposals) > 32:
                raise ValueError("Invalid patch list")
            parsed = []
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    raise ValueError("Expected patch object")
                proposal = proposal.copy()
                if "start" not in proposal and "end" not in proposal:
                    source = proposal.get("source")
                    if not isinstance(source, str) or not source:
                        raise ValueError("Expected exact source text")
                    start = original.find(source)
                    if start < 0 or original.find(source, start + 1) >= 0:
                        continue  # No guessing when the source is ambiguous.
                    proposal.update(start=start, end=start + len(source))
                parsed.append(Patch(**proposal))
            candidates = result.get("terms", [])
            if not isinstance(candidates, list) or len(candidates) > 100:
                raise ValueError("Invalid term list")
            applied = self.apply(segment_id, revision, parsed)
            # Only literal evidence from RAW can enter dynamic bias. Never use
            # repaired text as evidence: that would reinforce hallucinations.
            evidence = payload["previous_text"] + " " + segment.raw
            grounded = [t for t in candidates if isinstance(t, str) and 1 <= len(t) <= 120
                        and contains_phrase(evidence, t)]
            before = self.glossary()
            self.dynamic_terms = list(dict.fromkeys(grounded + self.dynamic_terms))[:100]
            if before != self.glossary():
                self.emit({"type": "glossary", "phrases": self.glossary()})
            segment.review_status = "reviewed"
            return applied
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            segment.review_status = "error"
            # Do not log transcript, prompts, response, or potentially sensitive
            # exception messages. Fail open to original ASR text.
            self.emit({"type": "controller_error", "error": type(exc).__name__})
            return False
        finally:
            self._busy = False

    def apply(self, segment_id: int, revision: int, patches: list[Patch]) -> bool:
        self.tick()
        segment = self.segments.get(segment_id)
        if segment is None or segment.committed or segment.revision != revision:
            return False
        accepted: list[Patch] = []
        for p in patches:
            if (type(p.start) is not int or type(p.end) is not int
                or not 0 <= p.start < p.end <= len(segment.text)
                or not isinstance(p.source, str) or not isinstance(p.replacement, str)
                or len(p.replacement) > 240 or not p.replacement.strip()
                or type(p.confidence) not in (int, float) or not math.isfinite(p.confidence)
                or not self.threshold <= p.confidence <= 1
                or segment.text[p.start:p.end] != p.source):
                continue
            # Formatting preserves letters; numeric and negation guards below
            # additionally protect meaning-bearing signs, units and words.
            formatting = re.sub(r"[^\w]", "", p.source.casefold()) == re.sub(r"[^\w]", "", p.replacement.casefold())
            alias = any(a.casefold() == p.source.casefold() and c == p.replacement
                        for a, c in self.context.aliases.items())
            boundary = ((p.start == 0 or not segment.text[p.start - 1].isalnum())
                        and (p.end == len(segment.text) or not segment.text[p.end].isalnum()))
            numbers_unchanged = re.findall(r"\d+(?:[.,:/-]\d+)*", p.source) == re.findall(r"\d+(?:[.,:/-]\d+)*", p.replacement)
            # Numeric spans preserve signs, percentage marks and units, too.
            if re.search(r"\d", p.source + p.replacement):
                numbers_unchanged = (re.sub(r"\s+", "", p.source)
                                     == re.sub(r"\s+", "", p.replacement) and numbers_unchanged)
            # Removing spaces must not silently change 'inte' to another word.
            protected = {"inte", "ej", "icke", "aldrig", "ingen", "inget", "inga", "utan"}
            negation_unchanged = [w for w in words(p.source) if w in protected] == [w for w in words(p.replacement) if w in protected]
            # Live context repairs must remain orthographically close to an
            # active term. This is a heuristic, not acoustic verification.
            source_key = "".join(words(p.source))
            replacement_key = "".join(words(p.replacement))
            number_words = {"noll", "ett", "en", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio", "tio",
                            "elva", "tolv", "hundra", "tusen", "miljon", "miljoner", "miljard", "miljarder"}
            number_words_unchanged = [w for w in words(p.source) if w in number_words] == [w for w in words(p.replacement) if w in number_words]
            similarity = SequenceMatcher(None, source_key, replacement_key, autojunk=False)
            edit_budget = sum(max(a2 - a1, b2 - b1) for tag, a1, a2, b1, b2 in similarity.get_opcodes() if tag != "equal")
            near_term = (self.allow_context_repairs and p.replacement in self.glossary()
                         and 6 <= len(source_key) <= 60 and len(words(p.source)) <= 4
                         and len(words(p.replacement)) <= 4
                         and similarity.ratio() >= .88 and edit_budget <= 2
                         and number_words_unchanged)
            valid_content = numbers_unchanged and negation_unchanged and number_words_unchanged
            # Model classification is not evidence: every lexical correction
            # must independently pass the pronunciation-proximity gate below.
            semantic = (self.allow_semantic_repairs and p.kind == "asr_error"
                        and isinstance(p.reason, str) and bool(p.reason.strip())
                        and len(p.source) <= 80
                        and len(p.replacement) <= 80
                        and 1 <= len(words(p.source)) <= 4
                        and 1 <= len(words(p.replacement)) <= 4)
            phonetic = phonetically_close(p.source, p.replacement)
            if (formatting or ((alias or near_term or semantic) and phonetic)) and boundary and valid_content and p.replacement != p.source:
                accepted.append(p)
            elif self.emit_suggestions and boundary and valid_content and p.replacement != p.source:
                self.emit({"type": "suggestion", "segment_id": segment_id, "base_revision": revision,
                           "patch": asdict(p), "status": "unverified", "text_changed": False})
        accepted.sort(key=lambda p: (p.start, p.end))
        if not accepted or any(a.end > b.start for a, b in zip(accepted, accepted[1:])):
            return False  # Reject ambiguous overlapping batches atomically.
        text = segment.text
        for p in reversed(accepted):
            text = text[:p.start] + p.replacement + text[p.end:]
        if not changes_stay_close(segment.raw, text):
            return False  # Repeated small repairs must not drift from RAW speech.
        segment.text, segment.revision = text, revision + 1
        self.emit({"type": "patch", "segment_id": segment_id, "base_revision": revision,
                   "revision": segment.revision, "patches": [asdict(p) for p in accepted], "text": text})
        return True
