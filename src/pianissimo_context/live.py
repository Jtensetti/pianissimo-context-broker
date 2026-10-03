"""Adaptive context memory and a coalescing worker for live transcripts."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass, field
import math

from .broker import Broker, Segment, contains_phrase, words


@dataclass
class ActiveContext:
    topic: str = ""
    summary: str = ""
    terms: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    revision: int = 0


class AdaptiveBroker(Broker):
    """Initial free text is background, not an instruction to rewrite speech.

    Context is reviewed on wall-clock cadence when new speech has arrived.
    Recent raw speech lives independently of the shorter mutable text window.
    """
    def __init__(self, *args, initial_context: str = "", review_seconds: float = 25,
                 history_seconds: float = 30, semantic_repairs: bool = True, **kwargs):
        kwargs.setdefault("allow_context_repairs", True)
        kwargs.setdefault("emit_suggestions", True)
        super().__init__(*args, **kwargs)
        self.allow_semantic_repairs = semantic_repairs and self.allow_context_repairs
        if not isinstance(initial_context, str) or len(initial_context) > 4000:
            raise ValueError("Initial context must be text of at most 4000 characters")
        if not math.isfinite(review_seconds) or not 20 <= review_seconds <= 30:
            raise ValueError("Review interval must be between 20 and 30 seconds")
        if not math.isfinite(history_seconds) or not review_seconds <= history_seconds <= 120:
            raise ValueError("History must cover at least one review interval, at most 120 seconds")
        self.initial_context = initial_context
        self.review_seconds, self.history_seconds = review_seconds, history_seconds
        self.active = ActiveContext(topic=self.context.topic or initial_context[:200])
        # Session topic memory outlives raw rolling history, but never becomes
        # a transcript or proof of speaker facts. Initial background stays separate.
        self.remembered_topics: deque[dict] = deque()
        self.history: deque[tuple[float, int, str]] = deque(maxlen=200)
        self._reviewed_at = self.clock()
        self._reviewed_id = -1
        self._bootstrapped = False
        self._finished = False
        self.manual_context: str | None = None
        self._manual_revision = 0

    def context_version(self) -> int:
        return self.active.revision

    def context_snapshot(self) -> dict:
        return {"background": self.manual_context if self.manual_context is not None else self.initial_context,
                "active_context": asdict(self.active),
                "remembered_topics": list(self.remembered_topics)}

    def _emit_context(self, mode: str):
        self.emit({"type": "context", "mode": mode, **self.context_snapshot(),
                   "through_segment_id": self._next_id - 1})

    def _remember_active(self):
        if not self.active.revision or not self.active.topic.strip():
            return
        self.remembered_topics = deque(t for t in self.remembered_topics
            if t["topic"].casefold() != self.active.topic.casefold())
        self.remembered_topics.append({"topic": self.active.topic, "summary": self.active.summary,
            "evidence": self.active.evidence, "context_revision": self.active.revision})

    def set_context_stack(self, background: str, current: str, memory: list) -> None:
        """Atomically replace an operator-reviewed stack; invalidate in-flight work."""
        if self._finished:
            raise ValueError("Ingen aktiv transkription")
        if not isinstance(background, str) or len(background) > 4000:
            raise ValueError("Bakgrunden får innehålla högst 4 000 tecken")
        if not isinstance(current, str) or not current.strip() or len(current) > 1000:
            raise ValueError("Aktuell kontext måste innehålla 1–1 000 tecken")
        if not isinstance(memory, list) or len(memory) > 500:
            raise ValueError("Ogiltigt ämnesminne")
        entries = []
        seen = set()
        for row in memory:
            if not isinstance(row, (list, tuple)) or len(row) != 2 or not all(isinstance(v, str) for v in row):
                raise ValueError("Varje ämne behöver en rubrik och en kontext")
            topic, summary = (v.strip() for v in row)
            if not topic and not summary:
                continue
            if not topic or len(topic) > 200 or len(summary) > 1000 or topic.casefold() in seen:
                raise ValueError("Ämnesrubriker måste vara unika och högst 200 tecken; kontext högst 1 000")
            seen.add(topic.casefold())
            entries.append({"topic": topic, "summary": summary, "evidence": [],
                            "context_revision": self.active.revision + 1, "source": "manual"})
        self.manual_context = background.strip()
        self._manual_revision += 1
        self.active = ActiveContext(current.splitlines()[0][:200], current.strip(), [], [], self.active.revision + 1)
        self.remembered_topics = deque(entries)
        self.dynamic_terms = []
        self._bootstrapped = True
        self._reviewed_at = self.clock()
        self._reviewed_id = self._next_id - 1
        self._emit_context("manual")
        self.emit({"type": "glossary", "phrases": self.glossary()})

    def set_manual_context(self, text: str) -> None:
        if self._finished:
            raise RuntimeError("Conversation is closed")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Manual context must contain 1–4000 characters")
        self._remember_active()
        self.manual_context = text.strip()
        self._manual_revision += 1
        self.active = ActiveContext(self.manual_context.splitlines()[0][:200], self.manual_context,
                                    [], [], self.active.revision + 1)
        self.dynamic_terms = []
        self._bootstrapped = False
        self._emit_context("manual")
        self.emit({"type": "glossary", "phrases": self.glossary()})

    def add(self, raw: str, **kwargs) -> Segment:
        if self._finished:
            raise RuntimeError("Conversation is closed")
        segment = super().add(raw, **kwargs)
        self.history.append((segment.received_at, segment.id, segment.raw))
        self._prune()
        return segment

    def _prune(self):
        cutoff = self.clock() - self.history_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def recent_raw(self, before_id: int | None = None) -> str:
        self._prune()
        return "\n".join(raw for _, i, raw in self.history
                         if before_id is None or i < before_id)[-8000:]

    def review_due(self) -> bool:
        return (not self._finished and (not self._bootstrapped or
                (self.clock() - self._reviewed_at >= self.review_seconds
                 and self._next_id - 1 > self._reviewed_id)))

    def memory_payload(self) -> list[dict]:
        # Retain the full session archive, but send at most eight topics:
        # four recent threads plus four selected by lexical relevance to speech.
        archive = list(self.remembered_topics)
        query = set(words(self.recent_raw() + " " + self.active.topic))
        ranked = sorted(range(max(0, len(archive) - 4)),
                        key=lambda i: len(query & set(words(archive[i]["topic"] + " " + archive[i]["summary"]))),
                        reverse=True)[:4]
        selected = sorted(set(ranked + list(range(max(0, len(archive) - 4), len(archive)))))
        return [{"topic": archive[i]["topic"], "summary": archive[i]["summary"][:600],
                 "evidence": [e[:200] for e in archive[i]["evidence"][:1]],
                 "context_revision": archive[i]["context_revision"]} for i in selected]

    async def review_context(self) -> bool:
        if not self.review_due() or self._busy:
            return False
        initial = not self._bootstrapped
        recent = self.recent_raw()
        snapshot_id = self._next_id - 1
        manual_revision = self._manual_revision
        self._reviewed_at = self.clock()
        review_started = self._reviewed_at
        self._reviewed_id = -1 if initial else snapshot_id
        self._bootstrapped = True
        reviewer = getattr(self.controller, "review_context", None)
        if reviewer is None or (not initial and not recent.strip()):
            return False
        mode = "manual" if initial and self.manual_context is not None else "initial" if initial else "review"
        payload = {"mode": mode,
                   "initial_context": self.initial_context,
                   "manual_context": self.manual_context,
                   "explicit_context": asdict(self.context),
                   "active_context": asdict(self.active), "recent_raw": recent,
                   "remembered_topics": self.memory_payload()}
        self._busy = True
        try:
            result = await reviewer(payload)
            if (self._finished or self._manual_revision != manual_revision
                or self.clock() - review_started >= self.review_seconds):
                return False
            if not isinstance(result, dict):
                raise ValueError("Expected context object")
            topic, summary = result.get("topic", ""), result.get("summary", "")
            terms, evidence = result.get("terms", []), result.get("evidence", [])
            if (not isinstance(topic, str) or not topic.strip() or len(topic) > 200
                or not isinstance(summary, str) or len(summary) > 1000
                or not isinstance(terms, list) or len(terms) > 100
                or not isinstance(evidence, list) or len(evidence) > 5):
                raise ValueError("Invalid context response")
            source = (self.manual_context if self.manual_context is not None else self.initial_context) if initial else recent
            grounded_evidence = [e for e in evidence if isinstance(e, str) and 1 <= len(e) <= 300 and e in source]
            # A new topic needs a literal raw evidence quote. During bootstrap,
            # free text was supplied by the user, so no speech proof is needed.
            if not initial and not grounded_evidence:
                return False
            term_source = source if initial else "\n".join(grounded_evidence)
            grounded_terms = list(dict.fromkeys(t for t in terms
                              if isinstance(t, str) and 1 <= len(t) <= 120 and contains_phrase(term_source, t)))[:20]
            before = self.glossary()
            if self.active.topic.casefold() == topic.casefold() and not summary.strip():
                summary = self.active.summary
            if self.active.revision and self.active.topic.casefold() != topic.casefold():
                self._remember_active()
            self.active = ActiveContext(topic, summary, grounded_terms, grounded_evidence,
                                        self.active.revision + 1)
            # Replace rather than append: stale inferred terms expire on review.
            self.dynamic_terms = grounded_terms.copy()
            self._emit_context(payload["mode"])
            if before != self.glossary():
                self.emit({"type": "glossary", "phrases": self.glossary()})
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.emit({"type": "controller_error", "operation": "context", "error": type(exc).__name__})
            return False
        finally:
            self._busy = False

    def repair_payload(self, segment: Segment, revision: int, original: str) -> dict:
        payload = super().repair_payload(segment, revision, original)
        payload.update(initial_context=self.initial_context, active_context=asdict(self.active),
                       previous_text=self.recent_raw(before_id=segment.id)[-4000:],
                       remembered_topics=self.memory_payload(), manual_context=self.manual_context)
        return payload

    def finish(self):
        self._finished = True
        super().finish()


class LiveSession:
    """Raw ingestion never awaits LLM; one worker serializes all model work.

    Session must be used on a single event loop. Callback output must not block.
    The timer locks segments during silence and wakes context reviews on cadence.
    """
    def __init__(self, broker: AdaptiveBroker):
        self.broker = broker
        self._wake = asyncio.Event()
        self._closing = False
        self._worker_task = None
        self._timer_task = None
        self._processed: dict[int, tuple[int, int]] = {}
        self._published: dict[int, int] = {}

    async def __aenter__(self):
        self._worker_task = asyncio.create_task(self._worker())
        self._timer_task = asyncio.create_task(self._timer())
        self._wake.set()
        return self

    def push(self, raw: str, **kwargs) -> Segment:
        if self._worker_task is None or self._closing:
            raise RuntimeError("Live session is not accepting input")
        segment = self.broker.add(raw, **kwargs)
        self._wake.set()
        return segment

    def _publish(self, segment, status):
        if self._published.get(segment.id) == segment.revision:
            return
        self._published[segment.id] = segment.revision
        self.broker.emit({"type": "display", "segment_id": segment.id,
                          "revision": segment.revision, "text": segment.text,
                          "review_status": status})

    def tick(self):
        self.broker.tick()
        for segment in self.broker.segments.values():
            if segment.committed:
                self._publish(segment, "expired" if segment.review_status == "pending" else segment.review_status)
        if self.broker.review_due() or any(
            not s.committed and self._processed.get(i) != (s.revision, self.broker.active.revision)
            for i, s in self.broker.segments.items()
        ):
            self._wake.set()
        self.broker.release_committed()
        self._processed = {i: v for i, v in self._processed.items() if i in self.broker.segments}
        self._published = {i: v for i, v in self._published.items() if i in self.broker.segments}

    async def _timer(self):
        while True:
            await asyncio.sleep(.1)
            self.tick()

    async def _worker(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            await self.broker.review_context()
            for i in list(self.broker.segments):
                s = self.broker.segments.get(i)
                if s is None or s.committed:
                    continue
                state = (s.revision, self.broker.active.revision)
                if self._processed.get(i) != state:
                    s.review_status = "pending"
                    await self.broker.repair(i)
                    self._processed[i] = (s.revision, state[1])
                    if self.broker.active.revision != state[1]:
                        self._wake.set()
                    else:
                        self._publish(s, "expired" if s.review_status == "pending" else s.review_status)
            # A cadence review may become due during inference.
            if self.broker.review_due():
                self._wake.set()
            if self._closing and not self._wake.is_set():
                return

    async def __aexit__(self, exc_type, exc, tb):
        self._closing = True
        try:
            if exc_type:
                self._worker_task.cancel()
            else:
                self._wake.set()
            await asyncio.gather(self._worker_task, return_exceptions=bool(exc_type))
        finally:
            self._timer_task.cancel()
            await asyncio.gather(self._timer_task, return_exceptions=True)
            for segment in self.broker.segments.values():
                self._publish(segment, "interrupted" if segment.review_status == "pending" else segment.review_status)
            self.broker.finish()
