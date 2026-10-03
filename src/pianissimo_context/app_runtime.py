"""Single local recording runtime, independent of the GUI toolkit."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Callable

from .live import AdaptiveBroker
from .llm import Ollama
from .microphone import transcribe_microphone


@dataclass(frozen=True)
class Snapshot:
    running: bool
    status: str
    transcript: str
    context: str
    revision: int
    updated: str
    message: str
    background: str = ""
    memory: list = field(default_factory=list)


class AppRuntime:
    """All broker mutations happen on one owned asyncio loop.

    Gradio callbacks communicate with that loop; a lock protects display state.
    One active recording per local app process. Models are not loaded at import.
    """
    def __init__(self, capture: Callable = transcribe_microphone):
        self.capture = capture
        self._lock = Lock()
        self._ready = Event()
        self._loop = None
        self._future: Future | None = None
        self._broker: AdaptiveBroker | None = None
        self._running = False
        self._closed = False
        self._stop_event: asyncio.Event | None = None
        self._status = "Redo"
        self._message = ""
        self._segments: dict[int, str] = {}
        self._context = ""
        self._background = ""
        self._memory = []
        self._revision = 0
        self._updated = ""
        self._thread = Thread(target=self._serve, name="pianissimo-app", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("Could not start audio runtime")

    def _serve(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.run_until_complete(self._loop.shutdown_default_executor())
        self._loop.close()

    def _event(self, event):
        with self._lock:
            kind = event["type"]
            if kind == "display":
                self._segments[event["segment_id"]] = event["text"]
                if event["review_status"] != "reviewed":
                    self._message = "Text visad utan slutförd språkgranskning"

            elif kind == "context":
                self._background = event.get("background", self._background)
                self._memory = [[t["topic"], t["summary"]] for t in event["remembered_topics"]]
                active = event["active_context"]
                topic, summary = active["topic"], active["summary"]
                self._context = summary if summary.startswith(topic) else "\n\n".join(t for t in (topic, summary) if t)
                self._revision += 1
                self._updated = datetime.now().astimezone().strftime("%H:%M:%S")
            elif kind == "audio_ready" and self._status != "Stoppar":
                self._status = "Lyssnar"
            elif kind == "audio_drop":
                self._message = "Ljud tappat"
            elif kind == "audio_warning":
                self._message = "Ljudavbrott"
            elif kind == "controller_error":
                self._message = "Kontextfel (" + event["error"] + ")"
            elif kind == "asr_warning":
                self._message = "Ordlistan kunde inte aktiveras"

    def start(self, device: int | None, model: str, initial_context: str) -> None:
        if device is None:
            raise ValueError("Välj en ljudkälla")
        if type(device) is not int or device < 0:
            raise ValueError("Välj en giltig ljudkälla")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Välj en språkmodell")
        if not isinstance(initial_context, str) or len(initial_context) > 4000:
            raise ValueError("Kontexten är för lång")
        with self._lock:
            if self._closed:
                raise ValueError("Applikationen är stängd")
            if self._running:
                raise ValueError("Transkriptionen är redan startad")
            self._running, self._status, self._message = True, "Laddar", ""
            self._stop_event = asyncio.Event()
            self._segments = {}
            self._context, self._updated = initial_context, ""
            self._background, self._memory = initial_context, []
            self._revision += 1
            self._broker = None
            self._future = asyncio.run_coroutine_threadsafe(
                self._record(device, model.strip(), initial_context, self._stop_event), self._loop)

    async def _record(self, device: int, model: str, initial: str, stop_event: asyncio.Event):
        try:
            broker = AdaptiveBroker(initial_context=initial, controller=Ollama(model), emit=self._event)
            self._broker = broker
            if stop_event.is_set():
                return
            await self.capture(broker, model_name="KlangAI/pianissimo-sv", device=device, stop_event=stop_event)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            with self._lock:
                self._message = "Kunde inte starta (" + type(exc).__name__ + ")"
        finally:
            if self._broker is not None:
                self._broker.finish()
            with self._lock:
                self._running, self._status = False, "Stoppad"

    def stop(self):
        with self._lock:
            if not self._running or self._status == "Stoppar":
                return
            self._status = "Stoppar"
            stop_event = self._stop_event
        # The event belongs to this recording, so a late stop cannot cancel a
        # later recording. Capture closes the input and drains accepted audio.
        self._loop.call_soon_threadsafe(stop_event.set)

    def update_context(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Ange en kontext på högst 4 000 tecken")
        with self._lock:
            if self._closed or not self._running or self._status == "Stoppar":
                raise ValueError("Ingen aktiv transkription")
        async def update():
            if self._broker is None or self._broker._finished:
                raise ValueError("Ingen aktiv transkription")
            self._broker.set_manual_context(text)
        future = asyncio.run_coroutine_threadsafe(update(), self._loop)
        try:
            future.result(timeout=5)
        except TimeoutError:
            future.cancel()
            raise RuntimeError("Kontexten kunde inte sparas") from None

    def update_stack(self, background: str, current: str, memory: list, expected_revision: int):
        async def update():
            with self._lock:
                if self._closed or not self._running or self._status == "Stoppar":
                    raise ValueError("Ingen aktiv transkription")
                if self._revision != expected_revision:
                    raise ValueError("Kontexten har uppdaterats. Avbryt och öppna redigeringen igen.")
            if self._broker is None or self._broker._finished:
                raise ValueError("Ingen aktiv transkription")
            self._broker.set_context_stack(background, current, memory)
        with self._lock:
            if self._closed:
                raise ValueError("Applikationen är stängd")
        future = asyncio.run_coroutine_threadsafe(update(), self._loop)
        try:
            future.result(timeout=5)
        except TimeoutError:
            future.cancel()
            raise RuntimeError("Kontexten kunde inte sparas") from None

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(self._running, self._status, "\n".join(self._segments[i] for i in sorted(self._segments)),
                            self._context, self._revision, self._updated, self._message,
                            self._background, [row.copy() for row in self._memory])

    def notice(self, message: str):
        with self._lock:
            self._message = message

    def shutdown(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.stop()
        # This may wait for a model load/decoding already executing in a thread.
        # It runs only on application shutdown, not a GUI event callback.
        if self._future is not None:
            self._future.result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
