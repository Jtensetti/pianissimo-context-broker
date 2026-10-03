"""Single local recording runtime, independent of the GUI toolkit."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
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
        self._record_task: asyncio.Task | None = None
        self._stop_requested = False
        self._status = "Redo"
        self._message = ""
        self._segments: dict[int, str] = {}
        self._context = ""
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
            if kind == "raw":
                self._segments[event["segment"]["id"]] = event["segment"]["text"]
            elif kind in {"patch", "commit"}:
                self._segments[event["segment_id"]] = event["text"]
            elif kind == "context":
                active = event["active_context"]
                topic, summary = active["topic"], active["summary"]
                self._context = summary if summary.startswith(topic) else "\n\n".join(t for t in (topic, summary) if t)
                self._revision = active["revision"]
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
            self._stop_requested = False
            self._segments = {}
            self._context, self._revision, self._updated = initial_context, 0, ""
            self._broker = None
            self._future = asyncio.run_coroutine_threadsafe(self._record(int(device), model.strip(), initial_context), self._loop)

    async def _record(self, device: int, model: str, initial: str):
        self._record_task = asyncio.current_task()
        try:
            broker = AdaptiveBroker(initial_context=initial, controller=Ollama(model), emit=self._event)
            self._broker = broker
            if self._stop_requested:
                return
            await self.capture(broker, model_name="KlangAI/pianissimo-sv", device=device)
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
            self._stop_requested = True
        # Cancel the actual task, not its concurrent Future. The latter marks
        # itself done before capture cleanup has finished and would permit a
        # premature second recording or shutdown of an active decoding thread.
        asyncio.run_coroutine_threadsafe(self._cancel_recording(), self._loop)

    async def _cancel_recording(self):
        if self._record_task is not None and not self._record_task.done():
            self._record_task.cancel()

    def update_context(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Ange en kontext på högst 4 000 tecken")
        async def update():
            if self._broker is None or self._broker._finished:
                raise ValueError("Ingen aktiv transkription")
            self._broker.set_manual_context(text)
        asyncio.run_coroutine_threadsafe(update(), self._loop).result(timeout=5)

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(self._running, self._status, "\n".join(self._segments.values()),
                            self._context, self._revision, self._updated, self._message)

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
