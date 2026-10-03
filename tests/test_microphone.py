import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from threading import Event
import wave

from pianissimo_context import AdaptiveBroker, Context
from pianissimo_context.microphone import transcribe_microphone
from pianissimo_context.asr import PhraseBoostingUnavailable


class MicrophoneTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_queue_overload_timestamps_and_stop_with_doubles(self):
        events = []
        raw_seen = asyncio.Event()
        def emit(event):
            events.append(event)
            if event["type"] == "raw":
                raw_seen.set()

        class Stream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
            def __enter__(self):
                # Five chunks arrive before the event loop drains the bounded
                # three-chunk queue. This deterministically exercises overload.
                self.callback(bytes(5 * 64000), 5 * 32000, None,
                              SimpleNamespace(input_overflow=False))
                return self
            def __exit__(self, *args):
                pass

        class ASR:
            def __init__(self):
                self.glossaries = []
            def update_glossary(self, phrases):
                self.glossaries.append(phrases)
            def transcribe(self, path):
                self_path = Path(path)
                assert self_path.exists()
                with wave.open(str(self_path)) as wav:
                    assert wav.getnframes() == 32000
                return "Svea"

        asr = ASR()
        broker = AdaptiveBroker(Context(terms=("Svea",)), emit=emit)
        with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(RawInputStream=Stream)}), \
                patch("pianissimo_context.microphone.NemoASR.load", return_value=asr):
            task = asyncio.create_task(transcribe_microphone(broker, model_name="double", chunk_seconds=2))
            try:
                await asyncio.wait_for(raw_seen.wait(), 5)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertEqual(len([e for e in events if e["type"] == "audio_drop"]), 2)
        first = next(e["segment"] for e in events if e["type"] == "raw")
        self.assertEqual((first["start"], first["end"]), (0, 2))
        self.assertEqual(asr.glossaries[0], ["Svea"])
        self.assertTrue(broker._finished)
        self.assertTrue(any(e["type"] == "commit" for e in events))

    async def test_graceful_stop_closes_input_and_preserves_inflight_and_tail_audio(self):
        loop = asyncio.get_running_loop()
        stop, started, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        release = Event()
        events, lengths = [], []
        class Stream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
            def __enter__(self):
                self.callback(bytes(80000), 40000, None, SimpleNamespace(input_overflow=False))
                return self
            def __exit__(self, *args):
                closed.set()
        class ASR:
            def update_glossary(self, phrases):
                pass
            def transcribe(self, path):
                with wave.open(path) as wav:
                    lengths.append(wav.getnframes())
                if len(lengths) == 1:
                    loop.call_soon_threadsafe(started.set)
                    if not release.wait(3):
                        raise RuntimeError("Test release missing")
                return "full" if len(lengths) == 1 else "tail"
        broker = AdaptiveBroker(emit=events.append)
        with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(RawInputStream=Stream)}), \
                patch("pianissimo_context.microphone.NemoASR.load", return_value=ASR()):
            task = asyncio.create_task(transcribe_microphone(broker, model_name="double", chunk_seconds=2, stop_event=stop))
            try:
                await asyncio.wait_for(started.wait(), 2)
                stop.set()
                await asyncio.wait_for(closed.wait(), 2)
                self.assertFalse(task.done())
                release.set()
                await asyncio.wait_for(task, 3)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(lengths, [32000, 8000])
        raw = [e["segment"] for e in events if e["type"] == "raw"]
        self.assertEqual([(s["start"], s["end"]) for s in raw], [(0, 2), (2, 2.5)])
        self.assertEqual([e["text"] for e in events if e["type"] == "commit"], ["full", "tail"])

    async def test_only_recovered_boost_failures_allow_transcription(self):
        for failure in [PhraseBoostingUnavailable("recovered"), RuntimeError("restore failed")]:
            with self.subTest(error=type(failure).__name__):
                stop = asyncio.Event()
                calls, events = [], []
                class Stream:
                    def __init__(self, **kwargs):
                        self.callback = kwargs["callback"]
                    def __enter__(self):
                        self.callback(bytes(64000), 32000, None, SimpleNamespace(input_overflow=False))
                        stop.set()
                        return self
                    def __exit__(self, *args):
                        pass
                class ASR:
                    def update_glossary(self, phrases):
                        raise failure
                    def transcribe(self, path):
                        calls.append(path)
                        return "text"
                broker = AdaptiveBroker(emit=events.append)
                with patch.dict("sys.modules", {"sounddevice": SimpleNamespace(RawInputStream=Stream)}), \
                        patch("pianissimo_context.microphone.NemoASR.load", return_value=ASR()):
                    run = transcribe_microphone(broker, model_name="double", chunk_seconds=2, stop_event=stop)
                    if isinstance(failure, PhraseBoostingUnavailable):
                        await run
                        self.assertEqual(len(calls), 1)
                        self.assertTrue(any(e["type"] == "asr_warning" for e in events))
                    else:
                        with self.assertRaises(RuntimeError):
                            await run
                        self.assertEqual(calls, [])
                        self.assertFalse(any(e["type"] == "asr_warning" for e in events))
