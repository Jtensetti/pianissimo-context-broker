import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from pianissimo_context import AdaptiveBroker, Context
from pianissimo_context.microphone import transcribe_microphone


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
