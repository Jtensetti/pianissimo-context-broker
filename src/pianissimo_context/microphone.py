"""Optional bounded PCM microphone capture and independent-chunk NeMo decoding.

Capture runs in a PortAudio callback while ASR and context work off-loop.
This is chunked live transcription, not cache-aware neural streaming.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import wave

from .asr import NemoASR
from .live import AdaptiveBroker, LiveSession


async def transcribe_microphone(broker: AdaptiveBroker, *, model_name: str,
                               chunk_seconds: int = 4, device: str | int | None = None):
    import sounddevice as sd
    # Initial model download/load happens before microphone capture is started.
    load_task = asyncio.create_task(asyncio.to_thread(NemoASR.load, model_name))
    try:
        asr = await asyncio.shield(load_task)
    except asyncio.CancelledError:
        await load_task
        raise
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue(maxsize=3)
    block_size = 1600
    chunk_bytes = chunk_seconds * 32000
    buffer = bytearray()
    captured_frames = 0
    accepting = True

    def enqueue(data, start, end, overflow):
        if not accepting:
            return
        if overflow:
            broker.emit({"type": "audio_warning", "message": "Microphone input overflow; audio may be missing"})
        if audio_queue.full():
            broker.emit({"type": "audio_drop", "start": start, "end": end,
                         "message": "ASR fell behind; this audio chunk was dropped"})
            return
        audio_queue.put_nowait((data, start, end))

    def callback(indata, frames, callback_time, status):
        nonlocal captured_frames
        buffer.extend(bytes(indata))
        captured_frames += frames
        if status.input_overflow:
            loop.call_soon_threadsafe(broker.emit, {"type": "audio_warning", "message": "Microphone input overflow"})
        while len(buffer) >= chunk_bytes:
            data = bytes(buffer[:chunk_bytes])
            del buffer[:chunk_bytes]
            end = (captured_frames - len(buffer) // 2) / 16000
            loop.call_soon_threadsafe(enqueue, data, end - chunk_seconds, end, False)

    try:
        async with LiveSession(broker) as session:
            with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", device=device,
                                   blocksize=block_size, callback=callback):
                broker.emit({"type": "audio_ready", "sample_rate": 16000, "chunk_seconds": chunk_seconds})
                with tempfile.TemporaryDirectory() as directory:
                    while True:
                        data, start, end = await audio_queue.get()
                        path = str(Path(directory) / "chunk.wav")
                        with wave.open(path, "wb") as chunk:
                            chunk.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                            chunk.writeframes(data)
                        try:
                            await asyncio.to_thread(asr.update_glossary, broker.glossary())
                        except RuntimeError:
                            broker.emit({"type": "asr_warning", "message": "Phrase boosting unavailable; using base decoder"})
                        # Don't leave a decoding thread accessing a deleted WAV
                        # after cancellation/closing the tempfile directory.
                        task = asyncio.create_task(asyncio.to_thread(asr.transcribe, path))
                        try:
                            raw = await asyncio.shield(task)
                        except asyncio.CancelledError:
                            await task
                            raise
                        session.push(raw, start=start, end=end)
    finally:
        accepting = False
