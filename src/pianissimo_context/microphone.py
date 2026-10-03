"""Bounded PCM capture with graceful stop and independent-chunk NeMo decoding."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import wave

from .asr import NemoASR, PhraseBoostingUnavailable
from .live import AdaptiveBroker, LiveSession


async def _finish_thread(function, *args):
    """Keep models and temporary audio alive until their worker has finished."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def transcribe_microphone(broker: AdaptiveBroker, *, model_name: str,
                               chunk_seconds: int = 4, device: str | int | None = None,
                               stop_event: asyncio.Event | None = None):
    import sounddevice as sd
    if type(chunk_seconds) is not int or not 1 <= chunk_seconds <= 30:
        raise ValueError("chunk_seconds must be an integer between 1 and 30")
    stop_event = stop_event if stop_event is not None else asyncio.Event()
    asr = await _finish_thread(NemoASR.load, model_name)
    if stop_event.is_set():
        broker.finish()
        return
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue(maxsize=3)
    chunk_bytes = chunk_seconds * 32000
    buffer = bytearray()
    captured_frames = 0
    accepting = True
    producer_error: Exception | None = None

    def enqueue(data, start, end):
        if not accepting:
            return
        if audio_queue.full():
            broker.emit({"type": "audio_drop", "start": start, "end": end,
                         "message": "ASR fell behind; this audio chunk was dropped"})
        else:
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
            loop.call_soon_threadsafe(enqueue, data, end - chunk_seconds, end)

    async def capture():
        nonlocal accepting, producer_error
        try:
            with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", device=device,
                                   blocksize=1600, callback=callback):
                broker.emit({"type": "audio_ready", "sample_rate": 16000, "chunk_seconds": chunk_seconds})
                await stop_event.wait()
            # The device callback has stopped. Drain already posted callbacks
            # before appending the final partial chunk and sentinel.
            await asyncio.sleep(0)
            accepting = False
            if buffer:
                end = captured_frames / 16000
                await audio_queue.put((bytes(buffer), end - len(buffer) / 32000, end))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            producer_error = exc
        finally:
            accepting = False
        await audio_queue.put(None)

    async with LiveSession(broker) as session:
        producer = asyncio.create_task(capture())
        try:
            with tempfile.TemporaryDirectory() as directory:
                while (item := await audio_queue.get()) is not None:
                    data, start, end = item
                    path = str(Path(directory) / "chunk.wav")
                    with wave.open(path, "wb") as chunk:
                        chunk.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        chunk.writeframes(data)
                    try:
                        await _finish_thread(asr.update_glossary, broker.glossary())
                    except PhraseBoostingUnavailable:
                        broker.emit({"type": "asr_warning", "message": "Phrase boosting unavailable; using base decoder"})
                    raw = await _finish_thread(asr.transcribe, path)
                    session.push(raw, start=start, end=end)
            await producer
            if producer_error is not None:
                raise producer_error
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
