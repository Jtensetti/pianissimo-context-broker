from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import wave

from .broker import Broker, Context
from .llm import Ollama


def output(event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


async def run(args):
    context = Context(**json.loads(Path(args.context).read_text(encoding="utf-8"))) if args.context else Context()
    controller = Ollama(args.ollama_model) if args.ollama_model else None
    broker = Broker(context, controller, output, args.mutable_seconds)
    if args.command == "demo":
        broker.context = Context(topic="Kommunal AI-utveckling", organisations=("AI Sweden",),
                                 terms=("Svea", "pseudonymisering"), aliases={"aj sveden": "AI Sweden", "svea": "Svea"})
        for text in ["vi har samarbetat med aj sveden inom projekt svea",
                     "Nästa punkt är pseudonymisering inom socialtjänsten."]:
            segment = broker.add(text)
            await broker.repair(segment.id)
        broker.finish()
    elif args.command == "text":
        # Read stdin off-loop while a timer enforces locking during silence.
        async def timer():
            while True:
                await asyncio.sleep(.1)
                broker.tick()
        timer_task = asyncio.create_task(timer())
        notifications = asyncio.Queue(maxsize=1)
        async def worker():
            last_id = -1
            while True:
                await notifications.get()
                try:
                    for segment_id in list(broker.segments):
                        if segment_id > last_id:
                            last_id = segment_id
                            await broker.repair(segment_id)
                finally:
                    notifications.task_done()
        worker_task = asyncio.create_task(worker())
        try:
            while line := await asyncio.to_thread(sys.stdin.readline):
                segment = broker.add(line.rstrip("\r\n"))
                if notifications.empty():
                    notifications.put_nowait(None)
                broker.release_committed()
            await notifications.join()
            broker.finish()
        finally:
            timer_task.cancel()
            worker_task.cancel()
            await asyncio.gather(timer_task, worker_task, return_exceptions=True)
    else:
        from .asr import NemoASR
        asr = NemoASR.load(args.asr_model)
        with wave.open(args.file, "rb") as source, tempfile.TemporaryDirectory() as directory:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
                raise ValueError("Use a 16 kHz mono PCM16 WAV file")
            position = 0
            while data := source.readframes(16000 * args.chunk_seconds):
                path = str(Path(directory) / "chunk.wav")
                with wave.open(path, "wb") as chunk:
                    chunk.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    chunk.writeframes(data)
                try:
                    asr.update_glossary(broker.glossary())
                except RuntimeError:
                    output({"type": "asr_warning", "message": "Phrase boosting unavailable; using base decoder"})
                text = await asyncio.to_thread(asr.transcribe, path)
                duration = len(data) / 32000
                segment = broker.add(text, start=position, end=position + duration)
                await broker.repair(segment.id)
                position += duration
                broker.release_committed()
        broker.finish()


def main():
    parser = argparse.ArgumentParser(description="Local conservative ASR context broker")
    parser.add_argument("--context", help="User-supplied context JSON")
    parser.add_argument("--ollama-model", help="Installed local Ollama model name; omitted = aliases only")
    parser.add_argument("--mutable-seconds", type=float, default=15)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    commands.add_parser("text", help="Read transcript segments from stdin, emit JSONL events")
    audio = commands.add_parser("audio", help="Experimental independent WAV chunk decoding")
    audio.add_argument("file")
    audio.add_argument("--asr-model", default="KlangAI/pianissimo-sv")
    audio.add_argument("--chunk-seconds", type=int, choices=range(1, 31), default=10)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
