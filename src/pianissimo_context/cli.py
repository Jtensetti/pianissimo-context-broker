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
from .live import AdaptiveBroker, LiveSession


def output(event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


async def run(args):
    context = Context(**json.loads(Path(args.context).read_text(encoding="utf-8"))) if args.context else Context()
    controller = Ollama(args.ollama_model) if args.ollama_model else None
    if args.command in {"text", "live", "microphone"}:
        broker = AdaptiveBroker(context, controller, output, args.mutable_seconds,
                                initial_context=args.initial_context, review_seconds=args.review_seconds,
                                allow_context_repairs=not args.suggestions_only)
    else:
        broker = Broker(context, controller, output, args.mutable_seconds)
    if args.command == "demo":
        broker.context = Context(topic="Kommunal AI-utveckling", organisations=("AI Sweden",),
                                 terms=("Svea", "pseudonymisering"), aliases={"aj sveden": "AI Sweden", "svea": "Svea"})
        for text in ["vi har samarbetat med aj sveden inom projekt svea",
                     "Nästa punkt är pseudonymisering inom socialtjänsten."]:
            segment = broker.add(text)
            await broker.repair(segment.id)
        broker.finish()
    elif args.command in {"text", "live"}:
        async with LiveSession(broker) as session:
            while line := await asyncio.to_thread(sys.stdin.readline):
                session.push(line.rstrip("\r\n"))
    elif args.command == "microphone":
        from .microphone import transcribe_microphone
        await transcribe_microphone(broker, model_name=args.asr_model,
                                   chunk_seconds=args.chunk_seconds, device=args.device)
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
    parser.add_argument("--initial-context", default="", help="Free-text background for the live conversation")
    parser.add_argument("--review-seconds", type=float, default=25, help="Live topic review cadence, 20–30 seconds")
    parser.add_argument("--suggestions-only", action="store_true", help="Disable automatic context term substitutions")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    commands.add_parser("text", aliases=["live"], help="Live raw ASR segments from stdin; adaptive context; JSONL output")
    microphone = commands.add_parser("microphone", help="Chunked live capture with Pianissimo and adaptive context")
    microphone.add_argument("--asr-model", default="KlangAI/pianissimo-sv")
    microphone.add_argument("--chunk-seconds", type=int, choices=range(2, 11), default=4)
    microphone.add_argument("--device", help="PortAudio input device name")
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
