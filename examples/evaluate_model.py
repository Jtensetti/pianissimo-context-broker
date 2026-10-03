"""Small synthetic Swedish text benchmark, not an acoustic/WER evaluation.

Run against local Ollama: python examples/evaluate_model.py --models qwen3.5:4b gemma4:e4b
"""
import argparse
import asyncio
import json
import math
import statistics
import time

from pianissimo_context import AdaptiveBroker, Ollama
from pianissimo_context.llm import RECOMMENDED_MODEL


INITIAL = ("Intervju med AI Sweden och Klang om Pianissimo i Svea. "
           "Samtalet gäller transkribering, pseudonymisering och informationssäkerhet.")
CASES = [
    ("Vi samarbetar med aj sveden.", "Vi samarbetar med AI Sweden."),
    ("Vi använder pianisimo i Svea.", "Vi använder Pianissimo i Svea."),
    ("Vi behöver psedonymisering.", "Vi behöver pseudonymisering."),
    ("Vi diskuterar informations säkerhet.", "Vi diskuterar informationssäkerhet."),
    ("Klang utvecklar Pianissimo.", "Klang utvecklar Pianissimo."),
    ("Jag odlar tomater i mitt växthus.", "Jag odlar tomater i mitt växthus."),
    ("Vi har inte godkänt förslaget.", "Vi har inte godkänt förslaget."),
    ("Förändringen var -5 %.", "Förändringen var -5 %."),
    ("Det tog tolv minuter.", "Det tog tolv minuter."),
    ("Nu byter vi ämne och pratar om semester.", "Nu byter vi ämne och pratar om semester."),
]


async def evaluate(model):
    rows, latencies = [], []
    for raw, expected in CASES:
        # Deliberately use the production request settings and timeout. The
        # background is sufficient for repair; no scripted model answers/aliases.
        broker = AdaptiveBroker(initial_context=INITIAL, controller=Ollama(model))
        segment = broker.add(raw)
        started = time.perf_counter()
        await broker.repair(segment.id)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        rows.append({"raw": raw, "expected": expected, "actual": segment.text,
                     "review_status": segment.review_status, "seconds": round(elapsed, 3),
                     "pass": segment.review_status == "reviewed" and segment.text == expected})
        broker.finish()
    return {"model": model, "synthetic_text_only": True, "cases": rows,
            "passed": sum(r["pass"] for r in rows), "total": len(rows),
            "median_seconds": round(statistics.median(latencies), 3),
            "p95_seconds": round(sorted(latencies)[math.ceil(.95 * len(latencies)) - 1], 3)}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[RECOMMENDED_MODEL])
    args = parser.parse_args()
    for model in args.models:
        print(json.dumps(await evaluate(model), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
