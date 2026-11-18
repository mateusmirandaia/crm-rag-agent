"""Retrieval eval: hit-rate@k and MRR on answerable questions, refusal rate on the rest.

Run after ingest:  python eval/run_eval.py --k 5 --min-hit-rate 0.8

A question counts as a hit when any of its expected record ids is among the top k
retrieved records. Questions with an empty `expected` list are off-topic on purpose;
for those the agent should say it doesn't know.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from crm_rag.agent import RagAgent
from crm_rag.config import get_settings
from crm_rag.providers import build_providers
from crm_rag.store import VectorStore

DEFAULT_QUESTIONS = Path(__file__).with_name("questions.jsonl")


@dataclass
class Report:
    answerable: int = 0
    hits: int = 0
    reciprocal_rank_sum: float = 0.0
    unanswerable: int = 0
    refusals: int = 0
    filtered_hits: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.answerable if self.answerable else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank_sum / self.answerable if self.answerable else 0.0

    @property
    def refusal_rate(self) -> float:
        return self.refusals / self.unanswerable if self.unanswerable else 1.0


def evaluate(agent: RagAgent, questions: list[dict], k: int, verbose: bool = False) -> Report:
    report = Report()
    for item in questions:
        expected = set(item["expected"])
        if not expected:
            report.unanswerable += 1
            refused = not agent.ask(item["question"], k).sources
            report.refusals += refused
            if verbose and not refused:
                print(f"  ANSWERED  {item['question']}")
            continue

        report.answerable += 1
        ranked = [r.chunk.doc_id for r in agent.retrieve(item["question"], k)]
        rank = next((i + 1 for i, doc_id in enumerate(ranked) if doc_id in expected), None)
        if rank:
            report.hits += 1
            report.reciprocal_rank_sum += 1 / rank
        # A hit in retrieval can still be thrown away by the score threshold.
        if rank and not agent.ask(item["question"], k).sources:
            report.filtered_hits += 1
        if verbose:
            print(f"  {'HIT ' if rank else 'MISS'} @{rank or '-'}  {item['question']}  {ranked}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--min-hit-rate", type=float, default=0.0, help="exit non-zero below this hit-rate (for CI)"
    )
    parser.add_argument("--min-refusal-rate", type=float, default=0.0)
    parser.add_argument(
        "--min-score", type=float, help="override the score threshold, useful for calibrating it"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    embedder, generator = build_providers(settings)
    store = VectorStore.load(settings.index_dir, expected_embedder=embedder.name)
    min_score = args.min_score if args.min_score is not None else settings.min_score
    agent = RagAgent(embedder, generator, store, min_score)

    with args.questions.open(encoding="utf-8") as fh:
        questions = [json.loads(line) for line in fh if line.strip()]

    report = evaluate(agent, questions, args.k, args.verbose)
    print(f"embedder        {embedder.name} (min_score={agent.min_score})")
    print(f"hit-rate@{args.k:<6} {report.hit_rate:.2f}  ({report.hits}/{report.answerable})")
    print(f"MRR@{args.k:<11} {report.mrr:.2f}")
    print(
        f"refusal rate    {report.refusal_rate:.2f}  ({report.refusals}/{report.unanswerable})"
        "  on off-topic questions"
    )
    if report.filtered_hits:
        print(f"note: {report.filtered_hits} hits were filtered out by the threshold")

    ok = report.hit_rate >= args.min_hit_rate and report.refusal_rate >= args.min_refusal_rate
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
