"""Audit a real local judge with known ground truth.

Builds a small arithmetic benchmark where the correct answer is known by
construction, asks an Ollama model to judge answer pairs in BOTH presentation
orders (plus a few identical pairs), and writes the verdicts to JSONL for
``judgekit report``.

The pairs are adversarial on purpose: half the time the wrong answer is the
long, worked-through one and the correct answer is one bare line, so a judge
that grades effort or padding instead of correctness shows up immediately.

Run:  python examples/audit_ollama.py --model qwen2.5:14b
Then: judgekit report examples/results/<model>.jsonl \
          --human examples/results/human.jsonl --out examples/results/audit

Standard library only; expects an Ollama server on localhost:11434.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from judgekit.io import dump_verdicts
from judgekit.schema import PairwiseVerdict, Verdict

HERE = Path(__file__).parent

PROMPT = """You are judging two candidate answers to a question.

Question: {question}

Answer 1:
{first}

Answer 2:
{second}

Which answer is correct? Reply with JSON: {{"choice": "1" | "2" | "tie"}}."""

SCHEMA = {"type": "object", "properties": {"choice": {"type": "string"}}, "required": ["choice"]}


def _concise(value: int) -> str:
    return f"The answer is {value}."


def _verbose(a: int, b: int, value: int) -> str:
    return (
        f"Let me work through this carefully step by step. We start with {a}. "
        f"The problem asks us to combine it with {b}, so I will add them, "
        f"double-checking each digit as I go and carrying where needed. "
        f"After verifying the arithmetic twice and considering edge cases, "
        f"I am confident the final result is {value}."
    )


def build_items(n_pairs: int, n_identical: int) -> list[dict[str, object]]:
    """Deterministic items; no RNG so reruns build the identical benchmark."""
    items: list[dict[str, object]] = []
    for i in range(n_pairs):
        a = 137 + 31 * i
        b = 249 + 17 * i
        correct = a + b
        wrong = correct + (7 if i % 2 == 0 else -13)
        question = f"What is {a} + {b}?"
        # Alternate which style the CORRECT answer gets, so correctness and
        # verbosity are unconfounded by construction.
        if i % 2 == 0:
            text_a, text_b = _concise(correct), _verbose(a, b, wrong)
        else:
            text_a, text_b = _verbose(a, b, correct), _concise(wrong)
        items.append(
            {
                "item_id": f"sum{i}",
                "question": question,
                "a": text_a,
                "b": text_b,
                "correct": "a",
                "identical": False,
            }
        )
    for i in range(n_identical):
        a = 500 + 41 * i
        b = 311 + 23 * i
        text = _concise(a + b)
        items.append(
            {
                "item_id": f"same{i}",
                "question": f"What is {a} + {b}?",
                "a": text,
                "b": text,
                "correct": "tie",
                "identical": True,
            }
        )
    return items


def ask(model: str, question: str, first: str, second: str, host: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(question=question, first=first, second=second),
            }
        ],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        content = json.loads(response.read())["message"]["content"]
    choice = str(json.loads(content).get("choice", "")).strip().lower()
    if choice in {"1", "2", "tie"}:
        return choice
    match = re.search(r"[12]", choice)
    return match.group(0) if match else "tie"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--pairs", type=int, default=90)
    parser.add_argument("--identical", type=int, default=15)
    args = parser.parse_args()

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    items = build_items(args.pairs, args.identical)

    human: list[Verdict] = [
        PairwiseVerdict(item_id=str(item["item_id"]), judge_id="human", choice=item["correct"])
        for item in items
    ]
    dump_verdicts(human, out_dir / "human.jsonl")

    slug = args.model.replace(":", "-").replace("/", "-")
    judge_path = out_dir / f"{slug}.jsonl"
    done: set[tuple[str, bool]] = set()
    if judge_path.exists():  # resume an interrupted run
        for line in judge_path.read_text().splitlines():
            record = json.loads(line)
            done.add((record["item_id"], record["swapped"]))

    with judge_path.open("a", encoding="utf-8") as sink:
        for index, item in enumerate(items):
            for swapped in (False, True):
                if (item["item_id"], swapped) in done:
                    continue
                first, second = (item["b"], item["a"]) if swapped else (item["a"], item["b"])
                slot = ask(args.model, str(item["question"]), str(first), str(second), args.host)
                if slot == "tie":
                    canonical = "tie"
                elif swapped:
                    canonical = "b" if slot == "1" else "a"
                else:
                    canonical = "a" if slot == "1" else "b"
                verdict = PairwiseVerdict(
                    item_id=str(item["item_id"]),
                    judge_id=args.model,
                    choice=canonical,
                    swapped=swapped,
                    a_len=len(str(item["a"])),
                    b_len=len(str(item["b"])),
                    meta={"identical": True} if item["identical"] else {},
                )
                sink.write(verdict.model_dump_json(exclude_none=True) + "\n")
                sink.flush()
            print(f"{index + 1}/{len(items)} items judged", flush=True)

    print(f"verdicts in {judge_path}")


if __name__ == "__main__":
    main()
