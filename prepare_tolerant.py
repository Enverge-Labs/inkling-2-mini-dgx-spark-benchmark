import collections
import sys
from pathlib import Path

from datasets import load_dataset

import speed_bench_prepare as sb

config, out_dir = sys.argv[1], Path(sys.argv[2])
original = sb._fetch_all_turns_data


def tolerant(example, speed_config):
    try:
        resolved = original(example, speed_config)
        resolved["resolved"] = True
        return resolved
    except Exception:
        example["resolved"] = False
        return example


dataset = load_dataset("nvidia/SPEED-Bench", config, split="test")
dataset = dataset.map(tolerant, fn_kwargs={"speed_config": config})
dropped = collections.Counter(r["source"] for r in dataset if not r["resolved"])
dataset = dataset.filter(lambda r: r["resolved"]).remove_columns(["resolved"])
dataset = dataset.map(
    lambda r: {"messages": [{"role": "user", "content": t} for t in r["turns"]]},
    remove_columns=["turns"],
)
out_dir.mkdir(parents=True, exist_ok=True)
dataset.to_json(out_dir / f"{config}.jsonl")
print(f"kept {len(dataset)} rows; dropped {sum(dropped.values())}: {dict(dropped)}")
print("kept by category:", dict(collections.Counter(dataset["category"])))
