#File 1: src/build_dataset.py
import json
import re
from pathlib import Path

from datasets import load_dataset

# ---------------- settings ----------------
DATASET = "qiaojin/PubMedQA"
CONFIG = "pqa_labeled"
N_DEV = 200          # ids 0 to 199 are the development set
N_FRESH = 300        # fresh test questions taken after the development ids
# ------------------------------------------

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:            # running inside a notebook cell
    ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def split_sentences(text):
    """Simple sentence splitter. Keep identical across runs for comparability."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def to_item(index, row):
    sentences = []
    for paragraph in row["context"]["contexts"]:
        sentences.extend(split_sentences(paragraph))
    return {
        "id": index,
        "question": row["question"],
        "gold_answer": row["final_decision"],       # yes, no or maybe
        "doc": {f"S{k}": s for k, s in enumerate(sentences)},
    }


def main():
    ds = load_dataset(DATASET, CONFIG, split="train")

    dev, fresh = [], []
    for i, row in enumerate(ds):
        item = to_item(i, row)
        if i < N_DEV:
            dev.append(item)
        elif item["gold_answer"] in ("yes", "no") and len(fresh) < N_FRESH:
            fresh.append(item)

    dev_ids = {it["id"] for it in dev}
    assert not any(it["id"] in dev_ids for it in fresh), "dev and fresh overlap"
    assert len(fresh) == N_FRESH, f"only found {len(fresh)} fresh questions"

    for name, items in (("dev200", dev), ("fresh300", fresh)):
        (DATA_DIR / f"{name}.json").write_text(json.dumps(items, indent=2))
        (DATA_DIR / f"{name}_ids.json").write_text(json.dumps([it["id"] for it in items]))

    n_yes = sum(it["gold_answer"] == "yes" for it in fresh)
    print(f"dev questions: {len(dev)}")
    print(f"fresh questions: {len(fresh)} | yes: {n_yes} ({n_yes / len(fresh):.0%}) | no: {len(fresh) - n_yes}")
    print(f"fresh id range: {fresh[0]['id']} to {fresh[-1]['id']}")
    print(f"files written to: {DATA_DIR}")


if __name__ == "__main__":
    main()
