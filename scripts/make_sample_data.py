"""Generate a tiny SYNTHETIC dataset to smoke-test the pipeline (not for real results)."""
import random
from pathlib import Path

import pandas as pd

random.seed(0)
out = Path(__file__).resolve().parent.parent / "data" / "sample"
out.mkdir(parents=True, exist_ok=True)

fake_bits = ["shocking", "secret", "they don't want you to know", "hoax", "exposed", "miracle",
             "unbelievable", "conspiracy", "cover-up", "outrage", "stunning", "bombshell"]
real_bits = ["officials said", "according to the report", "committee", "budget", "statement",
             "lawmakers", "spokesman", "quarterly", "negotiations", "regulators", "ministry", "agreement"]
common = ["president", "government", "economy", "election", "policy", "court", "senate", "market", "tax", "border"]


def make(bits, n, prefix=""):
    rows = []
    for i in range(n):
        words = random.choices(bits, k=6) + random.choices(common, k=6) + random.choices(bits + common, k=4)
        random.shuffle(words)
        rows.append({"title": f"Story {i} " + random.choice(common),
                     "text": prefix + " ".join(words) + f" number {i}.", "subject": "x", "date": "x"})
    return pd.DataFrame(rows)

make(fake_bits, 600).to_csv(out / "Fake.csv", index=False)
make(real_bits, 600, "WASHINGTON (Reuters) - ").to_csv(out / "True.csv", index=False)
print("wrote", out)
