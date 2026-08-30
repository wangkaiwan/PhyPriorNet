"""Patient-level, site-stratified K-fold splits (deterministic)."""
from __future__ import annotations

import random
from collections import defaultdict


def make_kfold_splits(patients: list[str], sites: dict[str, str],
                      k: int = 5, seed: int = 42) -> dict[str, dict[str, list[str]]]:
    by_site: dict[str, list[str]] = defaultdict(list)
    for p in sorted(patients):
        by_site[sites[p]].append(p)

    rng = random.Random(seed)
    # assign each patient a fold index, round-robin within each site
    fold_of: dict[str, int] = {}
    for site, plist in sorted(by_site.items()):
        shuffled = plist[:]
        rng.shuffle(shuffled)
        for i, p in enumerate(shuffled):
            fold_of[p] = i % k

    folds: dict[str, dict[str, list[str]]] = {}
    for f in range(k):
        val = sorted(p for p, fi in fold_of.items() if fi == f)
        train = sorted(p for p, fi in fold_of.items() if fi != f)
        folds[f"fold_{f}"] = {"train": train, "val": val}
    return folds
