import numpy as np


def validate_dataset(X, y, idx):
    """Basic quality checks. AssertionError aaye to dataset me problem hai."""
    assert not np.isnan(X).any(), "X me NaN hai"
    assert not np.isinf(X).any(), "X me inf hai"
    tr, va, te = set(idx["train"]), set(idx["val"]), set(idx["test"])
    assert not (tr & va or tr & te or va & te), "Splits overlap kar rahe hain (leakage)"
    print("OK: no NaN/inf, no split overlap")
    if y is not None:
        for s, ids in idx.items():
            vals, cnt = np.unique(y[ids], return_counts=True)
            print(s, dict(zip(vals.tolist(), cnt.tolist())))
