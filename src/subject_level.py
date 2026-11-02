import numpy as np
from scipy import stats

from pipeline import build_inputs, cross_validate, load, save, task_data

TASKS = ["AD vs. CN", "FTD vs. CN", "AD vs. FTD"]
COMPARISONS = [("D2", "D1"), ("Concat", "D1"), ("Concat", "D2")]


def delong(y, p1, p2):
    def components(p):
        pos, neg = p[y == 1], p[y == 0]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        return psi.mean(), psi.mean(1), psi.mean(0)

    a1, v10_1, v01_1 = components(p1)
    a2, v10_2, v01_2 = components(p2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    s = s10 / (y == 1).sum() + s01 / (y == 0).sum()
    z = (a1 - a2) / np.sqrt(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    return float(a1), float(a2), float(2 * stats.norm.sf(abs(z)))


def main():
    df, d1, d2 = load()
    inputs = build_inputs(df, d1, d2)
    results = {}
    for task in TASKS:
        scores = {}
        for name in ["D1", "D2", "Concat"]:
            X, y = task_data(inputs[name], df.group.values, task)
            _, proba = cross_validate(X, y, 30)
            scores[name] = proba[:, 1]
        for a, b in COMPARISONS:
            auc_a, auc_b, p = delong(y, scores[a], scores[b])
            results[f"{task}|{a} vs {b}"] = {"auc_a": round(100 * auc_a, 1), "auc_b": round(100 * auc_b, 1), "p": round(p, 4)}
            print(f"{task:11s} {a} vs {b}: {results[f'{task}|{a} vs {b}']}", flush=True)
    save(results, "subject_level.json")


if __name__ == "__main__":
    main()
