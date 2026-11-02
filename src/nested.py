import numpy as np
from sklearn.metrics import accuracy_score

from pipeline import SEEDS, auc, build_inputs, extra_trees, load, preprocess, save, splits, task_data

TASKS = ["AD vs. CN", "FTD vs. CN", "AD vs. FTD"]
INPUTS = ["D1", "D2", "Mean"]
K_GRID = [10, 20, 30, 40, 60]


def select_k(X, y, seed):
    scores = {}
    for k in K_GRID:
        acc = []
        for train, test in splits(y, seed + 1000, n_splits=3):
            X_train, X_test = preprocess(X[train], X[test], y[train], k, seed)
            model = extra_trees(seed, 100).fit(X_train, y[train])
            acc.append(accuracy_score(y[test], model.predict(X_test)))
        scores[k] = np.mean(acc)
    return max(K_GRID, key=lambda k: (scores[k], -k))


def main():
    df, d1, d2 = load()
    inputs = build_inputs(df, d1, d2)
    results = {}
    for task in TASKS:
        for name in INPUTS:
            X, y = task_data(inputs[name], df.group.values, task)
            acc, aucs, chosen = [], [], []
            for seed in SEEDS:
                for train, test in splits(y, seed):
                    k = select_k(X[train], y[train], seed)
                    X_train, X_test = preprocess(X[train], X[test], y[train], k, seed)
                    model = extra_trees(seed).fit(X_train, y[train])
                    acc.append(100 * accuracy_score(y[test], model.predict(X_test)))
                    aucs.append(100 * auc(y[test], model.predict_proba(X_test)))
                    chosen.append(k)
            results[f"{task}|{name}"] = {
                "acc": round(float(np.mean(acc)), 2),
                "auc": round(float(np.mean(aucs)), 2),
                "k_mode": int(np.bincount(chosen).argmax()),
            }
            print(f"{task:11s} {name:5s} {results[f'{task}|{name}']}", flush=True)
    save(results, "nested.json")


if __name__ == "__main__":
    main()
