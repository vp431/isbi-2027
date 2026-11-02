import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from pipeline import SEEDS, build_inputs, cross_validate, extra_trees, load, save, task_data

TASKS = {"AD vs. CN": 30, "FTD vs. CN": 30, "3-class": 40}
INPUTS = ["D1", "D2", "Concat"]
MODELS = {
    "ET": extra_trees,
    "RF": lambda s: RandomForestClassifier(
        n_estimators=500, max_depth=10, class_weight="balanced", random_state=s, n_jobs=-1
    ),
    "SVM-C1": lambda s: SVC(C=1, kernel="rbf", gamma="scale", class_weight="balanced", random_state=s),
    "SVM-C10": lambda s: SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced", random_state=s),
    "LR": lambda s: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=s),
    "GBM": lambda s: GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=s),
    "MLP": lambda s: MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=s
    ),
    "kNN": lambda s: KNeighborsClassifier(n_neighbors=7),
}


def main():
    warnings.filterwarnings("ignore")
    df, d1, d2 = load()
    inputs = build_inputs(df, d1, d2)
    results = {}
    for task, k in TASKS.items():
        for model, make in MODELS.items():
            for name in INPUTS:
                X, y = task_data(inputs[name], df.group.values, task)
                folds, _ = cross_validate(X, y, k, make_model=make)
                per_seed = folds[:, 0].reshape(len(SEEDS), -1).mean(1)
                results[f"{task}|{model}|{name}"] = [round(float(per_seed.mean()), 2), round(float(per_seed.std(ddof=1)), 2)]
            row = [results[f"{task}|{model}|{n}"][0] for n in INPUTS]
            print(f"{task:11s} {model:8s} " + " ".join(f"{n}={a:5.1f}" for n, a in zip(INPUTS, row)), flush=True)
    for task in TASKS:
        gain = [results[f"{task}|{m}|Concat"][0] - results[f"{task}|{m}|D1"][0] for m in MODELS]
        print(f"{task:11s} mean accuracy gain of Concat over D1: {np.mean(gain):.1f}")
    save(results, "classifiers.json")


if __name__ == "__main__":
    main()
