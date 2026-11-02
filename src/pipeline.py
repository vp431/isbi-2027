import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features.csv"
RESULTS = ROOT / "results"

SEEDS = [42, 0, 1, 7, 123, 17, 99, 31, 55, 77]
TASKS = {"AD vs. CN": 30, "FTD vs. CN": 30, "AD vs. FTD": 30, "3-class": 40}
CLASSES = {"AD": 0, "FTD": 1, "CN": 2}
N_COMPONENTS = 20
METRICS = ["acc", "prec", "rec", "f1", "auc"]


def load():
    df = pd.read_csv(FEATURES)
    d1 = [c for c in df.columns if c.startswith("D1_")]
    d2 = ["D2_" + c[3:] for c in d1]
    return df, d1, d2


def build_inputs(df, d1, d2, perm=None):
    f1 = df[d1].to_numpy(float)
    f2 = df[d2].to_numpy(float)
    if perm is not None:
        f2 = f2[perm]
    return {"D1": f1, "D2": f2, "Concat": np.hstack([f1, f2]), "Mean": (f1 + f2) / 2}


def task_data(X, groups, task):
    if task == "3-class":
        return np.nan_to_num(X), np.array([CLASSES[g] for g in groups])
    a, b = task.split(" vs. ")
    mask = np.isin(groups, [a, b])
    return np.nan_to_num(X[mask]), (groups[mask] == a).astype(int)


def preprocess(X_train, X_test, y_train, k, seed):
    scaler = RobustScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    keep = X_train.std(0) > 1e-8
    X_train, X_test = X_train[:, keep], X_test[:, keep]
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    X_train, X_test = selector.fit_transform(X_train, y_train), selector.transform(X_test)
    if X_train.shape[1] > N_COMPONENTS:
        pca = PCA(N_COMPONENTS, random_state=seed)
        X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)
    return X_train, X_test


def extra_trees(seed, n_trees=500):
    return ExtraTreesClassifier(
        n_estimators=n_trees, max_depth=10, class_weight="balanced", random_state=seed, n_jobs=-1
    )


def auc(y, proba):
    if proba.shape[1] == 2:
        return roc_auc_score(y, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr")


def classification_metrics(y, pred):
    return [
        accuracy_score(y, pred),
        precision_score(y, pred, average="macro", zero_division=0),
        recall_score(y, pred, average="macro"),
        f1_score(y, pred, average="macro"),
    ]


def splits(y, seed, n_splits=5):
    return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def cross_validate(X, y, k, seeds=SEEDS, make_model=extra_trees):
    rows = []
    proba = np.zeros((len(seeds), len(y), len(np.unique(y))))
    for i, seed in enumerate(seeds):
        for train, test in splits(y, seed):
            X_train, X_test = preprocess(X[train], X[test], y[train], k, seed)
            model = make_model(seed).fit(X_train, y[train])
            pred = model.predict(X_test)
            row = classification_metrics(y[test], pred)
            if hasattr(model, "predict_proba"):
                proba[i, test] = model.predict_proba(X_test)
                row.append(auc(y[test], proba[i, test]))
            rows.append(row)
    return 100 * np.array(rows), proba.mean(0)


def seed_sd(folds, n_seeds):
    return folds.reshape(n_seeds, -1, folds.shape[1]).mean(1).std(0, ddof=1)


def save(obj, name):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=1)
