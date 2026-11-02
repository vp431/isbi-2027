import numpy as np
from sklearn.feature_selection import f_classif

from pipeline import build_inputs, load, save, task_data

MARKER = "theta_alpha_ratio_mean_O"


def cohen_d(a, b):
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled)


def main():
    df, d1, d2 = load()
    groups = df.group.values
    results = {}
    for cond in ["D1", "D2"]:
        x = df[f"{cond}_{MARKER}"].to_numpy(float)
        for g in ["AD", "FTD"]:
            results[f"d|{cond}|{g} vs CN"] = round(float(cohen_d(x[groups == g], x[groups == "CN"])), 2)
    names = np.array(d1 + d2)
    concat = build_inputs(df, d1, d2)["Concat"]
    for task in ["AD vs. CN", "FTD vs. CN"]:
        X, y = task_data(concat, groups, task)
        F = np.nan_to_num(f_classif(X, y)[0])
        top = names[np.argsort(F)[::-1][:15]].tolist()
        results[f"top15|{task}"] = top
        results[f"top15_from|{task}"] = {c: sum(n.startswith(c) for n in top) for c in ["D1", "D2"]}
    for key, value in results.items():
        if not key.startswith("top15|"):
            print(key, value)
    save(results, "biomarkers.json")


if __name__ == "__main__":
    main()
