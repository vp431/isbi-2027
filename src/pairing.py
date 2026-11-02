import numpy as np
from scipy.stats import spearmanr

from pipeline import SEEDS, TASKS, build_inputs, cross_validate, extra_trees, load, save, task_data

N_PERMUTATIONS = 3
CONTROL_SEEDS = SEEDS[:5]
CONTROL_TREES = 300


def within_group_permutation(groups, seed):
    rng = np.random.default_rng(seed)
    perm = np.arange(len(groups))
    for g in ["AD", "CN", "FTD"]:
        idx = np.where(groups == g)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def pairing_control(df, d1, d2):
    groups = df.group.values
    perms = {"true": None}
    perms.update({f"shuffled{r}": within_group_permutation(groups, 100 + r) for r in range(N_PERMUTATIONS)})
    results = {}
    for label, perm in perms.items():
        inputs = build_inputs(df, d1, d2, perm)
        for task, k in TASKS.items():
            for name in ["Concat", "Mean"]:
                X, y = task_data(inputs[name], groups, task)
                folds, _ = cross_validate(X, y, k, CONTROL_SEEDS, lambda s: extra_trees(s, CONTROL_TREES))
                acc, auc = folds[:, 0].mean(), folds[:, -1].mean()
                results[f"{label}|{task}|{name}"] = {"acc": round(float(acc), 1), "auc": round(float(auc), 1)}
                print(f"{label:9s} {task:11s} {name:6s} acc={acc:5.1f} auc={auc:5.1f}", flush=True)
    return results


def identification(df, d1, d2):
    a = np.nan_to_num(df[d1].to_numpy(float))
    b = np.nan_to_num(df[d2].to_numpy(float))
    keep = (a.std(0) > 0) & (b.std(0) > 0)
    a, b = a[:, keep], b[:, keep]
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    n = len(a)
    sim = np.corrcoef(a, b)[:n, n:]
    top1 = np.mean(sim.argmax(1) == np.arange(n))
    rank = np.median([(sim[i] > sim[i, i]).sum() + 1 for i in range(n)])
    rho = np.array([spearmanr(a[:, j], b[:, j]).correlation for j in range(a.shape[1])])
    return {
        "top1": round(100 * float(top1), 1),
        "chance": round(100 / n, 1),
        "median_rank": float(rank),
        "n_candidates": n,
        "feature_rho_median": round(float(np.median(rho)), 3),
        "feature_rho_max": round(float(np.max(rho)), 3),
    }


def main():
    df, d1, d2 = load()
    results = {"identification": identification(df, d1, d2)}
    print(results["identification"], flush=True)
    results["control"] = pairing_control(df, d1, d2)
    save(results, "pairing.json")


if __name__ == "__main__":
    main()
