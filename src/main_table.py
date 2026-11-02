from pipeline import METRICS, SEEDS, TASKS, build_inputs, cross_validate, load, save, seed_sd, task_data


def main():
    df, d1, d2 = load()
    inputs = build_inputs(df, d1, d2)
    results = {}
    for task, k in TASKS.items():
        for name, X in inputs.items():
            X_task, y = task_data(X, df.group.values, task)
            folds, _ = cross_validate(X_task, y, k)
            mean, sd = folds.mean(0), seed_sd(folds, len(SEEDS))
            results[f"{task}|{name}"] = {
                m: {"mean": round(float(a), 4), "sd": round(float(b), 4)} for m, a, b in zip(METRICS, mean, sd)
            }
            print(f"{task:11s} {name:7s} " + " ".join(f"{m}={a:5.1f}" for m, a in zip(METRICS, mean)), flush=True)
    save(results, "main_table.json")


if __name__ == "__main__":
    main()
