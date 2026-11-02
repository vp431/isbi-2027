import argparse
import math
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.integrate import trapezoid
from scipy.sparse.csgraph import shortest_path
from scipy.stats import kurtosis, skew

from pipeline import FEATURES

FS = 256
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6", "FP1": "Fp1", "FP2": "Fp2"}
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
SUB_BANDS = {"low_alpha": (8, 10), "high_alpha": (10, 13), "low_beta": (13, 20), "high_beta": (20, 30)}
REGIONS = {
    "F": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "T": ["T3", "T4", "T5", "T6"],
    "C": ["C3", "Cz", "C4"],
    "P": ["P3", "Pz", "P4"],
    "O": ["O1", "O2"],
}
PAIRS = [("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("T3", "T4"),
         ("C3", "C4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2")]
GROUPS = {"A": "AD", "F": "FTD", "C": "CN"}


def divide(a, b, eps=1e-10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > eps)


def find_recordings(root):
    files = {}
    for path in sorted(Path(root, "derivatives").rglob("*.set")):
        files.setdefault(path.name.split("_")[0], path)
    return files


def load_recording(path):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    raw.rename_channels({c: ALIASES[c.upper()] for c in raw.ch_names
                         if c.upper() in ALIASES and ALIASES[c.upper()] != c})
    raw.pick(CHANNELS).reorder_channels(CHANNELS)
    raw.filter(0.5, 45.0, verbose=False)
    if raw.info["sfreq"] != FS:
        raw.resample(FS, verbose=False)
    return raw.get_data()


def relative_power(freqs, psd, bands):
    total = trapezoid(psd, freqs, axis=-1)
    out = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs <= hi)
        out[name] = divide(trapezoid(psd[:, idx], freqs[idx], axis=-1), total, eps=0)
    return out


def spectral_entropy(freqs, psd):
    p = psd[:, (freqs >= 0.5) & (freqs <= 45)]
    p = p / p.sum(1, keepdims=True)
    return np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in p])


def sample_entropy(x, m=2, r=0.2, n=2000):
    x = x[:n]
    tol = r * np.std(x)
    if tol == 0:
        return 0.0

    def matches(length):
        t = sliding_window_view(x, length)[: len(x) - length]
        return sum(np.sum(np.max(np.abs(t[j + 1:] - t[j]), axis=1) < tol) for j in range(len(t)))

    b, a = matches(m), matches(m + 1)
    return -np.log(a / b) if a > 0 and b > 0 else 0.0


def permutation_entropy(x, order=3, n=2000):
    patterns = np.argsort(sliding_window_view(x[:n], order), axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p)) / np.log2(math.factorial(order))


def phase_locking(data, band):
    lo, hi = BANDS[band]
    b, a = signal.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    z = np.exp(1j * np.angle(signal.hilbert(signal.filtfilt(b, a, data, axis=-1), axis=-1)))
    plv = np.abs(z @ z.conj().T) / z.shape[1]
    np.fill_diagonal(plv, 0)
    return plv


def graph_metrics(plv, threshold=0.5):
    adj = (plv > threshold).astype(float)
    np.fill_diagonal(adj, 0)
    degree = adj.sum(1)
    triangles = np.diag(adj @ adj @ adj)
    clustering = np.mean(divide(triangles, degree * (degree - 1)) * (degree >= 2))
    dist = shortest_path(adj, unweighted=True)
    off = ~np.eye(len(adj), dtype=bool)
    reachable = dist[off][np.isfinite(dist[off])]
    path_length = reachable.mean() if reachable.size else 10.0
    efficiency = np.mean(np.where(np.isfinite(dist[off]), 1 / dist[off], 0))
    return clustering, path_length, efficiency


def hjorth(data):
    dx, ddx = np.diff(data, axis=1), np.diff(data, n=2, axis=1)
    activity = data.var(1)
    mobility = np.sqrt(divide(dx.var(1), activity, eps=0))
    complexity = divide(np.sqrt(divide(ddx.var(1), dx.var(1), eps=0)), mobility, eps=0)
    return activity, mobility, complexity


def extract(data, prefix):
    freqs, psd = signal.welch(data, fs=FS, nperseg=2 * FS, noverlap=FS, axis=-1)
    bands = relative_power(freqs, psd, BANDS)
    sub_bands = relative_power(freqs, psd, SUB_BANDS)
    region_idx = {r: [CHANNELS.index(c) for c in chs] for r, chs in REGIONS.items()}
    f = {}

    for name, power in {**bands, **sub_bands}.items():
        f.update({f"{name}_{c}": v for c, v in zip(CHANNELS, power)})

    entropies = {
        "SE": spectral_entropy(freqs, psd),
        "SampEn": [sample_entropy(x) for x in data],
        "PE": [permutation_entropy(x) for x in data],
    }
    for name, values in entropies.items():
        f.update({f"{name}_{c}": v for c, v in zip(CHANNELS, values)})

    for band in ["theta", "alpha"]:
        plv = phase_locking(data, band)
        cc, cpl, ge = graph_metrics(plv)
        upper = plv[np.triu_indices(len(CHANNELS), k=1)]
        f.update({f"PLV_{band}_CC": cc, f"PLV_{band}_CPL": cpl, f"PLV_{band}_GE": ge,
                  f"PLV_{band}_mean": upper.mean(), f"PLV_{band}_std": upper.std()})

    for name, power in bands.items():
        for region, idx in region_idx.items():
            f[f"{name}_mean_{region}"] = power[idx].mean()
            f[f"{name}_std_{region}"] = power[idx].std()

    ratios = {
        "theta_alpha_ratio": divide(bands["theta"], bands["alpha"]),
        "delta_alpha_ratio": divide(bands["delta"], bands["alpha"]),
        "slowing_ratio": divide(bands["delta"] + bands["theta"], bands["alpha"] + bands["beta"]),
        "alpha_beta_ratio": divide(bands["alpha"], bands["beta"]),
    }
    for name, values in ratios.items():
        f.update({f"{name}_{c}": v for c, v in zip(CHANNELS, values)})
        f.update({f"{name}_mean_{r}": values[idx].mean() for r, idx in region_idx.items()})

    for name, power in bands.items():
        for left, right in PAIRS:
            lv, rv = power[CHANNELS.index(left)], power[CHANNELS.index(right)]
            f[f"{name}_asym_{left}_{right}"] = divide(rv - lv, rv + lv)

    for c, act, mob, comp in zip(CHANNELS, *hjorth(data)):
        f.update({f"Hjorth_act_{c}": act, f"Hjorth_mob_{c}": mob, f"Hjorth_comp_{c}": comp})

    alpha = (freqs >= 8) & (freqs <= 13)
    paf = freqs[alpha][psd[:, alpha].argmax(1)]
    f.update({f"PAF_{c}": v for c, v in zip(CHANNELS, paf)})
    f["PAF_mean"] = paf.mean()

    for c, x in zip(CHANNELS, data):
        f.update({f"kurtosis_{c}": kurtosis(x), f"skewness_{c}": skew(x)})

    return {f"{prefix}_{k}": float(v) for k, v in f.items()}


def main():
    parser = argparse.ArgumentParser(description="Extract rest (D1) and photic (D2) EEG features.")
    parser.add_argument("--d1", required=True, help="path to OpenNeuro ds004504")
    parser.add_argument("--d2", required=True, help="path to OpenNeuro ds006036")
    parser.add_argument("--out", default=str(FEATURES))
    args = parser.parse_args()

    participants = pd.read_csv(Path(args.d1, "participants.tsv"), sep="\t")
    labels = dict(zip(participants.participant_id, participants.Group.map(GROUPS)))
    d1, d2 = find_recordings(args.d1), find_recordings(args.d2)
    subjects = sorted(set(d1) & set(d2) & set(labels))

    rows = []
    for sub in subjects:
        print(sub, flush=True)
        row = {"participant_id": sub}
        row.update(extract(load_recording(d1[sub]), "D1"))
        row.update(extract(load_recording(d2[sub]), "D2"))
        row["group"] = labels[sub]
        rows.append(row)

    table = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} subjects, {table.shape[1] - 2} features -> {args.out}")


if __name__ == "__main__":
    main()
