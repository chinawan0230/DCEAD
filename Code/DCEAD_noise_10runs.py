import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score, average_precision_score

from DCEAD import DCEAD


DATASETS = [
    ("Ann", "ann", 1.00),
    ("Bands_27", "band27", 0.65),
    ("Bands_34", "band34", 0.65),
    ("Bands_42", "band42", 0.70),
    ("Breast", "breast", 0.80),
    ("Cardio", "cardio", 0.75),
    ("CreditA", "credita", 0.95),
    ("Diabetes", "diabetes", 0.90),
    ("Heart", "heart", 0.30),
    ("Hepatitis", "hepatitis", 0.00),
    ("Ionosphere", "ionosphere", 0.60),
    ("Monks", "monks", 0.55),
    ("Thyroid", "thyroid", 0.95),
    ("Vote", "vote", 0.00),
    ("Wbc", "wbc", 0.05),
    ("Yeast", "yeast", 0.60),
]

NOISE_LEVELS = np.round(np.arange(0.0, 0.5001, 0.05), 2)


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_dataset_file(folder, token):
    files = sorted(Path(folder).rglob("*.mat"))
    target = normalize_name(token)

    for path in files:
        if target == normalize_name(path.stem):
            return path
    for path in files:
        if target in normalize_name(path.stem):
            return path

    raise FileNotFoundError(f"Cannot find dataset matching: {token}")


def load_mat_dataset(path):
    mat = sio.loadmat(path)

    if "trandata" in mat:
        arr = np.asarray(mat["trandata"])
        return arr[:, :-1].astype(float), np.asarray(arr[:, -1]).ravel()

    if "X" in mat and "y" in mat:
        return np.asarray(mat["X"]).astype(float), np.asarray(mat["y"]).ravel()

    arrays = [
        np.asarray(v)
        for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2
    ]
    arrays = [a for a in arrays if a.shape[0] >= 2 and a.shape[1] >= 2]
    if not arrays:
        raise ValueError(f"Cannot parse dataset: {path}")

    arr = max(arrays, key=lambda a: a.size)
    return arr[:, :-1].astype(float), np.asarray(arr[:, -1]).ravel()


def make_binary_labels(y_raw):
    y_raw = np.asarray(y_raw).ravel()
    values, counts = np.unique(y_raw, return_counts=True)

    if set(values.tolist()).issubset({0, 1}):
        return y_raw.astype(int)

    anomaly_value = values[np.argmin(counts)]
    return (y_raw == anomaly_value).astype(int)


def is_integer_like(values):
    unique_values = np.unique(values)
    return (
        unique_values.size <= 10
        and np.allclose(unique_values, np.round(unique_values))
    )


def corrupt_value(X, row, col, rng):
    values = X[:, col]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    if is_integer_like(values):
        X[row, col] = rng.choice(np.unique(values))
        return

    lower = float(np.min(values))
    upper = float(np.max(values))
    X[row, col] = lower if np.isclose(lower, upper) else rng.uniform(lower, upper)


def add_attribute_noise(X, noise_level, rng):
    X_noisy = np.array(X, dtype=float, copy=True)
    n, m = X_noisy.shape
    count = int(np.floor(noise_level * n))

    if count <= 0:
        return X_noisy

    rows = rng.choice(n, size=count, replace=False)
    for row in rows:
        col = int(rng.integers(0, m))
        corrupt_value(X_noisy, row, col, rng)

    return X_noisy


def run_once(X, y, epsilon):
    start = time.perf_counter()
    scores = DCEAD(X, epsilon=epsilon, verbose=False)
    elapsed = time.perf_counter() - start
    return (
        float(roc_auc_score(y, scores)),
        float(average_precision_score(y, scores)),
        float(elapsed),
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="DCEAD attribute-noise experiment with repeated random realizations."
    )
    parser.add_argument("--datasets-dir", default="../Datasets")
    parser.add_argument("--out-dir", default="DCEAD_noise_results")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    summary_rows = []

    for dataset_id, (name, token, epsilon) in enumerate(DATASETS, start=1):
        path = find_dataset_file(args.datasets_dir, token)
        X, y_raw = load_mat_dataset(path)
        y = make_binary_labels(y_raw)

        if np.unique(y).size != 2:
            raise ValueError(f"{name}: binary labels are required for evaluation")

        for noise_level in NOISE_LEVELS:
            repeats = 1 if np.isclose(noise_level, 0.0) else args.repeats
            auc_values = []
            ap_values = []
            time_values = []

            for repeat in range(repeats):
                seed = (
                    args.seed
                    + dataset_id * 100000
                    + int(round(float(noise_level) * 1000)) * 100
                    + repeat
                )
                rng = np.random.default_rng(seed)
                X_noisy = add_attribute_noise(X, float(noise_level), rng)
                auc_value, ap_value, elapsed = run_once(X_noisy, y, epsilon)

                auc_values.append(auc_value)
                ap_values.append(ap_value)
                time_values.append(elapsed)
                run_rows.append({
                    "dataset": name,
                    "epsilon": epsilon,
                    "noise_level": float(noise_level),
                    "repeat": repeat + 1,
                    "seed": seed,
                    "auc": auc_value,
                    "ap": ap_value,
                    "time_sec": elapsed,
                })

            summary_rows.append({
                "dataset": name,
                "epsilon": epsilon,
                "noise_level": float(noise_level),
                "auc_mean": float(np.mean(auc_values)),
                "auc_std": float(np.std(auc_values, ddof=1)) if repeats > 1 else 0.0,
                "ap_mean": float(np.mean(ap_values)),
                "ap_std": float(np.std(ap_values, ddof=1)) if repeats > 1 else 0.0,
                "time_mean": float(np.mean(time_values)),
                "time_std": float(np.std(time_values, ddof=1)) if repeats > 1 else 0.0,
                "repeats": repeats,
            })

            print(
                f"{name:12s} noise={noise_level:.2f} "
                f"AUC={np.mean(auc_values):.6f}±{np.std(auc_values, ddof=1) if repeats > 1 else 0.0:.6f} "
                f"AP={np.mean(ap_values):.6f}±{np.std(ap_values, ddof=1) if repeats > 1 else 0.0:.6f}"
            )

    write_csv(
        output_dir / "DCEAD_noise_runs.csv",
        run_rows,
        ["dataset", "epsilon", "noise_level", "repeat", "seed", "auc", "ap", "time_sec"],
    )
    write_csv(
        output_dir / "DCEAD_noise_summary.csv",
        summary_rows,
        [
            "dataset", "epsilon", "noise_level", "auc_mean", "auc_std",
            "ap_mean", "ap_std", "time_mean", "time_std", "repeats",
        ],
    )


if __name__ == "__main__":
    main()
