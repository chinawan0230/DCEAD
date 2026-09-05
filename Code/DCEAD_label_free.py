import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score

from DCEAD import DCEAD


EPSILON_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def load_mat_dataset(path):
    mat = sio.loadmat(path)

    if "trandata" in mat:
        arr = np.asarray(mat["trandata"])
        return arr[:, :-1].astype(float), np.asarray(arr[:, -1]).ravel()

    for x_key, y_key in [
        ("X", "y"),
        ("data", "label"),
        ("data", "y"),
        ("features", "labels"),
        ("Data", "Label"),
    ]:
        if x_key in mat and y_key in mat:
            return np.asarray(mat[x_key]).astype(float), np.asarray(mat[y_key]).ravel()

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


def clean_scores(scores):
    scores = np.asarray(scores, dtype=float).ravel()
    if np.all(np.isfinite(scores)):
        return scores

    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.zeros_like(scores)

    return np.nan_to_num(
        scores,
        nan=float(np.min(finite)),
        neginf=float(np.min(finite)),
        posinf=float(np.max(finite)),
    )


def kendall_centrality(score_vectors):
    parameters = sorted(score_vectors)
    count = len(parameters)
    tau_matrix = np.eye(count, dtype=float)

    for i in range(count):
        for j in range(i + 1, count):
            result = kendalltau(
                score_vectors[parameters[i]],
                score_vectors[parameters[j]],
                variant="b",
                nan_policy="omit",
            )
            tau = result.correlation if hasattr(result, "correlation") else result[0]
            if tau is None or not np.isfinite(tau):
                tau = 0.0
            tau_matrix[i, j] = float(np.clip(tau, -1.0, 1.0))
            tau_matrix[j, i] = tau_matrix[i, j]

    centrality = np.zeros(count, dtype=float)
    for i in range(count):
        if count == 1:
            centrality[i] = 1.0
        else:
            centrality[i] = np.mean(np.delete(tau_matrix[i], i))

    best = np.max(centrality)
    candidates = np.where(np.isclose(centrality, best, rtol=1e-12, atol=1e-12))[0]

    if len(candidates) == 1:
        best_index = int(candidates[0])
    else:
        median_parameter = float(np.median(parameters))
        best_index = int(
            min(
                candidates,
                key=lambda idx: (
                    abs(parameters[idx] - median_parameter),
                    parameters[idx],
                ),
            )
        )

    return parameters, tau_matrix, centrality, float(parameters[best_index])


def select_epsilon_label_free(X):
    score_vectors = {}
    for epsilon in EPSILON_GRID:
        score_vectors[float(epsilon)] = clean_scores(
            DCEAD(X, epsilon=float(epsilon), verbose=False)
        )

    parameters, tau_matrix, centrality, selected = kendall_centrality(score_vectors)
    return selected, score_vectors[selected], parameters, tau_matrix, centrality


def save_selection(output_dir, parameters, tau_matrix, centrality, selected):
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = ["epsilon,mean_kendall_tau,selected"]
    for epsilon, value in zip(parameters, centrality):
        rows.append(f"{epsilon:.2f},{value:.12f},{int(np.isclose(epsilon, selected))}")
    (output_dir / "DCEAD_label_free_selection.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    header = ",".join(f"{v:.2f}" for v in parameters)
    matrix_lines = ["," + header]
    for epsilon, row in zip(parameters, tau_matrix):
        matrix_lines.append(
            f"{epsilon:.2f}," + ",".join(f"{v:.12f}" for v in row)
        )
    (output_dir / "DCEAD_kendall_tau_matrix.csv").write_text(
        "\n".join(matrix_lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Label-free DCEAD parameter selection using Kendall tau-b centrality."
    )
    parser.add_argument("data", help="Path to one .mat dataset.")
    parser.add_argument("--out-dir", default="DCEAD_label_free_results")
    args = parser.parse_args()

    X, y_raw = load_mat_dataset(args.data)
    selected, scores, parameters, tau_matrix, centrality = select_epsilon_label_free(X)

    output_dir = Path(args.out_dir) / Path(args.data).stem
    save_selection(output_dir, parameters, tau_matrix, centrality, selected)

    y = make_binary_labels(y_raw)
    auc_value = roc_auc_score(y, scores)
    ap_value = average_precision_score(y, scores)

    summary = (
        f"Selected epsilon: {selected:.2f}\n"
        f"AUC: {auc_value:.6f}\n"
        f"AP: {ap_value:.6f}\n"
    )
    (output_dir / "DCEAD_label_free_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
