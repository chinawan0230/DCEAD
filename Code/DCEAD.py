import numpy as np


def normalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def should_normalize(col_data):
    try:
        col_float = col_data.astype(float)
    except (TypeError, ValueError):
        return False, "non_numeric"

    data_min = np.min(col_float)
    data_max = np.max(col_float)
    data_range = data_max - data_min

    if data_range < 1e-10:
        return False, "constant"
    if data_range > 1.5:
        return True, "large_range"
    if data_min < -0.1 or data_max > 1.1:
        return True, "out_of_range"
    return False, "already_normalized"


def single_attribute_difference(a, x, flag):
    if flag == 0:
        return 0.0 if a == x else 1.0
    return abs(a - x)


def mixed_distance(op, oq, delta):
    m = len(op)
    numeric_sum = 0.0
    categorical_sum = 0.0

    for col in range(m):
        diff = single_attribute_difference(op[col], oq[col], delta[col])
        if delta[col] == 1:
            numeric_sum += diff ** 2
        else:
            categorical_sum += diff

    return np.sqrt(numeric_sum + categorical_sum) / np.sqrt(m)


def fuzzy_similarity(op, oq, delta):
    return max(0.0, 1.0 - mixed_distance(op, oq, delta))


def fuzzy_entropy_single(data, col, epsilon, delta):
    n = data.shape[0]
    entropy = 0.0
    data_single = data[:, col:col + 1]
    delta_single = delta[col:col + 1]

    for i in range(n):
        neighborhood_sum = 0.0
        for j in range(n):
            sim = fuzzy_similarity(data_single[i], data_single[j], delta_single)
            if sim >= epsilon:
                neighborhood_sum += sim
        if neighborhood_sum > 0:
            entropy -= np.log2(neighborhood_sum / n)

    return entropy / n


def compute_similarity_matrix(data, attr_subset, epsilon, delta):
    n = data.shape[0]
    sim_matrix = np.zeros((n, n), dtype=float)
    data_subset = data[:, attr_subset]
    delta_subset = delta[attr_subset]

    for i in range(n):
        for j in range(i, n):
            sim = fuzzy_similarity(data_subset[i], data_subset[j], delta_subset)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


def class_margin(i, sim_matrix, epsilon):
    row = sim_matrix[i]
    idx = np.arange(sim_matrix.shape[0]) != i
    similar = row[idx & (row >= epsilon)]
    dissimilar = row[idx & (row < epsilon)]

    gamma = np.mean(similar) if similar.size else 0.0
    gamma_bar = np.mean(1.0 - dissimilar) if dissimilar.size else 0.0
    return (gamma + gamma_bar) / 2.0


def fuzzy_local_entropy(i, sim_matrix, epsilon):
    neighborhood_sims = sim_matrix[i, sim_matrix[i] >= epsilon]
    m_i = neighborhood_sims.size

    if m_i <= 1:
        return 0.0

    total_sim = np.sum(neighborhood_sims)
    if total_sim == 0:
        return 0.0

    probs = neighborhood_sims / total_sim
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return entropy / np.log(m_i)


def dual_view_metric(i, sim_matrix, epsilon):
    fle = fuzzy_local_entropy(i, sim_matrix, epsilon)
    margin = class_margin(i, sim_matrix, epsilon)
    return np.sqrt(fle * (1.0 - margin))


def granule_anomaly_extent(i, sim_matrix, epsilon):
    n = sim_matrix.shape[0]
    row = sim_matrix[i]
    neighborhood_sum = np.sum(row[row >= epsilon])
    h_value = dual_view_metric(i, sim_matrix, epsilon)
    return 1.0 - (neighborhood_sum / n) * (1.0 - h_value)


def granule_weight(i, sim_matrix, epsilon):
    n = sim_matrix.shape[0]
    row = sim_matrix[i]
    neighborhood_sum = np.sum(row[row >= epsilon])
    return 1.0 - np.power(neighborhood_sum / n, 1.0 / 3.0)


def DCEAD(data, epsilon, verbose=False):
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix")

    n, m = data.shape
    if n == 0 or m == 0:
        raise ValueError("data must contain at least one sample and one attribute")

    delta = np.zeros(m, dtype=int)
    data_normalized = np.zeros_like(data, dtype=float)

    for col in range(m):
        col_data = data[:, col]
        try:
            col_float = col_data.astype(float)
        except (TypeError, ValueError):
            delta[col] = 0
            data_normalized[:, col] = col_data
            continue

        need_norm, reason = should_normalize(col_float)
        if reason == "constant":
            delta[col] = 0
            data_normalized[:, col] = col_float
        elif need_norm:
            delta[col] = 1
            data_normalized[:, col] = normalize(col_float)
        else:
            delta[col] = 1
            data_normalized[:, col] = col_float

    entropy_list = []
    for col in range(m):
        entropy_list.append((col, fuzzy_entropy_single(data_normalized, col, epsilon, delta)))

    entropy_list.sort(key=lambda item: item[1])
    attribute_order = [col for col, _ in entropy_list]

    single_attr_subsets = [[attr] for attr in attribute_order]
    forward_attr_subsets = [attribute_order[:t] for t in range(1, m + 1)]

    scores = np.zeros(n, dtype=float)

    for j in range(m):
        sim_single = compute_similarity_matrix(
            data_normalized, single_attr_subsets[j], epsilon, delta
        )
        sim_forward = compute_similarity_matrix(
            data_normalized, forward_attr_subsets[j], epsilon, delta
        )

        for i in range(n):
            gae_single = granule_anomaly_extent(i, sim_single, epsilon)
            weight_single = granule_weight(i, sim_single, epsilon)
            gae_forward = granule_anomaly_extent(i, sim_forward, epsilon)
            weight_forward = granule_weight(i, sim_forward, epsilon)

            scores[i] += (
                (gae_single + gae_forward)
                * (weight_single + weight_forward)
            )

        if verbose:
            print(f"granularity {j + 1}/{m}")

    return scores / (4.0 * m)
