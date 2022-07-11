import numpy as np


def solve_tsp_two_opt(positions: np.ndarray):
    n = positions.shape[0]
    distance_matrix = np.linalg.norm(positions[None] - positions[:, None], axis=-1)
    index_range = np.arange(n)
    current_solution = index_range
    current_cost = np.diag(distance_matrix, 1).sum() + distance_matrix[0, -1]
    index_range_bc = np.broadcast_to(index_range[None], (n, n))
    index_pairs_raw = np.stack([index_range_bc, index_range_bc.T], axis=-1).reshape((-1, 2))
    index_pairs = index_pairs_raw[np.where(index_pairs_raw[:, 0] < index_pairs_raw[:, 1])]
    ipc = index_pairs.shape[0]
    solution_changed = True
    while solution_changed:
        solution_changed = False
        current_solution_bc = np.broadcast_to(current_solution[None], (index_pairs.shape[0], n)).copy()
        tmp = current_solution_bc[np.arange(ipc), index_pairs[:, 0]]
        current_solution_bc[np.arange(ipc), index_pairs[:, 0]] = current_solution_bc[np.arange(ipc), index_pairs[:, 1]]
        current_solution_bc[np.arange(ipc), index_pairs[:, 1]] = tmp
        current_solution_aug = np.concatenate([current_solution_bc, current_solution_bc[:, 0, None]], axis=-1)
        transitions = np.stack([current_solution_aug[:, :-1], current_solution_aug[:, 1:]], axis=-1)
        costs = distance_matrix[transitions[:, :, 0], transitions[:, :, 1]].sum(-1)
        best_idx = np.argmin(costs)
        if costs[best_idx] < current_cost:
            solution_changed = True
            current_cost = costs[best_idx]
            current_solution = current_solution_bc[best_idx]
    return current_solution
