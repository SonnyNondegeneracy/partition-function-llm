"""Task definitions and the edge-flow decomposition used by Figure 1."""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def task_spec(task):
    d = json.load(open(f"{REPO}/evaluation/main/tasks/{task}_direct_formal_v1.json"))
    u = np.array([s["risk_urgency_reward"] if task == "T1" else s["reward"] for s in d["states"]], float)
    ids = [s["id"] for s in d["states"]]
    desc = [s["description"] for s in d["states"]]
    return u, ids, desc


def hodge(y, E, n):
    """Least-squares split of edge flow y on edges E into gradient (phi_i - phi_j) and residual."""
    A = np.zeros((len(E), n))
    for k, (i, j) in enumerate(E):
        A[k, i], A[k, j] = 1, -1
    phi, *_ = np.linalg.lstsq(A, y, rcond=None)
    phi -= phi.mean()
    g = A @ phi
    return phi, g, y - g
