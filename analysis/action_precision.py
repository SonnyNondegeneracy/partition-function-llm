"""Diagnostic minimizer of the SAME exponential action, in tree coordinates.

Raw simulation checkpoints are never overwritten. A maximum-weight spanning
tree makes each fundamental cut have at least one resolvable edge. Dividing
each Newton equation by that tree-edge weight avoids accepting tiny absolute
gradients as convergence. Both node and fundamental-cut relative balances,
and the final potential step, are reported.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.special import logsumexp


SOLVER_VERSION = "convex_cut_newton_v1"


def balance_action(model, t, initial=None, *, tol=2e-10, maxiter=600):
    src, dst, n = (model[k] for k in ("src", "dst", "n"))
    src, dst, t = np.asarray(src), np.asarray(dst), np.asarray(t, float)
    if n < 2 or t.ndim != 1 or src.shape != t.shape or dst.shape != t.shape:
        raise ValueError("action requires matching arc arrays and at least two nodes")
    if not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("action requires finite strictly positive candidate weights")
    support = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    if connected_components(support, directed=True, connection="strong")[0] != 1:
        raise ValueError("a unique finite action minimizer requires strongly connected support")
    if initial is not None and (np.shape(initial) != (n,) or not np.isfinite(initial).all()):
        raise ValueError("initial potential must be a finite node vector")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    logt = np.log(t)
    z = np.zeros(n) if initial is None else np.asarray(initial, float).copy() / 2
    z -= z.mean()
    low, high = np.minimum(src, dst), np.maximum(src, dst)
    keys, inverse = np.unique(low * n + high, return_inverse=True)
    es, ed = keys // n, keys % n
    converged = False
    step_size = np.inf
    for iteration in range(maxiter):
        logw = logt + z[dst] - z[src]
        edge_logw = np.full(len(keys), -np.inf)
        np.logaddexp.at(edge_logw, inverse, logw)
        costs = edge_logw.max() - edge_logw + 1
        tree = minimum_spanning_tree(coo_matrix((costs, (es, ed)), shape=(n, n))).tocoo()
        if tree.nnz != n - 1:
            raise ValueError("disconnected undirected support")
        adjacency = [[] for _ in range(n)]
        for a, b in zip(tree.row, tree.col):
            adjacency[a].append(b); adjacency[b].append(a)
        parent = np.full(n, -1, int)
        order = [0]; parent[0] = 0
        for i in order:
            for j in adjacency[i]:
                if parent[j] < 0:
                    parent[j] = i; order.append(j)
        nodes = np.asarray(order[1:])
        # z=A d, where d is the child-minus-parent potential on the tree.
        a = np.zeros((n, n - 1))
        for column, node in enumerate(nodes):
            a[node] = a[parent[node]]
            a[node, column] = 1
        incidence = coo_matrix((np.tile([-1., 1.], len(src)),
                                (np.repeat(np.arange(len(src)), 2),
                                 np.column_stack((src, dst)).ravel())),
                               shape=(len(src), n)).tocsr()
        p = incidence @ csr_matrix(a)
        tree_keys = np.minimum(nodes, parent[nodes]) * n + np.maximum(nodes, parent[nodes])
        scale = edge_logw[np.searchsorted(keys, tree_keys)]
        # For an arc crossing a fundamental cut, logw <= its tree scale.
        signed = p.T.tocsr()
        cut_ids = np.repeat(np.arange(n - 1), np.diff(signed.indptr))
        signed.data *= np.exp(np.minimum(logw[signed.indices] - scale[cut_ids], 0))
        gradient = np.asarray(signed.sum(axis=1)).ravel()
        cut_weight = np.asarray(abs(signed).sum(axis=1)).ravel()
        cut_balance = float(np.max(np.abs(gradient) / cut_weight))
        hessian = (signed @ p).toarray()
        delta = solve(hessian, -gradient, check_finite=False)
        dz = a @ delta
        dz -= dz.mean()
        step_size = float(np.max(np.abs(dz)))
        if cut_balance < tol and step_size < 1e-7:
            converged = True
            break
        max_arc_step = np.max(np.abs(dz[dst] - dz[src]))
        damping = min(1., 2. / max(max_arc_step, 1e-30))
        z += damping * dz
        z -= z.mean()
    logw = logt + z[dst] - z[src]
    log_out = np.full(n, -np.inf); log_in = np.full(n, -np.inf)
    np.logaddexp.at(log_out, src, logw); np.logaddexp.at(log_in, dst, logw)
    node_balance = float(np.max(np.abs(np.tanh((log_out - log_in) / 2))))
    return 2 * z, float(np.exp(logsumexp(logw) - np.log(n))), {
        "converged": bool(converged), "iterations": iteration + 1,
        "max_relative_cut_balance": cut_balance,
        "max_relative_node_balance": node_balance,
        "max_newton_step": step_size,
    }
