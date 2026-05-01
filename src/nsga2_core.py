from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray
ObjectiveList = list[Callable[[Array], Array]]


@dataclass(frozen=True)
class NSGAConfig:
    pop_size: int = 100
    n_gen: int = 250
    seed: int = 42
    eta_c: float = 20.0   # SBX distribution index
    eta_m: float = 20.0   # polynomial mutation distribution index
    p_cross: float = 0.9  # crossover probability


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

def dominates(p: Array, q: Array) -> bool:
    """True if p Pareto-dominates q (all objectives minimised)."""
    return bool(np.all(p <= q) and np.any(p < q))


# ---------------------------------------------------------------------------
# Fast non-dominated sort  –  O(MN²)
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(F: Array) -> list[list[int]]:
    """Sort population into non-domination fronts.

    Parameters
    ----------
    F : (N, M) objective values.

    Returns
    -------
    List of fronts.  front[0] is the Pareto front (rank 1).
    Each front is a list of row indices into F.
    """
    N = len(F)
    dominated_set: list[list[int]] = [[] for _ in range(N)]  # S_p
    domination_count = np.zeros(N, dtype=int)                 # n_p

    for p in range(N):
        for q in range(p + 1, N):
            if dominates(F[p], F[q]):
                dominated_set[p].append(q)
                domination_count[q] += 1
            elif dominates(F[q], F[p]):
                dominated_set[q].append(p)
                domination_count[p] += 1

    fronts: list[list[int]] = []
    current = [p for p in range(N) if domination_count[p] == 0]

    while current:
        fronts.append(current)
        nxt: list[int] = []
        for p in current:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        current = nxt

    return fronts


# ---------------------------------------------------------------------------
# Crowding distance  –  O(M N log N) per front
# ---------------------------------------------------------------------------

def crowding_distance(F: Array) -> Array:
    """Crowding distance assignment for one front.

    Parameters
    ----------
    F : (l, M) objective values for l individuals in the front.

    Returns
    -------
    (l,) crowding distances.  Boundary solutions get +inf.
    """
    l, M = F.shape
    if l <= 2:
        return np.full(l, np.inf)

    distances = np.zeros(l)
    for m in range(M):
        idx = np.argsort(F[:, m])
        distances[idx[0]] = np.inf
        distances[idx[-1]] = np.inf
        f_range = F[idx[-1], m] - F[idx[0], m]
        if f_range < 1e-30:
            continue
        for i in range(1, l - 1):
            distances[idx[i]] += (F[idx[i + 1], m] - F[idx[i - 1], m]) / f_range

    return distances


# ---------------------------------------------------------------------------
# Crowded comparison operator  ≺_n
# ---------------------------------------------------------------------------

def crowded_better(rank_i: int, dist_i: float, rank_j: int, dist_j: float) -> bool:
    """Return True if solution i is preferred over j under crowded comparison."""
    if rank_i < rank_j:
        return True
    return rank_i == rank_j and dist_i > dist_j


# ---------------------------------------------------------------------------
# Genetic operators: SBX crossover + polynomial mutation
# ---------------------------------------------------------------------------

def _sbx_variable(
    x1: float, x2: float, lb: float, ub: float, eta_c: float, rng: np.random.Generator
) -> tuple[float, float]:
    """SBX crossover for one variable (x1 ≤ x2 guaranteed by caller)."""
    if x2 - x1 < 1e-14:
        return x1, x2

    beta1 = 1.0 + 2.0 * (x1 - lb) / (x2 - x1)
    beta2 = 1.0 + 2.0 * (ub - x2) / (x2 - x1)

    u = rng.random()

    alpha = 2.0 - beta1 ** (-(eta_c + 1.0))
    betaq1 = (u * alpha) ** (1.0 / (eta_c + 1.0)) if u <= 1.0 / alpha \
        else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

    alpha = 2.0 - beta2 ** (-(eta_c + 1.0))
    betaq2 = (u * alpha) ** (1.0 / (eta_c + 1.0)) if u <= 1.0 / alpha \
        else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

    c1 = 0.5 * ((x1 + x2) - betaq1 * (x2 - x1))
    c2 = 0.5 * ((x1 + x2) + betaq2 * (x2 - x1))
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def sbx_crossover(
    parent1: Array, parent2: Array, bounds: Array, eta_c: float, rng: np.random.Generator
) -> tuple[Array, Array]:
    """Simulated Binary Crossover."""
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if rng.random() <= 0.5:
            x1, x2 = (parent1[i], parent2[i]) if parent1[i] <= parent2[i] \
                else (parent2[i], parent1[i])
            c1, c2 = _sbx_variable(x1, x2, bounds[i, 0], bounds[i, 1], eta_c, rng)
            if rng.random() <= 0.5:
                child1[i], child2[i] = c1, c2
            else:
                child1[i], child2[i] = c2, c1
    return child1, child2


def polynomial_mutation(
    x: Array, bounds: Array, eta_m: float, p_m: float, rng: np.random.Generator
) -> Array:
    """Polynomial mutation."""
    y = x.copy()
    for i in range(len(x)):
        if rng.random() < p_m:
            lb, ub = bounds[i, 0], bounds[i, 1]
            delta1 = (y[i] - lb) / (ub - lb)
            delta2 = (ub - y[i]) / (ub - lb)
            u = rng.random()
            if u <= 0.5:
                deltaq = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (eta_m + 1.0)) ** (1.0 / (eta_m + 1.0)) - 1.0
            else:
                deltaq = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2) ** (eta_m + 1.0)) ** (1.0 / (eta_m + 1.0))
            y[i] = np.clip(y[i] + deltaq * (ub - lb), lb, ub)
    return y


# ---------------------------------------------------------------------------
# Offspring generation: tournament → crossover → mutation
# ---------------------------------------------------------------------------

def _tournament_select(
    pop: Array, ranks: Array, distances: Array, rng: np.random.Generator
) -> Array:
    """Binary tournament selection based on crowded comparison."""
    N = len(pop)
    i, j = rng.integers(0, N, size=2)
    if crowded_better(ranks[i], distances[i], ranks[j], distances[j]):
        return pop[i]
    return pop[j]


def make_offspring(
    pop: Array, ranks: Array, distances: Array,
    bounds: Array, config: NSGAConfig, rng: np.random.Generator
) -> Array:
    """Create offspring population Q of size N."""
    N = len(pop)
    n_var = bounds.shape[0]
    p_m = 1.0 / n_var
    children: list[Array] = []

    while len(children) < N:
        p1 = _tournament_select(pop, ranks, distances, rng)
        p2 = _tournament_select(pop, ranks, distances, rng)
        if rng.random() < config.p_cross:
            c1, c2 = sbx_crossover(p1, p2, bounds, config.eta_c, rng)
        else:
            c1, c2 = p1.copy(), p2.copy()
        c1 = polynomial_mutation(c1, bounds, config.eta_m, p_m, rng)
        c2 = polynomial_mutation(c2, bounds, config.eta_m, p_m, rng)
        children.append(c1)
        if len(children) < N:
            children.append(c2)

    return np.array(children[:N])


# ---------------------------------------------------------------------------
# Assign ranks and crowding distances to a population
# ---------------------------------------------------------------------------

def assign_rank_and_distance(F: Array) -> tuple[Array, Array, list[list[int]]]:
    """Return (ranks, distances, fronts) for population with objectives F."""
    N = len(F)
    fronts = fast_non_dominated_sort(F)
    ranks = np.zeros(N, dtype=int)
    distances = np.zeros(N)
    for rank, front in enumerate(fronts):
        fa = np.array(front)
        ranks[fa] = rank
        distances[fa] = crowding_distance(F[fa])
    return ranks, distances, fronts


# ---------------------------------------------------------------------------
# Main NSGA-II loop
# ---------------------------------------------------------------------------

def nsga2(
    objectives: ObjectiveList,
    bounds: Array,
    config: NSGAConfig,
) -> dict:
    """Run NSGA-II and return full history.

    Parameters
    ----------
    objectives : list of M callables, each (N, n) → (N,) (all minimised).
    bounds     : (n, 2) array of variable bounds.
    config     : NSGAConfig.

    Returns
    -------
    dict with keys:
        'populations'  – list of length n_gen+1, each (pop_size, n_var)
        'objectives'   – list of length n_gen+1, each (pop_size, M)
        'fronts'       – list of length n_gen+1, front decomposition
    """
    rng = np.random.default_rng(config.seed)
    N = config.pop_size
    bounds = np.asarray(bounds, dtype=float)
    n_var = bounds.shape[0]

    def evaluate(X: Array) -> Array:
        return np.column_stack([f(X) for f in objectives])

    # --- initialise parent population P_0 ---
    pop = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N, n_var))
    F = evaluate(pop)
    ranks, distances, fronts = assign_rank_and_distance(F)

    history: dict = {
        "populations": [pop.copy()],
        "objectives": [F.copy()],
        "fronts": [fronts],
    }

    for _ in range(config.n_gen):
        # --- create offspring Q ---
        Q = make_offspring(pop, ranks, distances, bounds, config, rng)
        F_Q = evaluate(Q)

        # --- combined population R = P ∪ Q of size 2N ---
        R = np.vstack([pop, Q])
        F_R = np.vstack([F, F_Q])

        all_ranks, all_distances, all_fronts = assign_rank_and_distance(F_R)

        # --- fill P_{t+1} front by front ---
        new_idx: list[int] = []
        for front in all_fronts:
            if len(new_idx) + len(front) <= N:
                new_idx.extend(front)
            else:
                remaining = N - len(new_idx)
                # sort by crowding distance descending
                front_sorted = sorted(front, key=lambda i: -all_distances[i])
                new_idx.extend(front_sorted[:remaining])
                break

        new_idx_arr = np.array(new_idx)
        pop = R[new_idx_arr]
        F = F_R[new_idx_arr]
        ranks, distances, fronts = assign_rank_and_distance(F)

        history["populations"].append(pop.copy())
        history["objectives"].append(F.copy())
        history["fronts"].append(fronts)

    return history


# ---------------------------------------------------------------------------
# Test problems (all objectives minimised)
# ---------------------------------------------------------------------------

def sch() -> tuple[ObjectiveList, Array]:
    """Schaffer problem: n=1, x ∈ [-1000, 1000]."""
    bounds = np.array([[-1e3, 1e3]])
    objectives = [
        lambda X: X[:, 0] ** 2,
        lambda X: (X[:, 0] - 2.0) ** 2,
    ]
    return objectives, bounds


def _zdt_g(X: Array) -> Array:
    """g function for ZDT1-3."""
    n = X.shape[1]
    return 1.0 + 9.0 * X[:, 1:].sum(axis=1) / (n - 1)


def zdt1(n: int = 30) -> tuple[ObjectiveList, Array]:
    """ZDT1: convex Pareto front."""
    bounds = np.zeros((n, 2)); bounds[:, 1] = 1.0
    objectives = [
        lambda X: X[:, 0],
        lambda X: _zdt_g(X) * (1.0 - np.sqrt(X[:, 0] / _zdt_g(X))),
    ]
    return objectives, bounds


def zdt2(n: int = 30) -> tuple[ObjectiveList, Array]:
    """ZDT2: nonconvex Pareto front."""
    bounds = np.zeros((n, 2)); bounds[:, 1] = 1.0
    objectives = [
        lambda X: X[:, 0],
        lambda X: _zdt_g(X) * (1.0 - (X[:, 0] / _zdt_g(X)) ** 2),
    ]
    return objectives, bounds


def zdt3(n: int = 30) -> tuple[ObjectiveList, Array]:
    """ZDT3: disconnected Pareto front."""
    bounds = np.zeros((n, 2)); bounds[:, 1] = 1.0
    objectives = [
        lambda X: X[:, 0],
        lambda X: _zdt_g(X) * (
            1.0 - np.sqrt(X[:, 0] / _zdt_g(X))
            - (X[:, 0] / _zdt_g(X)) * np.sin(10.0 * np.pi * X[:, 0])
        ),
    ]
    return objectives, bounds


def zdt4(n: int = 10) -> tuple[ObjectiveList, Array]:
    """ZDT4: many local Pareto-optimal fronts."""
    bounds = np.array([[0.0, 1.0]] + [[-5.0, 5.0]] * (n - 1))
    def g(X: Array) -> Array:
        return 1.0 + 10.0 * (n - 1) + (X[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * X[:, 1:])).sum(axis=1)
    objectives = [
        lambda X: X[:, 0],
        lambda X: g(X) * (1.0 - np.sqrt(X[:, 0] / g(X))),
    ]
    return objectives, bounds


# ---------------------------------------------------------------------------
# True Pareto fronts for reference
# ---------------------------------------------------------------------------

def true_front_sch(n: int = 200) -> Array:
    x = np.linspace(0.0, 2.0, n)
    return np.column_stack([x ** 2, (x - 2.0) ** 2])


def true_front_zdt1(n: int = 200) -> Array:
    f1 = np.linspace(0.0, 1.0, n)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])


def true_front_zdt2(n: int = 200) -> Array:
    f1 = np.linspace(0.0, 1.0, n)
    return np.column_stack([f1, 1.0 - f1 ** 2])


def true_front_zdt3(n: int = 500) -> Array:
    f1 = np.linspace(0.0, 1.0, n)
    f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
    # keep only non-dominated points
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if j != i and f1[j] <= f1[i] and f2[j] <= f2[i] and (f1[j] < f1[i] or f2[j] < f2[i]):
                mask[i] = False
                break
    return np.column_stack([f1[mask], f2[mask]])


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def convergence_metric(obtained_front: Array, true_front: Array) -> float:
    """Metric Υ: mean min-distance from true front samples to obtained front."""
    H = true_front
    dists = []
    for h in H:
        d = np.min(np.linalg.norm(obtained_front - h, axis=1))
        dists.append(d)
    return float(np.mean(dists))


def diversity_metric(obtained_front: Array) -> float:
    """Metric Δ: spread of obtained front (Eq. 1 from Deb et al. 2002)."""
    F = obtained_front[np.argsort(obtained_front[:, 0])]
    N = len(F)
    if N < 2:
        return np.inf
    dists = np.linalg.norm(np.diff(F, axis=0), axis=1)
    d_bar = dists.mean()
    d_f = dists[0]
    d_l = dists[-1]
    numerator = d_f + d_l + np.sum(np.abs(dists[1:-1] - d_bar))
    denominator = d_f + d_l + (N - 1) * d_bar
    if denominator < 1e-30:
        return 0.0
    return float(numerator / denominator)


def get_first_front(history: dict, gen: int) -> Array:
    """Return objective values of rank-0 individuals at generation gen."""
    F = history["objectives"][gen]
    front0 = history["fronts"][gen][0]
    return F[np.array(front0)]
