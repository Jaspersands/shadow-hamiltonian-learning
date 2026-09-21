"""
Derandomised classical shadows (Huang, Kueng & Preskill, PRL 127, 030503, 2021).

Instead of sampling Pauli bases at random, the bases of each shot are chosen
greedily, qubit by qubit, to minimise the *expected* confidence bound

    CONF_ε(P; O) = Σ_{o ∈ O} exp(−ε²/2 · h_o(P)),

where h_o counts the shots whose bases are compatible with observable o
(``o_i = I`` or ``o_i = P_i`` on every qubit of its support). With ν = 1 − e^{−ε²/2},
the cost of assigning Pauli W to qubit k of shot m, given the earlier
assignments, is

    Σ_o exp(−ε²/2 · h_o) · [1 − ν · 1[o compatible so far] · 3^{−(w_o − assigned_o)}] · (1 − ν/3^{w_o})^{M−m−1},

whose minimiser is taken (ties broken X, Y, Z). This is Algorithm 1 of the
paper; the v0.2 implementation was a fixed per-qubit majority vote that
returned the same basis for every shot.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

PAULIS = ("X", "Y", "Z")


def hits_per_observable(bases: Sequence[Tuple[str, ...]], observables: Sequence[str]) -> Dict[str, int]:
    """Number of shots whose bases are compatible with each observable."""
    out = {}
    for obs in observables:
        out[obs] = sum(1 for b in bases if all(o == "I" or o == w for o, w in zip(obs, b)))
    return out


class DerandomizedShadowSelector:
    """
    Parameters
    ----------
    target_observables : Pauli strings (``"XZI"`` …) of equal length ``n_qubits``
    n_qubits : register size
    epsilon : accuracy parameter of the confidence bound (0.9 works well in practice)
    """

    def __init__(self, target_observables: Sequence[str], n_qubits: int, epsilon: float = 0.9):
        self.observables = [str(o).upper() for o in target_observables]
        self.n_qubits = int(n_qubits)
        for o in self.observables:
            if len(o) != self.n_qubits or any(c not in "IXYZ" for c in o):
                raise ValueError(f"observable {o!r} must be a length-{n_qubits} string over I, X, Y, Z")
        self.epsilon = float(epsilon)
        self.weights = np.array([sum(c != "I" for c in o) for o in self.observables], dtype=float)

    def select_measurement_bases(self, n_snapshots: int) -> List[Tuple[str, ...]]:
        M = int(n_snapshots)
        nu = 1.0 - np.exp(-self.epsilon**2 / 2.0)
        obs = self.observables
        hits = np.zeros(len(obs))
        bases: List[Tuple[str, ...]] = []
        for m in range(M):
            assigned: List[str] = []
            # (1 − ν/3^w)^(M−m−1) can be ~1e-50 for large M; work with log-weights and
            # rescale so that comparisons between candidates are numerically meaningful.
            log_future = (M - m - 1) * np.log1p(-nu / 3.0 ** self.weights)
            log_weight = -self.epsilon**2 / 2.0 * hits + log_future
            weight = np.exp(log_weight - log_weight.max())
            for k in range(self.n_qubits):
                best_w, best_cost = "Z", np.inf
                for W in PAULIS:
                    trial = assigned + [W]
                    cost = 0.0
                    for j, o in enumerate(obs):
                        compatible = all(o[q] == "I" or o[q] == trial[q] for q in range(k + 1))
                        remaining = sum(1 for q in range(k + 1, self.n_qubits) if o[q] != "I")
                        cost += weight[j] * (1.0 - (nu * 3.0 ** (-remaining) if compatible else 0.0))
                    if cost < best_cost * (1.0 - 1e-9):
                        best_cost, best_w = cost, W
                assigned.append(best_w)
            shot = tuple(assigned)
            bases.append(shot)
            for j, o in enumerate(obs):
                if all(o[q] == "I" or o[q] == shot[q] for q in range(self.n_qubits)):
                    hits[j] += 1
        return bases

    def coverage(self, bases: Sequence[Tuple[str, ...]]) -> Dict[str, int]:
        return hits_per_observable(bases, self.observables)
