"""
Randomised Pauli classical shadows (Huang, Kueng & Preskill 2020).

Each snapshot measures every qubit in a uniformly random Pauli basis
(X, Y or Z) and records the outcome bits. Two acquisition paths:

* ``fast=True`` (default): the state-preparation circuit is simulated once to
  a state vector (or density matrix if a noise model is given); each snapshot
  is then drawn by rotating that state into the chosen product basis and
  sampling the computational-basis distribution — O(2ⁿ) per snapshot instead
  of a full circuit simulation.
* ``fast=False``: the original one-circuit-per-snapshot path through Cirq.

Both produce identically distributed snapshots (see ``tests/test_sampling``).
Readout errors can be injected through a :class:`ReadoutErrorModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cirq

    HAS_CIRQ = True
except ImportError:  # pragma: no cover
    HAS_CIRQ = False
    cirq = None

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
_SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)
_ROT = {"X": _H, "Y": _H @ _SDG, "Z": _I2}


@dataclass(frozen=True)
class ShadowSnapshot:
    """One randomised Pauli measurement: per-qubit bases and outcome bits."""

    bases: Tuple[str, ...]
    bits: Tuple[int, ...]


def basis_rotation_matrix(basis: str) -> np.ndarray:
    """Single-qubit unitary mapping the ±1 eigenstates of ``basis`` onto |0⟩, |1⟩."""
    return _ROT[basis]


def gibbs_state(hamiltonian: np.ndarray, beta: float) -> np.ndarray:
    """ρ = e^{−βH} / Tr e^{−βH}, computed stably through the eigendecomposition."""
    evals, evecs = np.linalg.eigh(hamiltonian)
    w = np.exp(-beta * (evals - evals.min()))
    w /= w.sum()
    return (evecs * w) @ evecs.conj().T


def sample_shadows_from_density_matrix(
    rho: np.ndarray,
    n_snapshots: int,
    rng: np.random.Generator,
    readout_error=None,
    bases: Optional[Sequence[Tuple[str, ...]]] = None,
) -> List[ShadowSnapshot]:
    """
    Draw ``n_snapshots`` Pauli shadows from a density matrix ``rho`` (2ⁿ × 2ⁿ).
    ``bases`` may fix the measurement bases per snapshot (derandomised shadows);
    otherwise each qubit's basis is uniform over X, Y, Z.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    n = int(round(np.log2(rho.shape[0])))
    choices = np.array(["X", "Y", "Z"])
    snapshots: List[ShadowSnapshot] = []
    cache = {}
    for s in range(int(n_snapshots)):
        b = tuple(str(x) for x in bases[s]) if bases is not None else tuple(str(x) for x in choices[rng.integers(0, 3, size=n)])
        if b not in cache:
            U = reduce(np.kron, [_ROT[x] for x in b])
            probs = np.real(np.einsum("ij,jk,ki->i", U, rho, U.conj().T))
            probs = np.clip(probs, 0.0, None)
            cache[b] = probs / probs.sum()
        idx = int(rng.choice(rho.shape[0], p=cache[b]))
        bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]
        if readout_error is not None:
            bits = [readout_error.apply_readout_noise(bit, rng) for bit in bits]
        snapshots.append(ShadowSnapshot(bases=b, bits=tuple(int(x) for x in bits)))
    return snapshots


def sample_shadows_from_state_vector(psi: np.ndarray, n_snapshots: int, rng: np.random.Generator, readout_error=None,
                                     bases: Optional[Sequence[Tuple[str, ...]]] = None) -> List[ShadowSnapshot]:
    psi = np.asarray(psi, dtype=np.complex128)
    return sample_shadows_from_density_matrix(np.outer(psi, psi.conj()), n_snapshots, rng, readout_error, bases)


class ClassicalShadowsProtocol:
    """Acquires randomised Pauli shadows from a Cirq state-preparation circuit."""

    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        if not HAS_CIRQ:
            raise RuntimeError("Cirq required.")
        self.n_qubits = int(n_qubits)
        self.qubits = cirq.LineQubit.range(self.n_qubits)
        self.rng = np.random.default_rng(seed)

    def measure_state(
        self,
        state_preparation_circuit: "cirq.Circuit",
        n_snapshots: int = 100,
        noise_model: Optional["cirq.NoiseModel"] = None,
        readout_error=None,
        fast: bool = True,
        bases: Optional[Sequence[Tuple[str, ...]]] = None,
    ) -> List[ShadowSnapshot]:
        if fast:
            circuit = state_preparation_circuit
            if noise_model is not None:
                circuit = cirq.Circuit(noise_model.noisy_moments(circuit, self.qubits))
                rho = cirq.DensityMatrixSimulator().simulate(circuit, qubit_order=self.qubits).final_density_matrix
            else:
                psi = cirq.Simulator().simulate(circuit, qubit_order=self.qubits).final_state_vector
                rho = np.outer(psi, psi.conj())
            return sample_shadows_from_density_matrix(rho, n_snapshots, self.rng, readout_error, bases)

        snapshots = []
        sim_seed = int(self.rng.integers(1 << 30))
        sim = cirq.DensityMatrixSimulator(seed=sim_seed) if noise_model else cirq.Simulator(seed=sim_seed)
        for s in range(int(n_snapshots)):
            chosen = list(bases[s]) if bases is not None else [self.rng.choice(["X", "Y", "Z"]) for _ in range(self.n_qubits)]
            c = state_preparation_circuit.copy()
            for i, b in enumerate(chosen):
                q = self.qubits[i]
                if b == "X":
                    c.append(cirq.H(q))
                elif b == "Y":
                    c.append(cirq.Z(q) ** -0.5)
                    c.append(cirq.H(q))
            c.append(cirq.measure(*self.qubits, key="m"))
            if noise_model:
                c = cirq.Circuit(noise_model.noisy_moments(c, self.qubits))
            raw = sim.run(c, repetitions=1).measurements["m"][0]
            bits = [int(b) for b in raw]
            if readout_error is not None:
                bits = [readout_error.apply_readout_noise(bit, self.rng) for bit in bits]
            snapshots.append(ShadowSnapshot(bases=tuple(chosen), bits=tuple(bits)))
        return snapshots


def measure_random_pauli_shadows(
    prep_circuit: "cirq.Circuit",
    n_qubits: int,
    n_snapshots: int = 200,
    seed: Optional[int] = None,
    readout_error=None,
    fast: bool = True,
) -> List[ShadowSnapshot]:
    """Convenience wrapper around :class:`ClassicalShadowsProtocol`."""
    return ClassicalShadowsProtocol(n_qubits=n_qubits, seed=seed).measure_state(
        prep_circuit, n_snapshots=n_snapshots, readout_error=readout_error, fast=fast
    )
