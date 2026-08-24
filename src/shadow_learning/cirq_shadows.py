"""
Classical Shadows Protocol in Cirq.

Implements randomized Pauli basis measurements (Huang, Kueng, Preskill 2020)
and snapshot collection on 1D/2D qubit registers.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Dict, Any

try:
    import cirq
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False
    cirq = None


@dataclass
class ShadowSnapshot:
    """Represents a single randomized Pauli measurement snapshot."""
    bases: Tuple[str, ...]   # 'X', 'Y', 'Z' for each qubit
    bits: Tuple[int, ...]    # 0 or 1 for each qubit outcome


class ClassicalShadowsProtocol:
    """
    Acquires randomized Pauli classical shadows from a target quantum state.
    """

    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        if not HAS_CIRQ:
            raise RuntimeError("Cirq required.")
        self.n_qubits = n_qubits
        self.qubits = cirq.LineQubit.range(n_qubits)
        self.rng = np.random.default_rng(seed)

    def measure_state(
        self,
        state_preparation_circuit: "cirq.Circuit",
        n_snapshots: int = 100,
        noise_model: Optional["cirq.NoiseModel"] = None,
    ) -> List[ShadowSnapshot]:
        """
        Executes randomized Pauli measurements for n_snapshots shots.
        """
        snapshots = []
        bases_choices = ["X", "Y", "Z"]

        sim = cirq.DensityMatrixSimulator() if noise_model else cirq.Simulator()

        for _ in range(n_snapshots):
            # Sample random basis for each qubit
            chosen_bases = [self.rng.choice(bases_choices) for _ in range(self.n_qubits)]

            # Build measurement circuit
            c = state_preparation_circuit.copy()

            # Apply basis rotation
            for i, b in enumerate(chosen_bases):
                q = self.qubits[i]
                if b == "X":
                    c.append(cirq.H(q))
                elif b == "Y":
                    # Rotate Y to Z: apply S^\dagger then H
                    c.append(cirq.Z(q) ** -0.5)
                    c.append(cirq.H(q))
                elif b == "Z":
                    pass # Already in Z basis

            # Computational basis measurement
            c.append(cirq.measure(*self.qubits, key="m"))

            if noise_model:
                c = cirq.Circuit(noise_model.noisy_moments(c, self.qubits))

            res = sim.run(c, repetitions=1)
            raw_bits = res.measurements["m"][0]
            bits_tuple = tuple(int(b) for b in raw_bits)
            snapshots.append(ShadowSnapshot(bases=tuple(chosen_bases), bits=bits_tuple))

        return snapshots


def measure_random_pauli_shadows(
    prep_circuit: "cirq.Circuit",
    n_qubits: int,
    n_snapshots: int = 200,
    seed: Optional[int] = None,
) -> List[ShadowSnapshot]:
    """Convenience helper to gather shadow snapshots."""
    proto = ClassicalShadowsProtocol(n_qubits=n_qubits, seed=seed)
    return proto.measure_state(prep_circuit, n_snapshots=n_snapshots)
