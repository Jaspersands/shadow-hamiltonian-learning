"""
shadow_learning
===============
Shadow-Tomography-Guided Hamiltonian Learning with Differentiable Post-Processing
in Cirq, PennyLane & JAX.
"""

from .cirq_shadows import (
    ClassicalShadowsProtocol,
    ShadowSnapshot,
    measure_random_pauli_shadows,
)
from .reconstruction import (
    ShadowObservableEstimator,
    estimate_pauli_string_expectation,
    estimate_many_observables,
)
from .differentiable_inversion import (
    DifferentiableHamiltonianLearner,
    HamiltonianLearningResult,
    learn_hamiltonian_from_shadows,
)
from .spam_mitigation import (
    SPAMNoiseMitigator,
    ReadoutErrorModel,
)

__version__ = "0.1.0"
__all__ = [
    "ClassicalShadowsProtocol",
    "ShadowSnapshot",
    "measure_random_pauli_shadows",
    "ShadowObservableEstimator",
    "estimate_pauli_string_expectation",
    "estimate_many_observables",
    "DifferentiableHamiltonianLearner",
    "HamiltonianLearningResult",
    "learn_hamiltonian_from_shadows",
    "SPAMNoiseMitigator",
    "ReadoutErrorModel",
]
