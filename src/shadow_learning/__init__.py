"""
shadow_learning
===============
Classical shadows in NumPy (Cirq optional): randomised and derandomised Pauli
shadows, matchgate (fermionic) shadows, readout-error mitigation, Gibbs-state
Hamiltonian learning with exact gradients, streaming EKF tracking, shadow
process tomography, and learning a Rydberg Hamiltonian (Ω, Δ, C6) from
computational-basis snapshots of quench dynamics.
"""

from .cirq_shadows import (
    ClassicalShadowsProtocol,
    ShadowSnapshot,
    measure_random_pauli_shadows,
    sample_shadows_from_density_matrix,
    sample_shadows_from_state_vector,
    gibbs_state,
    basis_rotation_matrix,
)
from .reconstruction import (
    ShadowObservableEstimator,
    estimate_pauli_string_expectation,
    estimate_pauli_expectation_median_of_means,
    estimate_many_observables,
    snapshots_to_arrays,
)
from .differentiable_inversion import (
    DifferentiableHamiltonianLearner,
    HamiltonianLearningResult,
    learn_hamiltonian_from_shadows,
)
from .spam_mitigation import SPAMNoiseMitigator, ReadoutErrorModel
from .derandomized import DerandomizedShadowSelector, hits_per_observable
from .kalman_tracker import StreamingKalmanHamiltonianTracker
from .fermionic_shadows import FermionicMatchgateShadows, MatchgateSnapshot, majorana_operators, slater_state
from .process_tomography import ShadowProcessTomographer, pauli_transfer_matrix_of_unitary, project_cptp
from .rydberg import (
    AQUILA,
    AQUILA_BRAKET,
    C6_RB87,
    DeviceNoise,
    QuenchData,
    RydbergDriftTracker,
    RydbergQuenchModel,
    blockade_radius,
)

__version__ = "0.5.0"
__all__ = [
    "ClassicalShadowsProtocol", "ShadowSnapshot", "measure_random_pauli_shadows",
    "sample_shadows_from_density_matrix", "sample_shadows_from_state_vector", "gibbs_state", "basis_rotation_matrix",
    "ShadowObservableEstimator", "estimate_pauli_string_expectation", "estimate_pauli_expectation_median_of_means",
    "estimate_many_observables", "snapshots_to_arrays",
    "DifferentiableHamiltonianLearner", "HamiltonianLearningResult", "learn_hamiltonian_from_shadows",
    "SPAMNoiseMitigator", "ReadoutErrorModel",
    "DerandomizedShadowSelector", "hits_per_observable",
    "StreamingKalmanHamiltonianTracker",
    "FermionicMatchgateShadows", "MatchgateSnapshot", "majorana_operators", "slater_state",
    "ShadowProcessTomographer", "pauli_transfer_matrix_of_unitary",
    "AQUILA", "AQUILA_BRAKET", "C6_RB87", "DeviceNoise", "QuenchData", "RydbergDriftTracker", "RydbergQuenchModel", "blockade_radius",
]
