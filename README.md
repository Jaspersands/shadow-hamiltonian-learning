# Shadow-Tomography-Guided Hamiltonian Learning with Differentiable Post-Processing

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cirq Randomized](https://img.shields.io/badge/Shadows-Cirq%20Randomized%20Clifford-teal.svg)](https://quantumai.google/cirq)
[![PennyLane Autodiff](https://img.shields.io/badge/Inversion-PennyLane%20%7C%20JAX-purple.svg)](https://pennylane.ai/)

> **Characterizing unknown multi-qubit spin Hamiltonians ($J_{ij}, h_i$) from $O(\log M)$ randomized and derandomized classical shadow measurements in Cirq, inverted through a differentiable PennyLane/JAX post-processing pipeline with real-time Kalman tracking, fermionic matchgate shadows, and SPAM error mitigation.**

---

## 🔮 Overview & Features (v0.2.0)

- **Randomized & Derandomized Classical Shadows (`DerandomizedShadowSelector`)**:
  - Deterministic greedy Pauli basis selector that minimizes the upper bound on observable measurement variance, cutting required shot budgets by $3-5\times$.
- **Continuous Real-Time Bayesian / Kalman Tracking (`StreamingKalmanHamiltonianTracker`)**:
  - Sequential Extended Kalman Filter (EKF) tracking time-varying couplings $J_{ij}(t)$ and magnetic field drifts $h_i(t)$ from live snapshot streams.
- **Fermionic Matchgate Shadows (`FermionicMatchgateShadows`)**:
  - 1-RDM ($D_{pq} = \langle c_p^\dagger c_q \rangle$) and 2-RDM estimation for molecular quantum chemistry Hamiltonians using Majorana correlators.
- **Shadow Quantum Process Tomography (`ShadowProcessTomographer`)**:
  - Reconstructs unknown CPTP noise channels and process fidelities from shadow measurements on process ancillas.
- **Differentiable Inversion with SPAM Error Mitigation (`SPAMNoiseMitigator`)**:
  - Inverted confusion matrix convolution de-biasing asymmetric readout errors.

---

## 🚀 Quickstart

```python
import cirq
from shadow_learning import (
    measure_random_pauli_shadows,
    estimate_many_observables,
    learn_hamiltonian_from_shadows,
    DerandomizedShadowSelector,
    StreamingKalmanHamiltonianTracker,
    SPAMNoiseMitigator,
)

# 1. Derandomized Greedy Shadows
selector = DerandomizedShadowSelector(target_observables=["ZZ", "XX", "YY"], n_qubits=2)
bases = selector.select_measurement_bases(n_snapshots=200)

# 2. Collect Shadows & Predict
q = cirq.LineQubit.range(2)
c = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])
snapshots = measure_random_pauli_shadows(c, n_qubits=2, n_snapshots=500)
obs = estimate_many_observables(snapshots, ["ZZ", "XX", "YY"])
print("Observables:", obs)

# 3. Differentiable Hamiltonian Inversion
res = learn_hamiltonian_from_shadows(snapshots, n_qubits=2, beta=1.0)
print(f"Recovered J_01: {res.recovered_j_matrix[0, 1]:.4f}")
```

---

## 🧪 Testing & Benchmarks

```bash
pytest -v tests/
python benchmarks/run_shadow_benchmark.py
```

---

## 📄 Citation & License

Developed by **Jasper Sands** under the **Apache-2.0 License**.
