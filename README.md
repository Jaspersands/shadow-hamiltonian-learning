# Shadow-Tomography-Guided Hamiltonian Learning with Differentiable Post-Processing

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cirq Randomized](https://img.shields.io/badge/Shadows-Cirq%20Randomized%20Clifford-teal.svg)](https://quantumai.google/cirq)
[![PennyLane Autodiff](https://img.shields.io/badge/Inversion-PennyLane%20%7C%20JAX-purple.svg)](https://pennylane.ai/)

> **Characterizing unknown multi-qubit spin Hamiltonians ($J_{ij}, h_i$) from $O(\log M)$ randomized Pauli/Clifford classical shadow measurements in Cirq, inverted through a differentiable PennyLane/JAX post-processing pipeline with SPAM error mitigation.**

---

## 🔮 Theoretical Overview

Characterizing an unknown Hamiltonian $\hat{H} = \sum_{ij} J_{ij} (\vec{\sigma}_i \cdot \vec{\sigma}_j) + \sum_i h_i \sigma_i^z$ traditionally requires full quantum state tomography ($\mathcal{O}(4^N)$ measurements).

### 1. Classical Shadows (Huang, Kueng, Preskill 2020)

By applying randomized single-qubit Clifford rotations $U_i \sim \text{Cl}(1)$ and measuring in the computational basis, each shot produces a classical snapshot:

$$\hat{\rho}^{(s)} = \bigotimes_{i=1}^N \left( 3 U_i^\dagger |b_i\rangle\langle b_i| U_i - \mathbb{I} \right)$$

This allows predicting $M$ arbitrary multi-qubit Pauli observables simultaneously from only:

$$S = \mathcal{O}\left( \frac{\log M}{\epsilon^2} \max_k \|O_k\|_{\text{shadow}}^2 \right)$$

### 2. Differentiable Inverse Optimization

We match the shadow expectations $\hat{o}_k$ against parameterized thermal Gibbs states $\rho(\mathbf{J}, \mathbf{h}) = \frac{e^{-\beta H(\mathbf{J},\mathbf{h})}}{\mathcal{Z}}$ via automatic differentiation:

$$\mathcal{L}(\mathbf{J}, \mathbf{h}) = \sum_{k=1}^M w_k \left| \text{Tr}\left( O_k \frac{e^{-\beta H(\mathbf{J}, \mathbf{h})}}{\mathcal{Z}} \right) - \hat{o}_k^{\text{shadow}} \right|^2 + \lambda \|\mathbf{J}\|_1$$

### 3. SPAM (State Preparation & Measurement) Error Mitigation

Inverted confusion matrix de-biasing corrects for asymmetric readout flip rates ($\epsilon_{0\to 1}, \epsilon_{1\to 0}$):

$$\langle O \rangle_{\text{mitigated}} = \frac{\langle O \rangle_{\text{raw}}}{(1 - \epsilon_{0\to 1} - \epsilon_{1\to 0})^{\text{weight}(O)}}$$

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/Jaspersands/shadow-hamiltonian-learning.git
cd shadow-hamiltonian-learning
pip install -e .
```

### Python API Example

```python
import cirq
from shadow_learning import (
    measure_random_pauli_shadows,
    estimate_many_observables,
    learn_hamiltonian_from_shadows,
    SPAMNoiseMitigator,
)

# 1. Collect Classical Shadows in Cirq
q = cirq.LineQubit.range(2)
prep_circuit = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])
snapshots = measure_random_pauli_shadows(prep_circuit, n_qubits=2, n_snapshots=1000)

# 2. Predict Multi-Qubit Observables
obs = estimate_many_observables(snapshots, ["ZZ", "XX", "YY", "ZI", "IZ"])
print("Shadow Observables:", obs)

# 3. Differentiable Hamiltonian Inversion
result = learn_hamiltonian_from_shadows(snapshots, n_qubits=2, beta=1.0)
print(f"Recovered Coupling J_01: {result.recovered_j_matrix[0, 1]:.4f}")
print(f"Recovered Field h: {result.recovered_h_vector}")
```

---

## 🧪 Testing & Benchmarks

Run unit tests:
```bash
pytest -v tests/
```

Run sample complexity benchmark:
```bash
python benchmarks/run_shadow_benchmark.py
```

---

## 🌐 Interactive Web Showcase

Launch `web/index.html` to explore:
- **Interactive Classical Shadow Visualizer**: Watch randomized Pauli bases ($X, Y, Z$) rotate and collapse into bitstrings live.
- **Real-Time Coupling Inversion Engine**: Reconstruct hidden Heisenberg coupling matrices $J_{ij}$ via browser gradient descent.
- **SPAM Error Mitigation Toggle**: Compare raw vs error-mitigated shadow predictions under 5% readout noise.
- **Log-Log Sample Complexity Scaling Charts**.

---

## 📄 Citation & License

Developed by **Jasper Sands** under the **Apache-2.0 License**.
