# Contributing to shadow-hamiltonian-learning

Thank you for contributing to **shadow-hamiltonian-learning**!

## Development Setup

```bash
git clone https://github.com/Jaspersands/shadow-hamiltonian-learning.git
cd shadow-hamiltonian-learning
pip install -e ".[dev]"
pytest -v tests/
```

## Pull Request Guidelines
- All unit tests must pass.
- Maintain unbiased shadow observable estimation formulas and SPAM mitigation kernels.

## Conventions
- Qubit 0 is the most significant bit of basis-state indices; Pauli strings are written qubit 0 first ("XZI").
- Randomised shadow estimates carry the 3^k weight; derandomised (fixed-basis) estimates are plain means over matching shots.
- Every stochastic routine takes a `seed` / `rng`.
- Run `pytest -q tests/` and `python benchmarks/run_shadow_benchmark.py --quick` before opening a PR.
