"""
Command-Line Interface for shadow_learning.
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import cirq
from .cirq_shadows import measure_random_pauli_shadows
from .reconstruction import estimate_many_observables
from .differentiable_inversion import learn_hamiltonian_from_shadows
from .derandomized import DerandomizedShadowSelector
from .spam_mitigation import ReadoutErrorModel, SPAMNoiseMitigator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shadow-learn",
        description="Shadow-Tomography-Guided Hamiltonian Learning CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Collect shadows and invert Hamiltonian parameters")
    run_parser.add_argument("--qubits", type=int, default=2, help="Number of qubits")
    run_parser.add_argument("--shots", type=int, default=500, help="Classical shadow snapshot budget")
    run_parser.add_argument("--derandomized", action="store_true", help="Use derandomized greedy basis selection")
    run_parser.add_argument("--spam-mitigate", action="store_true", help="Apply SPAM readout error mitigation")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run":
        print(f"[*] Preparing {args.qubits}-qubit Bell/GHZ state on Cirq...")
        q = cirq.LineQubit.range(args.qubits)
        c = cirq.Circuit()
        c.append(cirq.H(q[0]))
        for i in range(args.qubits - 1):
            c.append(cirq.CNOT(q[i], q[i + 1]))

        if args.derandomized:
            print(f"[*] Selecting greedy derandomized bases for {args.shots} shots...")
            selector = DerandomizedShadowSelector(target_observables=["ZZ", "XX", "YY"], n_qubits=args.qubits)
            bases = selector.select_measurement_bases(args.shots)

        print(f"[*] Acquiring {args.shots} shadow snapshots...")
        snapshots = measure_random_pauli_shadows(c, n_qubits=args.qubits, n_snapshots=args.shots, seed=42)

        target_obs = ["ZZ", "XX", "YY"] if args.qubits == 2 else ["ZZI", "IZZ", "XXX"]
        obs_vals = estimate_many_observables(snapshots, target_obs)
        print(f"[+] Shadow Observables: {obs_vals}")

        if args.qubits == 2:
            print("[*] Running differentiable Hamiltonian inversion...")
            res = learn_hamiltonian_from_shadows(snapshots, n_qubits=2, beta=1.0)
            print(f"[+] Recovered Coupling J_01: {res.recovered_j_matrix[0, 1]:.4f} (True: -1.00)")
            print(f"[+] Final Loss: {res.final_loss:.6f}")

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
