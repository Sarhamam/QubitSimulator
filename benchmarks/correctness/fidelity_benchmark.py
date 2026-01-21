"""
State Fidelity Benchmark

Establishes numerical correctness parity between NoeticEidos and
Qiskit/Cirq by comparing output statevectors for standard circuits.

Philosophy: We must match established simulators to floating-point precision
before demonstrating our unique geometric capabilities.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from benchmarks.shared.utils import print_table, format_scientific, ensure_results_dir, save_results


def compute_fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """
    Compute fidelity between two pure state vectors.

    F = |⟨ψ₁|ψ₂⟩|²

    Args:
        psi1: First statevector
        psi2: Second statevector

    Returns:
        Fidelity in [0, 1]
    """
    # Normalize both vectors
    psi1 = psi1 / np.linalg.norm(psi1)
    psi2 = psi2 / np.linalg.norm(psi2)

    # Compute overlap magnitude squared
    overlap = np.abs(np.vdot(psi1, psi2)) ** 2

    return float(overlap)


def compare_bell_state() -> Dict[str, float]:
    """Compare Bell state preparation across all three frameworks."""
    from benchmarks.shared.circuits import (
        build_bell_circuit_ours, build_bell_circuit_qiskit, build_bell_circuit_cirq,
        get_statevector_ours, get_statevector_qiskit, get_statevector_cirq
    )

    # Get statevectors
    psi_ours = get_statevector_ours(build_bell_circuit_ours())
    psi_qiskit = get_statevector_qiskit(build_bell_circuit_qiskit())
    psi_cirq = get_statevector_cirq(build_bell_circuit_cirq())

    return {
        'fidelity_vs_qiskit': compute_fidelity(psi_ours, psi_qiskit),
        'fidelity_vs_cirq': compute_fidelity(psi_ours, psi_cirq),
    }


def compare_ghz_state(n_qubits: int) -> Dict[str, float]:
    """Compare GHZ state preparation for n qubits."""
    from benchmarks.shared.circuits import (
        build_ghz_circuit_ours, build_ghz_circuit_qiskit, build_ghz_circuit_cirq,
        get_statevector_ours, get_statevector_qiskit, get_statevector_cirq
    )

    psi_ours = get_statevector_ours(build_ghz_circuit_ours(n_qubits))
    psi_qiskit = get_statevector_qiskit(build_ghz_circuit_qiskit(n_qubits))
    psi_cirq = get_statevector_cirq(build_ghz_circuit_cirq(n_qubits))

    return {
        'fidelity_vs_qiskit': compute_fidelity(psi_ours, psi_qiskit),
        'fidelity_vs_cirq': compute_fidelity(psi_ours, psi_cirq),
    }


def compare_qft(n_qubits: int) -> Dict[str, float]:
    """Compare QFT circuit for n qubits."""
    from benchmarks.shared.circuits import (
        build_qft_circuit_ours, build_qft_circuit_qiskit, build_qft_circuit_cirq,
        get_statevector_ours, get_statevector_qiskit, get_statevector_cirq
    )

    psi_ours = get_statevector_ours(build_qft_circuit_ours(n_qubits))
    psi_qiskit = get_statevector_qiskit(build_qft_circuit_qiskit(n_qubits))
    psi_cirq = get_statevector_cirq(build_qft_circuit_cirq(n_qubits))

    return {
        'fidelity_vs_qiskit': compute_fidelity(psi_ours, psi_qiskit),
        'fidelity_vs_cirq': compute_fidelity(psi_ours, psi_cirq),
    }


def compare_random_clifford(n_qubits: int, depth: int = 15, seed: int = 42) -> Dict[str, float]:
    """Compare random Clifford circuit."""
    from benchmarks.shared.circuits import (
        build_random_clifford_circuit_ours, build_random_clifford_circuit_qiskit,
        build_random_clifford_circuit_cirq, get_statevector_ours, get_statevector_qiskit,
        get_statevector_cirq
    )

    # Use same seed for all frameworks
    psi_ours = get_statevector_ours(build_random_clifford_circuit_ours(n_qubits, depth, seed))
    psi_qiskit = get_statevector_qiskit(build_random_clifford_circuit_qiskit(n_qubits, depth, seed))
    psi_cirq = get_statevector_cirq(build_random_clifford_circuit_cirq(n_qubits, depth, seed))

    return {
        'fidelity_vs_qiskit': compute_fidelity(psi_ours, psi_qiskit),
        'fidelity_vs_cirq': compute_fidelity(psi_ours, psi_cirq),
    }


def run_all_fidelity_tests() -> Dict[str, Dict]:
    """
    Run all fidelity benchmark tests.

    Returns:
        Dict mapping test name to results
    """
    results = {}

    # Bell state
    print("  Testing Bell state...", end=" ", flush=True)
    results['Bell (2q)'] = compare_bell_state()
    print("done")

    # GHZ states
    for n in [3, 4, 5]:
        print(f"  Testing GHZ ({n}q)...", end=" ", flush=True)
        results[f'GHZ ({n}q)'] = compare_ghz_state(n)
        print("done")

    # QFT
    for n in [3, 4]:
        print(f"  Testing QFT ({n}q)...", end=" ", flush=True)
        results[f'QFT ({n}q)'] = compare_qft(n)
        print("done")

    # Random Clifford
    for n in [3, 4]:
        print(f"  Testing Clifford ({n}q)...", end=" ", flush=True)
        results[f'Clifford ({n}q)'] = compare_random_clifford(n)
        print("done")

    return results


def print_fidelity_table(results: Dict[str, Dict]):
    """Print formatted fidelity results table."""
    headers = ["Circuit", "vs Qiskit", "vs Cirq"]
    rows = []

    for name, data in results.items():
        f_qiskit = format_scientific(data['fidelity_vs_qiskit'])
        f_cirq = format_scientific(data['fidelity_vs_cirq'])
        rows.append([name, f_qiskit, f_cirq])

    print_table(headers, rows, title="State Fidelity Benchmark: NoeticEidos vs Qiskit/Cirq")


def run_fidelity_benchmark(save: bool = True) -> Dict[str, Dict]:
    """
    Run the complete fidelity benchmark.

    Args:
        save: Whether to save results to file

    Returns:
        Benchmark results dict
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 1: State Fidelity Correctness")
    print("=" * 60)
    print("\nVerifying numerical parity with Qiskit and Cirq...")
    print()

    try:
        results = run_all_fidelity_tests()
        print()
        print_fidelity_table(results)

        # Check all fidelities are acceptable
        all_passed = True
        threshold = 0.9999999  # 7 nines

        for name, data in results.items():
            for key, fidelity in data.items():
                if fidelity < threshold:
                    print(f"  WARNING: {name} {key} = {fidelity:.9f} < {threshold}")
                    all_passed = False

        if all_passed:
            print("✓ All fidelity tests passed (F ≥ 0.9999999)")
        else:
            print("✗ Some fidelity tests below threshold")

        if save:
            save_results(results, 'fidelity_benchmark.json')

        return results

    except ImportError as e:
        print(f"\nERROR: Missing dependency - {e}")
        print("Install with: pip install qiskit qiskit-aer cirq")
        return {}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_fidelity_benchmark()
