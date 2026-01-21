"""
Noise Geometry Benchmark

Demonstrates unique geometric trajectory tracking during noisy evolution.
While Qiskit/Cirq can simulate noise, they don't track the full geometry.

Key capabilities demonstrated:
- Bures distance from initial state (proper metric on density matrices)
- Quantum Fisher Information decay (sensing precision loss)
- Purity evolution (decoherence tracking)
- Geodesic deviation (ideal vs noisy trajectory comparison)

This is what makes NoeticEidos distinct: we don't just simulate noise,
we characterize it through the lens of information geometry.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from state import QuantumState, plus_state, computational_basis
from lindblad import (
    evolve_lindblad, evolve_unitary, amplitude_damping_ops, phase_damping_ops,
    purity_decay, geodesic_deviation, LindbladOperator
)
from information_geometry import bures_distance, quantum_fisher_information
from benchmarks.shared.hamiltonians import single_qubit_hamiltonian, pauli_z
from benchmarks.shared.utils import (
    print_table, plot_bures_trajectory, plot_qfi_decay, plot_trajectory,
    get_figures_dir, save_results
)


def track_geometric_evolution(
    initial_state: QuantumState,
    hamiltonian: np.ndarray,
    lindblad_ops: List[LindbladOperator],
    generator: np.ndarray,
    dt: float = 0.05,
    steps: int = 60
) -> Dict[str, np.ndarray]:
    """
    Track geometric quantities during Lindblad evolution.

    This is the core unique capability: full trajectory tracking with
    information-geometric interpretation.

    Args:
        initial_state: Starting quantum state
        hamiltonian: System Hamiltonian
        lindblad_ops: Noise operators
        generator: Generator for QFI computation (e.g., σz)
        dt: Time step
        steps: Number of steps

    Returns:
        Dict with time, bures, qfi, purity arrays
    """
    # Evolve under noisy dynamics
    trajectory = evolve_lindblad(initial_state, hamiltonian, lindblad_ops, dt, steps)

    # Also get ideal (unitary) trajectory for comparison
    ideal_trajectory = evolve_unitary(initial_state, hamiltonian, dt, steps)

    # Extract geometric quantities
    times = np.arange(len(trajectory)) * dt
    bures_distances = np.zeros(len(trajectory))
    qfi_values = np.zeros(len(trajectory))
    purities = np.zeros(len(trajectory))
    fidelities_to_ideal = np.zeros(len(trajectory))

    initial_rho = initial_state.rho

    for i, state in enumerate(trajectory):
        # Bures distance from initial state
        bures_distances[i] = bures_distance(initial_rho, state.rho)

        # Quantum Fisher Information
        qfi_values[i] = quantum_fisher_information(state.rho, generator)

        # Purity
        purities[i] = state.purity

        # Fidelity to ideal trajectory
        fidelities_to_ideal[i] = state.fidelity(ideal_trajectory[i])

    return {
        'time': times,
        'bures_distance': bures_distances,
        'qfi': qfi_values,
        'purity': purities,
        'fidelity_to_ideal': fidelities_to_ideal,
        'decoherence_rate': estimate_decoherence_rate(times, purities),
    }


def estimate_decoherence_rate(times: np.ndarray, purities: np.ndarray) -> float:
    """
    Estimate decoherence rate from purity decay.

    For small times: Tr(ρ²) ≈ 1 - Γt
    """
    # Use first half where linear approximation holds
    n_fit = len(times) // 2
    if n_fit < 2:
        n_fit = len(times)

    purity_loss = 1 - purities[:n_fit]
    times_fit = times[:n_fit]

    # Linear regression: purity_loss = Γ * t
    if np.sum(times_fit ** 2) > 1e-12:
        gamma = np.sum(times_fit * purity_loss) / np.sum(times_fit ** 2)
    else:
        gamma = 0.0

    return max(0, gamma)


def benchmark_amplitude_damping(
    gamma: float = 0.1,
    T: float = 3.0,
    steps: int = 60
) -> Dict[str, np.ndarray]:
    """
    Benchmark geometric tracking under amplitude damping (T1 decay).

    Physical interpretation: Energy relaxation |1⟩ → |0⟩

    Args:
        gamma: Damping rate
        T: Total evolution time
        steps: Number of time steps
    """
    dt = T / steps

    # Start from |+⟩ state (superposition)
    initial_state = plus_state(1)

    # Simple Hamiltonian: ω σz / 2
    H = single_qubit_hamiltonian(omega=1.0)

    # Amplitude damping
    lindblad_ops = amplitude_damping_ops(gamma, n_qubits=1)

    # σz is our generator for QFI
    generator = pauli_z(n_qubits=1)

    return track_geometric_evolution(initial_state, H, lindblad_ops, generator, dt, steps)


def benchmark_phase_damping(
    gamma: float = 0.1,
    T: float = 3.0,
    steps: int = 60
) -> Dict[str, np.ndarray]:
    """
    Benchmark geometric tracking under phase damping (T2* dephasing).

    Physical interpretation: Pure dephasing without energy exchange

    Args:
        gamma: Dephasing rate
        T: Total evolution time
        steps: Number of time steps
    """
    dt = T / steps

    initial_state = plus_state(1)
    H = single_qubit_hamiltonian(omega=1.0)
    lindblad_ops = phase_damping_ops(gamma, n_qubits=1)
    generator = pauli_z(n_qubits=1)

    return track_geometric_evolution(initial_state, H, lindblad_ops, generator, dt, steps)


def print_geometry_summary(results: Dict[str, np.ndarray], noise_type: str):
    """Print summary of geometric evolution."""
    print(f"\n{noise_type} Noise Geometry Summary:")
    print("-" * 40)
    print(f"  Initial purity:        {results['purity'][0]:.6f}")
    print(f"  Final purity:          {results['purity'][-1]:.6f}")
    print(f"  Max Bures distance:    {np.max(results['bures_distance']):.6f}")
    print(f"  Initial QFI:           {results['qfi'][0]:.6f}")
    print(f"  Final QFI:             {results['qfi'][-1]:.6f}")
    print(f"  QFI decay ratio:       {results['qfi'][-1] / max(results['qfi'][0], 1e-10):.4f}")
    print(f"  Decoherence rate Γ:    {results['decoherence_rate']:.6f}")
    print(f"  Final fidelity to ideal: {results['fidelity_to_ideal'][-1]:.6f}")


def print_trajectory_table(results: Dict[str, np.ndarray], sample_points: int = 8):
    """Print sampled trajectory as table."""
    n_points = len(results['time'])
    indices = np.linspace(0, n_points - 1, sample_points, dtype=int)

    headers = ["Time", "Bures", "QFI", "Purity", "F(ideal)"]
    rows = []

    for i in indices:
        rows.append([
            f"{results['time'][i]:.2f}",
            f"{results['bures_distance'][i]:.4f}",
            f"{results['qfi'][i]:.4f}",
            f"{results['purity'][i]:.4f}",
            f"{results['fidelity_to_ideal'][i]:.4f}",
        ])

    print_table(headers, rows)


def run_noise_geometry_benchmark(
    save_plots: bool = True,
    save_data: bool = True
) -> Dict[str, Dict]:
    """
    Run the complete noise geometry benchmark.

    Demonstrates unique capability: full geometric trajectory tracking
    that goes beyond simple noise simulation.

    Args:
        save_plots: Whether to save figures
        save_data: Whether to save results to JSON

    Returns:
        Dict with results for each noise type
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Noise Geometry (Unique Capability)")
    print("=" * 60)
    print("\nTracking geometric quantities during noisy evolution...")
    print("This demonstrates what NoeticEidos does beyond plain simulation.")
    print()

    results = {}

    # Amplitude damping benchmark
    print("Testing amplitude damping (T1 decay)...", end=" ", flush=True)
    amp_results = benchmark_amplitude_damping(gamma=0.1, T=3.0, steps=60)
    results['amplitude_damping'] = amp_results
    print("done")
    print_geometry_summary(amp_results, "Amplitude Damping")
    print_trajectory_table(amp_results)

    # Phase damping benchmark
    print("Testing phase damping (T2* dephasing)...", end=" ", flush=True)
    phase_results = benchmark_phase_damping(gamma=0.15, T=3.0, steps=60)
    results['phase_damping'] = phase_results
    print("done")
    print_geometry_summary(phase_results, "Phase Damping")
    print_trajectory_table(phase_results)

    # Generate plots
    if save_plots:
        figures_dir = get_figures_dir()

        # Amplitude damping Bures trajectory
        plot_bures_trajectory(
            amp_results['time'],
            amp_results['bures_distance'],
            amp_results['purity'],
            title="Amplitude Damping: Bures Distance Evolution",
            save_path=str(figures_dir / "amplitude_damping_bures.png")
        )

        # Phase damping Bures trajectory
        plot_bures_trajectory(
            phase_results['time'],
            phase_results['bures_distance'],
            phase_results['purity'],
            title="Phase Damping: Bures Distance Evolution",
            save_path=str(figures_dir / "phase_damping_bures.png")
        )

        # QFI decay comparison
        plot_trajectory(
            amp_results['time'],
            {
                'QFI (Amplitude Damping)': amp_results['qfi'],
                'QFI (Phase Damping)': phase_results['qfi'],
            },
            title="Quantum Fisher Information Decay Comparison",
            xlabel="Time",
            save_path=str(figures_dir / "qfi_decay_comparison.png")
        )

        # Geodesic deviation (fidelity to ideal)
        plot_trajectory(
            amp_results['time'],
            {
                'Fidelity (Amp. Damping)': amp_results['fidelity_to_ideal'],
                'Fidelity (Phase Damping)': phase_results['fidelity_to_ideal'],
            },
            title="Geodesic Deviation: Fidelity to Ideal Trajectory",
            xlabel="Time",
            save_path=str(figures_dir / "geodesic_deviation.png")
        )

    if save_data:
        save_results(results, 'noise_geometry_benchmark.json')

    print("\n✓ Noise geometry benchmark complete")
    print("  Key insight: We track the FULL trajectory with geometric interpretation,")
    print("  not just end states. This enables decoherence characterization.")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_noise_geometry_benchmark()
