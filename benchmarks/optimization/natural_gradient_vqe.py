"""
Natural Gradient VQE Benchmark

Demonstrates the practical advantage of geometry-aware optimization
for variational quantum algorithms.

Four-Way Comparison:
1. Vanilla GD     - No geometry, no history (baseline)
2. Natural GD    - Geometry-aware (F⁻¹ metric)
3. Conjugate GD  - History-aware (conjugate directions)
4. Natural CG    - Both geometry and history (best of both worlds)

The key insight: Natural gradient follows the steepest descent on the
quantum state manifold, not the parameter space. This leads to:
- Faster convergence
- Parameterization invariance
- Better navigation of optimization landscapes

Additionally, spectral ζ-analysis of the QFI matrix predicts convergence behavior.
"""

import sys
import os
import numpy as np
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from state import QuantumState
from natural_gradient import (
    natural_gradient_descent, vanilla_gradient_descent,
    make_vqe_state_fn, make_vqe_energy_fn, simple_ansatz,
    quantum_fisher_information_matrix,
    OptimizationResult
)
from conjugate_gradient import (
    conjugate_gradient_descent,
    natural_conjugate_gradient_descent,
    compute_spectral_diagnostics
)
from benchmarks.shared.hamiltonians import zz_hamiltonian, ground_state_energy
from benchmarks.shared.utils import (
    print_table, plot_convergence, get_figures_dir, save_results
)
from benchmarks.optimization.optimizer_comparison import (
    compare_all_optimizers,
    spectral_diagnostics_dict,
    count_iterations_to_threshold
)


# =============================================================================
# VQE Benchmark
# =============================================================================

def run_vqe_comparison(
    n_iterations: int = 100,
    vanilla_lr: float = 0.01,
    natural_lr: float = 0.1,
    cg_lr: float = 0.05,
    ncg_lr: float = 0.1,
    seed: int = 42
) -> Dict[str, OptimizationResult]:
    """
    Run 4-way VQE optimization comparison on Z⊗Z Hamiltonian.

    Target: Find ground state with energy -1.

    Args:
        n_iterations: Max iterations for each optimizer
        vanilla_lr: Learning rate for vanilla GD
        natural_lr: Learning rate for natural GD
        cg_lr: Learning rate for conjugate GD
        ncg_lr: Learning rate for natural CG
        seed: Random seed for reproducibility

    Returns:
        Dict mapping optimizer name to OptimizationResult
    """
    np.random.seed(seed)

    # Setup
    n_qubits = 2
    H = zz_hamiltonian()
    target_energy = ground_state_energy(H)  # Should be -1

    # Initial parameters (random)
    initial_params = np.random.randn(n_qubits) * 0.5

    # Create state and energy functions
    state_fn = make_vqe_state_fn(simple_ansatz, n_qubits)
    energy_fn = make_vqe_energy_fn(simple_ansatz, H, n_qubits)

    print("  Running 4-way optimizer comparison...")

    # Use the centralized comparison function
    results = compare_all_optimizers(
        state_fn, energy_fn, initial_params,
        vanilla_lr=vanilla_lr,
        natural_lr=natural_lr,
        cg_lr=cg_lr,
        natural_cg_lr=ncg_lr,
        max_iterations=n_iterations,
        verbose=True
    )

    # Rename keys for display
    display_results = {
        'Vanilla GD': results['vanilla'],
        'Natural GD': results['natural'],
        'Conjugate GD': results['cg'],
        'Natural CG': results['natural_cg']
    }

    return display_results, target_energy, state_fn, initial_params


def print_vqe_results(results: Dict[str, OptimizationResult], target_energy: float):
    """Print VQE comparison results."""
    headers = ["Optimizer", "Final Energy", "Iters to 0.01", "Iters to 0.001", "Converged"]
    rows = []

    for name, result in results.items():
        iters_001 = count_iterations_to_threshold(result.energies, target_energy, 0.01)
        iters_0001 = count_iterations_to_threshold(result.energies, target_energy, 0.001)

        rows.append([
            name,
            f"{result.energies[-1]:.6f}",
            f"{iters_001}" if iters_001 < len(result.energies) else ">max",
            f"{iters_0001}" if iters_0001 < len(result.energies) else ">max",
            "Yes" if result.converged else "No"
        ])

    print_table(headers, rows, title="VQE Optimization Comparison (H = Z⊗Z, E₀ = -1)")


def print_spectral_analysis(state_fn, initial_params: np.ndarray):
    """Print spectral analysis of initial QFI."""
    F = quantum_fisher_information_matrix(state_fn, initial_params)
    diag = spectral_diagnostics_dict(F)

    print("\nSpectral Diagnostics of Initial QFI:")
    print("-" * 40)
    print(f"  Condition number κ:     {diag['condition_number']:.2f}")
    print(f"  Spectral gap λ_min:     {diag['spectral_gap']:.4f}")
    print(f"  Effective dimension:    {diag['effective_dim']:.2f}")
    print(f"  ζ_F(1) = Tr(F⁻¹):       {diag['zeta_1']:.4f}")
    print(f"  ζ_F(-1) = Tr(F):        {diag['zeta_neg1']:.4f}")
    print(f"  Predicted CG iters:     ~{diag['predicted_cg_iters']}")
    print()


def run_vqe_benchmark(
    save_plots: bool = True,
    save_data: bool = True
) -> Dict:
    """
    Run the complete VQE optimization benchmark.

    Demonstrates:
    1. Natural gradient's faster convergence
    2. Conjugate gradient's efficiency
    3. Natural CG combining both benefits
    4. Spectral ζ-analysis predicting performance

    Args:
        save_plots: Whether to save figures
        save_data: Whether to save results to JSON
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Natural Gradient VQE Optimization")
    print("=" * 60)
    print("\nComparing 4 optimizers on VQE for H = Z⊗Z:")
    print("  - Vanilla GD:    Standard gradient descent (baseline)")
    print("  - Natural GD:    Uses Fisher metric F⁻¹ (geometry)")
    print("  - Conjugate GD:  Uses previous directions (history)")
    print("  - Natural CG:    F-conjugate directions (both)")
    print()

    # Run comparison
    results, target_energy, state_fn, initial_params = run_vqe_comparison(
        n_iterations=150,
        vanilla_lr=0.02,
        natural_lr=0.15,
        cg_lr=0.08,
        ncg_lr=0.15
    )

    print()
    print_vqe_results(results, target_energy)
    print_spectral_analysis(state_fn, initial_params)

    # Generate convergence plot
    if save_plots:
        figures_dir = get_figures_dir()

        energy_histories = {
            name: result.energies for name, result in results.items()
        }

        plot_convergence(
            energy_histories,
            title="VQE Convergence: 4-Way Optimizer Comparison",
            ylabel="Energy ⟨H⟩",
            target_energy=target_energy,
            save_path=str(figures_dir / "vqe_convergence_comparison.png")
        )

    if save_data:
        save_dict = {
            name: {
                'energies': result.energies,
                'iterations': result.iterations,
                'converged': result.converged,
                'final_energy': result.energies[-1]
            }
            for name, result in results.items()
        }
        save_dict['target_energy'] = target_energy
        save_results(save_dict, 'vqe_benchmark.json')

    print("✓ VQE benchmark complete")
    print("  Key insight: Natural gradient follows state manifold geometry,")
    print("  leading to faster convergence than Euclidean methods.")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_vqe_benchmark()