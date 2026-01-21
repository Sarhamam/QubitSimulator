"""
Optimizer Comparison Utilities

Provides functions for comparing all four optimization methods:
1. Vanilla GD     - No geometry, no history (baseline)
2. Natural GD     - Geometry-aware (F⁻¹ metric)
3. Conjugate GD   - History-aware (conjugate directions)
4. Natural CG     - Both geometry and history

Also includes spectral diagnostics for predicting optimizer performance.
"""

import sys
import os
import numpy as np
from typing import Dict, Callable, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from state import QuantumState
from natural_gradient import (
    OptimizationResult,
    natural_gradient_descent,
    vanilla_gradient_descent,
    quantum_fisher_information_matrix
)
from conjugate_gradient import (
    conjugate_gradient_descent,
    natural_conjugate_gradient_descent,
    compute_spectral_diagnostics,
    SpectralDiagnostics
)


def compare_all_optimizers(
    state_fn: Callable[[np.ndarray], QuantumState],
    energy_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    natural_lr: float = 0.1,
    vanilla_lr: float = 0.01,
    cg_lr: float = 0.1,
    natural_cg_lr: float = 0.1,
    max_iterations: int = 100,
    verbose: bool = False
) -> Dict[str, OptimizationResult]:
    """
    Compare all four optimization methods.

    Methods:
    - Vanilla GD: θ' = θ - η∇E
    - Natural GD: θ' = θ - ηF⁻¹∇E
    - Conjugate GD: d = -∇E + βd_{prev}
    - Natural CG: d = -F⁻¹∇E + βd_{prev}

    Args:
        state_fn: Parameterized state function
        energy_fn: Energy function
        initial_params: Starting parameters
        natural_lr: Learning rate for natural GD
        vanilla_lr: Learning rate for vanilla GD
        cg_lr: Learning rate for CG
        natural_cg_lr: Learning rate for natural CG
        max_iterations: Maximum iterations
        verbose: Print progress

    Returns:
        Dict mapping method name to OptimizationResult
    """
    results = {}

    # Vanilla gradient descent
    if verbose:
        print("  Running Vanilla GD...", end=" ", flush=True)
    results['vanilla'] = vanilla_gradient_descent(
        energy_fn, initial_params.copy(),
        learning_rate=vanilla_lr,
        max_iterations=max_iterations,
        verbose=verbose
    )
    if verbose:
        print(f"done ({results['vanilla'].iterations} iters)")

    # Natural gradient descent
    if verbose:
        print("  Running Natural GD...", end=" ", flush=True)
    results['natural'] = natural_gradient_descent(
        state_fn, energy_fn, initial_params.copy(),
        learning_rate=natural_lr,
        max_iterations=max_iterations,
        verbose=verbose
    )
    if verbose:
        print(f"done ({results['natural'].iterations} iters)")

    # Conjugate gradient descent
    if verbose:
        print("  Running Conjugate GD...", end=" ", flush=True)
    results['cg'] = conjugate_gradient_descent(
        energy_fn, initial_params.copy(),
        learning_rate=cg_lr,
        max_iterations=max_iterations,
        verbose=verbose
    )
    if verbose:
        print(f"done ({results['cg'].iterations} iters)")

    # Natural conjugate gradient descent
    if verbose:
        print("  Running Natural CG...", end=" ", flush=True)
    results['natural_cg'] = natural_conjugate_gradient_descent(
        state_fn, energy_fn, initial_params.copy(),
        learning_rate=natural_cg_lr,
        max_iterations=max_iterations,
        verbose=verbose
    )
    if verbose:
        print(f"done ({results['natural_cg'].iterations} iters)")

    return results


def predict_optimizer_performance(
    state_fn: Callable[[np.ndarray], QuantumState],
    params: np.ndarray,
    epsilon: float = 1e-5
) -> Dict[str, Any]:
    """
    Use spectral analysis to predict optimizer performance.

    Args:
        state_fn: Parameterized state function
        params: Parameter point for analysis
        epsilon: Finite difference step size

    Returns:
        Dict with predictions for each optimizer
    """
    qfi = quantum_fisher_information_matrix(state_fn, params, epsilon)
    diagnostics = compute_spectral_diagnostics(qfi)

    kappa = diagnostics.condition_number

    return {
        'spectral_diagnostics': diagnostics,
        'predictions': {
            'vanilla_gd': {
                'convergence': 'Slow, depends on learning rate tuning',
                'iterations_order': f'O(κ) ≈ O({int(kappa)})',
                'best_when': 'Condition number near 1, simple landscapes'
            },
            'natural_gd': {
                'convergence': 'Parameterization invariant, geometry-aware',
                'iterations_order': f'O(κ) ≈ O({int(kappa)}) but larger steps',
                'best_when': 'Curved parameter space, QFI varies significantly'
            },
            'conjugate_gd': {
                'convergence': 'Linear convergence with rate ((√κ-1)/(√κ+1))',
                'iterations_order': f'O(√κ) ≈ O({diagnostics.predicted_cg_iterations})',
                'best_when': 'Large κ, quadratic-like landscape'
            },
            'natural_cg': {
                'convergence': 'Best of both: geometry + history awareness',
                'iterations_order': f'O(√κ) ≈ O({diagnostics.predicted_cg_iterations}) with better constants',
                'best_when': 'Curved space with moderate-to-large κ'
            }
        },
        'recommendation': _recommend_optimizer(diagnostics)
    }


def _recommend_optimizer(diagnostics: SpectralDiagnostics) -> str:
    """Choose optimizer based on spectral properties."""
    kappa = diagnostics.condition_number
    eff_dim = diagnostics.effective_dimension

    if kappa < 5:
        return 'vanilla_gd or natural_gd (both efficient for well-conditioned problems)'
    elif kappa < 50:
        if eff_dim < 3:
            return 'natural_gd (low effective dimension, geometry matters)'
        else:
            return 'conjugate_gd (moderate κ, history helps)'
    else:
        return 'natural_cg (large κ, need both geometry and history awareness)'


def spectral_diagnostics_dict(qfi_matrix: np.ndarray) -> Dict[str, float]:
    """
    Analyze QFI spectrum and return as dict (for benchmark reporting).

    Args:
        qfi_matrix: Quantum Fisher Information matrix

    Returns:
        Dict with spectral properties
    """
    diag = compute_spectral_diagnostics(qfi_matrix)

    return {
        'condition_number': diag.condition_number,
        'spectral_gap': diag.spectral_gap,
        'effective_dim': diag.effective_dimension,
        'zeta_1': diag.zeta_1,
        'zeta_neg1': diag.zeta_neg1,
        'predicted_cg_iters': diag.predicted_cg_iterations,
    }


def count_iterations_to_threshold(
    energies: list,
    target: float,
    threshold: float = 0.01
) -> int:
    """Count iterations to reach within threshold of target."""
    for i, e in enumerate(energies):
        if abs(e - target) < threshold:
            return i
    return len(energies)