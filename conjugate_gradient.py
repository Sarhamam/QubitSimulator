"""
Conjugate Gradient Optimization for Variational Quantum Algorithms

Implements conjugate gradient descent with optional QFI preconditioning.
CG builds conjugate search directions that exploit optimization history,
leading to faster convergence than steepest descent methods.

Mathematical Foundation:
    Standard CG:  d_k = -∇E + β d_{k-1}  (Euclidean conjugacy)
    Natural CG:   d_k = -F⁻¹∇E + β d_{k-1}  (Fisher metric conjugacy)

Key Properties:
    - History awareness: Uses previous directions to build conjugate directions
    - For quadratic problems: Converges in at most n iterations
    - Convergence rate: ((√κ - 1)/(√κ + 1))^k where κ = λ_max/λ_min

Spectral Connection:
    CG convergence is governed by the eigenspectrum of the Hessian (or QFI).
    The spectral zeta function ζ_F(s) = Σ λₙ^{-s} provides invariants:
    - ζ_F(1) = Tr(F⁻¹): average parameter sensitivity
    - ζ_F(-1) = Tr(F): total Fisher information
    - Condition number κ: predicts CG iteration count as O(√κ)
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass

from state import QuantumState
from natural_gradient import (
    OptimizationResult,
    compute_gradient,
    quantum_fisher_information_matrix
)


# ============================================================================
# Spectral Diagnostics
# ============================================================================

@dataclass
class SpectralDiagnostics:
    """
    Spectral analysis of QFI matrix for optimization prediction.

    Attributes:
        eigenvalues: Sorted eigenvalues of QFI matrix
        condition_number: κ = λ_max/λ_min (convergence rate indicator)
        spectral_gap: Smallest eigenvalue (stability indicator)
        effective_dimension: How many directions carry information
        zeta_1: ζ(1) = Tr(F⁻¹), average sensitivity
        zeta_neg1: ζ(-1) = Tr(F), total Fisher information
        predicted_cg_iterations: Estimated iterations for CG (≈√κ)
    """
    eigenvalues: np.ndarray
    condition_number: float
    spectral_gap: float
    effective_dimension: float
    zeta_1: float
    zeta_neg1: float
    predicted_cg_iterations: int


def compute_spectral_diagnostics(
    qfi_matrix: np.ndarray,
    regularization: float = 1e-12
) -> SpectralDiagnostics:
    """
    Analyze QFI spectrum to predict optimization behavior.

    The eigenspectrum of the QFI matrix determines:
    - CG convergence rate (via condition number)
    - Natural GD stability (via spectral gap)
    - Effective parameter dimensionality

    Args:
        qfi_matrix: Quantum Fisher Information matrix
        regularization: Threshold for near-zero eigenvalues

    Returns:
        SpectralDiagnostics with computed invariants
    """
    eigenvalues = np.linalg.eigvalsh(qfi_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Filter out numerically zero eigenvalues
    positive_eigs = eigenvalues[eigenvalues > regularization]

    if len(positive_eigs) == 0:
        # Degenerate case: all eigenvalues near zero
        return SpectralDiagnostics(
            eigenvalues=eigenvalues,
            condition_number=np.inf,
            spectral_gap=0.0,
            effective_dimension=0.0,
            zeta_1=np.inf,
            zeta_neg1=0.0,
            predicted_cg_iterations=len(eigenvalues)
        )

    lambda_max = positive_eigs[0]
    lambda_min = positive_eigs[-1]

    # Condition number determines CG convergence rate
    condition_number = lambda_max / lambda_min

    # Spectral gap: stability indicator for gradient methods
    spectral_gap = lambda_min

    # Effective dimension: participation ratio
    # Measures how many eigenvalues contribute significantly
    total = np.sum(positive_eigs)
    effective_dimension = total**2 / np.sum(positive_eigs**2)

    # Zeta function invariants
    zeta_1 = np.sum(1.0 / positive_eigs)      # Tr(F⁻¹)
    zeta_neg1 = np.sum(positive_eigs)          # Tr(F)

    # Predicted CG iterations (theoretical bound for quadratic)
    predicted_cg_iterations = int(np.ceil(np.sqrt(condition_number)))

    return SpectralDiagnostics(
        eigenvalues=eigenvalues,
        condition_number=condition_number,
        spectral_gap=spectral_gap,
        effective_dimension=effective_dimension,
        zeta_1=zeta_1,
        zeta_neg1=zeta_neg1,
        predicted_cg_iterations=predicted_cg_iterations
    )


# ============================================================================
# Conjugate Gradient Optimizers
# ============================================================================

def compute_beta_polak_ribiere(
    grad: np.ndarray,
    prev_grad: np.ndarray,
    metric_grad: Optional[np.ndarray] = None
) -> float:
    """
    Compute Polak-Ribière beta for CG direction update.

    β = (∇E_k - ∇E_{k-1})ᵀ M⁻¹ ∇E_k / (∇E_{k-1}ᵀ M⁻¹ ∇E_{k-1})

    For standard CG: M = I (identity)
    For natural CG: M = F (QFI matrix), so M⁻¹ gradient is natural gradient

    Args:
        grad: Current gradient
        prev_grad: Previous gradient
        metric_grad: M⁻¹ ∇E (natural gradient if using QFI metric)

    Returns:
        Beta coefficient (clamped to non-negative)
    """
    if metric_grad is None:
        metric_grad = grad

    delta_grad = grad - prev_grad
    numerator = np.dot(delta_grad, metric_grad)

    # For denominator, use previous metric gradient
    # In standard CG this is just prev_grad
    denominator = np.dot(prev_grad, prev_grad)

    if np.abs(denominator) < 1e-12:
        return 0.0

    beta = numerator / denominator

    # Polak-Ribière with restart: clamp negative beta to zero
    return max(0.0, beta)


def compute_beta_fletcher_reeves(
    grad: np.ndarray,
    prev_grad: np.ndarray
) -> float:
    """
    Compute Fletcher-Reeves beta for CG direction update.

    β = ||∇E_k||² / ||∇E_{k-1}||²

    Simpler than Polak-Ribière but may not restart as effectively.

    Args:
        grad: Current gradient
        prev_grad: Previous gradient

    Returns:
        Beta coefficient
    """
    denominator = np.dot(prev_grad, prev_grad)

    if np.abs(denominator) < 1e-12:
        return 0.0

    return np.dot(grad, grad) / denominator


def conjugate_gradient_step(
    params: np.ndarray,
    gradient: np.ndarray,
    prev_direction: Optional[np.ndarray],
    prev_gradient: Optional[np.ndarray],
    learning_rate: float,
    beta_method: str = 'polak_ribiere'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a single conjugate gradient update (Euclidean metric).

    d_k = -∇E + β d_{k-1}
    θ' = θ + η d_k

    Args:
        params: Current parameters
        gradient: Energy gradient ∇E
        prev_direction: Previous search direction (None for first step)
        prev_gradient: Previous gradient (None for first step)
        learning_rate: Step size η
        beta_method: 'polak_ribiere' or 'fletcher_reeves'

    Returns:
        (new_params, direction): Updated parameters and current direction
    """
    if prev_direction is None or prev_gradient is None:
        # First iteration: steepest descent
        direction = -gradient
    else:
        # Compute beta
        if beta_method == 'polak_ribiere':
            beta = compute_beta_polak_ribiere(gradient, prev_gradient)
        else:
            beta = compute_beta_fletcher_reeves(gradient, prev_gradient)

        # Conjugate direction
        direction = -gradient + beta * prev_direction

    new_params = params + learning_rate * direction
    return new_params, direction


def natural_conjugate_gradient_step(
    params: np.ndarray,
    gradient: np.ndarray,
    qfi_matrix: np.ndarray,
    prev_direction: Optional[np.ndarray],
    prev_gradient: Optional[np.ndarray],
    prev_natural_grad: Optional[np.ndarray],
    learning_rate: float,
    regularization: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a single natural conjugate gradient update (Fisher metric).

    Natural CG builds directions conjugate with respect to the QFI metric:
    d_k = -F⁻¹∇E + β d_{k-1}

    This combines:
    - Geometry awareness (natural gradient: F⁻¹∇E)
    - History awareness (conjugate directions)

    Args:
        params: Current parameters
        gradient: Energy gradient ∇E
        qfi_matrix: Quantum Fisher Information matrix F
        prev_direction: Previous search direction (None for first step)
        prev_gradient: Previous gradient (None for first step)
        prev_natural_grad: Previous F⁻¹∇E (None for first step)
        learning_rate: Step size η
        regularization: Tikhonov regularization for QFI inversion

    Returns:
        (new_params, direction, natural_grad): Updated params, direction, and natural gradient
    """
    n_params = len(params)

    # Regularize QFI for numerical stability
    qfi_reg = qfi_matrix + regularization * np.eye(n_params)

    # Compute natural gradient: F⁻¹∇E
    try:
        natural_grad = np.linalg.solve(qfi_reg, gradient)
    except np.linalg.LinAlgError:
        natural_grad = np.linalg.lstsq(qfi_reg, gradient, rcond=None)[0]

    if prev_direction is None or prev_gradient is None or prev_natural_grad is None:
        # First iteration: natural gradient descent
        direction = -natural_grad
    else:
        # Polak-Ribière variant for natural CG
        # β = (∇E_k - ∇E_{k-1})ᵀ F⁻¹∇E_k / (∇E_{k-1}ᵀ F⁻¹∇E_{k-1})
        delta_grad = gradient - prev_gradient
        numerator = np.dot(delta_grad, natural_grad)
        denominator = np.dot(prev_gradient, prev_natural_grad)

        if np.abs(denominator) < 1e-12:
            beta = 0.0
        else:
            beta = max(0.0, numerator / denominator)

        # Conjugate direction with respect to Fisher metric
        direction = -natural_grad + beta * prev_direction

    new_params = params + learning_rate * direction
    return new_params, direction, natural_grad


def conjugate_gradient_descent(
    energy_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    epsilon: float = 1e-5,
    beta_method: str = 'polak_ribiere',
    restart_every: Optional[int] = None,
    verbose: bool = False
) -> OptimizationResult:
    """
    Optimize parameters using conjugate gradient descent (Euclidean metric).

    CG builds search directions that are conjugate with respect to the
    Hessian, leading to faster convergence than steepest descent.

    For quadratic functions, CG converges in at most n iterations.

    Args:
        energy_fn: Function mapping parameters to energy
        initial_params: Starting parameters
        learning_rate: Step size
        max_iterations: Maximum iterations
        convergence_tol: Stop when |∇E| < tol
        epsilon: Finite difference step size
        beta_method: 'polak_ribiere' or 'fletcher_reeves'
        restart_every: Reset CG directions every N iterations (None = no restart)
        verbose: Print progress

    Returns:
        OptimizationResult with final parameters and history
    """
    params = initial_params.copy()
    energies = [energy_fn(params)]
    param_history = [params.copy()]
    converged = False

    prev_direction = None
    prev_gradient = None

    for iteration in range(max_iterations):
        # Compute gradient
        gradient = compute_gradient(energy_fn, params, epsilon)

        # Check convergence
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < convergence_tol:
            converged = True
            if verbose:
                print(f"CG converged at iteration {iteration}: |∇E| = {grad_norm:.2e}")
            break

        # Check for restart
        if restart_every is not None and iteration > 0 and iteration % restart_every == 0:
            prev_direction = None
            prev_gradient = None
            if verbose:
                print(f"CG restart at iteration {iteration}")

        # Conjugate gradient step
        params, direction = conjugate_gradient_step(
            params, gradient, prev_direction, prev_gradient,
            learning_rate, beta_method
        )

        # Update history for next iteration
        prev_direction = direction.copy()
        prev_gradient = gradient.copy()

        # Record energy
        energy = energy_fn(params)
        energies.append(energy)
        param_history.append(params.copy())

        if verbose and iteration % 10 == 0:
            print(f"CG Iter {iteration}: E = {energy:.6f}, |∇E| = {grad_norm:.2e}")

    return OptimizationResult(
        params=params,
        energies=energies,
        param_history=param_history,
        converged=converged,
        iterations=len(energies) - 1
    )


def natural_conjugate_gradient_descent(
    state_fn: Callable[[np.ndarray], QuantumState],
    energy_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    regularization: float = 1e-4,
    epsilon: float = 1e-5,
    restart_every: Optional[int] = None,
    verbose: bool = False
) -> OptimizationResult:
    """
    Optimize parameters using natural conjugate gradient descent.

    Combines:
    - Geometry awareness: Uses F⁻¹∇E (natural gradient) as base direction
    - History awareness: Builds F-conjugate directions

    This is the optimal choice when:
    - The parameter space has significant curvature (QFI varies)
    - The condition number κ is moderate to large

    Args:
        state_fn: Function mapping parameters to quantum state
        energy_fn: Function mapping parameters to energy
        initial_params: Starting parameters
        learning_rate: Step size
        max_iterations: Maximum iterations
        convergence_tol: Stop when |∇E| < tol
        regularization: Tikhonov regularization for QFI
        epsilon: Finite difference step size
        restart_every: Reset CG directions every N iterations
        verbose: Print progress

    Returns:
        OptimizationResult with final parameters and history
    """
    params = initial_params.copy()
    energies = [energy_fn(params)]
    param_history = [params.copy()]
    converged = False

    prev_direction = None
    prev_gradient = None
    prev_natural_grad = None

    for iteration in range(max_iterations):
        # Compute gradient
        gradient = compute_gradient(energy_fn, params, epsilon)

        # Check convergence
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < convergence_tol:
            converged = True
            if verbose:
                print(f"Natural CG converged at iteration {iteration}: |∇E| = {grad_norm:.2e}")
            break

        # Compute QFI matrix
        qfi = quantum_fisher_information_matrix(state_fn, params, epsilon)

        # Check for restart
        if restart_every is not None and iteration > 0 and iteration % restart_every == 0:
            prev_direction = None
            prev_gradient = None
            prev_natural_grad = None
            if verbose:
                print(f"Natural CG restart at iteration {iteration}")

        # Natural conjugate gradient step
        params, direction, natural_grad = natural_conjugate_gradient_step(
            params, gradient, qfi,
            prev_direction, prev_gradient, prev_natural_grad,
            learning_rate, regularization
        )

        # Update history for next iteration
        prev_direction = direction.copy()
        prev_gradient = gradient.copy()
        prev_natural_grad = natural_grad.copy()

        # Record energy
        energy = energy_fn(params)
        energies.append(energy)
        param_history.append(params.copy())

        if verbose and iteration % 10 == 0:
            print(f"Natural CG Iter {iteration}: E = {energy:.6f}, |∇E| = {grad_norm:.2e}")

    # Compute final QFI for diagnostics
    final_qfi = quantum_fisher_information_matrix(state_fn, params, epsilon)

    return OptimizationResult(
        params=params,
        energies=energies,
        param_history=param_history,
        converged=converged,
        iterations=len(energies) - 1,
        final_qfi=final_qfi
    )


# ============================================================================
# Optimizer Class Interface
# ============================================================================

class NaturalConjugateGradient:
    """
    Conjugate gradient optimizer with Fisher metric.

    Combines:
    - Geometry awareness (natural gradient)
    - History awareness (conjugate directions)
    - Spectral diagnostics (ζ-analysis of QFI)

    Usage:
        optimizer = NaturalConjugateGradient(compute_qfi_fn)
        for i in range(max_iter):
            grad = compute_gradient(params)
            params = optimizer.step(params, grad, learning_rate)
    """

    def __init__(
        self,
        compute_qfi_fn: Callable[[np.ndarray], np.ndarray],
        regularization: float = 1e-4
    ):
        """
        Initialize natural CG optimizer.

        Args:
            compute_qfi_fn: Function (params) -> QFI matrix
            regularization: Tikhonov regularization for QFI inversion
        """
        self.compute_qfi = compute_qfi_fn
        self.regularization = regularization
        self.reset()

    def reset(self):
        """Reset conjugate directions (e.g., after restart)."""
        self.prev_direction = None
        self.prev_gradient = None
        self.prev_natural_grad = None
        self._last_qfi = None

    def step(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
        learning_rate: float = 0.1
    ) -> np.ndarray:
        """
        Perform one natural CG step.

        Args:
            params: Current parameters
            gradient: Energy gradient
            learning_rate: Step size

        Returns:
            Updated parameters
        """
        qfi = self.compute_qfi(params)
        self._last_qfi = qfi

        n_params = len(params)
        qfi_reg = qfi + self.regularization * np.eye(n_params)

        # Compute natural gradient
        try:
            natural_grad = np.linalg.solve(qfi_reg, gradient)
        except np.linalg.LinAlgError:
            natural_grad = np.linalg.lstsq(qfi_reg, gradient, rcond=None)[0]

        # Compute direction
        if self.prev_direction is None:
            direction = -natural_grad
        else:
            # Polak-Ribière with Fisher metric
            delta_grad = gradient - self.prev_gradient
            numerator = np.dot(delta_grad, natural_grad)
            denominator = np.dot(self.prev_gradient, self.prev_natural_grad)

            if np.abs(denominator) < 1e-12:
                beta = 0.0
            else:
                beta = max(0.0, numerator / denominator)

            direction = -natural_grad + beta * self.prev_direction

        # Store for next iteration
        self.prev_direction = direction.copy()
        self.prev_gradient = gradient.copy()
        self.prev_natural_grad = natural_grad.copy()

        return params + learning_rate * direction

    def spectral_diagnostics(self, params: np.ndarray) -> SpectralDiagnostics:
        """
        Pre-optimization analysis using ζ-machinery.

        Computes spectral invariants that predict optimization behavior.

        Args:
            params: Parameter point for QFI evaluation

        Returns:
            SpectralDiagnostics with condition number, effective dimension, etc.
        """
        qfi = self.compute_qfi(params)
        return compute_spectral_diagnostics(qfi)

