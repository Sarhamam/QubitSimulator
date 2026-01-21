"""
Spectral Zeta Functions for Quantum Systems

Implements spectral zeta functions connecting quantum Hamiltonians and
Lindbladians to the NoeticEidos framework via Mellin transforms.

Mathematical Foundation:
    For a Hamiltonian H with eigenvalues {λₙ}:
        ζ_H(s) = Tr(H^{-s}) = Σₙ λₙ^{-s}    (for λₙ > 0)

    Heat kernel relation (Mellin transform):
        ζ_H(s) = (1/Γ(s)) ∫₀^∞ t^{s-1} Tr(e^{-tH}) dt

    For Lindbladian L (non-Hermitian superoperator):
        ζ_L(s) = Σₙ |λₙ|^{-s} e^{-is·arg(λₙ)}

    This encodes both decay rates (|λₙ|) and oscillation frequencies (arg(λₙ)).

Geodesic-Spectral Duality:
    - Ideal geodesic (unitary) ↔ ζ_H poles
    - Actual trajectory (noisy) ↔ ζ_L poles
    - Geodesic deviation = spectral difference

Connection to NoeticEidos:
    The Mellin transform at s=1/2 (critical line) provides the
    "balance point" between additive and multiplicative transport.
    Spectral signatures on this line connect to Fisher-Rao geometry.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
from scipy import integrate
from scipy.special import gamma as gamma_fn

from lindblad import LindbladOperator


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SpectralFingerprint:
    """
    Spectral fingerprint of a quantum system.

    The fingerprint is the spectral zeta function evaluated across
    a range of s values, providing a characteristic "signature"
    that uniquely identifies the spectrum.

    Attributes:
        values: ζ(s) values at each s point
        s_values: The s parameter values used
        eigenvalues: The underlying eigenvalues
        is_lindbladian: Whether this is from a Lindbladian (complex spectrum)
    """
    values: np.ndarray
    s_values: np.ndarray
    eigenvalues: np.ndarray
    is_lindbladian: bool = False


# ============================================================================
# Spectral Zeta for Hamiltonians (Hermitian)
# ============================================================================

def spectral_zeta_hamiltonian(
    H: np.ndarray,
    s: Union[float, complex],
    regularization: float = 1e-12
) -> Union[float, complex]:
    """
    Compute spectral zeta function for a Hermitian Hamiltonian.

    ζ_H(s) = Σₙ λₙ^{-s}  for eigenvalues λₙ > 0

    Args:
        H: Hermitian Hamiltonian matrix
        s: Complex parameter (Re(s) > 0 for convergence with positive eigenvalues)
        regularization: Threshold below which eigenvalues are skipped

    Returns:
        Spectral zeta value ζ_H(s)

    Note:
        Eigenvalues ≤ regularization are skipped to avoid divergence.
        For Hamiltonians with negative eigenvalues, shift H → H + cI first.
    """
    eigenvalues = np.linalg.eigvalsh(H)

    # Filter to positive eigenvalues
    positive_eigenvalues = eigenvalues[eigenvalues > regularization]

    if len(positive_eigenvalues) == 0:
        return 0.0

    # ζ(s) = Σ λ^{-s}
    zeta = np.sum(positive_eigenvalues ** (-s))

    return zeta


# ============================================================================
# Vectorized Lindbladian Superoperator
# ============================================================================

def vectorize_lindbladian(
    H: np.ndarray,
    lindblad_ops: List[LindbladOperator]
) -> np.ndarray:
    """
    Construct the vectorized Lindbladian superoperator.

    For density matrix ρ, the Lindblad equation is:
        dρ/dt = L[ρ] = -i[H, ρ] + Σₖ γₖ D[Lₖ](ρ)

    where D[L](ρ) = LρL† - (1/2){L†L, ρ}

    In vectorized form (vec(ρ) = column-stacking of ρ):
        d(vec(ρ))/dt = L_matrix @ vec(ρ)

    where L_matrix is a d²×d² matrix.

    Args:
        H: System Hamiltonian (d×d)
        lindblad_ops: List of Lindblad operators with rates

    Returns:
        L_matrix: Vectorized Lindbladian (d²×d²)

    Note:
        Uses the identity: vec(AXB) = (B^T ⊗ A) vec(X)
    """
    d = H.shape[0]
    d_squared = d * d

    # Identity matrix
    I = np.eye(d, dtype=np.complex128)

    # Hamiltonian contribution: -i[H, ρ] = -i(Hρ - ρH)
    # vec(-iHρ) = -i(I ⊗ H) vec(ρ)
    # vec(iρH) = i(H^T ⊗ I) vec(ρ)
    L_H = -1j * (np.kron(I, H) - np.kron(H.T, I))

    L_matrix = L_H.astype(np.complex128)

    # Dissipator contributions
    for lop in lindblad_ops:
        L = lop.operator
        gamma = lop.rate
        L_dag = L.conj().T
        L_dag_L = L_dag @ L

        # D[L](ρ) = LρL† - (1/2){L†L, ρ}
        # = LρL† - (1/2)L†Lρ - (1/2)ρL†L

        # vec(LρL†) = (L̄ ⊗ L) vec(ρ)  where L̄ = conj(L)
        term1 = np.kron(L.conj(), L)

        # vec(L†Lρ) = (I ⊗ L†L) vec(ρ)
        term2 = np.kron(I, L_dag_L)

        # vec(ρL†L) = (L†L)^T ⊗ I) vec(ρ) = ((L†L)^T ⊗ I) vec(ρ)
        term3 = np.kron(L_dag_L.T, I)

        L_matrix += gamma * (term1 - 0.5 * term2 - 0.5 * term3)

    return L_matrix


def lindbladian_spectrum(
    H: np.ndarray,
    lindblad_ops: List[LindbladOperator]
) -> np.ndarray:
    """
    Compute the spectrum of the Lindbladian superoperator.

    Args:
        H: System Hamiltonian
        lindblad_ops: Lindblad operators

    Returns:
        Array of (generally complex) eigenvalues
    """
    L_matrix = vectorize_lindbladian(H, lindblad_ops)
    eigenvalues = np.linalg.eigvals(L_matrix)
    return eigenvalues


# ============================================================================
# Spectral Zeta for Lindbladians (Non-Hermitian)
# ============================================================================

def spectral_zeta_lindbladian(
    H: np.ndarray,
    lindblad_ops: List[LindbladOperator],
    s: Union[float, complex],
    regularization: float = 1e-12
) -> complex:
    """
    Compute spectral zeta for a Lindbladian superoperator.

    For non-Hermitian operator with complex eigenvalues {λₙ}:
        ζ_L(s) = Σₙ |λₙ|^{-s} e^{-is·arg(λₙ)}

    This encodes both:
        - Decay rates: |λₙ| (modulus)
        - Oscillation frequencies: arg(λₙ) (phase)

    Args:
        H: System Hamiltonian
        lindblad_ops: Lindblad operators
        s: Complex parameter
        regularization: Threshold for skipping small eigenvalues

    Returns:
        Complex spectral zeta value
    """
    spectrum = lindbladian_spectrum(H, lindblad_ops)

    zeta = 0.0 + 0.0j

    for lam in spectrum:
        abs_lam = np.abs(lam)
        if abs_lam > regularization:
            # |λ|^{-s} * e^{-is·arg(λ)}
            arg_lam = np.angle(lam)
            zeta += (abs_lam ** (-s)) * np.exp(-1j * s * arg_lam)

    return zeta


# ============================================================================
# Heat Kernel Trace
# ============================================================================

def heat_kernel_trace(
    H: np.ndarray,
    t: float,
    lindblad_ops: Optional[List[LindbladOperator]] = None
) -> Union[float, complex]:
    """
    Compute the heat kernel trace K(t) = Tr(e^{-tH}) or Tr(e^{tL}).

    For Hamiltonian H:
        K(t) = Σₙ e^{-t·λₙ}

    For Lindbladian L:
        K(t) = Σₙ e^{t·λₙ}  (note: Re(λₙ) ≤ 0 for Lindbladian)

    The heat kernel is the Mellin partner of spectral zeta:
        ζ(s) = (1/Γ(s)) ∫₀^∞ t^{s-1} K(t) dt

    Args:
        H: Hamiltonian matrix
        t: Time parameter (t ≥ 0)
        lindblad_ops: If provided, compute for Lindbladian instead

    Returns:
        Heat kernel trace value
    """
    if lindblad_ops is not None:
        # Lindbladian case
        spectrum = lindbladian_spectrum(H, lindblad_ops)
        # For Lindbladian: e^{tL} decays because Re(λ) ≤ 0
        return np.sum(np.exp(t * spectrum))
    else:
        # Hamiltonian case
        eigenvalues = np.linalg.eigvalsh(H)
        return np.sum(np.exp(-t * eigenvalues))


def mellin_from_heat_kernel(
    H: np.ndarray,
    s: Union[float, complex],
    t_max: float = 50.0,
    lindblad_ops: Optional[List[LindbladOperator]] = None
) -> Union[float, complex]:
    """
    Compute spectral zeta via Mellin transform of heat kernel.

    ζ(s) = (1/Γ(s)) ∫₀^∞ t^{s-1} K(t) dt

    This provides an alternative computation path that connects
    to the additive-multiplicative duality in NoeticEidos.

    Args:
        H: Hamiltonian matrix
        s: Complex parameter (Re(s) > 0)
        t_max: Upper integration limit
        lindblad_ops: For Lindbladian computation

    Returns:
        Spectral zeta approximation via Mellin transform
    """
    def integrand_real(t):
        if t < 1e-10:
            return 0.0
        K_t = heat_kernel_trace(H, t, lindblad_ops)
        if lindblad_ops is not None:
            K_t = np.real(K_t)  # Take real part for integration
        return (t ** (np.real(s) - 1)) * K_t

    # Numerical integration
    result, _ = integrate.quad(integrand_real, 0, t_max, limit=100)

    # Divide by Γ(s)
    gamma_s = gamma_fn(s)
    if np.abs(gamma_s) < 1e-12:
        return np.inf

    return result / gamma_s


# ============================================================================
# Spectral Signatures and Fingerprinting
# ============================================================================

def spectral_signature(
    H: np.ndarray,
    s_values: np.ndarray,
    lindblad_ops: Optional[List[LindbladOperator]] = None
) -> SpectralFingerprint:
    """
    Compute spectral signature (fingerprint) of a quantum system.

    The signature is ζ(s) evaluated across a range of s values,
    providing a characteristic curve that uniquely identifies the spectrum.

    This connects to NoeticEidos spectral signatures:
    - Different systems with same spectrum → identical signature
    - Signature distance measures spectral similarity

    Args:
        H: Hamiltonian matrix
        s_values: Array of s parameter values to evaluate
        lindblad_ops: If provided, compute Lindbladian signature

    Returns:
        SpectralFingerprint containing signature values
    """
    if lindblad_ops is not None:
        # Lindbladian signature
        spectrum = lindbladian_spectrum(H, lindblad_ops)
        values = np.array([
            spectral_zeta_lindbladian(H, lindblad_ops, s)
            for s in s_values
        ])
        return SpectralFingerprint(
            values=values,
            s_values=s_values,
            eigenvalues=spectrum,
            is_lindbladian=True
        )
    else:
        # Hamiltonian signature
        eigenvalues = np.linalg.eigvalsh(H)
        values = np.array([
            spectral_zeta_hamiltonian(H, s)
            for s in s_values
        ])
        return SpectralFingerprint(
            values=values,
            s_values=s_values,
            eigenvalues=eigenvalues,
            is_lindbladian=False
        )


def spectral_distance(
    sig1: SpectralFingerprint,
    sig2: SpectralFingerprint,
    metric: str = 'l2'
) -> float:
    """
    Compute distance between two spectral signatures.

    This measures how "different" two quantum systems are spectrally,
    connecting to geodesic deviation in state space.

    Args:
        sig1: First spectral fingerprint
        sig2: Second spectral fingerprint
        metric: Distance metric ('l2', 'fisher_rao', 'js')

    Returns:
        Non-negative distance value
    """
    # Extract values (use magnitude for complex)
    v1 = np.abs(sig1.values) if np.iscomplexobj(sig1.values) else sig1.values
    v2 = np.abs(sig2.values) if np.iscomplexobj(sig2.values) else sig2.values

    if metric == 'l2':
        # L2 norm
        return np.linalg.norm(v1 - v2)

    elif metric == 'fisher_rao':
        # Fisher-Rao distance (treating normalized signature as distribution)
        # Normalize to probability distributions
        p1 = v1 / (np.sum(v1) + 1e-12)
        p2 = v2 / (np.sum(v2) + 1e-12)

        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(p1 * p2))
        bc = np.clip(bc, 0, 1)

        # Fisher-Rao distance
        return np.arccos(bc)

    elif metric == 'js':
        # Jensen-Shannon distance
        p1 = v1 / (np.sum(v1) + 1e-12)
        p2 = v2 / (np.sum(v2) + 1e-12)
        m = (p1 + p2) / 2

        # KL divergences (with regularization)
        kl_pm = np.sum(p1 * np.log((p1 + 1e-12) / (m + 1e-12)))
        kl_qm = np.sum(p2 * np.log((p2 + 1e-12) / (m + 1e-12)))

        return np.sqrt((kl_pm + kl_qm) / 2)

    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# Physical Interpretation Layer
# ============================================================================

@dataclass
class LindbladianDecomposition:
    """
    Physical interpretation of Lindbladian spectrum.

    Attributes:
        decay_rates: Real parts of eigenvalues (energy dissipation rates)
        oscillation_freqs: Imaginary parts (coherent oscillation frequencies)
        steady_state_eigenvalue: The eigenvalue closest to zero
        gap: Spectral gap (smallest non-zero |Re(λ)|)
    """
    decay_rates: np.ndarray
    oscillation_freqs: np.ndarray
    steady_state_eigenvalue: complex
    gap: float


def decompose_lindbladian_spectrum(
    H: np.ndarray,
    lindblad_ops: List[LindbladOperator]
) -> LindbladianDecomposition:
    """
    Extract physical quantities from Lindbladian spectrum.

    The Lindbladian eigenvalues encode:
    - Re(λ): Decay rate (should be ≤ 0)
    - Im(λ): Oscillation frequency

    The spectral gap determines the timescale of approach to steady state.

    Args:
        H: System Hamiltonian
        lindblad_ops: Lindblad operators

    Returns:
        LindbladianDecomposition with physical quantities
    """
    spectrum = lindbladian_spectrum(H, lindblad_ops)

    decay_rates = -np.real(spectrum)  # Negate so rates are positive
    oscillation_freqs = np.imag(spectrum)

    # Find eigenvalue closest to zero (steady state)
    abs_eigenvalues = np.abs(spectrum)
    ss_idx = np.argmin(abs_eigenvalues)
    steady_state_eigenvalue = spectrum[ss_idx]

    # Spectral gap: smallest non-zero decay rate
    nonzero_decays = decay_rates[decay_rates > 1e-10]
    gap = np.min(nonzero_decays) if len(nonzero_decays) > 0 else 0.0

    return LindbladianDecomposition(
        decay_rates=decay_rates,
        oscillation_freqs=oscillation_freqs,
        steady_state_eigenvalue=steady_state_eigenvalue,
        gap=gap
    )


# ============================================================================
# Geodesic-Spectral Connection
# ============================================================================

def spectral_geodesic_deviation(
    H: np.ndarray,
    lindblad_ops: List[LindbladOperator],
    s_values: np.ndarray
) -> Tuple[float, SpectralFingerprint, SpectralFingerprint]:
    """
    Compute geodesic deviation in spectral terms.

    The geodesic deviation between ideal (unitary) and actual (noisy)
    evolution is reflected in the difference between Hamiltonian and
    Lindbladian spectral signatures.

    This connects the Fubini-Study geodesic deviation (in state space)
    to spectral structure (in operator space).

    Args:
        H: System Hamiltonian
        lindblad_ops: Noise operators
        s_values: Range of s values for signature

    Returns:
        (deviation, ideal_signature, noisy_signature)
    """
    sig_ideal = spectral_signature(H, s_values)
    sig_noisy = spectral_signature(H, s_values, lindblad_ops=lindblad_ops)

    deviation = spectral_distance(sig_ideal, sig_noisy)

    return deviation, sig_ideal, sig_noisy
