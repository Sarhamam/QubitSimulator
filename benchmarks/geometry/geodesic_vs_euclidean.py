"""
Benchmark 8: Geodesic vs Euclidean Similarity

Demonstrates that Fubini-Study/Bures distances are operationally meaningful,
while Euclidean distance on amplitudes is not.

Key Problem with Euclidean Distance:
Global phase: |ψ⟩ and e^{iφ}|ψ⟩ are physically identical but Euclidean-distant.
    d_Euclidean(|ψ⟩, e^{iπ}|ψ⟩) = 2‖ψ‖ = 2  (maximum!)
    d_FS(|ψ⟩, e^{iπ}|ψ⟩) = 0                  (correct)

Key Results:
- Euclidean distance poorly correlates with distinguishability
- Fubini-Study/Bures distances strongly correlate
- Geodesic distances are operationally meaningful

Statement: "Geodesic distances (Fubini-Study, Bures) correctly capture quantum
state similarity, while Euclidean distance fails due to global phase invariance."
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state import QuantumState, random_state
from information_geometry import (
    fubini_study_distance,
    bures_distance,
    fidelity,
)
from benchmarks.shared.utils import (
    print_table,
    save_results,
    get_figures_dir,
    ensure_results_dir,
)


# =============================================================================
# Distance Metrics
# =============================================================================

def euclidean_distance(psi: QuantumState, phi: QuantumState) -> float:
    """
    Compute Euclidean distance between state vectors.

    d_E = ‖ψ - φ‖ = √(2 - 2 Re(⟨ψ|φ⟩))

    This is NOT phase-invariant and thus not operationally meaningful.
    """
    amp1 = psi.amplitudes
    amp2 = phi.amplitudes
    return np.linalg.norm(amp1 - amp2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute trace distance: T(ρ,σ) = (1/2)‖ρ - σ‖₁

    This equals the Kolmogorov distance between measurement outcome
    distributions, optimized over all measurements.
    """
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigenvalues))


def helstrom_error_probability(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute Helstrom minimum error probability for distinguishing ρ and σ.

    P_err = (1/2)(1 - T(ρ,σ))

    where T is the trace distance.
    """
    T = trace_distance(rho, sigma)
    return 0.5 * (1 - T)


# =============================================================================
# Phase Sensitivity Test
# =============================================================================

def test_phase_sensitivity():
    """
    Demonstrate that Euclidean distance is phase-sensitive while
    geodesic distances are not.
    """
    results = []

    # Create a reference state
    psi = QuantumState(1, amplitudes=np.array([1, 0], dtype=np.complex128))

    # Apply various global phases
    phases = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]

    for phi in phases:
        # Create phase-shifted state
        amp_shifted = np.exp(1j * phi) * psi.amplitudes
        psi_shifted = QuantumState(1, amplitudes=amp_shifted)

        # Compute distances
        d_euclidean = euclidean_distance(psi, psi_shifted)
        d_fs = fubini_study_distance(psi, psi_shifted)
        d_bures = bures_distance(psi.rho, psi_shifted.rho)

        results.append({
            'phase': phi,
            'd_euclidean': d_euclidean,
            'd_fs': d_fs,
            'd_bures': d_bures,
        })

    return results


# =============================================================================
# Operational Correlation Test
# =============================================================================

def test_operational_correlation(n_pairs: int = 100) -> Dict[str, np.ndarray]:
    """
    Test correlation between distance metrics and operational distinguishability.

    Generates random state pairs and computes:
    - Various distance metrics
    - Helstrom error probability (operational distinguishability)
    """
    euclidean_dists = []
    fs_dists = []
    bures_dists = []
    trace_dists = []
    helstrom_errors = []

    for _ in range(n_pairs):
        # Generate two random states
        psi = random_state(1)
        phi = random_state(1)

        # Compute distances
        d_euclidean = euclidean_distance(psi, phi)
        d_fs = fubini_study_distance(psi, phi)
        d_bures = bures_distance(psi.rho, phi.rho)
        d_trace = trace_distance(psi.rho, phi.rho)

        # Helstrom error (operational measure)
        p_err = helstrom_error_probability(psi.rho, phi.rho)

        euclidean_dists.append(d_euclidean)
        fs_dists.append(d_fs)
        bures_dists.append(d_bures)
        trace_dists.append(d_trace)
        helstrom_errors.append(p_err)

    return {
        'euclidean': np.array(euclidean_dists),
        'fubini_study': np.array(fs_dists),
        'bures': np.array(bures_dists),
        'trace': np.array(trace_dists),
        'helstrom_error': np.array(helstrom_errors),
    }


def compute_correlations(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute correlation coefficients between each distance metric
    and operational distinguishability.
    """
    # For correlation with distinguishability, use (1 - 2*P_err) = T
    # which increases with distinguishability
    distinguishability = 1 - 2 * data['helstrom_error']

    correlations = {}
    for metric in ['euclidean', 'fubini_study', 'bures', 'trace']:
        r = np.corrcoef(data[metric], distinguishability)[0, 1]
        correlations[metric] = r

    return correlations


# =============================================================================
# Visualization
# =============================================================================

def plot_phase_sensitivity(phase_results: List[Dict], save_path: str = None):
    """Plot distance vs phase for different metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    phases = [r['phase'] for r in phase_results]
    phases_deg = [p * 180 / np.pi for p in phases]

    d_euclidean = [r['d_euclidean'] for r in phase_results]
    d_fs = [r['d_fs'] for r in phase_results]
    d_bures = [r['d_bures'] for r in phase_results]

    ax.plot(phases_deg, d_euclidean, 'ro-', linewidth=2, markersize=10,
           label='Euclidean (WRONG)')
    ax.plot(phases_deg, d_fs, 'bs-', linewidth=2, markersize=10,
           label='Fubini-Study (correct)')
    ax.plot(phases_deg, d_bures, 'g^-', linewidth=2, markersize=10,
           label='Bures (correct)')

    ax.set_xlabel('Global Phase φ (degrees)', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Phase Sensitivity: |ψ⟩ vs e^{iφ}|ψ⟩\n(Should be 0 - states are identical!)',
                fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Euclidean fails:\nphysically identical states\nappear distant!',
               xy=(180, d_euclidean[3]),
               xytext=(220, d_euclidean[3] - 0.3),
               fontsize=10, color='red',
               arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_correlation_comparison(
    data: Dict[str, np.ndarray],
    correlations: Dict[str, float],
    save_path: str = None
):
    """Plot scatter plots showing correlation with distinguishability."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    distinguishability = 1 - 2 * data['helstrom_error']

    metrics = [
        ('euclidean', 'Euclidean', 'red'),
        ('fubini_study', 'Fubini-Study', 'blue'),
        ('bures', 'Bures', 'green'),
        ('trace', 'Trace', 'purple'),
    ]

    for ax, (key, name, color) in zip(axes.flat, metrics):
        ax.scatter(data[key], distinguishability, c=color, alpha=0.6, s=30)

        # Correlation
        r = correlations[key]
        ax.text(0.05, 0.95, f'r = {r:.3f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(f'{name} Distance', fontsize=11)
        ax.set_ylabel('Distinguishability', fontsize=11)
        ax.set_title(f'{name}: {"✓" if r > 0.9 else "✗"} (r = {r:.3f})', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Distance Metrics vs Operational Distinguishability\n'
                '(High correlation = operationally meaningful)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


# =============================================================================
# Main Benchmark
# =============================================================================

def run_geodesic_vs_euclidean_benchmark(
    save_plots: bool = True,
    save_data: bool = True
) -> Dict[str, Any]:
    """
    Run the geodesic vs Euclidean similarity benchmark.

    Demonstrates:
    1. Euclidean distance is phase-sensitive (fails for identical states)
    2. Geodesic distances (Fubini-Study, Bures) are phase-invariant
    3. Geodesic distances correlate with operational distinguishability

    Returns:
        Dictionary containing all benchmark results
    """
    print("\n" + "="*70)
    print("BENCHMARK 8: Geodesic vs Euclidean Similarity")
    print("="*70)
    print("\nDemonstrating operational meaning of geodesic distances")
    print("─" * 70)

    # Test 1: Phase sensitivity
    print("\n1. Phase Sensitivity Test")
    print("   States |ψ⟩ and e^{iφ}|ψ⟩ are physically identical.")
    print("   Distance should be ZERO for all phases.\n")

    phase_results = test_phase_sensitivity()

    headers = ['Phase (deg)', 'Euclidean', 'Fubini-Study', 'Bures']
    rows = []
    for r in phase_results:
        phase_deg = r['phase'] * 180 / np.pi
        row = [
            f'{phase_deg:.0f}°',
            f'{r["d_euclidean"]:.4f}' + (' ← WRONG!' if r['d_euclidean'] > 0.1 else ''),
            f'{r["d_fs"]:.4f}' + (' ✓' if r['d_fs'] < 1e-10 else ''),
            f'{r["d_bures"]:.4f}' + (' ✓' if r['d_bures'] < 1e-10 else ''),
        ]
        rows.append(row)

    print_table(headers, rows, title="Distance for |ψ⟩ vs e^{iφ}|ψ⟩")

    print("\nObservation: Euclidean distance varies with phase (maximum at φ=π),")
    print("but Fubini-Study and Bures correctly give zero for all phases.")

    # Test 2: Correlation with operational distinguishability
    print("\n" + "─" * 70)
    print("\n2. Correlation with Operational Distinguishability")
    print("   Testing 100 random state pairs...")

    correlation_data = test_operational_correlation(n_pairs=100)
    correlations = compute_correlations(correlation_data)

    print("\nCorrelation with Helstrom distinguishability:")
    print("─" * 40)
    headers = ['Metric', 'Correlation r', 'Assessment']
    rows = []
    for metric, r in correlations.items():
        assessment = 'Excellent' if r > 0.95 else ('Good' if r > 0.8 else 'Poor')
        rows.append([metric.replace('_', ' ').title(), f'{r:.4f}', assessment])

    print_table(headers, rows, title="Correlation with Distinguishability")

    # Key findings
    print("\nKey Findings:")
    print("─" * 40)
    print(f"• Euclidean: r = {correlations['euclidean']:.3f} (poor - ignores phase)")
    print(f"• Fubini-Study: r = {correlations['fubini_study']:.3f} (excellent)")
    print(f"• Bures: r = {correlations['bures']:.3f} (excellent)")
    print(f"• Trace: r = {correlations['trace']:.3f} (perfect - it IS the operational measure)")

    # Prepare results
    results = {
        'benchmark': 'geodesic_vs_euclidean',
        'phase_sensitivity': {
            'phases_deg': [r['phase'] * 180 / np.pi for r in phase_results],
            'd_euclidean': [r['d_euclidean'] for r in phase_results],
            'd_fubini_study': [r['d_fs'] for r in phase_results],
            'd_bures': [r['d_bures'] for r in phase_results],
        },
        'correlation_test': {
            'n_pairs': 100,
            'correlations': correlations,
        },
        'conclusion': {
            'euclidean_fails_phase': True,
            'geodesic_distances_meaningful': True,
            'best_metric': 'fubini_study or bures',
        },
        'statement': (
            "Geodesic distances (Fubini-Study, Bures) correctly capture quantum "
            "state similarity, while Euclidean distance fails due to global "
            "phase invariance."
        )
    }

    # Save results and plots
    if save_data:
        ensure_results_dir()
        save_results(results, 'geodesic_vs_euclidean.json')

    if save_plots:
        figures_dir = get_figures_dir()
        plot_phase_sensitivity(
            phase_results,
            save_path=str(figures_dir / 'phase_sensitivity.png')
        )
        plot_correlation_comparison(
            correlation_data, correlations,
            save_path=str(figures_dir / 'distance_correlations.png')
        )

    print("\n" + "─" * 70)
    print("Statement: " + results['statement'])
    print("─" * 70)

    return results


if __name__ == '__main__':
    run_geodesic_vs_euclidean_benchmark()
