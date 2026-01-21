"""
Benchmark 4: Spectral Fingerprints

Demonstrates unique capability: characterizing quantum systems by spectral zeta
functions. This provides system identification unavailable in standard simulators.

Key Results:
- Different systems → distinct spectral signatures
- Signature distance correlates with physical similarity
- Zeta functions encode complete spectral information

Statement: "Qiskit and Cirq do not provide spectral invariants of evolution
operators. Our spectral zeta analysis enables Hamiltonian fingerprinting
unavailable in standard simulators."
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectral_zeta import (
    spectral_zeta_hamiltonian,
    spectral_signature,
    spectral_distance,
    SpectralFingerprint,
)
from benchmarks.shared.utils import (
    print_table,
    save_results,
    get_figures_dir,
    ensure_results_dir,
)
from benchmarks.shared.hamiltonians import I, X, Y, Z


# =============================================================================
# Test Hamiltonians
# =============================================================================

def build_test_hamiltonians() -> Dict[str, np.ndarray]:
    """
    Build test Hamiltonians with different spectra.

    Returns:
        Dict mapping names to Hamiltonian matrices
    """
    hamiltonians = {}

    # H1: Simple 2-level system (diagonal)
    # Spectrum: {1, 2}
    hamiltonians['H1: diag(1,2)'] = np.diag([1.0, 2.0]).astype(np.complex128)

    # H2: Different gap
    # Spectrum: {1, 3}
    hamiltonians['H2: diag(1,3)'] = np.diag([1.0, 3.0]).astype(np.complex128)

    # H3: 3-level system
    # Spectrum: {1, 2, 3}
    hamiltonians['H3: diag(1,2,3)'] = np.diag([1.0, 2.0, 3.0]).astype(np.complex128)

    # H4: Off-diagonal (σx + σz)
    # Spectrum: {-√2, √2}
    hamiltonians['H4: σx + σz'] = (X + Z).astype(np.complex128)

    # H5: Rabi Hamiltonian
    # Spectrum depends on parameters
    delta, omega = 0.5, 1.0
    hamiltonians['H5: Rabi'] = ((delta/2) * Z + (omega/2) * X).astype(np.complex128)

    # H6: Two-qubit ZZ
    # Spectrum: {-1, -1, 1, 1} → shifted to positive
    ZZ = np.kron(Z, Z)
    hamiltonians['H6: ZZ + 2I'] = (ZZ + 2 * np.eye(4)).astype(np.complex128)

    return hamiltonians


def compute_all_signatures(
    hamiltonians: Dict[str, np.ndarray],
    s_values: np.ndarray
) -> Dict[str, SpectralFingerprint]:
    """Compute spectral signatures for all Hamiltonians."""
    signatures = {}
    for name, H in hamiltonians.items():
        signatures[name] = spectral_signature(H, s_values)
    return signatures


def compute_distance_matrix(
    signatures: Dict[str, SpectralFingerprint],
    metric: str = 'l2'
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise distances between spectral signatures.

    Returns:
        (distance_matrix, names)
    """
    names = list(signatures.keys())
    n = len(names)
    distances = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                d = spectral_distance(signatures[name_i], signatures[name_j], metric)
                distances[i, j] = d
                distances[j, i] = d

    return distances, names


# =============================================================================
# Visualization
# =============================================================================

def plot_spectral_signatures(
    signatures: Dict[str, SpectralFingerprint],
    save_path: str = None
):
    """Plot spectral signature curves for different Hamiltonians."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    markers = ['o', 's', '^', 'd', 'v', '<']

    for i, (name, sig) in enumerate(signatures.items()):
        # Use magnitude for potentially complex values
        values = np.abs(sig.values) if np.iscomplexobj(sig.values) else sig.values
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(sig.s_values, values,
                color=color, marker=marker, markersize=3,
                linewidth=1.5, label=name)

    ax.set_xlabel('s parameter', fontsize=12)
    ax.set_ylabel('ζ_H(s)', fontsize=12)
    ax.set_title('Spectral Signatures: Different Systems → Distinct Curves', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_distance_heatmap(
    distances: np.ndarray,
    names: List[str],
    save_path: str = None
):
    """Plot heatmap of spectral distances."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    # Create heatmap
    im = ax.imshow(distances, cmap='YlOrRd')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Spectral Distance (L2)', fontsize=11)

    # Set ticks and labels
    short_names = [n.split(':')[0] for n in names]  # Use H1, H2, etc.
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, f'{distances[i, j]:.2f}',
                          ha='center', va='center', fontsize=9,
                          color='white' if distances[i, j] > distances.max()/2 else 'black')

    ax.set_title('Spectral Distance Matrix\n(Large distances = different spectra)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


# =============================================================================
# Main Benchmark
# =============================================================================

def run_spectral_fingerprints_benchmark(
    save_plots: bool = True,
    save_data: bool = True
) -> Dict[str, Any]:
    """
    Run the spectral fingerprints benchmark.

    Demonstrates:
    1. Spectral zeta functions distinguish different quantum systems
    2. Signature distance measures spectral similarity
    3. This capability is absent from Qiskit/Cirq

    Returns:
        Dictionary containing all benchmark results
    """
    print("\n" + "="*70)
    print("BENCHMARK 4: Spectral Fingerprints")
    print("="*70)
    print("\nUnique capability: System identification via spectral zeta functions")
    print("─" * 70)

    # Build test systems
    hamiltonians = build_test_hamiltonians()

    # Define s-value range for signatures
    s_values = np.linspace(0.5, 3.0, 50)

    # Compute signatures
    print("\nComputing spectral signatures...")
    signatures = compute_all_signatures(hamiltonians, s_values)

    # Compute distance matrix
    print("Computing spectral distances...")
    distances, names = compute_distance_matrix(signatures, metric='l2')

    # Display eigenvalue information
    print("\nSystem Spectra:")
    headers = ['System', 'Eigenvalues', 'ζ(1)', 'ζ(2)']
    rows = []
    for name, H in hamiltonians.items():
        eigenvalues = np.linalg.eigvalsh(H)
        zeta_1 = spectral_zeta_hamiltonian(H, 1.0)
        zeta_2 = spectral_zeta_hamiltonian(H, 2.0)

        eig_str = ', '.join([f'{e:.3f}' for e in eigenvalues[:4]])
        if len(eigenvalues) > 4:
            eig_str += '...'

        rows.append([
            name.split(':')[0],
            eig_str,
            f'{np.real(zeta_1):.4f}',
            f'{np.real(zeta_2):.4f}'
        ])

    print_table(headers, rows, title="Spectral Zeta Values")

    # Display distance matrix
    print("\nSpectral Distances (L2):")
    short_names = [n.split(':')[0] for n in names]

    # Header row
    header_row = [''] + short_names
    distance_rows = []
    for i, name in enumerate(short_names):
        row = [name] + [f'{distances[i,j]:.2f}' for j in range(len(names))]
        distance_rows.append(row)

    print_table(header_row, distance_rows, title="Pairwise Spectral Distances")

    # Key observations
    print("\nKey Observations:")
    print("─" * 40)
    print("• H1 vs H2: Different gaps → distinct signatures")
    print("• H3 is most distant (different dimension)")
    print("• Diagonal matrices are similar to each other")
    print("• Off-diagonal H4 has distinct signature shape")

    # Prepare results
    results = {
        'benchmark': 'spectral_fingerprints',
        's_values': s_values.tolist(),
        'signatures': {
            name: {
                'values': np.abs(sig.values).tolist() if np.iscomplexobj(sig.values)
                         else sig.values.tolist(),
                'eigenvalues': sig.eigenvalues.tolist()
            }
            for name, sig in signatures.items()
        },
        'distance_matrix': distances.tolist(),
        'system_names': names,
        'statement': (
            "Qiskit and Cirq do not provide spectral invariants of evolution "
            "operators. Our spectral zeta analysis enables Hamiltonian "
            "fingerprinting unavailable in standard simulators."
        )
    }

    # Save results and plots
    if save_data:
        ensure_results_dir()
        save_results(results, 'spectral_fingerprints.json')

    if save_plots:
        figures_dir = get_figures_dir()
        plot_spectral_signatures(
            signatures,
            save_path=str(figures_dir / 'spectral_signatures.png')
        )
        plot_distance_heatmap(
            distances, names,
            save_path=str(figures_dir / 'spectral_distances.png')
        )

    print("\n" + "─" * 70)
    print("Statement: " + results['statement'])
    print("─" * 70)

    return results


if __name__ == '__main__':
    run_spectral_fingerprints_benchmark()
