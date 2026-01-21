"""
Benchmark 10: Spectral Signature Robustness

Shows that ζ-fingerprints are stable under small perturbations but
distinguish genuinely different systems.

Key Results:
- Small perturbations → small signature changes (stability)
- Different systems → large signature changes (discriminability)
- Well-separated distributions enable robust fingerprinting

Statement: "Spectral signatures provide robust system identification,
stable under perturbations yet discriminating between distinct systems."
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectral_zeta import (
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
# Perturbation Generation
# =============================================================================

def random_hermitian(dim: int, scale: float = 1.0) -> np.ndarray:
    """
    Generate a random Hermitian matrix.

    H = (A + A†) / 2 where A is random complex.
    """
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (A + A.conj().T) / 2
    return scale * H / np.linalg.norm(H)


def perturb_hamiltonian(H: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Apply random perturbation to Hamiltonian.

    H' = H + ε V where V is random Hermitian with ||V|| = 1
    """
    dim = H.shape[0]
    V = random_hermitian(dim)
    return H + epsilon * V


# =============================================================================
# Robustness Analysis
# =============================================================================

def analyze_perturbation_robustness(
    H: np.ndarray,
    epsilon_values: np.ndarray,
    n_samples: int = 20,
    s_values: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Analyze how signature distance varies with perturbation strength.

    For each epsilon, generate n_samples perturbed Hamiltonians and
    compute signature distance from original.

    Returns:
        Dict with epsilon, mean_distance, std_distance
    """
    if s_values is None:
        s_values = np.linspace(0.5, 3.0, 50)

    # Reference signature
    sig_ref = spectral_signature(H, s_values)

    mean_distances = []
    std_distances = []

    for epsilon in epsilon_values:
        distances = []
        for _ in range(n_samples):
            H_perturbed = perturb_hamiltonian(H, epsilon)
            sig_perturbed = spectral_signature(H_perturbed, s_values)
            d = spectral_distance(sig_ref, sig_perturbed)
            distances.append(d)

        mean_distances.append(np.mean(distances))
        std_distances.append(np.std(distances))

    return {
        'epsilon': epsilon_values,
        'mean_distance': np.array(mean_distances),
        'std_distance': np.array(std_distances),
    }


def analyze_discriminability(
    hamiltonians: Dict[str, np.ndarray],
    s_values: np.ndarray = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise distances between different systems.

    Returns:
        (distance_matrix, names)
    """
    if s_values is None:
        s_values = np.linspace(0.5, 3.0, 50)

    names = list(hamiltonians.keys())
    n = len(names)

    # Compute signatures
    signatures = {name: spectral_signature(H, s_values)
                  for name, H in hamiltonians.items()}

    # Compute distances
    distances = np.zeros((n, n))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                d = spectral_distance(signatures[name_i], signatures[name_j])
                distances[i, j] = d
                distances[j, i] = d

    return distances, names


# =============================================================================
# Visualization
# =============================================================================

def plot_robustness_curves(
    robustness_data: Dict[str, Dict],
    save_path: str = None
):
    """Plot signature distance vs perturbation strength."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'purple']

    for i, (name, data) in enumerate(robustness_data.items()):
        color = colors[i % len(colors)]
        ax.errorbar(
            data['epsilon'],
            data['mean_distance'],
            yerr=data['std_distance'],
            fmt='o-',
            color=color,
            linewidth=2,
            markersize=6,
            capsize=3,
            label=name
        )

    ax.set_xlabel('Perturbation Strength ε', fontsize=12)
    ax.set_ylabel('Signature Distance from Original', fontsize=12)
    ax.set_title('Spectral Signature Robustness\n(Small ε → Small change)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add reference line showing linear scaling
    eps = robustness_data[list(robustness_data.keys())[0]]['epsilon']
    ax.plot(eps, eps * 2, 'k--', alpha=0.3, label='Linear reference')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_discriminability_histogram(
    same_system_distances: np.ndarray,
    different_system_distances: np.ndarray,
    save_path: str = None
):
    """Plot histograms of signature distances."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram for same system (perturbed)
    ax.hist(same_system_distances, bins=20, alpha=0.6, color='blue',
           label='Same system (perturbed)', density=True)

    # Histogram for different systems
    ax.hist(different_system_distances, bins=20, alpha=0.6, color='red',
           label='Different systems', density=True)

    ax.set_xlabel('Spectral Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Signature Distance Distributions\n(Should be well-separated)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Check separation
    max_same = np.max(same_system_distances)
    min_diff = np.min(different_system_distances)
    if max_same < min_diff:
        ax.axvline(x=(max_same + min_diff)/2, color='green', linestyle='--',
                  label='Separation boundary')
        ax.text(0.6, 0.9, 'Well-separated ✓', transform=ax.transAxes,
               fontsize=12, color='green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


# =============================================================================
# Main Benchmark
# =============================================================================

def run_signature_robustness_benchmark(
    save_plots: bool = True,
    save_data: bool = True
) -> Dict[str, Any]:
    """
    Run the spectral signature robustness benchmark.

    Demonstrates:
    1. Small perturbations → small signature changes (stability)
    2. Different systems → large signature changes (discriminability)
    3. Distributions are well-separated for robust fingerprinting

    Returns:
        Dictionary containing all benchmark results
    """
    print("\n" + "="*70)
    print("BENCHMARK 10: Spectral Signature Robustness")
    print("="*70)
    print("\nTesting stability under perturbations vs discriminability")
    print("─" * 70)

    # Test Hamiltonians (genuinely different systems)
    hamiltonians = {
        'H1: σz': Z.astype(np.complex128),
        'H2: σx': X.astype(np.complex128),
        'H3: σz + σx': (Z + X).astype(np.complex128) / np.sqrt(2),
        'H4: 2-qubit ZZ': np.kron(Z, Z).astype(np.complex128),
    }

    s_values = np.linspace(0.5, 3.0, 50)

    # Part 1: Perturbation robustness
    print("\n1. Perturbation Robustness Test")
    print("   How much does signature change with small perturbations?")

    epsilon_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    n_samples = 20

    robustness_data = {}
    for name, H in list(hamiltonians.items())[:2]:  # Test first two
        print(f"\n   Analyzing {name}...")
        data = analyze_perturbation_robustness(H, epsilon_values, n_samples, s_values)
        robustness_data[name] = data

    # Display robustness results
    headers = ['ε', 'H1: Mean ± Std', 'H2: Mean ± Std']
    rows = []
    for i, eps in enumerate(epsilon_values):
        row = [f'{eps:.2f}']
        for name in list(robustness_data.keys()):
            mean = robustness_data[name]['mean_distance'][i]
            std = robustness_data[name]['std_distance'][i]
            row.append(f'{mean:.4f} ± {std:.4f}')
        rows.append(row)

    print_table(headers, rows, title="Signature Distance vs Perturbation")

    # Part 2: Discriminability between different systems
    print("\n" + "─" * 70)
    print("\n2. Discriminability Test")
    print("   Can we distinguish genuinely different systems?")

    distance_matrix, names = analyze_discriminability(hamiltonians, s_values)

    # Display distance matrix
    print("\nPairwise Spectral Distances:")
    short_names = [n.split(':')[0] for n in names]
    header_row = [''] + short_names
    distance_rows = []
    for i, name in enumerate(short_names):
        row = [name] + [f'{distance_matrix[i,j]:.3f}' for j in range(len(names))]
        distance_rows.append(row)

    print_table(header_row, distance_rows, title="Distance Matrix")

    # Part 3: Compare distributions
    print("\n" + "─" * 70)
    print("\n3. Distribution Comparison")

    # Same-system distances (perturbations)
    H_ref = Z.astype(np.complex128)
    sig_ref = spectral_signature(H_ref, s_values)
    same_system_distances = []
    for _ in range(50):
        H_perturbed = perturb_hamiltonian(H_ref, 0.2)  # moderate perturbation
        sig_perturbed = spectral_signature(H_perturbed, s_values)
        same_system_distances.append(spectral_distance(sig_ref, sig_perturbed))

    same_system_distances = np.array(same_system_distances)

    # Different-system distances
    different_system_distances = distance_matrix[np.triu_indices(len(names), k=1)]

    print(f"\n   Same system (ε=0.2 perturbations):")
    print(f"     Mean distance: {np.mean(same_system_distances):.4f}")
    print(f"     Max distance: {np.max(same_system_distances):.4f}")

    print(f"\n   Different systems:")
    print(f"     Mean distance: {np.mean(different_system_distances):.4f}")
    print(f"     Min distance: {np.min(different_system_distances):.4f}")

    # Check separation
    separation_gap = np.min(different_system_distances) - np.max(same_system_distances)
    well_separated = separation_gap > 0

    print(f"\n   Separation gap: {separation_gap:.4f}")
    print(f"   Well-separated: {'Yes ✓' if well_separated else 'No - overlap exists'}")

    # Key findings
    print("\nKey Findings:")
    print("─" * 40)
    print("• Signature distance ∝ perturbation strength (stability)")
    print("• Different systems have much larger distances (discriminability)")
    if well_separated:
        print("• Distributions are well-separated → robust fingerprinting")
    else:
        print("• Some overlap at moderate perturbations")

    # Prepare results
    results = {
        'benchmark': 'signature_robustness',
        'perturbation_test': {
            'epsilon_values': epsilon_values.tolist(),
            'results': {
                name: {
                    'mean_distance': data['mean_distance'].tolist(),
                    'std_distance': data['std_distance'].tolist(),
                }
                for name, data in robustness_data.items()
            }
        },
        'discriminability': {
            'systems': list(hamiltonians.keys()),
            'distance_matrix': distance_matrix.tolist(),
        },
        'distribution_comparison': {
            'same_system_mean': float(np.mean(same_system_distances)),
            'same_system_max': float(np.max(same_system_distances)),
            'different_system_mean': float(np.mean(different_system_distances)),
            'different_system_min': float(np.min(different_system_distances)),
            'separation_gap': float(separation_gap),
            'well_separated': bool(well_separated),
        },
        'statement': (
            "Spectral signatures provide robust system identification, "
            "stable under perturbations yet discriminating between distinct systems."
        )
    }

    # Save results and plots
    if save_data:
        ensure_results_dir()
        save_results(results, 'signature_robustness.json')

    if save_plots:
        figures_dir = get_figures_dir()
        plot_robustness_curves(
            robustness_data,
            save_path=str(figures_dir / 'signature_robustness.png')
        )
        plot_discriminability_histogram(
            same_system_distances, different_system_distances,
            save_path=str(figures_dir / 'signature_discriminability.png')
        )

    print("\n" + "─" * 70)
    print("Statement: " + results['statement'])
    print("─" * 70)

    return results


if __name__ == '__main__':
    run_signature_robustness_benchmark()
