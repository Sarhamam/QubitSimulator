"""
Benchmark Utilities

Plotting, formatting, and I/O utilities for benchmark results.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


# =============================================================================
# Directory Management
# =============================================================================

def ensure_results_dir() -> Path:
    """Ensure results directories exist and return base path."""
    base = Path(__file__).parent.parent / "results"
    (base / "figures").mkdir(parents=True, exist_ok=True)
    (base / "data").mkdir(parents=True, exist_ok=True)
    return base


def get_figures_dir() -> Path:
    """Get path to figures directory."""
    return ensure_results_dir() / "figures"


def get_data_dir() -> Path:
    """Get path to data directory."""
    return ensure_results_dir() / "data"


# =============================================================================
# Table Formatting
# =============================================================================

def print_table(headers: List[str], rows: List[List[Any]], title: Optional[str] = None):
    """
    Print a nicely formatted ASCII table.

    Args:
        headers: Column headers
        rows: List of rows, each row is a list of values
        title: Optional table title
    """
    # Convert all values to strings
    str_rows = [[str(v) for v in row] for row in rows]

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    # Build format string
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    separator = "─" * (sum(widths) + 2 * (len(widths) - 1))

    # Print
    if title:
        print(f"\n{title}")
        print("=" * len(title))

    print()
    print(fmt.format(*headers))
    print(separator)

    for row in str_rows:
        print(fmt.format(*row))

    print()


def format_scientific(value: float, precision: int = 9) -> str:
    """Format a value with high precision for fidelity display."""
    if abs(value - 1.0) < 1e-15:
        return "1.000000000"
    return f"{value:.{precision}f}"


def format_iterations(n: int) -> str:
    """Format iteration count."""
    return f"{n:>4d}"


# =============================================================================
# Plotting Utilities
# =============================================================================

def plot_convergence(
    results: Dict[str, List[float]],
    title: str = "VQE Convergence Comparison",
    xlabel: str = "Iteration",
    ylabel: str = "Energy",
    target_energy: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Plot convergence curves for multiple optimizers.

    Args:
        results: Dict mapping optimizer name to energy history
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        target_energy: Optional horizontal line for ground state
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color and marker scheme
    styles = {
        'Vanilla GD': {'color': 'red', 'marker': 'o', 'linestyle': '-'},
        'Natural GD': {'color': 'blue', 'marker': 's', 'linestyle': '-'},
        'Conjugate GD': {'color': 'green', 'marker': '^', 'linestyle': '-'},
        'Natural CG': {'color': 'purple', 'marker': 'd', 'linestyle': '-'},
    }

    for name, energies in results.items():
        style = styles.get(name, {'color': 'gray', 'marker': '.', 'linestyle': '-'})
        iterations = range(len(energies))
        ax.plot(iterations, energies, label=name,
                color=style['color'],
                marker=style['marker'],
                markersize=3,
                linestyle=style['linestyle'],
                linewidth=1.5)

    if target_energy is not None:
        ax.axhline(y=target_energy, color='black', linestyle='--',
                   label=f'Ground state ({target_energy:.4f})', alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_trajectory(
    times: np.ndarray,
    quantities: Dict[str, np.ndarray],
    title: str = "Geometric Trajectory",
    xlabel: str = "Time",
    save_path: Optional[str] = None
):
    """
    Plot multiple quantities along a trajectory.

    Args:
        times: Time array
        quantities: Dict mapping quantity name to values
        title: Plot title
        xlabel: X-axis label
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n_quantities = len(quantities)
    fig, axes = plt.subplots(n_quantities, 1, figsize=(10, 3 * n_quantities), sharex=True)

    if n_quantities == 1:
        axes = [axes]

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, (name, values) in enumerate(quantities.items()):
        ax = axes[i]
        ax.plot(times, values, color=colors[i % len(colors)], linewidth=2)
        ax.set_ylabel(name, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times[0], times[-1])

    axes[-1].set_xlabel(xlabel, fontsize=12)
    axes[0].set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_bures_trajectory(
    times: np.ndarray,
    bures_distances: np.ndarray,
    purities: np.ndarray,
    title: str = "Bures Distance from Initial State",
    save_path: Optional[str] = None
):
    """
    Plot Bures distance trajectory with purity subplot.

    Args:
        times: Time array
        bures_distances: Bures distance at each time
        purities: State purity at each time
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Bures distance
    ax1.plot(times, bures_distances, 'b-', linewidth=2, label='Bures distance')
    ax1.fill_between(times, 0, bures_distances, alpha=0.3)
    ax1.set_ylabel('Bures Distance', fontsize=11)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Purity
    ax2.plot(times, purities, 'r-', linewidth=2, label='Purity')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Maximally mixed (1-qubit)')
    ax2.set_ylabel('Purity Tr(ρ²)', fontsize=11)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_qfi_decay(
    times: np.ndarray,
    qfi_values: np.ndarray,
    title: str = "Quantum Fisher Information Decay",
    save_path: Optional[str] = None
):
    """
    Plot QFI decay under noise.

    Args:
        times: Time array
        qfi_values: QFI at each time
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(times, qfi_values, 'b-', linewidth=2)
    ax.fill_between(times, 0, qfi_values, alpha=0.3)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Quantum Fisher Information', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark initial QFI
    ax.axhline(y=qfi_values[0], color='gray', linestyle='--', alpha=0.5,
               label=f'Initial QFI = {qfi_values[0]:.3f}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


# =============================================================================
# Data I/O
# =============================================================================

def save_results(results: Dict[str, Any], filename: str):
    """Save benchmark results to JSON file."""
    import json

    path = get_data_dir() / filename

    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"Saved results: {path}")


def load_results(filename: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    import json

    path = get_data_dir() / filename

    with open(path) as f:
        return json.load(f)
