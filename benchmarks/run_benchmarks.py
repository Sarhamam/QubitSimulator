#!/usr/bin/env python3
"""
NoeticEidos Benchmark Suite Runner

Runs benchmarks demonstrating:

Tier 1 (Foundation):
1. State Fidelity - Correctness parity with Qiskit/Cirq
2. Noise Geometry - Bures/QFI trajectory tracking
3. Natural Gradient VQE - Optimization advantages

Tier 2 (Core Capabilities):
4. Spectral Fingerprints - System identification via zeta functions
5. Entanglement Dynamics - Sudden death phenomenon
6. Barren Plateau Detection - QFI predicts trainability
7. Spectral Gap Relaxation - Spectral-geometric bridge

Tier 3 (Advanced):
8. Geodesic vs Euclidean - Distance metric comparison
9. Berry Phase - QGT validation
10. Signature Robustness - Perturbation stability
11. Chaos Detection - Form factor analysis

Philosophy: Benchmark on geometry, stability, and interpretability—not raw speed.

Usage:
    python benchmarks/run_benchmarks.py                    # Run Tier 1 only
    python benchmarks/run_benchmarks.py --tier2            # Include Tier 2
    python benchmarks/run_benchmarks.py --all              # Run all benchmarks
    python benchmarks/run_benchmarks.py --benchmark spectral_fingerprints
"""

import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def print_banner():
    """Print benchmark suite banner."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  NoeticEidos Quantum Simulator - Benchmark Suite".center(58) + "║")
    print("║" + "  Geometry • Stability • Interpretability".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("Philosophy: Benchmark on geometry, stability, interpretability")
    print("            — not raw speed.")
    print()
    print("-" * 60)


def print_summary(results: dict, elapsed: float):
    """Print final summary of all benchmarks."""
    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print()

    # Count successes
    tier1_count = 0
    tier2_count = 0
    tier3_count = 0

    # Tier 1
    if results.get('fidelity'):
        tier1_count += 1
        print("1. State Fidelity:          ✓ PASSED")
    if results.get('geometry'):
        tier1_count += 1
        print("2. Noise Geometry:          ✓ COMPLETE")
    if results.get('vqe'):
        tier1_count += 1
        print("3. Natural Gradient VQE:    ✓ COMPLETE")

    # Tier 2
    if results.get('spectral_fingerprints'):
        tier2_count += 1
        print("4. Spectral Fingerprints:   ✓ COMPLETE")
    if results.get('entanglement'):
        tier2_count += 1
        print("5. Entanglement Dynamics:   ✓ COMPLETE")
    if results.get('barren_plateau'):
        tier2_count += 1
        print("6. Barren Plateau QFI:      ✓ COMPLETE")
    if results.get('spectral_gap'):
        tier2_count += 1
        print("7. Spectral Gap Relaxation: ✓ COMPLETE")

    # Tier 3
    if results.get('geodesic'):
        tier3_count += 1
        print("8. Geodesic vs Euclidean:   ✓ COMPLETE")
    if results.get('berry_phase'):
        tier3_count += 1
        print("9. Berry Phase:             ✓ COMPLETE")
    if results.get('robustness'):
        tier3_count += 1
        print("10. Signature Robustness:   ✓ COMPLETE")
    if results.get('chaos'):
        tier3_count += 1
        print("11. Chaos Detection:        ✓ COMPLETE")

    print()
    print(f"Tier 1 (Foundation):    {tier1_count}/3 completed")
    print(f"Tier 2 (Core):          {tier2_count}/4 completed")
    print(f"Tier 3 (Advanced):      {tier3_count}/4 completed")
    print()
    print(f"Total elapsed time: {elapsed:.1f}s")
    print()
    print("Results saved to: benchmarks/results/")
    print("  - data/       JSON result files")
    print("  - figures/    Generated plots")
    print()


def run_tier1(results: dict, skip_fidelity=False, skip_geometry=False, skip_vqe=False):
    """Run Tier 1 benchmarks."""
    print("\n" + "=" * 60)
    print("TIER 1: Foundation Benchmarks")
    print("=" * 60)

    # 1. State Fidelity Benchmark
    if not skip_fidelity:
        try:
            from benchmarks.correctness.fidelity_benchmark import run_fidelity_benchmark
            results['fidelity'] = run_fidelity_benchmark(save=True)
        except ImportError as e:
            print(f"\nSkipping fidelity benchmark: {e}")
            print("Install with: pip install qiskit qiskit-aer cirq")
            results['fidelity'] = None
    else:
        print("\nSkipping fidelity benchmark (--skip-fidelity)")
        results['fidelity'] = None

    # 2. Noise Geometry Benchmark
    if not skip_geometry:
        from benchmarks.geometry.noise_geometry import run_noise_geometry_benchmark
        results['geometry'] = run_noise_geometry_benchmark(save_plots=True, save_data=True)
    else:
        print("\nSkipping geometry benchmark (--skip-geometry)")
        results['geometry'] = None

    # 3. VQE Optimization Benchmark
    if not skip_vqe:
        from benchmarks.optimization.natural_gradient_vqe import run_vqe_benchmark
        results['vqe'] = run_vqe_benchmark(save_plots=True, save_data=True)
    else:
        print("\nSkipping VQE benchmark (--skip-vqe)")
        results['vqe'] = None


def run_tier2(results: dict):
    """Run Tier 2 benchmarks."""
    print("\n" + "=" * 60)
    print("TIER 2: Core Capability Benchmarks")
    print("=" * 60)

    # 4. Spectral Fingerprints
    from benchmarks.spectral.spectral_fingerprints import run_spectral_fingerprints_benchmark
    results['spectral_fingerprints'] = run_spectral_fingerprints_benchmark(
        save_plots=True, save_data=True
    )

    # 5. Entanglement Dynamics
    from benchmarks.geometry.entanglement_dynamics import run_entanglement_dynamics_benchmark
    results['entanglement'] = run_entanglement_dynamics_benchmark(
        save_plots=True, save_data=True
    )

    # 6. Barren Plateau Detection
    from benchmarks.optimization.barren_plateau_qfi import run_barren_plateau_benchmark
    results['barren_plateau'] = run_barren_plateau_benchmark(
        save_plots=True, save_data=True
    )

    # 7. Spectral Gap Relaxation
    from benchmarks.spectral.spectral_gap_relaxation import run_spectral_gap_benchmark
    results['spectral_gap'] = run_spectral_gap_benchmark(
        save_plots=True, save_data=True
    )


def run_tier3(results: dict):
    """Run Tier 3 benchmarks."""
    print("\n" + "=" * 60)
    print("TIER 3: Advanced Benchmarks")
    print("=" * 60)

    # 8. Geodesic vs Euclidean
    from benchmarks.geometry.geodesic_vs_euclidean import run_geodesic_vs_euclidean_benchmark
    results['geodesic'] = run_geodesic_vs_euclidean_benchmark(
        save_plots=True, save_data=True
    )

    # 9. Berry Phase
    from benchmarks.geometry.berry_phase import run_berry_phase_benchmark
    results['berry_phase'] = run_berry_phase_benchmark(
        save_plots=True, save_data=True
    )

    # 10. Signature Robustness
    from benchmarks.spectral.signature_robustness import run_signature_robustness_benchmark
    results['robustness'] = run_signature_robustness_benchmark(
        save_plots=True, save_data=True
    )

    # 11. Chaos Detection
    from benchmarks.spectral.chaos_detection import run_chaos_detection_benchmark
    results['chaos'] = run_chaos_detection_benchmark(
        save_plots=True, save_data=True
    )


def run_single_benchmark(name: str) -> dict:
    """Run a single benchmark by name."""
    benchmark_map = {
        # Tier 1
        'fidelity': ('benchmarks.correctness.fidelity_benchmark', 'run_fidelity_benchmark'),
        'geometry': ('benchmarks.geometry.noise_geometry', 'run_noise_geometry_benchmark'),
        'vqe': ('benchmarks.optimization.natural_gradient_vqe', 'run_vqe_benchmark'),
        # Tier 2
        'spectral_fingerprints': ('benchmarks.spectral.spectral_fingerprints', 'run_spectral_fingerprints_benchmark'),
        'entanglement': ('benchmarks.geometry.entanglement_dynamics', 'run_entanglement_dynamics_benchmark'),
        'barren_plateau': ('benchmarks.optimization.barren_plateau_qfi', 'run_barren_plateau_benchmark'),
        'spectral_gap': ('benchmarks.spectral.spectral_gap_relaxation', 'run_spectral_gap_benchmark'),
        # Tier 3
        'geodesic': ('benchmarks.geometry.geodesic_vs_euclidean', 'run_geodesic_vs_euclidean_benchmark'),
        'berry_phase': ('benchmarks.geometry.berry_phase', 'run_berry_phase_benchmark'),
        'robustness': ('benchmarks.spectral.signature_robustness', 'run_signature_robustness_benchmark'),
        'chaos': ('benchmarks.spectral.chaos_detection', 'run_chaos_detection_benchmark'),
    }

    if name not in benchmark_map:
        print(f"Unknown benchmark: {name}")
        print(f"Available: {', '.join(benchmark_map.keys())}")
        return {}

    module_name, func_name = benchmark_map[name]
    import importlib
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    if name == 'fidelity':
        return func(save=True)
    else:
        return func(save_plots=True, save_data=True)


def run_all(tier2=False, tier3=False, run_all_tiers=False,
            skip_fidelity=False, skip_geometry=False, skip_vqe=False,
            single_benchmark=None):
    """
    Run benchmark suites.

    Args:
        tier2: Include Tier 2 benchmarks
        tier3: Include Tier 3 benchmarks
        run_all_tiers: Run all tiers
        skip_fidelity: Skip fidelity benchmark (requires qiskit/cirq)
        skip_geometry: Skip noise geometry benchmark
        skip_vqe: Skip VQE optimization benchmark
        single_benchmark: Run only this specific benchmark
    """
    print_banner()

    results = {}
    start_time = time.time()

    # Ensure results directories exist
    from benchmarks.shared.utils import ensure_results_dir
    ensure_results_dir()

    # Run single benchmark if specified
    if single_benchmark:
        print(f"\nRunning single benchmark: {single_benchmark}")
        results[single_benchmark] = run_single_benchmark(single_benchmark)
        elapsed = time.time() - start_time
        print(f"\nElapsed time: {elapsed:.1f}s")
        return results

    # Run Tier 1 (always, unless single benchmark)
    run_tier1(results, skip_fidelity, skip_geometry, skip_vqe)

    # Run Tier 2 if requested
    if tier2 or run_all_tiers:
        run_tier2(results)

    # Run Tier 3 if requested
    if tier3 or run_all_tiers:
        run_tier3(results)

    elapsed = time.time() - start_time
    print_summary(results, elapsed)

    return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run NoeticEidos benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmarks/run_benchmarks.py                         # Tier 1 only
    python benchmarks/run_benchmarks.py --tier2                 # Tier 1 + 2
    python benchmarks/run_benchmarks.py --all                   # All tiers
    python benchmarks/run_benchmarks.py --benchmark chaos       # Single benchmark
    python benchmarks/run_benchmarks.py --skip-fidelity         # Skip Qiskit test

Available single benchmarks:
    Tier 1: fidelity, geometry, vqe
    Tier 2: spectral_fingerprints, entanglement, barren_plateau, spectral_gap
    Tier 3: geodesic, berry_phase, robustness, chaos
        """
    )

    parser.add_argument(
        '--tier2',
        action='store_true',
        help='Include Tier 2 benchmarks'
    )

    parser.add_argument(
        '--tier3',
        action='store_true',
        help='Include Tier 3 benchmarks'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all benchmark tiers'
    )

    parser.add_argument(
        '--benchmark', '-b',
        type=str,
        help='Run a single specific benchmark by name'
    )

    parser.add_argument(
        '--skip-fidelity',
        action='store_true',
        help='Skip fidelity benchmark (requires qiskit/cirq)'
    )

    parser.add_argument(
        '--skip-geometry',
        action='store_true',
        help='Skip noise geometry benchmark'
    )

    parser.add_argument(
        '--skip-vqe',
        action='store_true',
        help='Skip VQE optimization benchmark'
    )

    args = parser.parse_args()

    run_all(
        tier2=args.tier2,
        tier3=args.tier3,
        run_all_tiers=args.all,
        skip_fidelity=args.skip_fidelity,
        skip_geometry=args.skip_geometry,
        skip_vqe=args.skip_vqe,
        single_benchmark=args.benchmark
    )


if __name__ == "__main__":
    main()
