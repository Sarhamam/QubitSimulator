"""Optimization benchmarks demonstrating natural gradient advantage."""

from .natural_gradient_vqe import run_vqe_benchmark
from .optimizer_comparison import (
    compare_all_optimizers,
    predict_optimizer_performance,
    spectral_diagnostics_dict,
    count_iterations_to_threshold
)
from .barren_plateau_qfi import run_barren_plateau_benchmark

__all__ = [
    'run_vqe_benchmark',
    'compare_all_optimizers',
    'predict_optimizer_performance',
    'spectral_diagnostics_dict',
    'count_iterations_to_threshold',
    'run_barren_plateau_benchmark',
]
