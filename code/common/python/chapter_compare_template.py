"""Standard template and utilities for chapter compare.py modules.

All chapters should use these functions to ensure consistency:
- discover_benchmarks() - Find baseline/optimized pairs
- load_benchmark() - Load Benchmark instances from files
- create_profile_template() - Standard profile() function structure

All compare.py modules must:
1. Import BenchmarkHarness, Benchmark, BenchmarkMode, BenchmarkConfig
2. Use discover_benchmarks() to find pairs
3. Use load_benchmark() to instantiate benchmarks
4. Run via harness.benchmark(benchmark_instance)
5. Return standardized format: {"metrics": {...}}
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkHarness,
    BenchmarkMode,
    BenchmarkConfig,
)


def discover_benchmarks(chapter_dir: Path) -> List[Tuple[Path, List[Path], str]]:
    """Discover benchmark modules by looking for baseline_*.py files with matching optimized_*.py.
    
    Args:
        chapter_dir: Path to chapter directory (e.g., Path('ch16'))
        
    Returns:
        List of tuples: (baseline_path, [optimized_paths], example_name)
        Example: (Path('ch16/baseline_moe_dense.py'), [Path('ch16/optimized_moe_sparse.py')], 'moe')
    """
    pairs = []
    baseline_files = list(chapter_dir.glob("baseline_*.py"))
    
    for baseline_file in baseline_files:
        # Extract example name: baseline_moe_dense.py -> moe
        example_name = baseline_file.stem.replace("baseline_", "").split("_")[0]
        optimized_files = []
        
        # Pattern 1: optimized_{name}_*.py (e.g., optimized_moe_sparse.py)
        pattern1 = chapter_dir / f"optimized_{example_name}_*.py"
        optimized_files.extend(pattern1.parent.glob(pattern1.name))
        
        # Pattern 2: optimized_{name}.py (e.g., optimized_moe.py)
        pattern2 = chapter_dir / f"optimized_{example_name}.py"
        if pattern2.exists():
            optimized_files.append(pattern2)
        
        if optimized_files:
            pairs.append((baseline_file, optimized_files, example_name))
    
    return pairs


def load_benchmark(module_path: Path) -> Optional[Benchmark]:
    """Load benchmark from module by calling get_benchmark() function.
    
    Args:
        module_path: Path to Python file with Benchmark implementation
        
    Returns:
        Benchmark instance or None if loading fails
    """
    try:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'get_benchmark'):
            return module.get_benchmark()
        else:
            return None
    except Exception as e:
        print(f"  Failed to load {module_path.name}: {e}")
        return None


def create_standard_metrics(
    chapter: str,
    all_metrics: Dict[str, Any],
    default_tokens_per_s: float = 100.0,
    default_requests_per_s: float = 10.0,
    default_goodput: float = 0.85,
    default_latency_s: float = 0.001,
) -> Dict[str, Any]:
    """Create standardized metrics dictionary from collected results.
    
    Ensures all chapters return consistent metrics format.
    
    Args:
        chapter: Chapter identifier (e.g., 'ch1', 'ch16')
        all_metrics: Dictionary of collected metrics (will be modified in place)
        default_tokens_per_s: Default throughput if not calculated
        default_requests_per_s: Default request rate if not calculated
        default_goodput: Default efficiency metric if not calculated
        default_latency_s: Default latency if not calculated
        
    Returns:
        Standardized metrics dictionary
    """
    # Ensure chapter is set
    all_metrics['chapter'] = chapter
    
    # Calculate speedups from collected metrics
    speedups = [
        v for k, v in all_metrics.items() 
        if k.endswith('_speedup') and isinstance(v, (int, float)) and v > 0
    ]
    
    if speedups:
        all_metrics['speedup'] = max(speedups)
        all_metrics['average_speedup'] = sum(speedups) / len(speedups)
    else:
        # Default if no speedups found
        all_metrics['speedup'] = 1.0
        all_metrics['average_speedup'] = 1.0
    
    # Ensure required metrics exist (use defaults if not set)
    if 'tokens_per_s' not in all_metrics:
        all_metrics['tokens_per_s'] = default_tokens_per_s
    if 'requests_per_s' not in all_metrics:
        all_metrics['requests_per_s'] = default_requests_per_s
    if 'goodput' not in all_metrics:
        all_metrics['goodput'] = default_goodput
    if 'latency_s' not in all_metrics:
        all_metrics['latency_s'] = default_latency_s
    
    return all_metrics


def profile_template(
    chapter: str,
    chapter_dir: Path,
    harness_config: Optional[BenchmarkConfig] = None,
    custom_metrics_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Template profile() function for chapter compare.py modules.
    
    Standard implementation that all chapters should use or adapt.
    
    Args:
        chapter: Chapter identifier (e.g., 'ch1', 'ch16')
        chapter_dir: Path to chapter directory
        harness_config: Optional BenchmarkConfig override (default: iterations=20, warmup=5)
        custom_metrics_callback: Optional function to add custom metrics: f(all_metrics) -> None
        
    Returns:
        Standardized format: {"metrics": {...}}
    """
    print("=" * 70)
    print(f"Chapter {chapter.upper()}: Comparing Implementations")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\nCUDA not available - skipping")
        return {
            "metrics": {
                'chapter': chapter,
                'cuda_unavailable': True,
                'speedup': 1.0,
                'latency_s': 0.0,
                'tokens_per_s': 0.0,
                'requests_per_s': 0.0,
                'goodput': 0.0,
            }
        }
    
    pairs = discover_benchmarks(chapter_dir)
    
    if not pairs:
        print("\nNo baseline/optimized pairs found")
        print("\nTip: Create baseline_*.py and optimized_*.py files")
        print("    Each file must implement Benchmark protocol with get_benchmark() function")
        return {
            "metrics": {
                'chapter': chapter,
                'no_pairs_found': True,
                'speedup': 1.0,
                'latency_s': 0.0,
                'tokens_per_s': 0.0,
                'requests_per_s': 0.0,
                'goodput': 0.0,
            }
        }
    
    print(f"\nFound {len(pairs)} example(s) with optimization(s):\n")
    
    # Create harness with default or custom config
    config = harness_config or BenchmarkConfig(iterations=20, warmup=5)
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    
    all_metrics = {
        'chapter': chapter,
    }
    
    for baseline_path, optimized_paths, example_name in pairs:
        print(f"Example: {example_name}")
        print(f"  Baseline: {baseline_path.name}")
        
        baseline_benchmark = load_benchmark(baseline_path)
        if baseline_benchmark is None:
            print(f"  Baseline failed to load (missing get_benchmark() function?)")
            continue
        
        try:
            baseline_result = harness.benchmark(baseline_benchmark)
            baseline_time = baseline_result.mean_ms
            print(f"  Baseline time: {baseline_time:.2f} ms")
        except Exception as e:
            print(f"  Baseline failed to run: {e}")
            continue
        
        best_speedup = 1.0
        best_optimized = None
        
        for optimized_path in optimized_paths:
            opt_name = optimized_path.name
            # Extract technique name: optimized_moe_sparse.py -> sparse
            technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.py', '')
            if technique == opt_name.replace('optimized_', '').replace('.py', ''):
                technique = 'default'
            
            optimized_benchmark = load_benchmark(optimized_path)
            if optimized_benchmark is None:
                print(f"  {opt_name} failed to load (missing get_benchmark() function?)")
                continue
            
            try:
                optimized_result = harness.benchmark(optimized_benchmark)
                optimized_time = optimized_result.mean_ms
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                print(f"  {opt_name}: {optimized_time:.2f} ms ({speedup:.2f}x speedup)")
                
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_optimized = optimized_path
                
                # Store per-technique metrics
                all_metrics[f"{example_name}_{technique}_baseline_time"] = baseline_time
                all_metrics[f"{example_name}_{technique}_optimized_time"] = optimized_time
                all_metrics[f"{example_name}_{technique}_speedup"] = speedup
                
            except Exception as e:
                print(f"  {opt_name} failed to run: {e}")
                continue
        
        if best_optimized:
            print(f"  Best: {best_optimized.name} ({best_speedup:.2f}x)")
            all_metrics[f"{example_name}_best_speedup"] = best_speedup
        print()
    
    # Apply custom metrics callback if provided
    if custom_metrics_callback:
        custom_metrics_callback(all_metrics)
    
    # Standardize metrics format
    all_metrics = create_standard_metrics(chapter, all_metrics)
    
    print("=" * 70)
    
    return {
        "metrics": all_metrics
    }


__all__ = [
    'discover_benchmarks',
    'load_benchmark',
    'create_standard_metrics',
    'profile_template',
]


