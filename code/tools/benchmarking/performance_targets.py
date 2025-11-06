#!/usr/bin/env python3
"""
Performance targets and thresholds for all chapters.
Extracted from README.md Performance Targets Summary.

Peak performance values are loaded from benchmark_peak_results_*.json files
if available (created during setup.sh). If not found, falls back to hardcoded
baseline values.
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional

# Default baseline targets (used as fallback if benchmark_peak hasn't run)
_DEFAULT_TARGETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "overall": {
        "hbm3e_bandwidth_tbs": {"min": 3.0, "target": 3.5, "unit": "TB/s", "realistic_max": 4.0},
        "fp16_compute_tflops": {"min": 1000, "target": 2000, "unit": "TFLOPS", "realistic_max": 1300},
        "torch_compile_speedup_small": {"min": 1.1, "target": 1.2, "unit": "x"},
        "torch_compile_speedup_large": {"min": 1.0, "target": 1.3, "unit": "x"},
        "flex_attention_speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
        "deepseek_l2_speedup": {"min": 1.1, "target": 1.3, "unit": "x"},
    },
    "ch1": {
        "description": "Performance Basics",
        "metrics": {}
    },
    "ch2": {
        "description": "Hardware Overview",
        "metrics": {
            "nvlink_bandwidth_gbs": {"min": 650, "target": 725, "unit": "GB/s"},
            "hbm3e_bandwidth_tbs": {"min": 5.0, "target": 5.5, "unit": "TB/s"},
        }
    },
    "ch3": {
        "description": "System Setup",
        "metrics": {}
    },
    "ch4": {
        "description": "Distributed Networking",
        "metrics": {
            "allreduce_bandwidth_gbs": {"min": 700, "target": 800, "unit": "GB/s"},
            "p2p_bandwidth_gbs": {"min": 800, "target": 900, "unit": "GB/s"},
            "small_message_latency_us": {"min": 0, "target": 2, "unit": "μs", "lower_is_better": True},
            "scaling_efficiency_percent": {"min": 85, "target": 95, "unit": "%"},
        }
    },
    "ch5": {
        "description": "Storage & I/O",
        "metrics": {}
    },
    "ch6": {
        "description": "CUDA Kernels",
        "metrics": {}
    },
    "ch7": {
        "description": "Memory Access Patterns",
        "metrics": {
            "utilization_percent": {"min": 60, "target": 68, "unit": "%"},
        }
    },
    "ch8": {
        "description": "Occupancy & ILP",
        "metrics": {}
    },
    "ch9": {
        "description": "Kernel Fusion",
        "metrics": {}
    },
    "ch10": {
        "description": "Tensor Cores",
        "metrics": {
            "fp8_tflops": {"min": 500, "target": 550, "unit": "TFLOPS"},
            "fp16_tflops": {"min": 500, "target": 550, "unit": "TFLOPS"},
            "tensor_core_utilization_percent": {"min": 20, "target": 30, "unit": "%"},
        }
    },
    "ch11": {
        "description": "Streams & Concurrency",
        "metrics": {
            "stream_speedup": {"min": 1.5, "target": 1.7, "unit": "x"},
            "stream_overlap_percent": {"min": 30, "target": 35, "unit": "%"},
        }
    },
    "ch12": {
        "description": "CUDA Graphs",
        "metrics": {}
    },
    "ch13": {
        "description": "PyTorch Profiling",
        "metrics": {
            "compiled_autograd_speedup": {"min": 1.1, "target": 1.3, "unit": "x"},
        }
    },
    "ch14": {
        "description": "Compiler & Triton",
        "metrics": {
            "torch_compile_speedup_large": {"min": 1.0, "target": 1.3, "unit": "x"},
        }
    },
    "ch15": {
        "description": "Disaggregated Inference",
        "metrics": {
            "resource_utilization_improvement_percent": {"min": 15, "target": 30, "unit": "%"},
        }
    },
    "ch16": {
        "description": "Inference Optimization",
        "metrics": {
            "speedup": {"min": 1.05, "target": 1.10, "unit": "x"},
            "fp8_speedup": {"min": 0.90, "target": 1.00, "unit": "x"},
        }
    },
    "ch17": {
        "description": "Dynamic Routing",
        "metrics": {
            "routing_overhead_ms": {"min": 0, "target": 1.0, "unit": "ms", "lower_is_better": True},
            "load_balance_variance": {"min": 0, "target": 0.1, "unit": "", "lower_is_better": True},
        }
    },
    "ch18": {
        "description": "Attention Mechanisms",
        "metrics": {}
    },
    "ch19": {
        "description": "Advanced Training",
        "metrics": {
            "fp8_training_speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
            "memory_reduction_percent": {"min": 30, "target": 50, "unit": "%"},
        }
    },
    "ch20": {
        "description": "AI Kernel Generator",
        "metrics": {}
    },
}


def _load_peak_benchmark_results(search_dir: Path = None) -> Optional[Dict[str, Any]]:
    """Load peak performance results from benchmark_peak_results_*.json."""
    if search_dir is None:
        # Try to find the project root (assume we're in tools/benchmarking/)
        current = Path(__file__).parent
        # Go up to code/ directory
        search_dir = current.parent.parent
    
    # Find the most recent benchmark results file (support both uppercase and lowercase)
    json_files = list(search_dir.glob("benchmark_peak_results_*.json"))
    if not json_files:
        # Fallback to old uppercase pattern for backwards compatibility
        json_files = list(search_dir.glob("BENCHMARK_PEAK_RESULTS_*.json"))
    if not json_files:
        return None
    
    # Sort by modification time, get most recent
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    try:
        with open(json_files[0]) as f:
            return json.load(f)
    except Exception:
        return None


def _get_peak_values() -> Dict[str, float]:
    """Get measured peak values from benchmark results."""
    benchmark_data = _load_peak_benchmark_results()
    if not benchmark_data:
        return {}
    
    peak_values = {}
    
    # Extract HBM memory bandwidth (previously hbm3e, now hbm)
    hbm_data = benchmark_data.get("hbm") or benchmark_data.get("hbm3e")  # Support both
    if hbm_data and "peak_bandwidth_tbs" in hbm_data:
        peak_values["hbm_bandwidth_tbs"] = hbm_data["peak_bandwidth_tbs"]
        # Also keep old name for compatibility
        peak_values["hbm3e_bandwidth_tbs"] = hbm_data["peak_bandwidth_tbs"]
    
    # Extract FP4 compute
    if "fp4_compute" in benchmark_data and "peak_tflops" in benchmark_data["fp4_compute"]:
        peak_values["fp4_compute_tflops"] = benchmark_data["fp4_compute"]["peak_tflops"]
    
    # Extract FP6 compute
    if "fp6_compute" in benchmark_data and "peak_tflops" in benchmark_data["fp6_compute"]:
        peak_values["fp6_compute_tflops"] = benchmark_data["fp6_compute"]["peak_tflops"]
    
    # Extract FP8 compute
    if "fp8_compute" in benchmark_data and "peak_tflops" in benchmark_data["fp8_compute"]:
        peak_values["fp8_compute_tflops"] = benchmark_data["fp8_compute"]["peak_tflops"]
    
    # Extract FP16 compute
    if "fp16_compute" in benchmark_data and "peak_tflops" in benchmark_data["fp16_compute"]:
        peak_values["fp16_compute_tflops"] = benchmark_data["fp16_compute"]["peak_tflops"]
    
    # Extract torch.compile speedup
    if "torch_compile" in benchmark_data and "speedup" in benchmark_data["torch_compile"]:
        peak_values["torch_compile_speedup"] = benchmark_data["torch_compile"]["speedup"]
    
    return peak_values


def _build_targets() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build TARGETS dict with measured peak values if available."""
    targets = copy.deepcopy(_DEFAULT_TARGETS)
    peak_values = _get_peak_values()
    
    if not peak_values:
        return targets
    
    # Update overall targets with measured peak values
    # Use peak as target, and set min to 85% of peak
    if "hbm_bandwidth_tbs" in peak_values or "hbm3e_bandwidth_tbs" in peak_values:
        peak_tbs = peak_values.get("hbm_bandwidth_tbs") or peak_values.get("hbm3e_bandwidth_tbs")
        targets["overall"]["hbm3e_bandwidth_tbs"] = {
            "min": peak_tbs * 0.85,
            "target": peak_tbs,
            "unit": "TB/s",
            "realistic_max": peak_tbs * 1.1,  # Allow 10% overhead
        }
        # Also update ch2 target
        if "ch2" in targets and "metrics" in targets["ch2"]:
            targets["ch2"]["metrics"]["hbm3e_bandwidth_tbs"] = {
                "min": peak_tbs * 0.85,
                "target": peak_tbs,
                "unit": "TB/s",
            }
    
    if "fp4_compute_tflops" in peak_values:
        peak_tflops = peak_values["fp4_compute_tflops"]
        # Update ch10 FP4 target if it exists, or add to overall
        if "ch10" in targets and "metrics" in targets["ch10"]:
            targets["ch10"]["metrics"]["fp4_tflops"] = {
                "min": peak_tflops * 0.85,
                "target": peak_tflops,
                "unit": "TFLOPS",
            }
    
    if "fp6_compute_tflops" in peak_values:
        peak_tflops = peak_values["fp6_compute_tflops"]
        # Update ch10 FP6 target if it exists, or add to overall
        if "ch10" in targets and "metrics" in targets["ch10"]:
            targets["ch10"]["metrics"]["fp6_tflops"] = {
                "min": peak_tflops * 0.85,
                "target": peak_tflops,
                "unit": "TFLOPS",
            }
    
    if "fp16_compute_tflops" in peak_values:
        peak_tflops = peak_values["fp16_compute_tflops"]
        targets["overall"]["fp16_compute_tflops"] = {
            "min": peak_tflops * 0.85,
            "target": peak_tflops,
            "unit": "TFLOPS",
            "realistic_max": peak_tflops * 1.1,
        }
        # Also update ch10 target
        if "ch10" in targets and "metrics" in targets["ch10"]:
            targets["ch10"]["metrics"]["fp16_tflops"] = {
                "min": peak_tflops * 0.85,
                "target": peak_tflops,
                "unit": "TFLOPS",
            }
    
    if "fp8_compute_tflops" in peak_values:
        peak_tflops = peak_values["fp8_compute_tflops"]
        # Update ch10 FP8 target
        if "ch10" in targets and "metrics" in targets["ch10"]:
            targets["ch10"]["metrics"]["fp8_tflops"] = {
                "min": peak_tflops * 0.85,
                "target": peak_tflops,
                "unit": "TFLOPS",
            }
    
    if "torch_compile_speedup" in peak_values:
        peak_speedup = peak_values["torch_compile_speedup"]
        targets["overall"]["torch_compile_speedup_small"] = {
            "min": peak_speedup * 0.85,
            "target": peak_speedup,
            "unit": "x",
        }
        targets["overall"]["torch_compile_speedup_large"] = {
            "min": peak_speedup * 0.85,
            "target": peak_speedup,
            "unit": "x",
        }
        # Also update ch14 target
        if "ch14" in targets and "metrics" in targets["ch14"]:
            targets["ch14"]["metrics"]["torch_compile_speedup_large"] = {
                "min": peak_speedup * 0.85,
                "target": peak_speedup,
                "unit": "x",
            }
    
    return targets


# Build TARGETS with measured peak values (if available)
TARGETS = _build_targets()


def get_target(chapter: str, metric: str) -> Dict[str, Any]:
    """Get target definition for a specific chapter/metric."""
    chapter_lower = chapter.lower()
    if chapter_lower not in TARGETS:
        return {}
    
    chapter_data = TARGETS[chapter_lower]
    if "metrics" not in chapter_data:
        return {}
    
    return chapter_data["metrics"].get(metric, {})


def get_all_chapters() -> list:
    """Get list of all chapters with targets."""
    return [ch for ch in TARGETS.keys() if ch != "overall"]


def get_chapter_description(chapter: str) -> str:
    """Get chapter description."""
    chapter_lower = chapter.lower()
    if chapter_lower not in TARGETS:
        return ""
    if chapter_lower == "overall":
        return "Overall System Performance"
    return TARGETS[chapter_lower].get("description", "")


def get_chapter_metrics(chapter: str) -> Dict[str, Dict[str, Any]]:
    """Get all metrics for a chapter."""
    chapter_lower = chapter.lower()
    if chapter_lower not in TARGETS:
        return {}
    # Overall stores metrics directly, not in a "metrics" sub-dict
    if chapter_lower == "overall":
        return TARGETS[chapter_lower]
    return TARGETS[chapter_lower].get("metrics", {})


def compute_status(actual: float, target_def: Dict[str, Any]) -> str:
    """
    Compute status (PASS/WARN/FAIL) based on actual value and target.
    
    Args:
        actual: Measured value
        target_def: Target definition with 'min', 'target', 'lower_is_better' fields
    
    Returns:
        Status string: "PASS", "WARN", or "FAIL"
    """
    if not target_def or "target" not in target_def:
        return "UNKNOWN"
    
    target_value = target_def["target"]
    min_value = target_def.get("min", target_value * 0.85)
    lower_is_better = target_def.get("lower_is_better", False)
    
    if lower_is_better:
        # For metrics where lower is better (latency, overhead)
        if actual <= target_value:
            return "PASS"
        elif actual <= min_value or actual <= target_value * 1.15:
            return "WARN"
        else:
            return "FAIL"
    else:
        # For metrics where higher is better (throughput, speedup)
        if actual >= target_value or actual >= min_value:
            return "PASS"
        elif actual >= target_value * 0.85:
            return "WARN"
        else:
            return "FAIL"


def format_value(value: float, unit: str) -> str:
    """Format a value with its unit."""
    if unit == "x":
        return f"{value:.2f}x"
    elif unit == "%":
        return f"{value:.1f}%"
    elif "FLOPS" in unit:
        return f"{value:.0f} {unit}"
    elif unit in ["GB/s", "TB/s"]:
        return f"{value:.2f} {unit}"
    elif unit in ["ms", "μs"]:
        return f"{value:.2f} {unit}"
    else:
        return f"{value:.2f} {unit}" if unit else f"{value:.2f}"
