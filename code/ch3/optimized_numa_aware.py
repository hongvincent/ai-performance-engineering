"""optimized_numa_aware.py - NUMA-aware memory allocation (optimized).

Actively binds the current process and memory policy to the GPU's NUMA node so
host allocations stay local to the GPU performing H2D/D2H transfers.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Set

import numpy as np

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def _parse_cpu_list(raw: str) -> Set[int]:
    """Parse Linux cpulist format (e.g., "0-7,16-23")."""
    cpus: Set[int] = set()
    for part in raw.strip().split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _node_cpu_affinity(node_id: int) -> Set[int]:
    """Return CPU IDs for the requested NUMA node."""
    cpulist_path = Path(f"/sys/devices/system/node/node{node_id}/cpulist")
    if not cpulist_path.exists():
        return set(range(os.cpu_count() or 1))
    return _parse_cpu_list(cpulist_path.read_text())


def _bus_id_for_gpu(gpu_index: int) -> Optional[str]:
    """Return PCI bus id for the target GPU using nvidia-smi or sysfs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) > gpu_index:
            return lines[gpu_index].lower()
    except Exception:
        pass
    # Fallback to sysfs numbering (best effort)
    by_path = Path("/sys/class/drm")
    for device in by_path.glob("card*/device" ):
        try:
            minor = int(device.parent.name.replace("card", ""))
        except ValueError:
            continue
        if minor == gpu_index:
            return device.resolve().name.lower()
    try:
        import torch

        if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(gpu_index)
            return f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    except Exception:
        pass
    return None


def _gpu_numa_node(gpu_index: int) -> Optional[int]:
    """Read NUMA node for the GPU via sysfs."""
    bus_id = _bus_id_for_gpu(gpu_index)
    if not bus_id:
        return None
    node_file = Path("/sys/bus/pci/devices") / bus_id / "numa_node"
    try:
        value = int(node_file.read_text().strip())
        return value if value >= 0 else None
    except Exception:
        return None


def _count_numa_nodes() -> int:
    try:
        nodes = [p for p in Path("/sys/devices/system/node").glob("node*") if p.is_dir()]
        return max(1, len(nodes))
    except Exception:
        return 1


class NUMABinder:
    """Bind the current process to the GPU's NUMA node."""

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index
        self.original_affinity: Optional[Set[int]] = None
        self.bound_node: Optional[int] = None
        self._libnuma = self._load_libnuma()
        self._last_error: Optional[str] = None

    @staticmethod
    def _load_libnuma():
        try:
            lib = ctypes.CDLL("libnuma.so.1")
            if lib.numa_available() >= 0:
                return lib
        except OSError:
            pass
        return None

    def bind(self) -> bool:
        """Bind scheduler affinity + preferred memory policy."""
        if self.bound_node is not None:
            return True
        node = _gpu_numa_node(self.gpu_index)
        if node is None:
            self._last_error = "GPU NUMA node unavailable"
            return False
        cpus = _node_cpu_affinity(node)
        if not cpus:
            self._last_error = f"No CPUs found for NUMA node {node}"
            return False
        try:
            self.original_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, cpus)
        except AttributeError:
            self._last_error = "sched_setaffinity unsupported on this platform"
            return False
        except PermissionError:
            self._last_error = "Insufficient permissions to set scheduler affinity"
            return False
        if self._libnuma is not None:
            self._libnuma.numa_run_on_node(ctypes.c_int(node))
            self._libnuma.numa_set_preferred(ctypes.c_int(node))
        self.bound_node = node
        self._last_error = None
        return True

    def restore(self) -> None:
        if self.original_affinity is not None:
            os.sched_setaffinity(0, self.original_affinity)
            self.original_affinity = None
        if self._libnuma is not None:
            self._libnuma.numa_set_preferred(ctypes.c_int(-1))
        self.bound_node = None

    @property
    def status(self) -> str:
        if self.bound_node is not None:
            return f"bound to NUMA node {self.bound_node}"
        if self._last_error:
            return self._last_error
        return "NUMA binding skipped"


class OptimizedNUMAAwareBenchmark:
    """Benchmark with enforced NUMA affinity for GPU-friendly locality."""

    def __init__(self, size_mb: int = 512, gpu_index: int = 0):
        self.size_mb = size_mb
        self.gpu_index = gpu_index
        self.data = None
        self.num_numa_nodes = _count_numa_nodes()
        self._binder = NUMABinder(gpu_index=gpu_index)
        self._binding_active = False

    def setup(self) -> None:
        self._binding_active = self._binder.bind()
        elements = self.size_mb * 1024 * 1024 // 8
        self.data = np.random.rand(elements).astype(np.float64)

    def benchmark_fn(self) -> None:
        torch_module = None
        try:
            import torch

            torch_module = torch
            if torch_module.cuda.is_available():
                torch_module.cuda.nvtx.range_push("optimized_numa_aware")
        except ImportError:
            torch_module = None
        try:
            _ = np.sum(self.data * 1.5 + 2.0)
        finally:
            try:
                if torch_module and torch_module.cuda.is_available():
                    torch_module.cuda.nvtx.range_pop()
            except (AttributeError, RuntimeError):
                pass

    def teardown(self) -> None:
        self.data = None
        self._binder.restore()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Data array not initialized"
        expected_size = self.size_mb * 1024 * 1024 // 8
        if self.data.size != expected_size:
            return f"Data size mismatch: expected {expected_size} elements, got {self.data.size}"
        if not np.isfinite(self.data).all():
            return "Data contains non-finite values"
        if self.num_numa_nodes > 1 and not self._binding_active:
            return f"NUMA binding failed: {self._binder.status}"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedNUMAAwareBenchmark(size_mb=512)


def main() -> None:
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedNUMAAwareBenchmark(size_mb=512)
    result = harness.benchmark(benchmark)

    print("=" * 70)
    print("Optimized: NUMA-Aware Memory Allocation")
    print("=" * 70)
    print(f"NUMA nodes: {benchmark.num_numa_nodes}")
    print(f"Binding status: {benchmark._binder.status}")
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
