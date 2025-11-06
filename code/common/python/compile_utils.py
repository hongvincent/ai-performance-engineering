"""Utilities for safely applying torch.compile and precision defaults."""

from __future__ import annotations

from typing import Any, Optional
from types import ModuleType
import warnings

import torch


def compile_model(module: torch.nn.Module, **_: Any) -> torch.nn.Module:
    """
    Placeholder compile helper.

    Historically these benchmarks relied on chapter-specific `arch_config`
    modules that exposed a `compile_model()` wrapper. Many chapters still
    import that helper defensively, but most workloads run uncompiled for
    stability. For now, keep behaviour identical to the legacy fallback by
    returning the module unchanged.
    """
    return module


def enable_tf32(
    *,
    matmul_precision: str = "tf32",
    cudnn_precision: str = "tf32",
    set_global_precision: bool = True,
) -> None:
    """
    Configure TF32 execution using the new PyTorch 2.9 APIs only.

    Parameters
    ----------
    matmul_precision:
        Precision setting forwarded to ``torch.backends.cuda.matmul.fp32_precision``.
    cudnn_precision:
        Precision setting forwarded to ``torch.backends.cudnn.conv.fp32_precision``.
    set_global_precision:
        When True, call ``torch.set_float32_matmul_precision('high')`` to
        ensure matmul kernels fall back to TF32-capable tensor cores.
    """
    # Suppress TF32 API deprecation warnings
    # PyTorch internally uses deprecated APIs when set_float32_matmul_precision is called
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*", category=UserWarning)
        
        if set_global_precision:
            try:
                torch.set_float32_matmul_precision("high")
            except (AttributeError, RuntimeError):
                pass

        matmul_backend: Optional[ModuleType] = getattr(torch.backends.cuda, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
            try:
                matmul_backend.fp32_precision = matmul_precision
            except RuntimeError:
                pass

        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            try:
                cudnn_conv.fp32_precision = cudnn_precision
            except RuntimeError:
                pass
