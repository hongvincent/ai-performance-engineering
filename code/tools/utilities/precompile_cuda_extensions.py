#!/usr/bin/env python3
"""Pre-compile CUDA extensions to avoid runtime segfaults.

This script compiles all CUDA extensions before running benchmarks,
ensuring hardware compatibility is checked upfront rather than during
benchmark execution.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from ch12.cuda_extensions import load_graph_bandwidth_extension


def precompile_extensions():
    """Pre-compile all CUDA extensions."""
    print("=" * 80)
    print("Pre-compiling CUDA Extensions")
    print("=" * 80)
    print()
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available - skipping extension compilation")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    success = True
    
    # Pre-compile graph bandwidth extension
    print("Compiling graph_bandwidth extension...")
    try:
        ext = load_graph_bandwidth_extension()
        print("  [OK] graph_bandwidth extension compiled successfully")
        print(f"  Module: {ext}")
    except Exception as e:
        print(f"  ERROR: Failed to compile graph_bandwidth extension: {e}")
        success = False
    
    print()
    
    if success:
        print("[OK] All CUDA extensions pre-compiled successfully")
        print("   Extensions are now cached and ready for use")
    else:
        print("WARNING: Some extensions failed to compile")
        print("   Benchmarks using these extensions may fail")
    
    return success


if __name__ == "__main__":
    success = precompile_extensions()
    sys.exit(0 if success else 1)

