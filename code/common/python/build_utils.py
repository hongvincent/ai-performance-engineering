"""Build utilities to prevent hangs from stale locks and stuck processes."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def cleanup_stale_build_locks(build_dir: Path, max_lock_age_seconds: int = 300) -> None:
    """Remove stale build lock files and kill processes holding them.
    
    Args:
        build_dir: Directory containing build artifacts (may contain lock files)
        max_lock_age_seconds: Consider locks older than this stale (default 5 minutes)
    """
    if not build_dir.exists():
        return
    
    lock_files = list(build_dir.glob("lock")) + list(build_dir.glob("*.lock"))
    lock_files.extend(build_dir.glob(".ninja_lock"))
    
    for lock_file in lock_files:
        if not lock_file.exists():
            continue
        
        try:
            # Check if lock file is stale
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age > max_lock_age_seconds:
                # Lock is stale - try to find and kill the process
                _kill_processes_using_directory(build_dir)
                # Remove stale lock
                try:
                    lock_file.unlink()
                except OSError:
                    pass  # May be locked by another process
        except OSError:
            pass  # File may have been deleted


def _kill_processes_using_directory(directory: Path) -> None:
    """Kill processes that are using files in the given directory."""
    if not PSUTIL_AVAILABLE:
        return  # Can't check processes without psutil
    
    try:
        directory_str = str(directory.resolve())
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if process is Python and might be compiling
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline:
                        cmdline_str = ' '.join(cmdline)
                        # Check if it's loading CUDA extensions
                        if any(keyword in cmdline_str.lower() for keyword in 
                               ['cuda_extensions', 'load', 'extension', 'compile']):
                            # Check if process has open files in this directory
                            try:
                                open_files = proc.open_files()
                                for file_info in open_files:
                                    if directory_str in file_info.path:
                                        print(f"Killing stale process {proc.info['pid']} using {directory_str}")
                                        proc.kill()
                                        break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass  # Fail silently if we can't check processes


def ensure_clean_build_directory(build_dir: Path, max_lock_age_seconds: int = 300) -> None:
    """Ensure build directory is clean before starting a build.
    
    This is a convenience function that:
    1. Cleans stale locks
    2. Removes old build artifacts if needed
    
    Args:
        build_dir: Directory containing build artifacts
        max_lock_age_seconds: Consider locks older than this stale
    """
    cleanup_stale_build_locks(build_dir, max_lock_age_seconds)
    
    # Optionally clean old .o files if directory is getting large
    if build_dir.exists():
        object_files = list(build_dir.glob("*.o"))
        if len(object_files) > 50:  # Too many object files
            # Clean old object files (keep .so files)
            for obj_file in object_files:
                try:
                    if time.time() - obj_file.stat().st_mtime > 3600:  # Older than 1 hour
                        obj_file.unlink()
                except OSError:
                    pass

