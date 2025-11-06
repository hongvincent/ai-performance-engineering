#!/usr/bin/env python3
"""Create detailed performance summary with actual timings."""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
performance_results_file = repo_root / 'performance_results.json'

with open(performance_results_file, 'r') as f:
    data = json.load(f)

results = data['results']
performance_data = data['performance_data']

print("="*120)
print("DETAILED PERFORMANCE SUMMARY - All Examples")
print("="*120)
print(f"{'Chapter':<8} {'Example':<35} {'Baseline (ms)':<18} {'Optimized (ms)':<18} {'Speedup':<12} {'Status':<15} {'Notes':<20}")
print("-"*120)

for data in sorted(performance_data, key=lambda x: x['speedup'], reverse=True):
    baseline = data['baseline_ms']
    optimized = data['optimized_ms']
    
    baseline_str = f"{baseline:.2f}" if isinstance(baseline, (int, float)) else "N/A"
    optimized_str = f"{optimized:.2f}" if isinstance(optimized, (int, float)) else "N/A"
    speedup = data['speedup']
    
    status = "Excellent" if speedup >= 1.5 else "Good" if speedup >= 1.2 else "Needs Work" if speedup >= 0.8 else "Critical"
    
    # Notes
    notes = ""
    if speedup < 0.5:
        notes = "Wrong comparison?"
    elif speedup < 0.8:
        notes = "Compilation overhead?"
    elif speedup < 1.1:
        notes = "Small workload?"
    
    print(f"{data['chapter']:<8} {data['example'][:33]:<35} {baseline_str:<18} {optimized_str:<18} {speedup:.2f}x{'':<8} {status:<15} {notes:<20}")

print("\n" + "="*120)
print("OPTIMIZATION PRIORITIES")
print("="*120)

critical = [d for d in performance_data if d['speedup'] < 0.8]
needs_work = [d for d in performance_data if 0.8 <= d['speedup'] < 1.2]
good = [d for d in performance_data if d['speedup'] >= 1.2]

print(f"\nERROR: Critical Issues ({len(critical)}): Speedup < 0.8x")
for d in sorted(critical, key=lambda x: x['speedup']):
    print(f"  - {d['chapter']}: {d['example']} ({d['speedup']:.2f}x)")

print(f"\nWARNING: Needs Work ({len(needs_work)}): Speedup 0.8-1.2x")
for d in sorted(needs_work, key=lambda x: x['speedup']):
    print(f"  - {d['chapter']}: {d['example']} ({d['speedup']:.2f}x)")

print(f"\nGood Performance ({len(good)}): Speedup >= 1.2x")
for d in sorted(good, key=lambda x: x['speedup'], reverse=True):
    print(f"  - {d['chapter']}: {d['example']} ({d['speedup']:.2f}x)")

print("\n" + "="*120)
print("OVERALL STATISTICS")
print("="*120)
print(f"Total examples: {len(performance_data)}")
print(f"Excellent (≥1.5x): {len([d for d in performance_data if d['speedup'] >= 1.5])}")
print(f"Good (≥1.2x): {len([d for d in performance_data if d['speedup'] >= 1.2])}")
print(f"Needs work (<1.2x): {len([d for d in performance_data if d['speedup'] < 1.2])}")
print(f"Average speedup: {sum(d['speedup'] for d in performance_data) / len(performance_data):.2f}x")
print(f"Best speedup: {max(d['speedup'] for d in performance_data):.2f}x")
print(f"Worst speedup: {min(d['speedup'] for d in performance_data):.2f}x")

