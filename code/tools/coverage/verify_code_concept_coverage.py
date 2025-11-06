#!/usr/bin/env python3
"""Verify that all book concepts are covered in code/chXX/ directories."""

import os
import re
from pathlib import Path
from collections import defaultdict

# Concept keywords mapped to implementation patterns
CONCEPT_PATTERNS = {
    'pinned_memory': [r'pin_memory', r'pinned.*memory', r'cudaHostAlloc', r'cudaMallocHost'],
    'cuda_graphs': [r'cudaGraph', r'cuda.*graph', r'cudaGraphLaunch', r'graph.*capture'],
    'batch_size': [r'batch.*size', r'batch_size', r'BATCH_SIZE'],
    'coalescing': [r'coalesc', r'coalesced', r'uncoalesced'],
    'vectorization': [r'vectoriz', r'vectorized', r'float4', r'float8', r'uint4', r'make_float4'],
    'tensor_cores': [r'tensor.*core', r'tcgen', r'wmma', r'mma\.sync', r'ldmatrix'],
    'shared_memory': [r'__shared__', r'shared.*memory', r'sharedMemory'],
    'tiling': [r'tiling', r'tiled', r'TILE'],
    'bank_conflicts': [r'bank.*conflict', r'padding', r'stride.*32'],
    'nvlink': [r'nvlink', r'NVLink', r'p2p', r'peer.*access'],
    'hbm3e': [r'hbm3e', r'HBM3e', r'bandwidth.*test'],
    'tma': [r'tma', r'TMA', r'tensor.*memory.*accelerator', r'cp\.async'],
    'fp8': [r'fp8', r'FP8', r'__nv_fp8'],
    'fp4': [r'fp4', r'FP4'],
    'fp6': [r'fp6', r'FP6'],
    'moe': [r'moe', r'MoE', r'mixture.*expert', r'expert.*routing'],
    'all_to_all': [r'all.*to.*all', r'alltoall', r'AllToAll'],
    'continuous_batching': [r'continuous.*batch', r'continuous_batching', r'vLLM'],
    'kv_cache': [r'kv.*cache', r'KV.*cache', r'key.*value.*cache', r'kv_cache', r'KVCache'],
    'attention': [r'attention', r'Attention', r'scaled.*dot.*product', r'multi.*head.*attention', r'mha'],
    'paged_attention': [r'paged.*attention', r'PagedAttention'],
    'flash_attention': [r'flash.*attention', r'FlashAttention', r'flash_attention'],
    'flex_attention': [r'flex.*attention', r'FlexAttention'],
    'guided_decoding': [r'guided.*decoding', r'guided_decoding', r'constrained.*decoding', r'json.*schema', r'structured.*decoding'],
    'torch_compile': [r'torch\.compile', r'torch_compile'],
    'triton': [r'@triton', r'import triton', r'triton\.jit'],
    'fsdp': [r'fsdp', r'FSDP', r'FullySharded'],
    'quantization': [r'quantiz', r'quantize', r'quantized'],
    'profiling': [r'profiling', r'profile', r'nsys', r'ncu', r'nsight'],
    'bandwidth': [r'bandwidth', r'Bandwidth', r'GB/s', r'TB/s'],
    'throughput': [r'throughput', r'Throughput', r'requests.*second', r'req/s'],
    'latency': [r'latency', r'Latency', r'ms/iter', r'us/iter'],
    'goodput': [r'goodput', r'Goodput'],
    'occupancy': [r'occupancy', r'Occupancy', r'cudaOccupancy'],
    'streams': [r'cudaStream', r'cuda.*stream', r'stream.*create'],
    'roofline': [r'roofline', r'Roofline'],
    'arithmetic_intensity': [r'arithmetic.*intensity', r'Arithmetic.*Intensity', r'AI\s*='],
    'loop_unrolling': [r'unroll', r'#pragma unroll'],
    'kernel_fusion': [r'fusion', r'fused', r'fuse'],
    'warp_specialization': [r'warp.*special', r'__activemask'],
    'dynamic_parallelism': [r'dynamic.*parallel', r'device.*launch'],
    'stream_ordered_allocator': [r'stream.*ordered', r'cudaMallocAsync'],
    'symmetric_memory': [r'symmetric.*memory', r'cuMemMap'],
    'disaggregated_inference': [r'disaggregat', r'prefill.*decode'],
    'early_exit': [r'early.*exit', r'early.*reject'],
    'dynamic_routing': [r'dynamic.*rout', r'route.*complexity'],
    'mla': [r'mla', r'MLA', r'multi.*head.*latent'],
    'sliding_window': [r'sliding.*window', r'local.*attention'],
    'gpudirect_storage': [r'gpudirect', r'GDS', r'cuFile', r'cufile'],
    'numa': [r'numa', r'NUMA', r'numa.*bind'],
    'distributed_training': [r'distributed', r'DDP', r'DataParallel', r'torchrun'],
    'tensor_parallelism': [r'tensor.*parallel', r'TP'],
    'pipeline_parallelism': [r'pipeline.*parallel', r'PP'],
    'nccl': [r'nccl', r'NCCL', r'ncclComm'],
    'cutlass': [r'cutlass', r'CUTLASS'],
    'compiled_autograd': [r'compiled.*autograd', r'CompiledAutograd'],
    'gradient_checkpointing': [r'gradient.*checkpoint', r'checkpoint'],
    'mixed_precision': [r'mixed.*precision', r'amp', r'autocast'],
    'dataloader_optimization': [r'dataloader', r'DataLoader', r'num_workers'],
    'double_buffering': [r'double.*buffer', r'ping.*pong'],
    'thread_block_clusters': [r'cluster', r'cluster.*dim'],
    'warp_divergence': [r'divergence', r'__syncwarp'],
    'predication': [r'predicate', r'predicated'],
    'instruction_level_parallelism': [r'ilp', r'ILP', r'independent.*operation'],
}

def detect_concepts_in_file(file_path: Path) -> set:
    """Detect concepts implemented in a code file."""
    concepts = set()
    
    try:
        content = file_path.read_text()
        content_lower = content.lower()
        
        # Check each concept pattern
        for concept, patterns in CONCEPT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    concepts.add(concept)
                    break
        
        # Also check filename
        filename_lower = file_path.name.lower()
        for concept, patterns in CONCEPT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename_lower, re.IGNORECASE):
                    concepts.add(concept)
                    break
        
    except Exception as e:
        pass
    
    return concepts

def extract_book_concepts_improved(md_file: Path) -> set:
    """Extract key concepts from book chapter - improved version."""
    if not md_file.exists():
        return set()
    
    text = md_file.read_text()
    text_lower = text.lower()
    
    concepts = set()
    
    # Check for each concept in the book text
    for concept, patterns in CONCEPT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                concepts.add(concept)
                break
    
    # Also extract from headings
    headings = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
    for heading in headings[:50]:  # First 50 headings
        heading_lower = heading.lower()
        for concept, patterns in CONCEPT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, heading_lower, re.IGNORECASE):
                    concepts.add(concept)
                    break
    
    return concepts

def main():
    repo_root = Path(__file__).parent.parent.parent
    book_dir = repo_root / "book"
    
    print("=" * 80)
    print("CODE CONCEPT COVERAGE VERIFICATION")
    print("=" * 80)
    print()
    
    coverage_report = {}
    
    # Analyze each chapter
    for ch_num in range(1, 21):
        ch_id = f"ch{ch_num}"
        md_file = book_dir / f"{ch_id}.md"
        code_dir = repo_root / ch_id
        
        book_concepts = extract_book_concepts_improved(md_file)
        
        # Analyze all code files
        code_concepts = set()
        code_files = []
        
        if code_dir.exists():
            for ext in ['.py', '.cu', '.cpp', '.c', '.h', '.hpp', '.sh']:
                # Use recursive glob to avoid duplicate file counts
                code_files.extend(list(code_dir.glob(f'**/*{ext}')))
            
            # Filter out common non-code files
            code_files = [f for f in code_files if not any(
                x in f.name.lower() for x in ['__pycache__', '.pyc', '__init__', 'requirements', 'setup']
            )]
            
            # Detect concepts in each file
            for f in code_files:
                file_concepts = detect_concepts_in_file(f)
                code_concepts.update(file_concepts)
        
        # Calculate coverage
        covered = book_concepts & code_concepts
        missing = book_concepts - code_concepts
        extra = code_concepts - book_concepts
        
        coverage_pct = (len(covered) / len(book_concepts) * 100) if book_concepts else 0
        
        coverage_report[ch_id] = {
            'book_concepts': len(book_concepts),
            'code_concepts': len(code_concepts),
            'covered': len(covered),
            'missing': len(missing),
            'file_count': len(code_files),
            'coverage_pct': coverage_pct,
            'missing_concepts': sorted(list(missing))[:15],
            'covered_concepts': sorted(list(covered))[:15],
        }
    
    # Print summary
    print("1. CHAPTER-BY-CHAPTER COVERAGE")
    print("-" * 80)
    
    total_book_concepts = 0
    total_code_concepts = 0
    total_covered = 0
    total_missing = 0
    
    for ch_id in sorted(coverage_report.keys()):
        report = coverage_report[ch_id]
        total_book_concepts += report['book_concepts']
        total_code_concepts += report['code_concepts']
        total_covered += report['covered']
        total_missing += report['missing']
        
        status = "[OK]" if report['coverage_pct'] >= 70 else "WARNING: " if report['coverage_pct'] >= 40 else "ERROR:"
        
        print(f"{status} {ch_id}: {report['file_count']:2d} files | "
              f"Book: {report['book_concepts']:2d} | "
              f"Code: {report['code_concepts']:2d} | "
              f"Covered: {report['covered']:2d} ({report['coverage_pct']:.0f}%)")
    
    print()
    print("2. OVERALL STATISTICS")
    print("-" * 80)
    avg_coverage = (total_covered / total_book_concepts * 100) if total_book_concepts > 0 else 0
    print(f"Total book concepts: {total_book_concepts}")
    print(f"Total code concepts found: {total_code_concepts}")
    print(f"Concepts covered: {total_covered} ({avg_coverage:.1f}%)")
    print(f"Concepts missing: {total_missing}")
    print()
    
    # Find chapters with low coverage
    print("3. CHAPTERS NEEDING MORE CODE EXAMPLES")
    print("-" * 80)
    low_coverage = [(ch_id, r) for ch_id, r in coverage_report.items() if r['coverage_pct'] < 50]
    
    if low_coverage:
        for ch_id, report in sorted(low_coverage, key=lambda x: x[1]['coverage_pct']):
            print(f"\n{ch_id}: {report['coverage_pct']:.0f}% coverage ({report['covered']}/{report['book_concepts']})")
            if report['missing_concepts']:
                print(f"  Missing: {', '.join(report['missing_concepts'][:10])}")
    else:
        print("[OK] All chapters have >50% coverage!")
    
    print()
    
    # Find chapters with high coverage
    print("4. CHAPTERS WITH GOOD CODE COVERAGE")
    print("-" * 80)
    high_coverage = [(ch_id, r) for ch_id, r in coverage_report.items() if r['coverage_pct'] >= 70]
    
    if high_coverage:
        for ch_id, report in sorted(high_coverage, key=lambda x: -x[1]['coverage_pct']):
            print(f"[OK] {ch_id}: {report['coverage_pct']:.0f}% coverage ({report['covered']}/{report['book_concepts']} concepts)")
            if report['covered_concepts']:
                print(f"   Covered: {', '.join(report['covered_concepts'][:10])}")
    else:
        print("WARNING: No chapters have >70% coverage")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Overall concept coverage: {avg_coverage:.1f}%")
    print(f"Chapters with >70% coverage: {len(high_coverage)}/20")
    print(f"Chapters with <50% coverage: {len(low_coverage)}/20")
    
    if avg_coverage >= 70:
        print("\n[OK] Excellent concept coverage across code examples!")
    elif avg_coverage >= 50:
        print("\nWARNING: Good concept coverage with some gaps")
    elif avg_coverage >= 30:
        print("\nWARNING: Moderate concept coverage - consider adding more examples")
    else:
        print("\nERROR: Low concept coverage - significant gaps need code examples")

if __name__ == "__main__":
    main()

