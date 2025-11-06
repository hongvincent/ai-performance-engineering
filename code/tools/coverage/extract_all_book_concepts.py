#!/usr/bin/env python3
"""Extract all concepts from all book chapters."""

import os
import re
from pathlib import Path
from collections import defaultdict, OrderedDict

# Comprehensive concept definitions - extracted from book analysis
ALL_CONCEPTS = {
    # Performance Fundamentals
    'roofline': ['roofline', 'arithmetic intensity', 'roofline model', 'roofline analysis'],
    'benchmarking': ['benchmark', 'profiling', 'performance', 'metrics', 'nsight', 'ncu', 'profiler'],
    'goodput': ['goodput', 'useful throughput', 'throughput', 'latency', 'tput'],
    
    # Hardware Concepts
    'tensor_cores': ['tensor core', 'mma', 'wmma', 'tensor engine', 'transformer engine', 'te'],
    'nvlink': ['nvlink', 'nvswitch', 'nvlink topology', 'p2p', 'peer access'],
    'hbm': ['hbm', 'hbm3e', 'hbm3', 'memory bandwidth', 'hbm bandwidth'],
    'cuda': ['cuda', 'cuda kernel', '__global__', '__device__', '__host__', 'cuda device'],
    'memory': ['memory', 'malloc', 'cudamalloc', 'memory bandwidth', 'gpu memory', 'device memory'],
    
    # Infrastructure
    'docker': ['docker', 'container', 'dockerfile'],
    'kubernetes': ['kubernetes', 'k8s', 'pod', 'deployment', 'service'],
    
    # Memory Optimization
    'coalescing': ['coalescing', 'coalesced', 'memory coalescing', 'uncoalesced'],
    'shared_memory': ['shared memory', '__shared__', 'sharedmem', 'smem'],
    'bank_conflicts': ['bank conflict', 'bank padding', 'stride', 'bank stride'],
    'pinned_memory': ['pinned memory', 'pin_memory', 'cudahostalloc', 'cudamallochost'],
    'double_buffering': ['double buffer', 'ping pong', 'async copy', 'double buffering'],
    
    # Compute Optimization
    'tiling': ['tiling', 'tiled', 'tile size', 'tile', 'block tiling'],
    'vectorization': ['vectorization', 'vectorized', 'float4', 'make_float4', 'uint4', 'vectorized load'],
    'ilp': ['ilp', 'instruction level parallelism', 'independent operation', 'ilp optimization'],
    'occupancy': ['occupancy', 'cudaoccupancy', 'warp', 'thread block', 'block size', 'occupancy calculator'],
    'warp_divergence': ['warp divergence', '__syncwarp', 'divergence', 'branch divergence'],
    'warp_specialization': ['warp specialization', '__activemask', 'warp special', 'specialized warp'],
    
    # Kernels and Operations
    'gemm': ['gemm', 'matrix multiplication', 'matmul', 'sgemm', 'dgemm'],
    'batch': ['batch', 'batching', 'batch size', 'batched', 'batch processing'],
    'streams': ['stream', 'cuda stream', 'cudastream', 'cudastreamcreate', 'stream synchronization'],
    'cuda_graphs': ['cuda graph', 'cudagraph', 'graph capture', 'graph launch', 'cudagraphlaunch'],
    'stream_ordered': ['stream ordered', 'cudamallocasync', 'async memory', 'stream ordered allocator'],
    
    # Frameworks and Libraries
    'triton': ['triton', '@triton', 'triton.jit', 'triton kernel'],
    'cutlass': ['cutlass', 'cutlass gemm', 'cutlass library'],
    'nccl': ['nccl', 'collective', 'allreduce', 'allgather', 'alltoall', 'broadcast', 'ncclcomm'],
    
    # Attention and Transformers
    'attention': ['attention', 'self attention', 'scaled dot product attention', 'multi head attention', 'mha'],
    'flash_attention': ['flash attention', 'flashattention', 'flash_attn', 'flash attention 2'],
    'flex_attention': ['flex attention', 'flexattention', 'flex_attn', 'flexible attention'],
    'kv_cache': ['kv cache', 'key value cache', 'kv_cache', 'kv cache management'],
    'kv_cache_management': ['kv cache management', 'kv cache reuse', 'kv cache transfer', 'kv cache pooling'],
    
    # Inference Optimizations
    'continuous_batching': ['continuous batching', 'paged attention', 'vllm', 'continuous batch'],
    'paged_attention': ['paged attention', 'pagedattention', 'paged attention v2'],
    'disaggregated': ['disaggregated', 'disagg', 'prefill decode', 'prefill-decode', 'disaggregated inference'],
    'speculative_decoding': ['speculative decoding', 'speculative', 'medusa', 'eagle', 'draft model', 'target model'],
    'guided_decoding': ['guided decoding', 'constrained decoding', 'json schema', 'structured decoding'],
    
    # Model Architectures
    'moe': ['moe', 'mixture of experts', 'expert', 'expert routing', 'expert parallelism', 'moe layer'],
    
    # Parallelism Strategies
    'tensor_parallelism': ['tensor parallelism', 'tp', 'tensor parallel', 'tensor sharding'],
    'pipeline_parallelism': ['pipeline parallelism', 'pp', 'pipeline parallel', 'pipeline stage'],
    'expert_parallelism': ['expert parallelism', 'ep', 'expert parallel', 'expert sharding'],
    'data_parallelism': ['data parallelism', 'dp', 'data parallel', 'ddp', 'distributeddataparallel'],
    'context_parallelism': ['context parallelism', 'cp', 'context parallel', 'sequence parallelism'],
    'distributed': ['distributed', 'multi node', 'multinode', 'distributed training', 'distributed inference'],
    
    # Precision and Quantization
    'quantization': ['quantization', 'quantize', 'int8', 'fp8', 'fp4', 'nvfp4', 'quantized model'],
    'mixed_precision': ['mixed precision', 'amp', 'autocast', 'bf16', 'fp16', 'bfp16'],
    
    # Advanced Optimizations
    'adaptive': ['adaptive', 'dynamic', 'runtime optimization', 'adaptive optimization'],
    'autotuning': ['autotuning', 'autotune', 'kernel tuning', 'auto tuning', 'kernel autotuning'],
    'ai_optimization': ['ai optimization', 'rl', 'reinforcement learning', 'alphatensor', 'ai assisted optimization'],
}

def extract_concepts_from_text(text: str) -> set:
    """Extract all concepts from text."""
    text_lower = text.lower()
    found_concepts = set()
    
    for concept, keywords in ALL_CONCEPTS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_concepts.add(concept)
                break
    
    return found_concepts

def get_chapter_concepts(ch_num: int, repo_root: Path) -> dict:
    """Get concepts from a chapter."""
    ch_id = f"ch{ch_num}"
    md_file = repo_root / "book" / f"{ch_id}.md"
    
    result = {
        'chapter': ch_id,
        'concepts': set(),
        'concept_count': 0,
    }
    
    if md_file.exists():
        book_text = md_file.read_text()
        result['concepts'] = extract_concepts_from_text(book_text)
        result['concept_count'] = len(result['concepts'])
    
    return result

def main():
    repo_root = Path(__file__).parent.parent.parent
    
    print("=" * 80)
    print("EXTRACTING ALL BOOK CONCEPTS")
    print("=" * 80)
    print()
    
    # Extract concepts from all chapters
    all_chapter_concepts = {}
    all_unique_concepts = set()
    
    for ch_num in range(1, 21):
        result = get_chapter_concepts(ch_num, repo_root)
        all_chapter_concepts[result['chapter']] = result
        all_unique_concepts.update(result['concepts'])
    
    # Print summary
    print("1. CONCEPTS BY CHAPTER")
    print("-" * 80)
    
    for ch_id in sorted(all_chapter_concepts.keys()):
        result = all_chapter_concepts[ch_id]
        print(f"{ch_id}: {result['concept_count']:2d} concepts")
    
    print()
    print("2. ALL UNIQUE CONCEPTS IN BOOK")
    print("-" * 80)
    print(f"Total unique concepts: {len(all_unique_concepts)}")
    print()
    
    # Group concepts by category
    categories = {
        'Performance Fundamentals': ['roofline', 'benchmarking', 'goodput'],
        'Hardware': ['tensor_cores', 'nvlink', 'hbm', 'cuda', 'memory'],
        'Infrastructure': ['docker', 'kubernetes'],
        'Memory Optimization': ['coalescing', 'shared_memory', 'bank_conflicts', 'pinned_memory', 'double_buffering'],
        'Compute Optimization': ['tiling', 'vectorization', 'ilp', 'occupancy', 'warp_divergence', 'warp_specialization'],
        'Kernels & Operations': ['gemm', 'batch', 'streams', 'cuda_graphs', 'stream_ordered'],
        'Frameworks & Libraries': ['triton', 'cutlass', 'nccl'],
        'Attention & Transformers': ['attention', 'flash_attention', 'flex_attention', 'kv_cache', 'kv_cache_management'],
        'Inference Optimizations': ['continuous_batching', 'paged_attention', 'disaggregated', 'speculative_decoding', 'guided_decoding'],
        'Model Architectures': ['moe'],
        'Parallelism': ['tensor_parallelism', 'pipeline_parallelism', 'expert_parallelism', 'data_parallelism', 'context_parallelism', 'distributed'],
        'Precision': ['quantization', 'mixed_precision'],
        'Advanced': ['adaptive', 'autotuning', 'ai_optimization'],
    }
    
    # Organize concepts by category
    concepts_by_category = defaultdict(list)
    uncategorized = []
    
    for concept in sorted(all_unique_concepts):
        found = False
        for category, concept_list in categories.items():
            if concept in concept_list:
                concepts_by_category[category].append(concept)
                found = True
                break
        if not found:
            uncategorized.append(concept)
    
    if uncategorized:
        concepts_by_category['Other'] = uncategorized
    
    print("Concepts organized by category:")
    print()
    for category in sorted(concepts_by_category.keys()):
        concepts = sorted(concepts_by_category[category])
        print(f"{category}: {len(concepts)} concepts")
        for concept in concepts:
            print(f"  - {concept}")
        print()
    
    # Save to file
    output_file = repo_root / "ALL_BOOK_CONCEPTS.md"
    
    with open(output_file, 'w') as f:
        f.write("# All Concepts Covered in AI Systems Performance Engineering\n\n")
        f.write(f"**Total Unique Concepts: {len(all_unique_concepts)}**\n\n")
        
        f.write("## Concepts by Category\n\n")
        for category in sorted(concepts_by_category.keys()):
            concepts = sorted(concepts_by_category[category])
            f.write(f"### {category} ({len(concepts)} concepts)\n\n")
            for concept in concepts:
                keywords = ALL_CONCEPTS.get(concept, [])
                f.write(f"- **{concept}**: {', '.join(keywords[:5])}\n")
            f.write("\n")
        
        f.write("## Concepts by Chapter\n\n")
        for ch_id in sorted(all_chapter_concepts.keys()):
            result = all_chapter_concepts[ch_id]
            concepts = sorted(list(result['concepts']))
            f.write(f"### {ch_id}: {len(concepts)} concepts\n\n")
            f.write(", ".join(concepts))
            f.write("\n\n")
    
    print()
    print(f"[OK] Saved complete concept list to: {output_file}")
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique concepts: {len(all_unique_concepts)}")

if __name__ == "__main__":
    main()

