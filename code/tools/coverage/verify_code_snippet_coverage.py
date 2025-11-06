#!/usr/bin/env python3
"""Verify that all code snippets from book chapters are present in code/ch* directories."""

import os
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import ast
import hashlib

def normalize_code(code):
    """Normalize code for comparison (remove comments, whitespace, etc.)."""
    # Remove comments
    lines = []
    for line in code.split('\n'):
        # Remove single-line comments (but preserve strings)
        if '#' in line:
            # Simple approach: remove everything after # if not in a string
            in_string = False
            quote_char = None
            new_line = []
            for i, char in enumerate(line):
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif char == '#' and not in_string:
                    break
                new_line.append(char)
            line = ''.join(new_line)
        lines.append(line)
    
    code = '\n'.join(lines)
    
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)
    code = re.sub(r'\s*([{}();,=+\-*/])\s*', r'\1', code)
    
    return code.strip().lower()

def extract_code_blocks(md_file: Path) -> list:
    """Extract all code blocks from markdown file."""
    if not md_file.exists():
        return []
    
    text = md_file.read_text()
    
    # Match code blocks: ```language\ncode\n```
    code_blocks = []
    pattern = r'```(\w+)?\n(.*?)```'
    
    for match in re.finditer(pattern, text, re.DOTALL):
        language = match.group(1) or ''
        code = match.group(2).strip()
        
        if code and len(code) > 10:  # Ignore very short snippets
            code_blocks.append({
                'language': language.lower(),
                'code': code,
                'normalized': normalize_code(code),
                'hash': hashlib.md5(code.encode()).hexdigest()[:8],
                'first_line': code.split('\n')[0][:60],
            })
    
    return code_blocks

def find_code_in_directory(code_dir: Path, target_code: str, target_lang: str = None) -> list:
    """Search for code snippet in directory files."""
    matches = []
    
    if not code_dir.exists():
        return matches
    
    # Determine file extensions to search
    extensions = []
    if target_lang in ['python', 'py', '']:
        extensions.extend(['.py'])
    if target_lang in ['cuda', 'cu', 'c++', 'cpp', 'c', '']:
        extensions.extend(['.cu', '.cpp', '.c', '.h', '.hpp'])
    if not target_lang or target_lang == '':
        extensions.extend(['.sh', '.yaml', '.yml', '.json'])
    
    target_normalized = normalize_code(target_code)
    
    for ext in extensions:
        for code_file in code_dir.rglob(f'*{ext}'):
            # Skip common non-code files
            if any(x in code_file.name.lower() for x in ['__pycache__', '.pyc', '__init__', 'requirements', 'setup']):
                continue
            
            try:
                file_content = code_file.read_text()
                file_normalized = normalize_code(file_content)
                
                # Check if target code is contained in file
                similarity = SequenceMatcher(None, target_normalized, file_normalized).ratio()
                
                # Also check if key lines/patterns match
                target_lines = [l.strip() for l in target_code.split('\n') if l.strip() and len(l.strip()) > 5]
                matching_lines = sum(1 for line in target_lines if normalize_code(line) in file_normalized)
                
                if similarity > 0.3 or (len(target_lines) > 0 and matching_lines / len(target_lines) > 0.5):
                    matches.append({
                        'file': code_file,
                        'similarity': similarity,
                        'matching_lines': matching_lines,
                        'total_lines': len(target_lines),
                    })
            except Exception:
                pass
    
    return sorted(matches, key=lambda x: -x['similarity'])

def extract_function_signatures(code: str, language: str) -> list:
    """Extract function/class signatures from code."""
    signatures = []
    
    if language in ['python', 'py', '']:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = ', '.join([arg.arg for arg in node.args.args])
                    signatures.append(f"{node.name}({args})")
                elif isinstance(node, ast.ClassDef):
                    signatures.append(f"class {node.name}")
        except:
            pass
    
    # Also extract from patterns
    if language in ['cuda', 'cu', 'c++', 'cpp', 'c', '']:
        # Match function definitions
        func_pattern = r'(__global__|__device__|__host__)?\s*\w+\s+\w+\s*\([^)]*\)'
        for match in re.finditer(func_pattern, code):
            signatures.append(match.group(0)[:80])
    
    return signatures

def analyze_chapter(ch_num: int, repo_root: Path) -> dict:
    """Analyze a single chapter for code coverage."""
    ch_id = f"ch{ch_num}"
    md_file = repo_root / "book" / f"{ch_id}.md"
    code_dir = repo_root / ch_id
    
    result = {
        'chapter': ch_id,
        'book_code_blocks': [],
        'found_code_blocks': [],
        'missing_code_blocks': [],
        'total_blocks': 0,
        'found_blocks': 0,
        'missing_blocks': 0,
    }
    
    # Extract code blocks from book
    code_blocks = extract_code_blocks(md_file)
    result['total_blocks'] = len(code_blocks)
    result['book_code_blocks'] = code_blocks
    
    # Check each code block
    for block in code_blocks:
        matches = find_code_in_directory(code_dir, block['code'], block['language'])
        
        if matches and matches[0]['similarity'] > 0.5:
            result['found_blocks'] += 1
            result['found_code_blocks'].append({
                'block': block,
                'matches': matches[:3],  # Top 3 matches
            })
        else:
            result['missing_blocks'] += 1
            result['missing_code_blocks'].append(block)
    
    return result

def main():
    repo_root = Path(__file__).parent.parent.parent
    
    print("=" * 80)
    print("CODE SNIPPET COVERAGE VERIFICATION")
    print("=" * 80)
    print()
    
    all_results = []
    
    # Analyze each chapter
    for ch_num in range(1, 21):
        result = analyze_chapter(ch_num, repo_root)
        all_results.append(result)
    
    # Print summary
    print("1. CHAPTER-BY-CHAPTER CODE SNIPPET COVERAGE")
    print("-" * 80)
    
    total_blocks = 0
    total_found = 0
    total_missing = 0
    
    for result in all_results:
        total_blocks += result['total_blocks']
        total_found += result['found_blocks']
        total_missing += result['missing_blocks']
        
        if result['total_blocks'] == 0:
            # No code blocks found in markdown (likely non-standard format)
            status = "N/A"
            print(f"{status} {result['chapter']}: No code blocks detected in markdown "
                  f"(book uses non-standard markdown format)")
        else:
            coverage_pct = (result['found_blocks'] / result['total_blocks'] * 100)
            status = "[OK]" if coverage_pct >= 80 else "WARNING: " if coverage_pct >= 50 else "ERROR:"
            print(f"{status} {result['chapter']}: {result['found_blocks']:2d}/{result['total_blocks']:2d} "
                  f"code blocks found ({coverage_pct:.0f}%)")
    
    print()
    print("2. OVERALL STATISTICS")
    print("-" * 80)
    overall_coverage = (total_found / total_blocks * 100) if total_blocks > 0 else 0
    print(f"Total code blocks in book: {total_blocks}")
    print(f"Code blocks found in code/: {total_found} ({overall_coverage:.1f}%)")
    print(f"Code blocks missing: {total_missing}")
    print()
    
    # Show chapters with missing code
    print("3. CHAPTERS WITH MISSING CODE SNIPPETS")
    print("-" * 80)
    
    missing_chapters = [r for r in all_results if r['missing_blocks'] > 0]
    
    if missing_chapters:
        for result in sorted(missing_chapters, key=lambda x: -x['missing_blocks']):
            print(f"\n{result['chapter']}: {result['missing_blocks']} missing code blocks")
            
            # Show first few missing blocks
            for block in result['missing_code_blocks'][:5]:
                lang = block['language'] or 'text'
                print(f"  - [{lang}] {block['first_line']}... (hash: {block['hash']})")
            
            if len(result['missing_code_blocks']) > 5:
                print(f"  ... and {len(result['missing_code_blocks']) - 5} more")
    else:
        print("[OK] All code blocks found!")
    
    print()
    
    # Show chapters with good coverage
    print("4. CHAPTERS WITH GOOD CODE COVERAGE")
    print("-" * 80)
    
    good_coverage = [r for r in all_results if r['total_blocks'] > 0 and (r['found_blocks'] / r['total_blocks']) >= 0.8]
    
    if good_coverage:
        for result in sorted(good_coverage, key=lambda x: -(x['found_blocks'] / x['total_blocks'])):
            coverage_pct = (result['found_blocks'] / result['total_blocks'] * 100)
            print(f"[OK] {result['chapter']}: {result['found_blocks']}/{result['total_blocks']} "
                  f"({coverage_pct:.0f}%)")
    else:
        print("WARNING: No chapters have >80% code coverage")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    chapters_with_blocks = len([r for r in all_results if r['total_blocks'] > 0])
    
    if total_blocks == 0:
        print(f"Overall code snippet coverage: N/A (no code blocks detected in markdown)")
        print(f"Note: Book markdown files use non-standard format without ``` code fences")
        print(f"Chapters with code blocks: {chapters_with_blocks}/20")
    else:
        print(f"Overall code snippet coverage: {overall_coverage:.1f}%")
        print(f"Chapters with >80% coverage: {len(good_coverage)}/{chapters_with_blocks}")
        print(f"Chapters with missing code: {len(missing_chapters)}/{chapters_with_blocks}")
    
    if total_blocks == 0:
        print("\nWARNING: No code blocks detected - book markdown uses non-standard format")
    elif overall_coverage >= 80:
        print("\n[OK] Excellent code snippet coverage!")
    elif overall_coverage >= 60:
        print("\nWARNING: Good code snippet coverage with some gaps")
    elif overall_coverage >= 40:
        print("\nWARNING: Moderate code snippet coverage - consider adding missing examples")
    else:
        print("\nERROR: Low code snippet coverage - significant gaps need code examples")
    
    # Detailed report for chapters with low coverage
    print()
    print("=" * 80)
    print("DETAILED ANALYSIS OF MISSING CODE")
    print("=" * 80)
    
    for result in sorted(missing_chapters, key=lambda x: -(x['missing_blocks'] / max(x['total_blocks'], 1))):
        if result['missing_blocks'] == 0:
            continue
        
        print(f"\n{result['chapter']}:")
        print(f"  Missing: {result['missing_blocks']}/{result['total_blocks']} code blocks")
        
        # Group by language
        by_lang = defaultdict(list)
        for block in result['missing_code_blocks']:
            lang = block['language'] or 'unknown'
            by_lang[lang].append(block)
        
        for lang, blocks in sorted(by_lang.items()):
            print(f"    [{lang}]: {len(blocks)} blocks")
            for block in blocks[:3]:
                # Extract first meaningful line
                first_lines = [l.strip() for l in block['code'].split('\n')[:3] if l.strip()]
                preview = ' | '.join(first_lines[:2])
                if len(preview) > 100:
                    preview = preview[:100] + "..."
                print(f"      - {preview}")

if __name__ == "__main__":
    main()

