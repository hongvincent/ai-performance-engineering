#!/usr/bin/env python3
"""Update coverage documentation from authoritative tool output."""

import subprocess
import sys
import re
from pathlib import Path


def get_coverage_data():
    """Run comprehensive_coverage_check.py and parse output."""
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "tools/coverage/comprehensive_coverage_check.py")],
        capture_output=True,
        text=True,
        cwd=repo_root
    )
    
    if result.returncode != 0:
        print(f"Error running coverage check: {result.stderr}")
        return None
    
    output = result.stdout
    
    # Parse overall coverage
    overall_match = re.search(r'Concepts covered: (\d+) \((\d+\.\d+)%\)', output)
    overall_pct = overall_match.group(2) if overall_match else "unknown"
    
    # Parse chapter coverage
    chapters = {}
    for line in output.split('\n'):
        if 'ch' in line and 'Covered:' in line:
            # Extract chapter and percentage
            ch_match = re.search(r'(ch\d+):.*?Covered: \d+ \((\d+)%\)', line)
            if ch_match:
                ch_id = ch_match.group(1)
                pct = ch_match.group(2)
                chapters[ch_id] = int(pct)
    
    return {
        'overall': overall_pct,
        'chapters': chapters
    }


def update_coverage_status_md(data):
    """Update COVERAGE_STATUS.md with actual data."""
    status_file = Path(__file__).parent.parent / "COVERAGE_STATUS.md"
    
    if not status_file.exists():
        return False
    
    content = status_file.read_text()
    
    # Update overall coverage
    content = re.sub(
        r'Overall coverage: \*\*[\d.]+%\*\*',
        f'Overall coverage: **{data["overall"]}%**',
        content
    )
    
    # Count chapters at 100%
    chapters_100 = sum(1 for pct in data['chapters'].values() if pct == 100)
    content = re.sub(
        r'Chapters at 100%: \*\*\d+/20\*\*',
        f'Chapters at 100%: **{chapters_100}/20**',
        content
    )
    
    # List chapters at 100%
    ch_100_list = [ch for ch, pct in sorted(data['chapters'].items()) if pct == 100]
    if ch_100_list:
        ch_list_str = ', '.join(ch_100_list)
        # Update the list
        content = re.sub(
            r'\(ch\d+(?:, ch\d+)*\)',
            f'({ch_list_str})',
            content
        )
    
    status_file.write_text(content)
    return True


def update_coverage_report_md(data):
    """Update COVERAGE_REPORT.md with actual data."""
    report_file = Path(__file__).parent.parent / "COVERAGE_REPORT.md"
    
    if not report_file.exists():
        return False
    
    content = report_file.read_text()
    
    # Update overall coverage
    content = re.sub(
        r'\*\*Overall Coverage: [\d.]+%\*\*',
        f'**Overall Coverage: {data["overall"]}%**',
        content
    )
    
    # Update status line
    chapters_100 = sum(1 for pct in data['chapters'].values() if pct == 100)
    chapters_remaining = 20 - chapters_100
    ch_100_list = [ch for ch, pct in sorted(data['chapters'].items()) if pct == 100]
    ch_list_str = ', '.join(ch_100_list) if ch_100_list else 'none'
    content = re.sub(
        r'\*\*Status:\*\* \d+ chapters at 100%.*?still need',
        f'**Status:** {chapters_100} chapters at 100% ({ch_list_str}). {chapters_remaining} chapters still need',
        content
    )
    
    report_file.write_text(content)
    return True


def main():
    """Main entry point."""
    data = get_coverage_data()
    if not data:
        return 1
    
    print(f"Overall coverage: {data['overall']}%")
    print(f"Chapters at 100%: {sum(1 for pct in data['chapters'].values() if pct == 100)}")
    
    updated_status = update_coverage_status_md(data)
    updated_report = update_coverage_report_md(data)
    
    if updated_status and updated_report:
        print("Updated coverage documentation")
        return 0
    else:
        print("Some documentation files not updated")
        return 1


if __name__ == '__main__':
    sys.exit(main())

