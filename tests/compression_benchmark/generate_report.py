#!/usr/bin/env python3
"""
Generate comprehensive markdown report from benchmark results.

This script processes collected benchmark results and generates a detailed
markdown report with tables, statistics, and analysis.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Try to import optional dependencies
try:
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas/tabulate not available. Using basic formatting.")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Charts will be skipped.")

from collect_metrics import load_results, generate_summary


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    return f"{value:.{decimals}f}"


def create_markdown_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
    """Create a markdown table."""
    if HAS_PANDAS:
        df = pd.DataFrame(rows, columns=headers)
        return f"\n### {title}\n\n" + df.to_markdown(index=False) + "\n"
    else:
        # Basic markdown table
        table = f"\n### {title}\n\n"
        table += "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        return table + "\n"


def generate_executive_summary(summary: Dict) -> str:
    """Generate executive summary section."""
    content = "# Compression Benchmark Report\n\n"
    content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    content += "## Executive Summary\n\n"
    
    total = summary.get("total_tests", 0)
    successful = summary.get("successful", 0)
    failed = summary.get("failed", 0)
    success_rate = summary.get("success_rate", 0)
    
    content += f"- **Total Tests**: {total}\n"
    content += f"- **Successful**: {successful} ({success_rate:.1f}%)\n"
    content += f"- **Failed**: {failed}\n\n"
    
    # Overall compression metrics
    compression = summary.get("overall_metrics", {}).get("compression", {})
    if compression:
        ratio_stats = compression.get("compression_ratio", {})
        if ratio_stats:
            content += "### Overall Compression Performance\n\n"
            content += f"- **Average Compression**: {ratio_stats.get('mean', 0):.1f}%\n"
            content += f"- **Median Compression**: {ratio_stats.get('median', 0):.1f}%\n"
            content += f"- **Range**: {ratio_stats.get('min', 0):.1f}% - {ratio_stats.get('max', 0):.1f}%\n"
            content += f"- **Standard Deviation**: {ratio_stats.get('std_dev', 0):.1f}%\n\n"
    
    # Overall quality metrics
    quality = summary.get("overall_metrics", {}).get("quality", {})
    if quality:
        overall_score = quality.get("overall_score", {})
        if overall_score:
            content += "### Overall Quality Performance\n\n"
            content += f"- **Average Quality Score**: {overall_score.get('mean', 0):.1f}%\n"
            content += f"- **Median Quality Score**: {overall_score.get('median', 0):.1f}%\n"
            content += f"- **Range**: {overall_score.get('min', 0):.1f}% - {overall_score.get('max', 0):.1f}%\n\n"
    
    return content


def generate_test_configuration(results_data: Dict) -> str:
    """Generate test configuration section."""
    content = "## Test Configuration\n\n"
    
    results = results_data.get("results", [])
    if results:
        # Get unique values
        categories = set(r.get("category") for r in results if r.get("category"))
        levels = set(r.get("compression_level") for r in results if r.get("compression_level"))
        scoring_modes = set(r.get("scoring_mode") for r in results if r.get("scoring_mode"))
        models = set(r.get("model") for r in results if r.get("model"))
        
        content += "### Test Matrix\n\n"
        content += f"- **Categories**: {', '.join(sorted(categories))}\n"
        content += f"- **Compression Levels**: {', '.join(sorted(levels))}\n"
        content += f"- **Scoring Modes**: {', '.join(sorted(scoring_modes))}\n"
        content += f"- **Models**: {', '.join(sorted(models))}\n"
        content += f"- **Total Combinations**: {len(results)}\n\n"
    
    content += "### Test Environment\n\n"
    content += "- **Tool**: Tokuin compression feature\n"
    content += "- **Feature Flags**: compression, compression-embeddings\n"
    content += "- **Test Dataset**: Real-world prompts from public sources\n\n"
    
    return content


def generate_compression_performance(summary: Dict) -> str:
    """Generate compression performance section."""
    content = "## Compression Performance\n\n"
    
    by_level = summary.get("by_dimension", {}).get("by_level", {})
    by_scoring = summary.get("by_dimension", {}).get("by_scoring", {})
    by_category = summary.get("by_dimension", {}).get("by_category", {})
    
    # Table by compression level
    if by_level:
        content += "### Performance by Compression Level\n\n"
        headers = ["Level", "Avg Compression %", "Median %", "Min %", "Max %", "Std Dev %", "Tests"]
        rows = []
        for level in ["light", "medium", "aggressive"]:
            if level in by_level:
                comp = by_level[level].get("compression", {}).get("compression_ratio", {})
                if comp:
                    rows.append([
                        level.capitalize(),
                        format_number(comp.get("mean", 0)),
                        format_number(comp.get("median", 0)),
                        format_number(comp.get("min", 0)),
                        format_number(comp.get("max", 0)),
                        format_number(comp.get("std_dev", 0)),
                        by_level[level].get("count", 0),
                    ])
        if rows:
            content += create_markdown_table(headers, rows)
    
    # Table by scoring mode
    if by_scoring:
        content += "### Performance by Scoring Mode\n\n"
        headers = ["Scoring Mode", "Avg Compression %", "Median %", "Min %", "Max %", "Std Dev %", "Tests"]
        rows = []
        for scoring in ["heuristic", "semantic", "hybrid"]:
            if scoring in by_scoring:
                comp = by_scoring[scoring].get("compression", {}).get("compression_ratio", {})
                if comp:
                    rows.append([
                        scoring.capitalize(),
                        format_number(comp.get("mean", 0)),
                        format_number(comp.get("median", 0)),
                        format_number(comp.get("min", 0)),
                        format_number(comp.get("max", 0)),
                        format_number(comp.get("std_dev", 0)),
                        by_scoring[scoring].get("count", 0),
                    ])
        if rows:
            content += create_markdown_table(headers, rows)
    
    # Table by category
    if by_category:
        content += "### Performance by Prompt Category\n\n"
        headers = ["Category", "Avg Compression %", "Median %", "Min %", "Max %", "Std Dev %", "Tests"]
        rows = []
        for category in sorted(by_category.keys()):
            comp = by_category[category].get("compression", {}).get("compression_ratio", {})
            if comp:
                rows.append([
                    category.capitalize(),
                    format_number(comp.get("mean", 0)),
                    format_number(comp.get("median", 0)),
                    format_number(comp.get("min", 0)),
                    format_number(comp.get("max", 0)),
                    format_number(comp.get("std_dev", 0)),
                    by_category[category].get("count", 0),
                ])
        if rows:
            content += create_markdown_table(headers, rows)
    
    return content


def generate_quality_analysis(summary: Dict) -> str:
    """Generate quality analysis section."""
    content = "## Quality Analysis\n\n"
    
    quality = summary.get("overall_metrics", {}).get("quality", {})
    if not quality:
        content += "No quality metrics available.\n\n"
        return content
    
    # Overall quality metrics table
    headers = ["Metric", "Mean %", "Median %", "Min %", "Max %", "Std Dev %"]
    rows = []
    
    for metric_name, metric_key in [
        ("Overall Score", "overall_score"),
        ("Semantic Similarity", "semantic_similarity"),
        ("Critical Instruction Preservation", "critical_instruction_preservation"),
        ("Information Retention", "information_retention"),
        ("Structural Integrity", "structural_integrity"),
    ]:
        if metric_key in quality:
            stats = quality[metric_key]
            rows.append([
                metric_name,
                format_number(stats.get("mean", 0)),
                format_number(stats.get("median", 0)),
                format_number(stats.get("min", 0)),
                format_number(stats.get("max", 0)),
                format_number(stats.get("std_dev", 0)),
            ])
    
    if rows:
        content += "### Overall Quality Metrics\n\n"
        content += create_markdown_table(headers, rows)
    
    # Quality by compression level
    by_level = summary.get("by_dimension", {}).get("by_level", {})
    if by_level:
        content += "### Quality by Compression Level\n\n"
        headers = ["Level", "Avg Quality Score %", "Median %", "Min %", "Max %", "Tests"]
        rows = []
        for level in ["light", "medium", "aggressive"]:
            if level in by_level:
                qm = by_level[level].get("quality", {}).get("overall_score", {})
                if qm:
                    rows.append([
                        level.capitalize(),
                        format_number(qm.get("mean", 0)),
                        format_number(qm.get("median", 0)),
                        format_number(qm.get("min", 0)),
                        format_number(qm.get("max", 0)),
                        by_level[level].get("count", 0),
                    ])
        if rows:
            content += create_markdown_table(headers, rows)
    
    return content


def generate_performance_analysis(summary: Dict) -> str:
    """Generate performance analysis section."""
    content = "## Performance Analysis\n\n"
    
    compression = summary.get("overall_metrics", {}).get("compression", {})
    if compression:
        time_stats = compression.get("compression_time_ms", {})
        if time_stats:
            content += "### Compression Speed\n\n"
            content += f"- **Average Time**: {time_stats.get('mean', 0):.2f} ms\n"
            content += f"- **Median Time**: {time_stats.get('median', 0):.2f} ms\n"
            content += f"- **Min Time**: {time_stats.get('min', 0):.2f} ms\n"
            content += f"- **Max Time**: {time_stats.get('max', 0):.2f} ms\n"
            content += f"- **Standard Deviation**: {time_stats.get('std_dev', 0):.2f} ms\n\n"
    
    return content


def generate_scenario_analysis(summary: Dict) -> str:
    """Generate scenario analysis section."""
    content = "## Scenario Analysis\n\n"
    
    best_worst = summary.get("best_worst", {})
    
    # Best compression
    best_comp = best_worst.get("compression", {}).get("best")
    if best_comp:
        content += "### Best Compression Performance\n\n"
        content += f"- **Prompt**: {best_comp.get('prompt_name', 'Unknown')}\n"
        content += f"- **Category**: {best_comp.get('category', 'Unknown')}\n"
        content += f"- **Level**: {best_comp.get('compression_level', 'Unknown')}\n"
        content += f"- **Scoring**: {best_comp.get('scoring_mode', 'Unknown')}\n"
        content += f"- **Compression**: {best_comp.get('compression_percentage', 0):.1f}%\n"
        content += f"- **Tokens Saved**: {best_comp.get('tokens_saved', 0)}\n\n"
    
    # Worst compression
    worst_comp = best_worst.get("compression", {}).get("worst")
    if worst_comp:
        content += "### Worst Compression Performance\n\n"
        content += f"- **Prompt**: {worst_comp.get('prompt_name', 'Unknown')}\n"
        content += f"- **Category**: {worst_comp.get('category', 'Unknown')}\n"
        content += f"- **Level**: {worst_comp.get('compression_level', 'Unknown')}\n"
        content += f"- **Scoring**: {worst_comp.get('scoring_mode', 'Unknown')}\n"
        content += f"- **Compression**: {worst_comp.get('compression_percentage', 0):.1f}%\n"
        content += f"- **Tokens Saved**: {worst_comp.get('tokens_saved', 0)}\n\n"
    
    return content


def generate_recommendations(summary: Dict) -> str:
    """Generate recommendations section."""
    content = "## Recommendations\n\n"
    
    by_level = summary.get("by_dimension", {}).get("by_level", {})
    by_scoring = summary.get("by_dimension", {}).get("by_scoring", {})
    
    # Analyze best combinations
    recommendations = []
    
    if by_level and by_scoring:
        # Find best level
        best_level = None
        best_level_score = 0
        for level, data in by_level.items():
            comp = data.get("compression", {}).get("compression_ratio", {}).get("mean", 0)
            quality = data.get("quality", {}).get("overall_score", {}).get("mean", 0)
            # Combined score (weighted: 60% compression, 40% quality)
            score = comp * 0.6 + quality * 0.4
            if score > best_level_score:
                best_level_score = score
                best_level = level
        
        if best_level:
            recommendations.append(
                f"- **Best Compression Level**: {best_level.capitalize()} - "
                f"Provides optimal balance between compression ratio and quality"
            )
        
        # Find best scoring mode
        best_scoring = None
        best_scoring_score = 0
        for scoring, data in by_scoring.items():
            comp = data.get("compression", {}).get("compression_ratio", {}).get("mean", 0)
            quality = data.get("quality", {}).get("overall_score", {}).get("mean", 0)
            score = comp * 0.6 + quality * 0.4
            if score > best_scoring_score:
                best_scoring_score = score
                best_scoring = scoring
        
        if best_scoring:
            recommendations.append(
                f"- **Best Scoring Mode**: {best_scoring.capitalize()} - "
                f"Provides best overall performance"
            )
    
    if not recommendations:
        recommendations.append("- Analysis pending - run benchmarks to generate recommendations")
    
    content += "\n".join(recommendations) + "\n\n"
    
    # Usage recommendations
    content += "### Usage Recommendations\n\n"
    content += "- **For Maximum Compression**: Use `aggressive` level with `heuristic` scoring\n"
    content += "- **For Best Quality**: Use `light` level with `semantic` or `hybrid` scoring\n"
    content += "- **For Balanced Performance**: Use `medium` level with `hybrid` scoring\n"
    content += "- **For Structured Content**: Enable `--structured` flag for better JSON/HTML preservation\n\n"
    
    return content


def generate_thresholds(summary: Dict) -> str:
    """Generate success thresholds based on results."""
    content = "## Success Thresholds\n\n"
    
    compression = summary.get("overall_metrics", {}).get("compression", {})
    quality = summary.get("overall_metrics", {}).get("quality", {})
    
    if compression and quality:
        comp_mean = compression.get("compression_ratio", {}).get("mean", 0)
        quality_mean = quality.get("overall_score", {}).get("mean", 0)
        
        content += "Based on empirical results from this benchmark:\n\n"
        content += f"- **Minimum Acceptable Compression**: {max(30.0, comp_mean * 0.7):.1f}% "
        content += f"(70% of average: {comp_mean:.1f}%)\n"
        content += f"- **Minimum Acceptable Quality**: {max(60.0, quality_mean * 0.85):.1f}% "
        content += f"(85% of average: {quality_mean:.1f}%)\n"
        content += f"- **Target Compression Range**: {comp_mean * 0.8:.1f}% - {comp_mean * 1.2:.1f}%\n"
        content += f"- **Target Quality Range**: {quality_mean * 0.9:.1f}% - 100%\n\n"
    else:
        content += "Run benchmarks to generate thresholds based on empirical data.\n\n"
    
    return content


def main():
    """Generate the complete report."""
    results_file = Path(__file__).parent / "results" / "benchmark_results.json"
    output_file = Path(__file__).parent.parent.parent / "COMPRESSION_BENCHMARK_REPORT.md"
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run 'python run_benchmark.py' first to generate results.")
        sys.exit(1)
    
    print(f"Loading results from {results_file}...")
    results_data = load_results(results_file)
    
    print("Generating summary...")
    summary = generate_summary(results_data)
    
    print("Generating report sections...")
    report = ""
    report += generate_executive_summary(summary)
    report += generate_test_configuration(results_data)
    report += generate_compression_performance(summary)
    report += generate_quality_analysis(summary)
    report += generate_performance_analysis(summary)
    report += generate_scenario_analysis(summary)
    report += generate_recommendations(summary)
    report += generate_thresholds(summary)
    
    # Write report
    output_file.write_text(report, encoding='utf-8')
    print(f"\n✓ Report generated: {output_file}")
    print(f"  Total length: {len(report)} characters")
    print(f"  Sections: 8")


if __name__ == "__main__":
    main()

