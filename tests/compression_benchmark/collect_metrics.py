#!/usr/bin/env python3
"""
Collect and aggregate metrics from benchmark results.

This module provides functions to parse, aggregate, and analyze
compression benchmark results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import statistics


def load_results(results_file: Path) -> Dict:
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def aggregate_by_category(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by category."""
    by_category = defaultdict(list)
    for result in results:
        if result.get("success"):
            by_category[result.get("category", "unknown")].append(result)
    return dict(by_category)


def aggregate_by_level(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by compression level."""
    by_level = defaultdict(list)
    for result in results:
        if result.get("success"):
            by_level[result.get("compression_level", "unknown")].append(result)
    return dict(by_level)


def aggregate_by_scoring(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by scoring mode."""
    by_scoring = defaultdict(list)
    for result in results:
        if result.get("success"):
            by_scoring[result.get("scoring_mode", "unknown")].append(result)
    return dict(by_scoring)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistical measures for a list of values."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_dev": 0.0,
            "count": 0,
        }
    
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def analyze_compression_metrics(results: List[Dict]) -> Dict:
    """Analyze compression performance metrics."""
    successful = [r for r in results if r.get("success")]
    
    if not successful:
        return {}
    
    compression_ratios = [r.get("compression_percentage", 0) for r in successful]
    tokens_saved = [r.get("tokens_saved", 0) for r in successful]
    compression_times = [r.get("compression_time_ms", 0) for r in successful]
    
    return {
        "compression_ratio": calculate_statistics(compression_ratios),
        "tokens_saved": calculate_statistics(tokens_saved),
        "compression_time_ms": calculate_statistics(compression_times),
    }


def analyze_quality_metrics(results: List[Dict]) -> Dict:
    """Analyze quality metrics."""
    successful = [r for r in results if r.get("success") and r.get("quality_metrics")]
    
    if not successful:
        return {}
    
    quality_data = {
        "overall_score": [],
        "semantic_similarity": [],
        "critical_instruction_preservation": [],
        "information_retention": [],
        "structural_integrity": [],
    }
    
    for result in successful:
        qm = result.get("quality_metrics", {})
        for key in quality_data.keys():
            if key in qm:
                quality_data[key].append(qm[key] * 100.0)  # Convert to percentage
    
    return {
        key: calculate_statistics(values)
        for key, values in quality_data.items()
        if values
    }


def analyze_by_dimension(results: List[Dict]) -> Dict:
    """Analyze results grouped by different dimensions."""
    analysis = {}
    
    # By category
    by_category = aggregate_by_category(results)
    analysis["by_category"] = {
        cat: {
            "compression": analyze_compression_metrics(cat_results),
            "quality": analyze_quality_metrics(cat_results),
            "count": len(cat_results),
        }
        for cat, cat_results in by_category.items()
    }
    
    # By compression level
    by_level = aggregate_by_level(results)
    analysis["by_level"] = {
        level: {
            "compression": analyze_compression_metrics(level_results),
            "quality": analyze_quality_metrics(level_results),
            "count": len(level_results),
        }
        for level, level_results in by_level.items()
    }
    
    # By scoring mode
    by_scoring = aggregate_by_scoring(results)
    analysis["by_scoring"] = {
        scoring: {
            "compression": analyze_compression_metrics(scoring_results),
            "quality": analyze_quality_metrics(scoring_results),
            "count": len(scoring_results),
        }
        for scoring, scoring_results in by_scoring.items()
    }
    
    return analysis


def get_best_worst_scenarios(results: List[Dict], metric: str = "compression_percentage") -> Dict:
    """Get best and worst performing scenarios."""
    successful = [r for r in results if r.get("success")]
    
    if not successful:
        return {"best": None, "worst": None}
    
    # Sort by metric
    sorted_results = sorted(
        successful,
        key=lambda x: x.get(metric, 0),
        reverse=True
    )
    
    return {
        "best": sorted_results[0] if sorted_results else None,
        "worst": sorted_results[-1] if sorted_results else None,
    }


def generate_summary(results_data: Dict) -> Dict:
    """Generate comprehensive summary of results."""
    results = results_data.get("results", [])
    successful = [r for r in results if r.get("success")]
    
    summary = {
        "total_tests": results_data.get("total_tests", 0),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "success_rate": len(successful) / len(results) * 100 if results else 0,
        "overall_metrics": {
            "compression": analyze_compression_metrics(results),
            "quality": analyze_quality_metrics(results),
        },
        "by_dimension": analyze_by_dimension(results),
        "best_worst": {
            "compression": get_best_worst_scenarios(results, "compression_percentage"),
            "quality": get_best_worst_scenarios(results, "quality_metrics.overall_score"),
        },
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    results_file = Path(__file__).parent / "results" / "benchmark_results.json"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run 'python run_benchmark.py' first.")
    else:
        results_data = load_results(results_file)
        summary = generate_summary(results_data)
        print(json.dumps(summary, indent=2))

