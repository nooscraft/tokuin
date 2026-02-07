#!/usr/bin/env python3
"""
Run compression benchmarks across all scenarios.

This script executes compression tests across all combinations of:
- Compression levels: light, medium, aggressive
- Scoring modes: heuristic, semantic, hybrid
- Prompt categories: instructions, structured, conversations, technical, mixed
"""

import json
import subprocess
import time
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import os

# Configuration
PROMPTS_DIR = Path(__file__).parent / "prompts"
RESULTS_DIR = Path(__file__).parent / "results"
TOKUIN_BINARY = "tokuin"  # Assumes tokuin is in PATH, or use full path

# Test matrix
COMPRESSION_LEVELS = ["light", "medium", "aggressive"]
SCORING_MODES = ["heuristic", "semantic", "hybrid"]
CATEGORIES = ["instructions", "structured", "conversations", "technical", "mixed"]


def find_tokuin_binary() -> str:
    """Find the tokuin binary and return absolute path (cross-platform)."""
    project_root = Path(__file__).parent.parent.parent
    
    # On Windows, binaries have .exe extension
    binary_name = "tokuin.exe" if sys.platform == "win32" else "tokuin"
    
    # Try common locations relative to project root
    possible_paths = [
        project_root / "target" / "release" / binary_name,
        project_root / "target" / "debug" / binary_name,
        Path(binary_name),  # In PATH
    ]
    
    for path in possible_paths:
        try:
            # Resolve to absolute path if it's a relative path
            if path.is_absolute():
                abs_path = path
            else:
                abs_path = path.resolve()
            
            # Check if file exists (for absolute paths)
            if abs_path.is_absolute() and not abs_path.exists():
                continue
            
            result = subprocess.run(
                [str(abs_path), "--version"],
                capture_output=True,
                timeout=5,
                cwd=str(project_root)
            )
            if result.returncode == 0:
                return str(abs_path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # Check if it's in PATH (cross-platform)
    # shutil.which() handles .exe extension automatically on Windows
    tokuin_path = shutil.which("tokuin")
    if tokuin_path:
        return tokuin_path
    
    raise FileNotFoundError(
        "Could not find tokuin binary. Please build it first:\n"
        "  cargo build --release --features compression,compression-embeddings"
    )


def count_tokens(prompt_file: Path, tokuin_bin: str, model: str = "gpt-4") -> int:
    """Count tokens in a prompt file using tokuin."""
    try:
        abs_prompt_file = prompt_file.resolve()
        result = subprocess.run(
            [tokuin_bin, str(abs_prompt_file), "--model", model, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(prompt_file.parent.parent.parent)
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("total_tokens", 0)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, Exception):
        pass
    return 0


def get_prompt_files(tokuin_bin: str, model: str = "gpt-4", min_tokens: int = 50) -> Dict[str, List[Path]]:
    """Get all prompt files organized by category, filtering out very short prompts.
    
    Args:
        tokuin_bin: Path to tokuin binary for token counting
        model: Model to use for token counting
        min_tokens: Minimum token count to include (default: 50)
                   Prompts shorter than this won't benefit from compression
                   due to Hieratic format overhead (~10 tokens)
    
    Returns:
        Dictionary mapping category names to lists of prompt file paths
    """
    prompts = {}
    filtered_count = 0
    
    for category in CATEGORIES:
        category_dir = PROMPTS_DIR / category
        if category_dir.exists():
            all_files = list(category_dir.glob("*.txt"))
            filtered_files = []
            
            for prompt_file in all_files:
                token_count = count_tokens(prompt_file, tokuin_bin, model)
                if token_count >= min_tokens:
                    filtered_files.append(prompt_file)
                else:
                    filtered_count += 1
            
            prompts[category] = filtered_files
        else:
            prompts[category] = []
    
    if filtered_count > 0:
        print(f"⚠️  Filtered out {filtered_count} prompt(s) with < {min_tokens} tokens")
        print(f"   (Very short prompts don't benefit from compression due to format overhead)")
    
    return prompts


def run_compression(
    prompt_file: Path,
    level: str,
    scoring: str,
    tokuin_bin: str,
    model: str = "gpt-4",
    llm_judge: bool = False,
    evaluation_model: Optional[str] = None,
    judge_model: str = "openai/gpt-4",
) -> Dict:
    """Run compression on a single prompt and return results."""
    # Resolve to absolute path to avoid issues with working directory
    abs_prompt_file = prompt_file.resolve()
    
    result = {
        "prompt_file": str(abs_prompt_file),
        "prompt_name": prompt_file.name,
        "category": prompt_file.parent.name,
        "compression_level": level,
        "scoring_mode": scoring,
        "model": model,
        "success": False,
        "error": None,
    }
    
    # Build command - use absolute path for input file
    # When format is json, output goes to a file, so we need to specify output path
    project_root = Path(__file__).parent.parent.parent
    json_output_file = RESULTS_DIR / f"temp_{prompt_file.stem}_{level}_{scoring}.json"
    json_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        tokuin_bin,
        "compress",
        str(abs_prompt_file),
        "--level", level,
        "--scoring", scoring,
        "--format", "json",
        "--output", str(json_output_file),
        "--quality",
        "--model", model,
    ]
    
    # Add LLM judge flags if enabled
    if llm_judge:
        cmd.append("--llm-judge")
        if evaluation_model:
            cmd.extend(["--evaluation-model", evaluation_model])
        cmd.extend(["--judge-model", judge_model])
    
    # Measure execution time and memory
    start_time = time.time()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Run compression - set cwd to project root to ensure paths work
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(project_root),  # Run from project root
        )
        
        end_time = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        result["compression_time_ms"] = (end_time - start_time) * 1000
        result["memory_usage_mb"] = max(0, mem_after - mem_before)
        
        if proc.returncode != 0:
            # Capture full error message
            error_msg = proc.stderr.strip() if proc.stderr else proc.stdout.strip()
            if not error_msg:
                error_msg = f"Command failed with return code {proc.returncode}"
            result["error"] = error_msg
            # Clean up temp file if it exists
            if json_output_file.exists():
                json_output_file.unlink()
            return result
        
        # Read JSON from output file (not stdout)
        try:
            if not json_output_file.exists():
                result["error"] = f"JSON output file not created: {json_output_file}"
                return result
            
            output_data = json.loads(json_output_file.read_text(encoding='utf-8'))
            # Clean up temp file after reading
            json_output_file.unlink()
            result["success"] = True
            result.update({
                "original_tokens": output_data.get("original_tokens", 0),
                "compressed_tokens": output_data.get("compressed_tokens", 0),
                "compression_ratio": output_data.get("compression_ratio", 0.0),
                "tokens_saved": output_data.get("tokens_saved", 0),
                "compression_percentage": output_data.get("compression_ratio", 0.0) * 100.0,
            })
            
            # Quality metrics
            if "quality_metrics" in output_data and output_data["quality_metrics"]:
                qm = output_data["quality_metrics"]
                result["quality_metrics"] = {
                    "overall_score": qm.get("overall_score", 0.0),
                    "semantic_similarity": qm.get("semantic_similarity", 0.0),
                    "critical_instruction_preservation": qm.get("critical_instruction_preservation", 0.0),
                    "information_retention": qm.get("information_retention", 0.0),
                    "structural_integrity": qm.get("structural_integrity", 0.0),
                    "critical_patterns_found": qm.get("critical_patterns_found", 0),
                    "critical_patterns_preserved": qm.get("critical_patterns_preserved", 0),
                }
                
                # LLM judge metrics
                if "llm_judge" in qm and qm["llm_judge"]:
                    lj = qm["llm_judge"]
                    result["llm_judge_metrics"] = {
                        "output_equivalence": lj.get("output_equivalence", 0.0),
                        "instruction_compliance": lj.get("instruction_compliance", 0.0),
                        "information_completeness": lj.get("information_completeness", 0.0),
                        "quality_preservation": lj.get("quality_preservation", 0.0),
                        "overall_fidelity": lj.get("overall_fidelity", 0.0),
                        "justification": lj.get("justification", ""),
                        "key_differences": lj.get("key_differences", []),
                        "evaluation_model": lj.get("evaluation_model", ""),
                        "judge_model": lj.get("judge_model", ""),
                        "evaluation_cost": lj.get("evaluation_cost"),
                        "original_output": lj.get("original_output"),
                        "compressed_output": lj.get("compressed_output"),
                    }
                else:
                    result["llm_judge_metrics"] = None
            else:
                result["quality_metrics"] = None
                result["llm_judge_metrics"] = None
            
            # Context references
            result["context_refs_count"] = len(output_data.get("context_refs", []))
            
            # Extractive stats
            extractive = output_data.get("extractive_stats", {})
            result["extractive_stats"] = {
                "low_relevance_removed": extractive.get("low_relevance_removed", 0),
                "redundant_removed": extractive.get("redundant_removed", 0),
                "sections_compressed": extractive.get("sections_compressed", 0),
            }
            
        except json.JSONDecodeError as e:
            result["error"] = f"Failed to parse JSON output: {e}"
            if json_output_file.exists():
                result["raw_output"] = json_output_file.read_text(encoding='utf-8')[:500]
                json_output_file.unlink()
            else:
                result["raw_output"] = proc.stdout[:500] if proc.stdout else "No output"
        
    except subprocess.TimeoutExpired:
        result["error"] = "Compression timed out after 5 minutes"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_all_benchmarks(
    tokuin_bin: str,
    model: str = "gpt-4",
    llm_judge: bool = False,
    evaluation_model: Optional[str] = None,
    judge_model: str = "openai/gpt-4",
) -> List[Dict]:
    """Run all benchmark combinations."""
    prompts = get_prompt_files(tokuin_bin, model)
    results = []
    
    total_tests = sum(
        len(prompts[cat]) * len(COMPRESSION_LEVELS) * len(SCORING_MODES)
        for cat in CATEGORIES
    )
    
    print(f"Running {total_tests} benchmark tests...")
    print(f"Prompts: {sum(len(prompts[cat]) for cat in CATEGORIES)}")
    print(f"Scenarios: {len(COMPRESSION_LEVELS)} levels × {len(SCORING_MODES)} scoring modes")
    print()
    
    test_num = 0
    for category in CATEGORIES:
        if not prompts[category]:
            print(f"⚠️  No prompts found in category: {category}")
            continue
        
        for prompt_file in prompts[category]:
            for level in COMPRESSION_LEVELS:
                for scoring in SCORING_MODES:
                    test_num += 1
                    print(f"[{test_num}/{total_tests}] {prompt_file.name} | {level} | {scoring}", end=" ... ")
                    sys.stdout.flush()
                    
                    result = run_compression(
                        prompt_file, level, scoring, tokuin_bin, model,
                        llm_judge, evaluation_model, judge_model
                    )
                    results.append(result)
                    
                    if result["success"]:
                        ratio = result.get("compression_percentage", 0)
                        print(f"✓ {ratio:.1f}% compression")
                    else:
                        error = result.get('error', 'Unknown error')
                        # Show first line of error or first 100 chars
                        error_preview = error.split('\n')[0][:100] if error else "Unknown error"
                        print(f"✗ {error_preview}")
    
    return results


def save_results(results: List[Dict], output_file: Path):
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Run compression benchmarks")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Enable LLM-as-a-judge evaluation")
    parser.add_argument("--evaluation-model", type=str, default=None,
                        help="Model to use for generating outputs (default: same as --model)")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-4",
                        help="Model to use for judging outputs (default: openai/gpt-4)")
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="Model to use for tokenization (default: gpt-4)")
    
    args = parser.parse_args()
    
    # Find tokuin binary
    try:
        tokuin_bin = find_tokuin_binary()
        print(f"Using tokuin binary: {tokuin_bin}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Check if prompts exist
    if not PROMPTS_DIR.exists():
        print(f"Error: Prompts directory not found: {PROMPTS_DIR}")
        print("Run 'python collect_prompts.py' first to collect prompts.")
        sys.exit(1)
    
    if args.llm_judge:
        print("⚠️  LLM judge evaluation enabled - this will make API calls and incur costs")
        print(f"   Evaluation model: {args.evaluation_model or args.model}")
        print(f"   Judge model: {args.judge_model}")
        print("   Make sure OPENROUTER_API_KEY is set in your environment")
    
    # Run benchmarks
    results = run_all_benchmarks(
        tokuin_bin,
        args.model,
        args.llm_judge,
        args.evaluation_model,
        args.judge_model,
    )
    
    # Save results
    output_file = RESULTS_DIR / "benchmark_results.json"
    save_results(results, output_file)
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        avg_compression = sum(
            r.get("compression_percentage", 0) for r in results if r["success"]
        ) / successful
        print(f"Average compression: {avg_compression:.1f}%")
        
        if args.llm_judge:
            llm_judge_results = [r for r in results if r.get("llm_judge_metrics")]
            if llm_judge_results:
                avg_fidelity = sum(
                    r["llm_judge_metrics"].get("overall_fidelity", 0)
                    for r in llm_judge_results
                ) / len(llm_judge_results)
                total_cost = sum(
                    r["llm_judge_metrics"].get("evaluation_cost", 0) or 0
                    for r in llm_judge_results
                )
                print(f"Average LLM judge fidelity: {avg_fidelity:.1f}/100")
                print(f"Total evaluation cost: ${total_cost:.4f}")
    
    print(f"\nResults saved to: {output_file}")
    print("Run 'python generate_report.py' to generate the report.")


if __name__ == "__main__":
    main()

