#!/usr/bin/env python3
"""
Script to find JSON files with Trajectory Tokens > threshold and convert them.
"""

import json
import os
import sys
import argparse
from pathlib import Path
import glob

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_tokens_approximate(text: str) -> int:
    """Approximate token count."""
    words = len(text.split())
    return int(words / 0.75)


def count_tokens_tiktoken(text: str) -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def extract_text_from_trajectory(trajectory: list) -> str:
    """Extract all text content from trajectory."""
    text_parts = []
    
    for item in trajectory:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, dict):
                    text_parts.append(json.dumps(value))
                elif isinstance(value, list):
                    for sub_item in value:
                        if isinstance(sub_item, str):
                            text_parts.append(sub_item)
                        elif isinstance(sub_item, dict):
                            text_parts.append(json.dumps(sub_item))
    
    return " ".join(text_parts)


def count_trajectory_tokens(file_path: str) -> int:
    """Count tokens in trajectory only."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "trajectory" not in data:
        return 0
    
    trajectory_text = extract_text_from_trajectory(data["trajectory"])
    
    if TIKTOKEN_AVAILABLE:
        return count_tokens_tiktoken(trajectory_text)
    else:
        return count_tokens_approximate(trajectory_text)


def convert_format(input_data: dict) -> dict:
    """Convert from source format to target format.
    
    Target format:
    {
        "episode_id": "xxx",
        "task": "xxx",
        "file": "relative path",
        "success": true/false,
        "num_turns": xxx,
        "total_tokens": estimate,
        "trajectory": [
            {"turn_idx": 0, "action": "Tool name + args", "explanation": "reasoning trace", "observation": "tool result"},
            ...
        ],
        "source": "gaia_level2"
    }
    """
    import re
    
    # Extract basic fields
    episode_id = input_data.get("task_id", "unknown")
    task = input_data.get("question", "")
    success = input_data.get("correct", False)
    level = input_data.get("level", "1")
    
    # Extract file from question if present (look for file path pattern)
    file = ""
    if "[Note: There is an attached file at path:" in task:
        match = re.search(r'\[Note: There is an attached file at path: ([^\]]+)\]', task)
        if match:
            file = match.group(1)
    
    # Convert trajectory
    source_trajectory = input_data.get("trajectory", [])
    converted_trajectory = []
    turn_idx = 0
    
    # Process trajectory events in pairs (llm -> tools)
    pending_action = None
    pending_explanation = None

    for event in source_trajectory:
        node = event.get("node", "")

        if node == "llm":
            # Extract reasoning trace as explanation
            actor_reasoning = event.get("actor_reasoning", "")
            pending_explanation = actor_reasoning if actor_reasoning else ""

            # Extract tool calls as action
            tool_calls = event.get("actor_tool_calls", [])
            if tool_calls:
                actions = []
                for tc in tool_calls:
                    tool_name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    args_str = json.dumps(args, ensure_ascii=False)
                    actions.append(f"{tool_name}({args_str})")
                pending_action = "; ".join(actions)
            elif event.get("final_answer"):
                # Final answer step
                pending_action = f"FINAL_ANSWER: {event.get('final_answer', '')}"
                converted_trajectory.append({
                    "turn_idx": turn_idx,
                    "action": pending_action,
                    "explanation": pending_explanation,
                    "observation": ""
                })
                turn_idx += 1
                pending_action = None
                pending_explanation = None

        elif node == "tools" and pending_action:
            # Extract tool result as observation
            tool_executed = event.get("tool_executed", {})
            observation = tool_executed.get("result", "")

            converted_trajectory.append({
                "turn_idx": turn_idx,
                "action": pending_action,
                "explanation": pending_explanation,
                "observation": observation
            })
            turn_idx += 1
            pending_action = None
            pending_explanation = None
    
    # Estimate total tokens from trajectory text
    trajectory_text = extract_text_from_trajectory(source_trajectory)
    if TIKTOKEN_AVAILABLE:
        total_tokens = count_tokens_tiktoken(trajectory_text)
    else:
        total_tokens = count_tokens_approximate(trajectory_text)
    
    return {
        "episode_id": episode_id,
        "task": input_data.get("question", "").split("\n\n[Note:")[0],  # Remove file note
        "file": file,
        "success": success,
        "num_turns": len(converted_trajectory),
        "total_tokens": total_tokens,
        "trajectory": converted_trajectory,
        "source": f"gaia_level{level}"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter and convert JSON files based on trajectory token count"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="converted_high_token_trajectories",
        help="Output directory (default: converted_high_token_trajectories)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=4000,
        help="Minimum trajectory token count (default: 4000)"
    )
    parser.add_argument(
        "--pattern",
        default="result_*.json",
        help="File pattern to match (default: result_*.json)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show files that would be converted, don't actually convert"
    )
    
    args = parser.parse_args()
    
    # Find all JSON files
    search_path = os.path.join(args.input_dir, args.pattern)
    input_files = sorted(glob.glob(search_path))
    
    if not input_files:
        print(f"No files found in {args.input_dir} matching {args.pattern}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(input_files)} file(s) to scan")
    print(f"Threshold: {args.threshold} trajectory tokens")
    print(f"Counting method: {'tiktoken' if TIKTOKEN_AVAILABLE else 'approximate'}")
    print("="*70)
    
    # Filter files by token count
    filtered_files = []
    stats = {
        "total_scanned": 0,
        "above_threshold": 0,
        "below_threshold": 0,
        "errors": 0
    }
    
    for file_path in input_files:
        stats["total_scanned"] += 1
        try:
            token_count = count_trajectory_tokens(file_path)
            
            if token_count > args.threshold:
                filtered_files.append((file_path, token_count))
                stats["above_threshold"] += 1
                
                if args.verbose:
                    print(f"✓ {os.path.basename(file_path)}: {token_count:,} tokens")
            else:
                stats["below_threshold"] += 1
                if args.verbose:
                    print(f"  {os.path.basename(file_path)}: {token_count:,} tokens (skipped)")
                    
        except Exception as e:
            stats["errors"] += 1
            print(f"✗ Error processing {file_path}: {e}", file=sys.stderr)
    
    print("\n" + "="*70)
    print("FILTERING SUMMARY")
    print("="*70)
    print(f"Total scanned: {stats['total_scanned']}")
    print(f"Above threshold (>{args.threshold}): {stats['above_threshold']}")
    print(f"Below threshold: {stats['below_threshold']}")
    print(f"Errors: {stats['errors']}")
    
    if not filtered_files:
        print("\nNo files found above the threshold.")
        return
    
    print(f"\n{len(filtered_files)} file(s) to convert:")
    for file_path, token_count in filtered_files:
        print(f"  - {os.path.basename(file_path)}: {token_count:,} tokens")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were converted.")
        return
    
    # Convert filtered files
    print(f"\n{'='*70}")
    print("CONVERTING FILES")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    conversion_stats = {
        "successful": 0,
        "failed": 0
    }
    
    for file_path, token_count in filtered_files:
        try:
            # Read input
            with open(file_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            # Convert
            output_data = convert_format(input_data)

            # Generate output filename: task_gaia_level_i_idx.json
            # Extract from filename: result_level1_example_000_baseline.json
            base_name = os.path.basename(file_path)

            import re
            # Extract level and example index from filename
            match = re.search(r'level(\d+)_example_(\d+)', base_name)
            if match:
                level = match.group(1)
                idx = match.group(2)
            else:
                # Fallback to data if filename doesn't match expected pattern
                level = input_data.get("level", "1")
                idx = "000"

            output_name = f"task_gaia_level{level}_{idx}.json"
            output_path = os.path.join(args.output_dir, output_name)
            
            # Write output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            conversion_stats["successful"] += 1
            print(f"✓ Converted: {base_name} -> {output_name}")
            
        except Exception as e:
            conversion_stats["failed"] += 1
            print(f"✗ Failed to convert {os.path.basename(file_path)}: {e}", file=sys.stderr)
    
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Successful: {conversion_stats['successful']}")
    print(f"Failed: {conversion_stats['failed']}")
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()

