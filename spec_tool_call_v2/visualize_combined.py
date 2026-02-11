#!/usr/bin/env python3
"""Generate combined level1 + level2 visualizations for blog post."""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def load_data(results_dir):
    """Load and process all result files."""
    results_path = Path(results_dir)
    
    all_tool_calls = defaultdict(list)
    correct_tool_calls = defaultdict(list)
    incorrect_tool_calls = defaultdict(list)
    
    num_correct = 0
    num_incorrect = 0
    
    json_files = sorted(results_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'trajectory' not in data or not data['trajectory']:
                continue
            
            is_correct = data.get('correct', False)
            
            if is_correct:
                num_correct += 1
            else:
                num_incorrect += 1
            
            for entry in data['trajectory']:
                if entry.get('node') == 'tools':
                    tool_time = entry.get('tool_time', 0)
                    tool_executed = entry.get('tool_executed', {})
                    tool_name = tool_executed.get('name', 'unknown')
                    
                    all_tool_calls[tool_name].append(tool_time)
                    if is_correct:
                        correct_tool_calls[tool_name].append(tool_time)
                    else:
                        incorrect_tool_calls[tool_name].append(tool_time)
                        
        except Exception as e:
            continue
    
    return all_tool_calls, correct_tool_calls, incorrect_tool_calls, num_correct, num_incorrect


def plot_1_combined_overview(level1_data, level2_data, output_dir):
    """Plot 1: Tool latency overview for both levels."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    for idx, (data, level_name, ax) in enumerate([
        (level1_data, 'Level 1', axes[0]),
        (level2_data, 'Level 2', axes[1])
    ]):
        all_tool_calls, _, _, num_correct, num_incorrect = data
        
        # Calculate averages
        tools_data = []
        for tool_name, latencies in all_tool_calls.items():
            avg = np.mean(latencies)
            count = len(latencies)
            tools_data.append((tool_name, avg, count))
        
        # Sort by average latency
        tools_data.sort(key=lambda x: x[1], reverse=True)
        
        tool_names = [t[0] for t in tools_data]
        avg_latencies = [t[1] for t in tools_data]
        counts = [t[2] for t in tools_data]
        
        # Color based on speculation viability
        colors = []
        for avg in avg_latencies:
            if avg >= 10:
                colors.append('#2ecc71')  # Green - viable
            elif avg >= 5:
                colors.append('#f39c12')  # Orange - maybe
            elif avg >= 1:
                colors.append('#e67e22')  # Dark orange - marginal
            else:
                colors.append('#e74c3c')  # Red - too fast
        
        # Create bar plot
        bars = ax.barh(tool_names, avg_latencies, color=colors, alpha=0.8)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{count} calls', 
                    ha='left', va='center', fontsize=8)
        
        # Add speculation threshold line
        ax.axvline(x=10, color='black', linestyle='--', linewidth=2, 
                   label='Spec Model Latency (~10-20s)')
        
        # Shaded regions
        ax.axvspan(0, 1, alpha=0.1, color='red', label='Too Fast (< 1s)')
        ax.axvspan(10, ax.get_xlim()[1], alpha=0.1, color='green', label='Viable (> 10s)')
        
        ax.set_xlabel('Average Latency (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Tool Name', fontsize=11, fontweight='bold')
        ax.set_title(f'{level_name}: Tool Latencies (Correct: {num_correct}, Incorrect: {num_incorrect})', 
                     fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Tool Execution Latencies on GAIA', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_tool_latency_overview_combined.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/1_tool_latency_overview_combined.png")
    plt.close()


def plot_2_combined_paradox(level1_data, level2_data, output_dir):
    """Plot 2: The Paradox - Correct vs Incorrect for both levels."""
    tools_to_compare = ['enhanced_search', 'web_search', 'vision_analyze']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for row_idx, (data, level_name) in enumerate([
        (level1_data, 'Level 1'),
        (level2_data, 'Level 2')
    ]):
        _, correct_tool_calls, incorrect_tool_calls, _, _ = data
        
        for col_idx, tool_name in enumerate(tools_to_compare):
            ax = axes[row_idx, col_idx]
            
            correct_lats = correct_tool_calls.get(tool_name, [])
            incorrect_lats = incorrect_tool_calls.get(tool_name, [])
            
            if not correct_lats and not incorrect_lats:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Calculate means
            correct_mean = np.mean(correct_lats) if correct_lats else 0
            incorrect_mean = np.mean(incorrect_lats) if incorrect_lats else 0
            
            # Bar plot
            x = ['Correct\nTasks', 'Incorrect\nTasks']
            means = [correct_mean, incorrect_mean]
            colors_bar = ['#2ecc71', '#e74c3c']
            
            bars = ax.bar(x, means, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar, mean, count in zip(bars, means, 
                                          [len(correct_lats), len(incorrect_lats)]):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{mean:.2f}s\n({count} calls)',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add spec model threshold line
            if tool_name in ['enhanced_search', 'vision_analyze']:
                ax.axhline(y=10, color='black', linestyle='--', linewidth=2, alpha=0.5,
                          label='Spec threshold')
                ax.legend(fontsize=7, loc='upper right')
            
            ax.set_ylabel('Avg Latency (s)', fontsize=10, fontweight='bold')
            
            # Title with level and tool name
            title = f'{level_name}: {tool_name.replace("_", " ").title()}'
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add ratio text
            if correct_mean > 0 and incorrect_mean > 0:
                ratio = incorrect_mean / correct_mean
                ax.text(0.5, 0.95, f'{ratio:.1f}x slower\non failures',
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('The Speculation Paradox: Tool Latencies on Correct vs Incorrect Tasks', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_correct_vs_incorrect_paradox_combined.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/2_correct_vs_incorrect_paradox_combined.png")
    plt.close()


def plot_3_combined_distributions(level1_data, level2_data, output_dir):
    """Plot 3: Distribution of enhanced_search latencies for both levels."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    tool = 'enhanced_search'
    
    for row_idx, (data, level_name) in enumerate([
        (level1_data, 'Level 1'),
        (level2_data, 'Level 2')
    ]):
        all_tool_calls, correct_tool_calls, incorrect_tool_calls, _, _ = data
        
        all_lats = all_tool_calls.get(tool, [])
        correct_lats = correct_tool_calls.get(tool, [])
        incorrect_lats = incorrect_tool_calls.get(tool, [])
        
        # Plot 1: All
        ax = axes[row_idx, 0]
        if all_lats:
            ax.hist(all_lats, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axvline(np.median(all_lats), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(all_lats):.2f}s')
            ax.axvline(10, color='black', linestyle='--', linewidth=2, alpha=0.5,
                       label='Spec threshold (10s)')
            ax.set_title(f'{level_name}: All Tasks (n={len(all_lats)})', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Latency (seconds)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 50)
        
        # Plot 2: Correct
        ax = axes[row_idx, 1]
        if correct_lats:
            ax.hist(correct_lats, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax.axvline(np.median(correct_lats), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(correct_lats):.2f}s')
            ax.axvline(10, color='black', linestyle='--', linewidth=2, alpha=0.5,
                       label='Spec threshold (10s)')
            ax.set_title(f'{level_name}: Correct Tasks (n={len(correct_lats)})', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Latency (seconds)', fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 50)
        
        # Plot 3: Incorrect
        ax = axes[row_idx, 2]
        if incorrect_lats:
            ax.hist(incorrect_lats, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.axvline(np.median(incorrect_lats), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(incorrect_lats):.2f}s')
            ax.axvline(10, color='black', linestyle='--', linewidth=2, alpha=0.5,
                       label='Spec threshold (10s)')
            ax.set_title(f'{level_name}: Incorrect Tasks (n={len(incorrect_lats)})', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Latency (seconds)', fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 50)
    
    fig.suptitle('Enhanced Search Latency Distributions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_latency_distributions_combined.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/3_latency_distributions_combined.png")
    plt.close()


def main():
    level1_dir = "results_level1_baseline_20251203_002006"
    level2_dir = "results_level2_baseline_20251203_032133"
    output_dir = "visualizations"
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n📊 Loading Level 1 data from {level1_dir}...")
    level1_data = load_data(level1_dir)
    
    print(f"📊 Loading Level 2 data from {level2_dir}...")
    level2_data = load_data(level2_dir)
    
    print(f"\n🎨 Generating combined visualizations...\n")
    
    plot_1_combined_overview(level1_data, level2_data, output_dir)
    plot_2_combined_paradox(level1_data, level2_data, output_dir)
    plot_3_combined_distributions(level1_data, level2_data, output_dir)
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print(f"\nGenerated 3 combined figures:")
    print(f"  1. Tool latency overview (Level 1 top, Level 2 bottom)")
    print(f"  2. The paradox: Correct vs Incorrect comparison (both levels)")
    print(f"  3. Latency distributions for enhanced_search (both levels)")


if __name__ == '__main__':
    main()

