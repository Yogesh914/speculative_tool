#!/usr/bin/env python3
"""Compare baseline vs speculation results for both Level 1 and Level 2."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def load_results(results_dir):
    """Load all result files and return as dict keyed by task_id."""
    results_path = Path(results_dir)
    results = {}
    
    for json_file in sorted(results_path.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            task_id = data.get('task_id')
            if task_id:
                results[task_id] = data
        except Exception as e:
            continue
    
    return results


def get_matched_data(baseline_results, spec_results):
    """Match examples by task_id and return comparison data."""
    matched_data = []
    for task_id, baseline in baseline_results.items():
        if task_id in spec_results:
            spec = spec_results[task_id]
            matched_data.append({
                'task_id': task_id,
                'baseline_time': baseline.get('time_seconds', 0),
                'spec_time': spec.get('time_seconds', 0),
                'baseline_correct': baseline.get('correct', False),
                'spec_correct': spec.get('correct', False),
                'baseline_steps': baseline.get('steps', 0),
                'spec_steps': spec.get('steps', 0),
                'spec_hits': spec.get('spec_hits', 0),
                'spec_misses': spec.get('spec_misses', 0),
                'spec_predictions': spec.get('spec_predictions', 0),
            })
    return matched_data


def plot_combined_comparison(level1_data, level2_data, output_dir):
    """Plot combined comparison for both levels."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for row_idx, (data, level_name) in enumerate([
        (level1_data, 'Level 1'),
        (level2_data, 'Level 2')
    ]):
        n_examples = len(data)
        baseline_accuracy = sum(1 for d in data if d['baseline_correct']) / n_examples * 100
        spec_accuracy = sum(1 for d in data if d['spec_correct']) / n_examples * 100
        
        baseline_avg_time = np.mean([d['baseline_time'] for d in data])
        spec_avg_time = np.mean([d['spec_time'] for d in data])
        
        total_hits = sum(d['spec_hits'] for d in data)
        total_predictions = sum(d['spec_predictions'] for d in data)
        hit_rate = (total_hits / total_predictions * 100) if total_predictions > 0 else 0
        
        # Plot 1: Accuracy comparison
        ax = axes[row_idx, 0]
        x = ['Baseline', 'Spec']
        accuracies = [baseline_accuracy, spec_accuracy]
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(x, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{level_name}: Accuracy (n={n_examples})', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight accuracy change
        acc_diff = spec_accuracy - baseline_accuracy
        color_text = 'green' if acc_diff >= 0 else 'red'
        sign = '+' if acc_diff >= 0 else ''
        ax.text(0.5, 0.85, f'{sign}{acc_diff:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=9, fontweight='bold', color=color_text)
        
        # Plot 2: Average time comparison
        ax = axes[row_idx, 1]
        times = [baseline_avg_time, spec_avg_time]
        bars = ax.bar(x, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}s',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add speedup/slowdown text
        speedup = ((baseline_avg_time - spec_avg_time) / baseline_avg_time * 100)
        color_text = 'green' if speedup > 0 else 'red'
        sign = '+' if speedup > 0 else ''
        status = 'speedup' if speedup > 0 else 'SLOWDOWN'
        ax.text(0.5, 0.95, f'{sign}{speedup:.1f}% {status}',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=9, fontweight='bold', color=color_text,
               bbox=dict(boxstyle='round', facecolor='yellow' if speedup > 0 else 'pink', alpha=0.5))
        
        ax.set_ylabel('Time (seconds)', fontsize=10, fontweight='bold')
        ax.set_title(f'{level_name}: Avg Time', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Hit rate
        ax = axes[row_idx, 2]
        hit_data = [total_hits, total_predictions - total_hits]
        labels = [f'Hits\n{total_hits}', f'Misses\n{total_predictions - total_hits}']
        colors_pie = ['#2ecc71', '#e74c3c']
        
        if total_predictions > 0:
            wedges, texts, autotexts = ax.pie(hit_data, labels=labels, autopct='%1.1f%%',
                                                colors=colors_pie, startangle=90,
                                                textprops={'fontsize': 9, 'fontweight': 'bold'})
            ax.set_title(f'{level_name}: Hit Rate {hit_rate:.1f}%', 
                        fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No predictions', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
            ax.set_title(f'{level_name}: Hit Rate N/A', fontsize=11, fontweight='bold')
    
    fig.suptitle('Baseline vs Speculation: Combined Level 1 & 2 Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_combined_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_combined_overview.png")
    plt.close()


def plot_time_scatter(level1_data, level2_data, output_dir):
    """Plot scatter comparison of baseline vs spec times for both levels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (data, level_name, ax) in enumerate([
        (level1_data, 'Level 1', axes[0]),
        (level2_data, 'Level 2', axes[1])
    ]):
        baseline_times = [d['baseline_time'] for d in data]
        spec_times = [d['spec_time'] for d in data]
        
        # Color by correctness: both correct (green), both wrong (red), mixed (orange)
        colors = []
        for d in data:
            if d['baseline_correct'] and d['spec_correct']:
                colors.append('green')
            elif not d['baseline_correct'] and not d['spec_correct']:
                colors.append('red')
            else:
                colors.append('orange')
        
        ax.scatter(baseline_times, spec_times, alpha=0.6, s=100, 
                  c=colors, edgecolors='black', linewidth=1)
        
        # Add diagonal line (y=x, no change)
        max_time = max(max(baseline_times), max(spec_times))
        ax.plot([0, max_time], [0, max_time], 'k--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Baseline Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Speculation Time (s)', fontsize=11, fontweight='bold')
        ax.set_title(f'{level_name}: Per-Example Time Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add regions
        ax.fill_between([0, max_time], [0, max_time], max_time, alpha=0.1, color='red')
        ax.text(0.25, 0.75, 'SLOWER\nwith spec', transform=ax.transAxes,
               fontsize=9, ha='center', color='red', alpha=0.7, fontweight='bold')
        ax.text(0.75, 0.25, 'FASTER\nwith spec', transform=ax.transAxes,
               fontsize=9, ha='center', color='green', alpha=0.7, fontweight='bold')
        
        # Count faster/slower
        faster = sum(1 for b, s in zip(baseline_times, spec_times) if s < b)
        slower = sum(1 for b, s in zip(baseline_times, spec_times) if s >= b)
        ax.text(0.5, 0.05, f'Faster: {faster} | Slower: {slower}',
               transform=ax.transAxes, ha='center', va='bottom',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Both Correct'),
        Patch(facecolor='orange', edgecolor='black', label='Mixed'),
        Patch(facecolor='red', edgecolor='black', label='Both Wrong')
    ]
    axes[1].legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    fig.suptitle('Time Comparison Scatter Plot', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_combined_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_combined_scatter.png")
    plt.close()


def plot_summary_bars(level1_data, level2_data, output_dir):
    """Plot summary bar chart comparing key metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate metrics for both levels
    metrics = []
    for data, level_name in [(level1_data, 'Level 1'), (level2_data, 'Level 2')]:
        n = len(data)
        baseline_acc = sum(1 for d in data if d['baseline_correct']) / n * 100
        spec_acc = sum(1 for d in data if d['spec_correct']) / n * 100
        baseline_time = np.mean([d['baseline_time'] for d in data])
        spec_time = np.mean([d['spec_time'] for d in data])
        speedup = (baseline_time - spec_time) / baseline_time * 100
        
        total_hits = sum(d['spec_hits'] for d in data)
        total_preds = sum(d['spec_predictions'] for d in data)
        hit_rate = (total_hits / total_preds * 100) if total_preds > 0 else 0
        
        metrics.append({
            'level': level_name,
            'acc_change': spec_acc - baseline_acc,
            'speedup': speedup,
            'hit_rate': hit_rate
        })
    
    # Create grouped bar chart
    x = np.arange(3)
    width = 0.35
    
    level1_vals = [metrics[0]['acc_change'], metrics[0]['speedup'], metrics[0]['hit_rate']]
    level2_vals = [metrics[1]['acc_change'], metrics[1]['speedup'], metrics[1]['hit_rate']]
    
    bars1 = ax.bar(x - width/2, level1_vals, width, label='Level 1', 
                   color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, level2_vals, width, label='Level 2',
                   color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            label_y = height if height >= 0 else height - 3
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.1f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title('Summary: Speculation Impact on Key Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy\nChange (%)', 'Speedup (%)', 'Hit Rate (%)'], fontsize=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation box
    interpretation = """
    Negative speedup = SLOWDOWN
    Level 1: -189% (nearly 3x slower)
    Level 2: -47% (1.5x slower)
    """
    ax.text(0.98, 0.95, interpretation.strip(),
           transform=ax.transAxes, ha='right', va='top',
           fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_combined_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_combined_summary.png")
    plt.close()


def main():
    level1_baseline_dir = "results_level1_baseline_20251203_002006"
    level1_spec_dir = "results_level1_spec_20251204_155043"
    level2_baseline_dir = "results_level2_baseline_20251203_032133"
    level2_spec_dir = "results_level2_spec_20251204_165317"
    output_dir = "visualizations"
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n📊 Loading Level 1 results...")
    level1_baseline = load_results(level1_baseline_dir)
    level1_spec = load_results(level1_spec_dir)
    level1_data = get_matched_data(level1_baseline, level1_spec)
    print(f"   Matched {len(level1_data)} Level 1 examples")
    
    print(f"📊 Loading Level 2 results...")
    level2_baseline = load_results(level2_baseline_dir)
    level2_spec = load_results(level2_spec_dir)
    level2_data = get_matched_data(level2_baseline, level2_spec)
    print(f"   Matched {len(level2_data)} Level 2 examples\n")
    
    if len(level1_data) == 0 and len(level2_data) == 0:
        print("❌ No matched examples found!")
        return
    
    print(f"🎨 Generating combined comparison visualizations...\n")
    
    plot_combined_comparison(level1_data, level2_data, output_dir)
    plot_time_scatter(level1_data, level2_data, output_dir)
    plot_summary_bars(level1_data, level2_data, output_dir)
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print(f"\nGenerated 3 combined comparison figures:")
    print(f"  1. Combined overview (accuracy, time, hit rate)")
    print(f"  2. Scatter plots (per-example time comparison)")
    print(f"  3. Summary bars (key metrics comparison)")


if __name__ == '__main__':
    main()

