#!/usr/bin/env python3
"""
Visualize host states and traffic evolution timeline for ATTACKS_HO experiments.

This script helps understand:
- How host states evolve during an experiment (normal, attacking, under attack, etc.)
- How traffic (packets/bytes) changes for each host over time
- Attack patterns and mitigation effectiveness

Usage:
    # Auto-detect latest ATTACKS_HO experiment
    python test_tools/plot_host_states_timeline.py
    
    # Use specific experiment
    python test_tools/plot_host_states_timeline.py --experiment_path _training/attacks_ho/20260428-142600_1_5_5/
    
    # Custom output directory
    python test_tools/plot_host_states_timeline.py --output_dir debug/ --states --traffic
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_tools.helpers import plot_host_states_evolution, plot_host_traffic_evolution
from utility.my_log import information, error


def find_latest_attacks_ho_experiment(training_dir="_training/attacks_ho"):
    """Find the latest ATTACKS_HO experiment folder."""
    attacks_ho_dir = Path(training_dir)
    
    if not attacks_ho_dir.exists():
        return None
    
    # Get all subdirectories sorted by modification time
    dirs = [d for d in attacks_ho_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None
    
    # Sort by modification time, newest first
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    return str(dirs[0])


def find_statuses_json(experiment_path):
    """Find statuses.json in experiment directory."""
    exp_path = Path(experiment_path)
    
    # Direct path
    statuses_path = exp_path / "statuses.json"
    if statuses_path.exists():
        return str(statuses_path)
    
    # Try parent directory (in case experiment_path points to a subfolder)
    parent_statuses = exp_path.parent / "statuses.json"
    if parent_statuses.exists():
        return str(parent_statuses)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize host states and traffic evolution for ATTACKS_HO experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect latest experiment
  python test_tools/plot_host_states_timeline.py
  
  # Use specific experiment with custom output
  python test_tools/plot_host_states_timeline.py --experiment_path _training/attacks_ho/20260428-142600_1_5_5/ --output_dir debug/
  
  # Only plot states, skip traffic
  python test_tools/plot_host_states_timeline.py --states --no-traffic
  
  # Only plot bytes traffic
  python test_tools/plot_host_states_timeline.py --traffic --metric bytes
        """
    )
    
    parser.add_argument(
        "--experiment_path",
        type=str,
        default=None,
        help="Path to ATTACKS_HO experiment directory (auto-detect if not provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="_training/_debug",
        help="Output directory for generated visualizations"
    )
    parser.add_argument(
        "--statuses_file",
        type=str,
        default=None,
        help="Direct path to statuses.json (bypasses experiment path search)"
    )
    parser.add_argument(
        "--states",
        action="store_true",
        default=True,
        help="Plot host states evolution (default: True)"
    )
    parser.add_argument(
        "--no-states",
        dest="states",
        action="store_false",
        help="Skip host states visualization"
    )
    parser.add_argument(
        "--traffic",
        action="store_true",
        default=True,
        help="Plot traffic evolution (default: True)"
    )
    parser.add_argument(
        "--no-traffic",
        dest="traffic",
        action="store_false",
        help="Skip traffic visualization"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["packets", "bytes"],
        default="packets",
        help="Traffic metric to plot (default: packets)"
    )
    
    args = parser.parse_args()
    
    # Determine statuses file path
    statuses_file = args.statuses_file
    
    if not statuses_file:
        # Try to find statuses.json from experiment path
        experiment_path = args.experiment_path or find_latest_attacks_ho_experiment()
        
        if not experiment_path:
            error("Could not find ATTACKS_HO experiment. Use --experiment_path to specify")
            sys.exit(1)
        
        information(f"Using experiment: {experiment_path}")
        
        statuses_file = find_statuses_json(experiment_path)
        if not statuses_file:
            error(f"Could not find statuses.json in {experiment_path}")
            sys.exit(1)
    
    if not os.path.exists(statuses_file):
        error(f"Statuses file not found: {statuses_file}")
        sys.exit(1)
    
    information(f"Loading statuses from: {statuses_file}")
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract experiment info from path for better organization
    try:
        parts = statuses_file.split(os.sep)
        scenario_idx = parts.index("attacks_ho") if "attacks_ho" in parts else -1
        if scenario_idx >= 0 and scenario_idx + 1 < len(parts):
            exp_folder = parts[scenario_idx + 1]
            organized_output = os.path.join(args.output_dir, f"attacks_ho/{exp_folder}")
            os.makedirs(organized_output, exist_ok=True)
        else:
            organized_output = args.output_dir
    except (ValueError, IndexError):
        organized_output = args.output_dir
    
    information(f"Output directory: {organized_output}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING HOST STATES & TRAFFIC VISUALIZATIONS")
    print("="*60 + "\n")
    
    if args.states:
        print("1. Plotting host states evolution...")
        try:
            result = plot_host_states_evolution(
                statuses_file,
                output_dir=organized_output,
                title="ATTACKS_HO: Host States Timeline"
            )
            if result:
                print(f"   ✓ Saved: host_states_evolution.png")
            else:
                print("   ✗ Failed to generate states visualization")
        except Exception as e:
            error(f"Error generating states visualization: {e}")
            import traceback
            print(traceback.format_exc())
    
    if args.traffic:
        print(f"\n2. Plotting host traffic evolution ({args.metric})...")
        try:
            result = plot_host_traffic_evolution(
                statuses_file,
                output_dir=organized_output,
                metric=args.metric,
                title=f"ATTACKS_HO: Host {args.metric.capitalize()} Timeline"
            )
            if result:
                print(f"   ✓ Saved: host_traffic_evolution_{args.metric}.png")
            else:
                print(f"   ✗ Failed to generate traffic visualization")
        except Exception as e:
            error(f"Error generating traffic visualization: {e}")
            import traceback
            print(traceback.format_exc())
    
    # Also generate the alternate metric if only one was requested
    if args.traffic:
        other_metric = "bytes" if args.metric == "packets" else "packets"
        print(f"\n3. Bonus: Plotting host traffic evolution ({other_metric})...")
        try:
            result = plot_host_traffic_evolution(
                statuses_file,
                output_dir=organized_output,
                metric=other_metric,
                title=f"ATTACKS_HO: Host {other_metric.capitalize()} Timeline"
            )
            if result:
                print(f"   ✓ Saved: host_traffic_evolution_{other_metric}.png")
        except Exception as e:
            information(f"Bonus metric skipped: {e}")
    
    print("\n" + "="*60)
    print(f"✓ Done! Check '{organized_output}' for visualizations")
    print("="*60 + "\n")
    
    # Print summary info
    try:
        with open(statuses_file, 'r') as f:
            statuses = json.load(f)
        
        if statuses:
            first_entry = statuses[0]
            hosts = first_entry.get("hostStatusesStructured", {})
            print(f"\nExperiment Summary:")
            print(f"  Timesteps: {len(statuses)}")
            print(f"  Hosts: {len(hosts)}")
            if hosts:
                print(f"  Host names: {', '.join(sorted(list(hosts.keys())))}")
    except Exception as e:
        information(f"Could not load summary info: {e}")


if __name__ == "__main__":
    main()
