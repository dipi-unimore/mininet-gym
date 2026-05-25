#!/usr/bin/env python3
"""
Test script to visualize prediction ground truth comparison with alternative charts.

This script helps compare different visualization approaches to understand
which one best shows prediction success rates.

Usage:
    python test_tools/test_chart_predictions.py --experiment_path <path/to/experiment> --output_dir debug/
"""

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utility.my_ho_statistics import plot_ho_agent_test_errors, plot_ho_success_rate_alternatives



def find_test_results(experiment_path):
    """
    Find test result JSON files in the experiment directory.
    
    Looks for:
    1. Folders starting with "TEST" containing test.json directly inside
    2. Folders starting with "TEST_" followed by agent name, containing test.json inside
    
    Pattern:
    - TEST/ -> test.json is inside
    - TEST_DQN_quick/ -> agent is "DQN_quick", test.json is inside
    """
    results = {}
    path = Path(experiment_path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {experiment_path}")
        return results
    
    # Look for TEST* directories
    for test_dir in path.iterdir():
        if not test_dir.is_dir():
            continue
        
        dir_name = test_dir.name
        
        # Check if directory starts with TEST
        if not dir_name.startswith("TEST"):
            continue
        
        # Extract agent name
        if dir_name == "TEST":
            # Plain TEST folder - test.json is inside
            agent_name = None
        elif dir_name.startswith("TEST_"):
            # TEST_<agent_name> pattern
            agent_name = dir_name[5:]  # Remove "TEST_" prefix
        else:
            continue
        
        # Look for test.json inside the TEST* folder
        test_json_path = test_dir / "test.json"
        if test_json_path.exists():
            try:
                with open(test_json_path, 'r') as f:
                    test_data = json.load(f)
                    
                # Use agent_name as key, or a generic name if not available
                key = agent_name if agent_name else f"test_{len(results)}"
                results[key] = test_data
                
            except Exception as e:
                print(f"Warning: Could not load {test_json_path}: {e}")
    
    return results

def extract_ground_truth_predictions(test_results):
    """
    Extract ground_truth and predicted lists from test results.
    
    Returns:
        Dict with 'ground_truth' (list) and 'predicted' (dict of agent -> list)
    """
    ground_truth = []
    predicted = {}
    
    for agent_name, result in test_results.items():
        if isinstance(result, dict):
            if "ground_truth" in result:
                gt = result["ground_truth"]
                if isinstance(gt, list):
                    ground_truth = gt
            
            if "predicted" in result:
                pred = result["predicted"]
                if isinstance(pred, list):
                    predicted[agent_name] = pred
                elif isinstance(pred, dict):
                    # If predicted is a dict of agents, extract this agent's predictions
                    if agent_name in pred:
                        predicted[agent_name] = pred[agent_name]
    
    return {
        "ground_truth": ground_truth,
        "predicted": predicted
    }


def create_test_data(n_steps=100, n_agents=2, success_rate=0.75):
    """
    Create synthetic test data for demonstration purposes.
    
    Args:
        n_steps: Number of prediction steps
        n_agents: Number of agents to simulate
        success_rate: Probability of correct prediction
    """
    ground_truth = np.random.randint(0, 3, n_steps).tolist()
    predicted = {}
    
    for i in range(n_agents):
        agent_name = f"TestAgent_{i+1}"
        predictions = []
        
        for gt in ground_truth:
            # success_rate chance of predicting correctly
            if np.random.random() < success_rate:
                predictions.append(gt)
            else:
                predictions.append(np.random.randint(0, 3))
        
        predicted[agent_name] = predictions
    
    return {
        "ground_truth": ground_truth,
        "predicted": predicted
    }


def generate_comparison_report(test_data):
    """
    Generate a comparison report showing success metrics for each agent.
    """
    ground_truth = np.array(test_data["ground_truth"], dtype=int)
    report = {
        "total_steps": len(ground_truth),
        "agents": {}
    }
    
    for agent_name, predictions in test_data["predicted"].items():
        pred_array = np.array(predictions, dtype=int)
        matches = (pred_array == ground_truth).astype(int)
        
        total_correct = np.sum(matches)
        total_incorrect = len(matches) - total_correct
        accuracy = (total_correct / len(matches) * 100) if len(matches) > 0 else 0
        
        report["agents"][agent_name] = {
            "total_steps": len(matches),
            "correct": int(total_correct),
            "incorrect": int(total_incorrect),
            "accuracy_percent": round(accuracy, 2)
        }
    
    return report


def print_report(report):
    """Pretty print the comparison report."""
    print("\n" + "="*60)
    print(f"PREDICTION SUCCESS REPORT")
    print("="*60)
    print(f"Total steps analyzed: {report['total_steps']}\n")
    
    for agent_name, metrics in report["agents"].items():
        print(f"Agent: {agent_name}")
        print(f"  ✓ Correct predictions: {metrics['correct']}/{metrics['total_steps']}")
        print(f"  ✗ Incorrect predictions: {metrics['incorrect']}/{metrics['total_steps']}")
        print(f"  Accuracy: {metrics['accuracy_percent']}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate alternative visualizations for prediction evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with real experiment
  python test_tools/test_chart_predictions.py --experiment_path experiments/2024_04_28/ --output_dir debug/
  
  # Generate synthetic test data
  python test_tools/test_chart_predictions.py --generate_test --output_dir debug/
  
  # Test with specific JSON file
  python test_tools/test_chart_predictions.py --json_file test_results.json --output_dir debug/
        """
    )
    
    parser.add_argument("--experiment_path", type=str, default="_training/attacks_ho/20260424-142600_1_5_5/", help="Path to experiment results directory")
    parser.add_argument("--json_file", type=str, help="Direct path to test results JSON file")
    parser.add_argument("--output_dir", type=str, default="_training/_debug", help="Output directory for generated charts")
    parser.add_argument("--generate_test", action="store_true", help="Generate synthetic test data")
    parser.add_argument("--test_size", type=int, default=100, help="Size of synthetic test data")
    parser.add_argument("--n_agents", type=int, default=2, help="Number of agents for synthetic test")
    parser.add_argument("--success_rate", type=float, default=0.75, help="Success rate for synthetic test (0-1)")
    parser.add_argument("--original_only", action="store_true", help="Only show original visualization")
    parser.add_argument("--new_only", action="store_true", help="Only show new alternative visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load or generate test data
    test_data = None
    
    if args.generate_test:
        print(f"\nGenerating synthetic test data...")
        print(f"  Steps: {args.test_size}")
        print(f"  Agents: {args.n_agents}")
        print(f"  Success rate: {args.success_rate*100:.0f}%")
        test_data = create_test_data(
            n_steps=args.test_size,
            n_agents=args.n_agents,
            success_rate=args.success_rate
        )
    
    elif args.json_file:
        print(f"\nLoading test data from: {args.json_file}")
        try:
            with open(args.json_file, 'r') as f:
                file_data = json.load(f)
            if isinstance(file_data, dict) and "ground_truth" in file_data:
                test_data = file_data
            else:
                test_data = extract_ground_truth_predictions({args.json_file: file_data})
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            sys.exit(1)
    
    elif args.experiment_path:
        print(f"\nSearching for test results in: {args.experiment_path}")
        results = find_test_results(args.experiment_path)
        
        if not results:
            print("No test results found. Try using --generate_test or --json_file")
            sys.exit(1)
            
        #from args.experiment_path string extract the  date folder and the scenario folder
        date_folder = args.experiment_path.split('/')[-2]
        scenario_folder = args.experiment_path.split('/')[-3]
        #update output dir, adding scenario e date folder
        args.output_dir = os.path.join(args.output_dir, f"{scenario_folder}/{date_folder}")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"New Output directory: {args.output_dir}")
        
        print(f"Found {len(results)} result file(s)")
        test_data = extract_ground_truth_predictions(results)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # Validate test data
    if not test_data or not test_data.get("ground_truth") or not test_data.get("predicted"):
        print("Error: Could not extract valid ground_truth and predicted data")
        sys.exit(1)
    
    print(f"\nLoaded data:")
    print(f"  Ground truth steps: {len(test_data['ground_truth'])}")
    print(f"  Agents: {len(test_data['predicted'])}")
    for agent_name in test_data['predicted']:
        print(f"    - {agent_name}: {len(test_data['predicted'][agent_name])} predictions")
    
    # Generate report
    report = generate_comparison_report(test_data)
    print_report(report)
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    if not args.new_only:
        print("  1. Original visualization (quadratic errors)...")
        try:
            plot_ho_agent_test_errors(test_data, args.output_dir, title="Original Evaluation")
            print("     ✓ Generated: test_quadratic_errors.png")
        except Exception as e:
            print(f"     ✗ Error: {e}")
    
    if not args.original_only:
        print("  2. New alternative visualizations...")
        try:
            plot_ho_success_rate_alternatives(test_data, args.output_dir, title="Alternative Evaluation")
            print("     ✓ Generated: success_scatter.png")
            print("     ✓ Generated: success_stacked_bars.png")
            print("     ✓ Generated: success_gauge_trend.png")
            if len(test_data['predicted']) <= 5:
                print("     ✓ Generated: success_heatmap.png")
        except Exception as e:
            print(f"     ✗ Error: {e}")
    
    print(f"\n✓ Done! Check '{args.output_dir}' for generated charts")


if __name__ == "__main__":
    main()
