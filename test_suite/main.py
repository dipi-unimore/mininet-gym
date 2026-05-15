"""Standardized entrypoint for ad-hoc test workflows.

Use this module instead of manually toggling code in ad-hoc scripts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from test_tools.helpers import (
    create_network_from_config,
    create_traffic_classification_env,
    create_traffic_csv_from_json,
    test_deep_learning,
    test_generate_attacks,
    test_generate_classification_traffic,
    test_results_from_saved_data,
    test_traffic_csv_with_qlearning,
)
from test_tools.qtable_coverage import plot_qtable_coverage_from_statuses
from utility.my_log import information, set_log_level
from utility.network_attacks import test_dos_attack
from utility.network_configurator import create_network, test_link_actions
from utility.params import read_config_file


@dataclass
class TestSuiteConfig:
    config_file: str = "config/default.yaml"
    qtable_coverage_statuses: str = "_training/attacks_ho/20260416-155149_1_5_5/statuses.json"
    qtable_coverage_output: str = "_training/_debug/qtable_coverage_preview.png"
    run_qtable_coverage: bool = False
    run_link_actions: bool = False
    run_create_traffic_classification_env: bool = False
    run_create_csv_traffic_from_json: bool = False
    run_generate_classification_traffic: bool = False
    run_generate_attacks: bool = False
    run_test_traffic_csv_with_qlearning: bool = False
    run_test_deep_learning: bool = False
    run_test_results_from_saved_data: bool = False
    run_dos_attack: bool = False


def load_suite_config(config_path: str | Path | None) -> TestSuiteConfig:
    base_config_path = Path(config_path) if config_path else Path(__file__).with_name("config/default.yaml")
    if not base_config_path.is_absolute():
        base_config_path = (Path.cwd() / base_config_path).resolve()

    raw = {}
    if base_config_path.exists():
        with open(base_config_path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

    defaults = raw.get("defaults", {}) if isinstance(raw.get("defaults", {}), dict) else {}
    run = raw.get("run", {}) if isinstance(raw.get("run", {}), dict) else {}
    paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}

    return TestSuiteConfig(
        config_file=str(defaults.get("config_file", "config/default.yaml")),
        qtable_coverage_statuses=str(paths.get("qtable_coverage_statuses", "")) or TestSuiteConfig.qtable_coverage_statuses,
        qtable_coverage_output=str(paths.get("qtable_coverage_output", "")) or TestSuiteConfig.qtable_coverage_output,
        run_qtable_coverage=bool(run.get("qtable_coverage", False)),
        run_link_actions=bool(run.get("link_actions", False)),
        run_create_traffic_classification_env=bool(run.get("create_traffic_classification_env", False)),
        run_create_csv_traffic_from_json=bool(run.get("create_csv_traffic_from_json", False)),
        run_generate_classification_traffic=bool(run.get("generate_classification_traffic", False)),
        run_generate_attacks=bool(run.get("generate_attacks", False)),
        run_test_traffic_csv_with_qlearning=bool(run.get("test_traffic_csv_with_qlearning", False)),
        run_test_deep_learning=bool(run.get("test_deep_learning", False)),
        run_test_results_from_saved_data=bool(run.get("test_results_from_saved_data", False)),
        run_dos_attack=bool(run.get("dos_attack", False)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standardized ad-hoc test runner")
    parser.add_argument("--config", default=None, help="Path to test-suite YAML config")
    parser.add_argument("--config-file", default=None, help="Override the application config/default.yaml passed to the legacy helpers")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    parser.add_argument("--run", action="append", default=[], help="Force-enable a task name (can be repeated)")
    parser.add_argument("--skip", action="append", default=[], help="Force-disable a task name (can be repeated)")
    return parser


def _task_enabled(task_name: str, suite_config: TestSuiteConfig, forced_run: set[str], forced_skip: set[str]) -> bool:
    if task_name in forced_skip:
        return False
    if task_name in forced_run:
        return True
    return bool(getattr(suite_config, f"run_{task_name}", False))


def run_suite(suite_config: TestSuiteConfig, config_override: str | None = None, forced_run=None, forced_skip=None):
    forced_run = set(forced_run or [])
    forced_skip = set(forced_skip or [])

    config_file = config_override or suite_config.config_file
    config, _ = read_config_file(config_file)
    set_log_level("info")

    tasks = [
        ("qtable_coverage", lambda: plot_qtable_coverage_from_statuses(
            statuses_path=suite_config.qtable_coverage_statuses,
            output_path=suite_config.qtable_coverage_output,
        )),
        ("link_actions", lambda: test_link_actions(config)),
        (
            "create_traffic_classification_env",
            lambda: create_traffic_classification_env(config, create_network_from_config(config_file=config_file)[1]),
        ),
        ("create_csv_traffic_from_json", lambda: create_traffic_csv_from_json(config.env_params.csv_file)),
        ("generate_classification_traffic", lambda: test_generate_classification_traffic(config)),
        ("generate_attacks", lambda: test_generate_attacks(config)),
        (
            "test_traffic_csv_with_qlearning",
            lambda: test_traffic_csv_with_qlearning(
                config.env_params.csv_file,
                create_network_from_config(config_file=config_file)[1],
                config.agents[4],
            ),
        ),
        ("test_deep_learning", lambda: test_deep_learning(config, create_network_from_config(config_file=config_file)[1])),
        (
            "test_results_from_saved_data",
            lambda: test_results_from_saved_data(
                "classification_from_dataset",
                "20250625-121343_1_10_1",
                "Q-learning_1",
            ),
        ),
        (
            "dos_attack",
            lambda: test_dos_attack(
                create_network(config.env_params.net_params, server_user=config.server_user)
            ),
        ),
    ]

    executed = []
    for task_name, task_fn in tasks:
        if _task_enabled(task_name, suite_config, forced_run, forced_skip):
            information(f"Running task: {task_name}\n")
            task_fn()
            executed.append(task_name)

    if not executed:
        information("No tasks enabled. Use --run <task> or update test_suite/config.yaml\n")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    suite_config = load_suite_config(args.config)

    if args.list_tasks:
        print("Available tasks:")
        print("- qtable_coverage")
        print("- link_actions")
        print("- create_traffic_classification_env")
        print("- create_csv_traffic_from_json")
        print("- generate_classification_traffic")
        print("- generate_attacks")
        print("- test_traffic_csv_with_qlearning")
        print("- test_deep_learning")
        print("- test_results_from_saved_data")
        print("- dos_attack")
        return

    run_suite(
        suite_config=suite_config,
        config_override=args.config_file,
        forced_run=args.run,
        forced_skip=args.skip,
    )


if __name__ == "__main__":
    main()
