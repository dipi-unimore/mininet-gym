# Test Suite

Use this folder as the entrypoint for ad-hoc experiments and helper tasks.

## Run

List available tasks:

```bash
python -m test_suite.main --list-tasks
```

Run tasks from `test_suite/config.yaml`:

```bash
python -m test_suite.main
```

Force-enable a single task from the command line:

```bash
python -m test_suite.main --run qtable_coverage
```

Override the application config file used by the legacy helpers:

```bash
python -m test_suite.main --config-file config.yaml
```
