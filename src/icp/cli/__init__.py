"""Command-line entry points for ICP workflows.

The main CLIs are exposed via modules in this package, e.g.:

- ``python -m icp.cli.score_accounts``
- ``python -m icp.cli.optimize_weights``
"""

__all__ = ["score_accounts", "optimize_weights", "update_matching", "build_playbooks", "build_pulse_artifacts"]
