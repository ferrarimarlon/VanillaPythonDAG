"""
Run both test scenarios:

  python scenarios.py          # run all
  python scenarios.py default  # strict SLA → quality_check fails
  python scenarios.py reprocess  # relax SLA → reprocess downstream_blocked
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import PipelineConfig
from pipeline import build_dag


def banner(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def scenario_default() -> object:
    banner("SCENARIO 1 — default run (strict SLA: 5%)")
    dag = build_dag(PipelineConfig(max_late_delivery_rate=0.05))
    dag.run()
    return dag


def scenario_reprocess(dag) -> None:
    banner("SCENARIO 2 — reprocess after relaxing SLA to 10%")
    dag.context["config"] = PipelineConfig(max_late_delivery_rate=0.10)
    dag.reprocess("downstream_blocked")


SCENARIOS = {
    "default":   lambda: scenario_default(),
    "reprocess": lambda: scenario_reprocess(scenario_default()),
}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "all":
        dag = scenario_default()
        scenario_reprocess(dag)
    elif arg in SCENARIOS:
        SCENARIOS[arg]()
    else:
        print(f"Unknown scenario '{arg}'. Choose: all | default | reprocess")
        sys.exit(1)
