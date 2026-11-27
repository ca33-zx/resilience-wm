"""Example: Plug `resilience-wm` directly into `verl` GRPO / RLOO RL Training.

Shows how to score LLM-generated urban emergency plans inside `verl`'s custom
reward function (`compute_score`) using `ResilienceEnv`, turning the Spatial World
Model into a verifiable physics-grounded RL reward verifier with zero hallucination.
"""

from __future__ import annotations

import json
import re

from resilience_wm.env import ResilienceEnv
from resilience_wm.physics import ActionKind, HazardType


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, object] | None = None,
    extra_info: dict[str, object] | None = None,
) -> float:
    """`verl`-compatible reward verifier scoring JSON intervention plans via `ResilienceEnv`.

    Expected model output contains a JSON array:
      `[{"kind": "mobile_generator", "target_id": "S_MIS"}, ...]`
    Reward = total urban resilience gain `R_plan - R_baseline` minus `-0.05` per
    hallucinated / invalid / budget-violating action.
    """
    del data_source, extra_info
    meta = ground_truth or {}
    city_name = str(meta.get("city", "sf"))
    hazard: HazardType = "earthquake" if meta.get("hazard") == "earthquake" else "flood"
    seed = int(meta.get("seed", 42))

    match = re.search(r"\[[\s\S]*\]", solution_str)
    if not match:
        return -0.20  # format violation
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError:
        return -0.20
    if not isinstance(plan, list):
        return -0.20

    env = ResilienceEnv(city=city_name, hazard=hazard, randomize=False)
    env.reset(seed=seed)
    total_reward = 0.0
    for item in plan[:6]:
        if not isinstance(item, dict) or "kind" not in item or "target_id" not in item:
            total_reward -= 0.05
            continue
        kind: ActionKind = item["kind"]
        tid = str(item["target_id"])
        _, r, done, _, _ = env.step((kind, tid))
        total_reward += r
        if done:
            break
    return round(float(total_reward), 4)


if __name__ == "__main__":
    sample_llm_output = """Based on the SF digital twin, I deploy:
```json
[
  {"kind": "mobile_generator", "target_id": "S_MIS"},
  {"kind": "repair_crew", "target_id": "E_CAS1"},
  {"kind": "repair_crew", "target_id": "E_GEA1"}
]
```"""
    score = compute_score(
        "resilience_wm",
        sample_llm_output,
        ground_truth={"city": "sf", "hazard": "earthquake", "seed": 42},
    )
    print(f"verl GRPO physics-grounded reward score: {score:+.4f}")
