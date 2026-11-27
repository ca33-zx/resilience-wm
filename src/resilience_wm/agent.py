"""Multimodal Perception Grounding & Spatial Reasoning Agent.

Bridges vision-language perception (aerial/satellite disaster imagery via `vLLM` /
`Qwen3-VL` or optional `vlm-zoom` for 4K orthophotos) directly into the Urban
Resilience World Model:
  1. Parses disaster detections (`flooded_underpass`, `collapsed_bridge`, `substation_flood`)
     in normalized [0, 1000] image coordinates and spatially snaps them to city twin nodes/edges.
  2. Executes deterministic World-Model rollouts & counterfactual optimization.
  3. Generates a zero-hallucination causal decision briefing (offline deterministic OR
     enriched via live OpenAI-compatible VLM endpoint).
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .city import UrbanTwin
from .physics import HazardType, ObservedDamage
from .planner import CounterfactualPlanner, PlanComparison, ResourceBudget


@dataclass(frozen=True)
class VisualDetection:
    """One disaster observation extracted from satellite/drone imagery."""

    label: str
    bbox_norm1000: tuple[float, float, float, float]  # [x1, y1, x2, y2] in [0, 1000]
    score: float = 0.9


def ground_visual_detections_to_city(
    city: UrbanTwin,
    detections: list[VisualDetection],
    *,
    hazard: HazardType = "flood",
) -> list[ObservedDamage]:
    """Spatially snap VLM image detections (`[0, 1000]`) onto city twin nodes & edges."""
    h, w = city.shape
    observations: list[ObservedDamage] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox_norm1000
        cx = ((x1 + x2) / 2.0) / 1000.0 * (w - 1)
        cy = ((y1 + y2) / 2.0) / 1000.0 * (h - 1)
        lbl = det.label.lower()

        if any(k in lbl for k in ("substation", "power", "transformer")):
            node = city.nearest_node(cy, cx, kinds={"substation"})
            observations.append(
                ObservedDamage(
                    target_id=node.node_id,
                    kind="substation_outage",
                    severity=1.0,
                    source=f"vlm:{det.label}",
                )
            )
        elif any(k in lbl for k in ("bridge", "road", "collapse", "block", "tunnel")):
            edge = city.nearest_edge(cy, cx)
            observations.append(
                ObservedDamage(
                    target_id=edge.edge_id,
                    kind="flooded_edge" if hazard == "flood" else "damaged_edge",
                    severity=0.75 if hazard == "flood" else 0.90,
                    source=f"vlm:{det.label}",
                )
            )
        elif hazard == "earthquake":
            edge = city.nearest_edge(cy, cx)
            observations.append(
                ObservedDamage(
                    target_id=edge.edge_id,
                    kind="damaged_edge",
                    severity=0.80,
                    source=f"vlm:{det.label}",
                )
            )
        else:
            node = city.nearest_node(cy, cx)
            observations.append(
                ObservedDamage(
                    target_id=node.node_id,
                    kind="flooded_node",
                    severity=0.65,
                    source=f"vlm:{det.label}",
                )
            )
    return observations


def synthesize_causal_briefing(
    city: UrbanTwin,
    comparison: PlanComparison,
    *,
    observations: list[ObservedDamage] | None = None,
) -> str:
    """Produce a deterministic, grounded spatial-reasoning briefing from rollout data."""
    base = comparison.baseline
    opt = comparison.optimal
    worst_b = base.metrics[base.worst_frame_index]
    worst_o = opt.metrics[opt.worst_frame_index]

    lines = [
        (
            f"=== Urban Resilience World-Model Briefing "
            f"({city.name} | {comparison.hazard.upper()}) ==="
        ),
        f"World-Model Rollouts Evaluated : {comparison.rollouts_evaluated}",
        (
            f"Mean Urban Resilience R(t)     : {base.mean_resilience:.3f} (Baseline) -> "
            f"{opt.mean_resilience:.3f} (Optimal Plan)  [{comparison.resilience_gain:+.3f}]"
        ),
        (
            f"Worst-Hour Resilience          : {worst_b.resilience_index:.3f} "
            f"@ t={worst_b.hour:.0f}h -> {worst_o.resilience_index:.3f} @ t={worst_o.hour:.0f}h"
        ),
        (
            f"Peak Isolated Residents        : {base.peak_isolated_population:,} -> "
            f"{opt.peak_isolated_population:,}  "
            f"(Rescued: {comparison.isolated_population_reduced:,})"
        ),
    ]
    if comparison.baseline_mc is not None and comparison.optimal_mc is not None:
        lines.append(
            f"Monte Carlo CVaR-10% (N={comparison.optimal_mc.n_samples:<2d})    : "
            f"{comparison.baseline_mc.cvar10_mean_r:.3f} (Baseline) -> "
            f"{comparison.optimal_mc.cvar10_mean_r:.3f} (Optimal Plan)"
        )

    if observations:
        lines.append("")
        lines.append("Grounded Multimodal / Sensor Observations:")
        for ob in observations:
            lines.append(
                f"  - [{ob.source}] {ob.kind} -> {ob.target_id} (severity={ob.severity:.2f})"
            )

    lines.append("")
    lines.append("Baseline Cascading Bottlenecks (Worst Hour):")
    lines.append(f"  - Offline Substations : {', '.join(worst_b.offline_substations) or 'None'}")
    lines.append(f"  - Severed Roads/Links : {', '.join(worst_b.blocked_edges) or 'None'}")
    lines.append(f"  - Isolated Districts  : {', '.join(worst_b.isolated_districts) or 'None'}")

    lines.append("")
    lines.append("Optimal Counterfactual Deployment & Leave-One-Out Attribution:")
    for idx, attr in enumerate(comparison.attributions, start=1):
        lines.append(
            f"  {idx}. {attr.action.display_name}  "
            f"(dR={attr.marginal_resilience_gain:+.3f}, "
            f"rescued={attr.marginal_isolated_rescued:,})"
        )
        lines.append(f"     {attr.rationale}")

    return "\n".join(lines)


class SpatialReasoningAgent:
    """Multimodal VLM + World-Model Agent for urban disaster perception & planning."""

    def __init__(
        self,
        city: UrbanTwin,
        *,
        base_url: str | None = None,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        api_key: str = "EMPTY",
        timeout: float = 90.0,
    ) -> None:
        self.city = city
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def perceive_disaster_image(self, image_path: str | Path) -> list[VisualDetection]:
        """Query an OpenAI-compatible VLM endpoint (`vLLM` / `Qwen3-VL`) to detect hazards."""
        if not self.base_url:
            raise ValueError("base_url must be configured to call a live VLM endpoint.")
        raw = Path(image_path).read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        prompt = (
            "Inspect this urban disaster aerial image. Detect flooded underpasses, "
            "collapsed/blocked bridges or roads, and flooded power substations. "
            'Return ONLY a JSON array of {"label": str, "bbox_2d": [x1, y1, x2, y2], '
            '"score": float} in normalized [0, 1000] coordinates.'
        )
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            text = str(resp.json()["choices"][0]["message"]["content"])
        return _parse_vlm_detections(text)

    def run_mission(
        self,
        *,
        hazard: HazardType = "flood",
        budget: ResourceBudget | None = None,
        visual_detections: list[VisualDetection] | None = None,
        image_path: str | Path | None = None,
        mc_samples: int = 0,
    ) -> tuple[PlanComparison, str]:
        """End-to-end Perception -> Spatial Grounding -> Counterfactual Planning -> Briefing."""
        dets = list(visual_detections or [])
        if image_path is not None and self.base_url:
            dets.extend(self.perceive_disaster_image(image_path))
        observations = ground_visual_detections_to_city(self.city, dets, hazard=hazard)
        planner = CounterfactualPlanner(self.city, hazard=hazard, observations=observations)
        comparison = planner.plan(budget=budget, mc_samples=mc_samples)
        briefing = synthesize_causal_briefing(self.city, comparison, observations=observations)
        return comparison, briefing


def _parse_vlm_detections(text: str) -> list[VisualDetection]:
    """Parse JSON `bbox_2d` arrays from VLM text output."""
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        items: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out: list[VisualDetection] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        box = it.get("bbox_2d") or it.get("bbox")
        if isinstance(box, list) and len(box) == 4:
            out.append(
                VisualDetection(
                    label=str(it.get("label") or "hazard"),
                    bbox_norm1000=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(it.get("score", 0.9)),
                )
            )
    return out
