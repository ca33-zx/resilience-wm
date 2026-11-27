"""Example: Closed-Loop Real San Francisco Satellite Orthophoto + vlm-zoom -> World Model.

Demonstrates connecting a real San Francisco Bayfront satellite orthophoto
(`assets/sf_bayfront_satellite.jpg`), optional `vlm-zoom` multi-region crop inspection,
spatial grounding onto the bundled real San Francisco digital twin (OSM + AWS 30m DEM),
LISFLOOD-FP 2D shallow-water simulation + Monte Carlo 90% CI, and counterfactual planning.
"""

from __future__ import annotations

from pathlib import Path

from resilience_wm import (
    ResourceBudget,
    SpatialReasoningAgent,
    VisualDetection,
    build_real_sf_city,
)
from resilience_wm.viz import render_multimodal_pipeline_banner


def main() -> None:
    city = build_real_sf_city()
    # Normalized [0, 1000] detections grounded from real SF Bayfront satellite orthophoto
    # (or query live Qwen3-VL / vlm-zoom via `base_url="http://localhost:8000/v1"`).
    detections = [
        VisualDetection("flooded_skyway_underpass", (260, 145, 420, 275), 0.94),
        VisualDetection("mission_bay_hospital_ponding", (415, 225, 565, 350), 0.92),
        VisualDetection("islais_creek_bridge_block", (365, 515, 520, 645), 0.91),
    ]
    agent = SpatialReasoningAgent(city)
    comparison, briefing = agent.run_mission(
        hazard="flood",
        budget=ResourceBudget.default_for("flood"),
        visual_detections=detections,
        mc_samples=16,
    )
    print(briefing)

    sat = Path("assets/sf_bayfront_satellite.jpg")
    if sat.exists():
        out = render_multimodal_pipeline_banner(
            city, comparison, sat, Path("assets/sf_vlm_world_model_pipeline.png")
        )
        print(f"\nSaved 3-stage multimodal world-model pipeline banner to {out}")


if __name__ == "__main__":
    main()
