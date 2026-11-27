"""Example 2: Multimodal VLM Disaster Grounding -> Spatial World-Model Rollout.

Demonstrates grounding normalized [0, 1000] VLM bounding boxes (from Qwen3-VL / vLLM
or `vlm-zoom` on 4K disaster orthophotos) directly onto the city digital twin graph,
followed by counterfactual intervention planning.
"""

from __future__ import annotations

from resilience_wm import (
    ResourceBudget,
    SpatialReasoningAgent,
    VisualDetection,
    build_riverfront_city,
)


def main() -> None:
    city = build_riverfront_city()

    # Example detections returned by Qwen3-VL (or pass `base_url="http://localhost:8000/v1"`
    # and `image_path="disaster_ortho.png"` to query a live vLLM endpoint).
    vlm_detections = [
        VisualDetection(
            label="flooded_substation_compound",
            bbox_norm1000=(375, 615, 425, 680),
            score=0.95,
        ),
        VisualDetection(
            label="submerged_bridge_approach",
            bbox_norm1000=(485, 505, 555, 555),
            score=0.91,
        ),
        VisualDetection(
            label="deep_underpass_ponding",
            bbox_norm1000=(245, 475, 295, 525),
            score=0.89,
        ),
    ]

    agent = SpatialReasoningAgent(city)
    comparison, briefing = agent.run_mission(
        hazard="flood",
        budget=ResourceBudget.default_for("flood"),
        visual_detections=vlm_detections,
    )
    print(briefing)
    print(
        f"\nResilience gain: {comparison.resilience_gain:+.3f} "
        f"(rescued {comparison.isolated_population_reduced:,} isolated residents)"
    )


if __name__ == "__main__":
    main()
