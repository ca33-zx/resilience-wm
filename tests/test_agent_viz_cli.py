"""Tests for multimodal VLM grounding, visual renderers (PNG/GIF/HTML), and CLI."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from resilience_wm.agent import (
    SpatialReasoningAgent,
    VisualDetection,
    _parse_vlm_detections,
    ground_visual_detections_to_city,
)
from resilience_wm.city import build_riverfront_city
from resilience_wm.cli import main as cli_main
from resilience_wm.planner import ResourceBudget
from resilience_wm.viz import (
    render_hero_banner,
    render_interactive_sandbox_html,
    render_rollout_gif,
)


def test_parse_vlm_detections_handles_markdown_fences() -> None:
    raw = """Here are the detected hazards:
```json
[
  {"label": "flooded_substation", "bbox_2d": [360, 600, 440, 680], "score": 0.94},
  {"label": "blocked_bridge", "bbox_2d": [460, 490, 560, 560], "score": 0.89}
]
```
"""
    dets = _parse_vlm_detections(raw)
    assert len(dets) == 2
    assert dets[0].label == "flooded_substation"
    assert dets[0].bbox_norm1000 == (360.0, 600.0, 440.0, 680.0)
    assert _parse_vlm_detections("no json here") == []


def test_ground_visual_detections_to_city_flood_and_earthquake() -> None:
    city = build_riverfront_city()
    dets = [
        # S1 is at row=41, col=38 -> norm1000 x~400, y~650
        VisualDetection("flooded_substation", (380, 620, 420, 680), 0.95),
        # Central bridge E_B2 spans row~33.5, col=44..56 -> norm1000 x~520, y~530
        VisualDetection("collapsed_bridge", (500, 510, 550, 550), 0.91),
        # General urban anomaly near U1 (row=32, col=26)
        VisualDetection("deep_ponding", (250, 480, 290, 520), 0.88),
    ]

    flood_obs = ground_visual_detections_to_city(city, dets, hazard="flood")
    assert flood_obs[0].kind == "substation_outage"
    assert flood_obs[0].target_id == "S1"
    assert flood_obs[1].kind == "flooded_edge"
    assert flood_obs[1].target_id == "E_B2"
    assert flood_obs[2].kind == "flooded_node"
    assert flood_obs[2].target_id == "U1"

    eq_obs = ground_visual_detections_to_city(city, dets, hazard="earthquake")
    assert eq_obs[0].kind == "substation_outage"
    assert eq_obs[1].kind == "damaged_edge"
    assert eq_obs[2].kind == "damaged_edge"
    assert eq_obs[2].target_id in city.edges


def test_agent_and_visual_renderers(tmp_path: Path) -> None:
    city = build_riverfront_city()
    agent = SpatialReasoningAgent(city)
    comparison, briefing = agent.run_mission(
        hazard="flood",
        budget=ResourceBudget(counts={"mobile_generator": 1, "mobile_pump": 1}),
        visual_detections=[VisualDetection("flooded_substation", (380, 620, 420, 680))],
    )
    assert "Urban Resilience World-Model Briefing" in briefing
    assert "Grounded Multimodal / Sensor Observations" in briefing

    png_path = render_hero_banner(city, comparison, tmp_path / "hero.png")
    gif_path = render_rollout_gif(city, comparison, tmp_path / "rollout.gif", scale_width=800)
    html_path = render_interactive_sandbox_html(city, comparison, tmp_path / "sandbox.html")

    assert png_path.exists()
    with Image.open(png_path) as im:
        assert im.size == (1600, 920)

    assert gif_path.exists()
    with Image.open(gif_path) as gif:
        assert getattr(gif, "is_animated", False)
        assert getattr(gif, "n_frames", 1) == len(comparison.baseline.frames)

    html = html_path.read_text(encoding="utf-8")
    assert "Interactive Urban Resilience World-Model Sandbox" in html
    assert "const DATA =" in html


def test_cli_plan_json_and_demo(tmp_path: Path, capsys) -> None:
    out_png = tmp_path / "plan.png"
    rc = cli_main(["plan", "--hazard", "earthquake", "--out-png", str(out_png), "--json"])
    assert rc == 0
    assert out_png.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["hazard"] == "earthquake"
    assert payload["resilience_gain"] > 0.0

    demo_dir = tmp_path / "demo_out"
    rc_demo = cli_main(["demo", "--out-dir", str(demo_dir)])
    assert rc_demo == 0
    assert (demo_dir / "flood_resilience_hero.png").exists()
    assert (demo_dir / "flood_resilience_rollout.gif").exists()
    assert (demo_dir / "flood_sandbox.html").exists()
    assert (demo_dir / "earthquake_resilience_hero.png").exists()
    assert (demo_dir / "earthquake_sandbox.html").exists()
