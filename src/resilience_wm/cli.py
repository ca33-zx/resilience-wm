"""Command-line interface for `resilience-wm`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import SpatialReasoningAgent
from .city import build_riverfront_city, load_city_from_geojson
from .planner import ResourceBudget
from .viz import render_hero_banner, render_interactive_sandbox_html, render_rollout_gif


def _load_city(args: argparse.Namespace):
    if getattr(args, "geojson", None):
        return load_city_from_geojson(args.geojson, dem_npy_path=getattr(args, "dem", None))
    return build_riverfront_city()


def cmd_plan(args: argparse.Namespace) -> int:
    city = _load_city(args)
    agent = SpatialReasoningAgent(
        city,
        base_url=args.vlm_url,
        model=args.model,
    )
    budget = ResourceBudget.default_for(args.hazard)
    comparison, briefing = agent.run_mission(
        hazard=args.hazard,
        budget=budget,
        image_path=args.image,
    )

    if args.out_png:
        render_hero_banner(city, comparison, args.out_png)
    if args.out_gif:
        render_rollout_gif(city, comparison, args.out_gif)
    if args.out_html:
        render_interactive_sandbox_html(city, comparison, args.out_html)

    if args.json:
        print(json.dumps(comparison.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(briefing)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    city = build_riverfront_city()
    agent = SpatialReasoningAgent(city)

    # 1. Extreme Cloudburst Flood scenario
    flood_cmp, flood_brief = agent.run_mission(hazard="flood")
    render_hero_banner(city, flood_cmp, out_dir / "flood_resilience_hero.png")
    render_rollout_gif(city, flood_cmp, out_dir / "flood_resilience_rollout.gif")
    render_interactive_sandbox_html(city, flood_cmp, out_dir / "flood_sandbox.html")

    # 2. Urban Earthquake scenario
    eq_cmp, eq_brief = agent.run_mission(hazard="earthquake")
    render_hero_banner(city, eq_cmp, out_dir / "earthquake_resilience_hero.png")
    render_interactive_sandbox_html(city, eq_cmp, out_dir / "earthquake_sandbox.html")

    print(flood_brief)
    print("")
    print(eq_brief)
    print(f"\nSaved visual assets & interactive HTML sandboxes to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resilience-wm",
        description="Neuro-Symbolic Spatial World Model for Urban Resilience.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Run counterfactual world-model rollout & optimization.")
    p_plan.add_argument("--hazard", choices=["flood", "earthquake"], default="flood")
    p_plan.add_argument("--geojson", type=str, default=None, help="Optional GeoJSON city graph.")
    p_plan.add_argument("--dem", type=str, default=None, help="Optional .npy DEM raster.")
    p_plan.add_argument("--image", type=str, default=None, help="Optional disaster aerial image.")
    p_plan.add_argument("--vlm-url", type=str, default=None, help="Optional vLLM OpenAI base URL.")
    p_plan.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    p_plan.add_argument("--out-png", type=str, default=None)
    p_plan.add_argument("--out-gif", type=str, default=None)
    p_plan.add_argument("--out-html", type=str, default=None)
    p_plan.add_argument("--json", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_demo = sub.add_parser(
        "demo", help="Generate flood & earthquake PNG/GIF/HTML showcase assets."
    )
    p_demo.add_argument("--out-dir", default="assets")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
