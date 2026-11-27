"""Command-line interface for `resilience-wm`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import SpatialReasoningAgent
from .benchmark import render_benchmark_chart, run_benchmark
from .city import UrbanTwin, build_riverfront_city, load_city_from_geojson
from .mcp_server import serve_stdio
from .osm import build_real_sf_city, fetch_city_from_osm
from .planner import CounterfactualPlanner, ResourceBudget
from .validation import render_validation_figure, run_validation_suite
from .viz import (
    render_hero_banner,
    render_interactive_sandbox_html,
    render_multimodal_pipeline_banner,
    render_rollout_gif,
)


def _load_city(args: argparse.Namespace) -> UrbanTwin:
    if getattr(args, "geojson", None):
        return load_city_from_geojson(args.geojson, dem_npy_path=getattr(args, "dem", None))
    city_key = getattr(args, "city", "riverfront")
    if city_key == "sf":
        return build_real_sf_city()
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
        mc_samples=args.mc_samples,
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
    mc_n = int(args.mc_samples)

    # 1. Riverfront Metropolis (Flood + Earthquake with Monte Carlo 90% CI)
    rf = build_riverfront_city()
    rf_agent = SpatialReasoningAgent(rf)
    flood_cmp, flood_brief = rf_agent.run_mission(hazard="flood", mc_samples=mc_n)
    render_hero_banner(rf, flood_cmp, out_dir / "flood_resilience_hero.png")
    render_rollout_gif(rf, flood_cmp, out_dir / "flood_resilience_rollout.gif")
    render_interactive_sandbox_html(rf, flood_cmp, out_dir / "flood_sandbox.html")

    eq_cmp, eq_brief = rf_agent.run_mission(hazard="earthquake", mc_samples=mc_n)
    render_hero_banner(rf, eq_cmp, out_dir / "earthquake_resilience_hero.png")
    render_interactive_sandbox_html(rf, eq_cmp, out_dir / "earthquake_sandbox.html")

    # 2. Real San Francisco Peninsula (OSM + AWS 30m DEM, Flood + Earthquake + MC CI)
    sf = build_real_sf_city()
    sf_agent = SpatialReasoningAgent(sf)
    sf_flood_cmp, sf_flood_brief = sf_agent.run_mission(hazard="flood", mc_samples=mc_n)
    render_hero_banner(sf, sf_flood_cmp, out_dir / "sf_real_flood_hero.png")
    render_interactive_sandbox_html(sf, sf_flood_cmp, out_dir / "sf_real_flood_sandbox.html")

    sf_eq_cmp, sf_eq_brief = sf_agent.run_mission(hazard="earthquake", mc_samples=mc_n)
    render_hero_banner(sf, sf_eq_cmp, out_dir / "sf_real_earthquake_hero.png")

    # 3. Spatial-Resilience Benchmark chart
    summaries = run_benchmark(beam_width=2)
    render_benchmark_chart(summaries, out_dir / "benchmark_comparison.png")

    # 4. Real SF Satellite Orthophoto + vlm-zoom -> Spatial World Model closed-loop banner
    sat_path = out_dir / "sf_bayfront_satellite.jpg"
    if sat_path.exists():
        render_multimodal_pipeline_banner(
            sf,
            sf_flood_cmp,
            sat_path,
            out_dir / "sf_vlm_world_model_pipeline.png",
        )

    print(flood_brief)
    print("")
    print(eq_brief)
    print("")
    print(sf_flood_brief)
    print("")
    print(sf_eq_brief)
    print("\n=== Spatial-Resilience Benchmark (12 Multi-City Scenarios) ===")
    for s in summaries:
        print(
            f"  {s.strategy:20s} | R={s.mean_resilience:.3f} ({s.mean_resilience_gain:+.3f}) | "
            f"Rescued={s.mean_rescued_population:8.0f} | "
            f"Wasted={s.wasted_action_rate * 100:4.1f}% | {s.avg_latency_ms:6.1f} ms"
        )
    print(f"\nSaved all v0.2.0 visual assets & interactive HTML sandboxes to {out_dir}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    summaries = run_benchmark(
        beam_width=args.beam_width,
        llm_url=args.llm_url,
        llm_model=args.model,
    )
    if args.out_png:
        render_benchmark_chart(summaries, args.out_png)
    if args.json:
        print(json.dumps([s.to_dict() for s in summaries], indent=2))
    else:
        print("=== Spatial-Resilience Benchmark Results ===")
        for s in summaries:
            print(
                f"  {s.strategy:20s} | Mean R={s.mean_resilience:.3f} "
                f"(dR={s.mean_resilience_gain:+.3f}) | "
                f"Rescued={s.mean_rescued_population:8.0f} | "
                f"Wasted={s.wasted_action_rate * 100:4.1f}% | {s.avg_latency_ms:6.1f} ms"
            )
    return 0


def cmd_fetch_city(args: argparse.Namespace) -> int:
    parts = [float(x) for x in args.bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be 'south,west,north,east'")
    bbox = (parts[0], parts[1], parts[2], parts[3])
    city = fetch_city_from_osm(bbox, name=args.name)
    cmp = CounterfactualPlanner(city, hazard=args.hazard).plan(beam_width=2)
    if args.out_png:
        render_hero_banner(city, cmp, args.out_png)
    print(json.dumps(cmp.to_dict(), indent=2))
    return 0


def cmd_mcp(_args: argparse.Namespace) -> int:
    return serve_stdio()


def cmd_validate(args: argparse.Namespace) -> int:
    report = run_validation_suite()
    if args.out_png:
        render_validation_figure(report, args.out_png)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_text())
    return 0 if report.numerical.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resilience-wm",
        description="Neuro-Symbolic Spatial World Model for Urban Resilience.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Run counterfactual world-model rollout & optimization.")
    p_plan.add_argument("--city", choices=["sf", "riverfront"], default="sf")
    p_plan.add_argument("--hazard", choices=["flood", "earthquake"], default="flood")
    p_plan.add_argument("--mc-samples", type=int, default=16, help="Monte Carlo samples (0=off).")
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
        "demo", help="Generate all SF & Riverfront PNG/GIF/HTML/Benchmark assets."
    )
    p_demo.add_argument("--out-dir", default="assets")
    p_demo.add_argument("--mc-samples", type=int, default=16)
    p_demo.set_defaults(func=cmd_demo)

    p_bench = sub.add_parser("benchmark", help="Run the 12-scenario Spatial-Resilience Benchmark.")
    p_bench.add_argument("--beam-width", type=int, default=2)
    p_bench.add_argument("--llm-url", type=str, default=None, help="Optional zero-shot LLM URL.")
    p_bench.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    p_bench.add_argument("--out-png", type=str, default=None)
    p_bench.add_argument("--json", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    p_fetch = sub.add_parser(
        "fetch-city", help="Fetch real OSM + AWS 30m DEM for any lat/lon bbox."
    )
    p_fetch.add_argument(
        "--bbox", required=True, help="south,west,north,east (e.g. 37.72,-122.46,37.80,-122.38)"
    )
    p_fetch.add_argument("--name", default="Custom OSM City")
    p_fetch.add_argument("--hazard", choices=["flood", "earthquake"], default="flood")
    p_fetch.add_argument("--out-png", type=str, default=None)
    p_fetch.set_defaults(func=cmd_fetch_city)

    p_mcp = sub.add_parser("mcp", help="Run the JSON-RPC 2.0 MCP stdio server for AI agents.")
    p_mcp.set_defaults(func=cmd_mcp)

    p_val = sub.add_parser(
        "validate", help="Run numerical verification & SF historical disaster hindcasts."
    )
    p_val.add_argument("--out-png", type=str, default=None)
    p_val.add_argument("--json", action="store_true")
    p_val.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
