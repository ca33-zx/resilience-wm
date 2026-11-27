"""Example 1: Counterfactual Urban Flood Resilience Planning & Visualization."""

from __future__ import annotations

from pathlib import Path

from resilience_wm import (
    CounterfactualPlanner,
    FloodForcing,
    ResourceBudget,
    build_riverfront_city,
    synthesize_causal_briefing,
)
from resilience_wm.viz import (
    render_hero_banner,
    render_interactive_sandbox_html,
    render_rollout_gif,
)


def main() -> None:
    city = build_riverfront_city()
    planner = CounterfactualPlanner(
        city,
        hazard="flood",
        flood_forcing=FloodForcing(total_rain_mm=240.0, duration_hours=6.0, peak_hour=2.0),
    )
    budget = ResourceBudget(counts={"mobile_pump": 2, "mobile_generator": 1, "flood_barrier": 1})
    comparison = planner.plan(budget=budget, beam_width=3)

    print(synthesize_causal_briefing(city, comparison))

    out_dir = Path("outputs")
    render_hero_banner(city, comparison, out_dir / "flood_hero.png")
    render_rollout_gif(city, comparison, out_dir / "flood_rollout.gif")
    render_interactive_sandbox_html(city, comparison, out_dir / "flood_sandbox.html")
    print(f"\nExported PNG, GIF, and interactive HTML sandbox to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
