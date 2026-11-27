"""Spatial-Resilience Benchmark Suite: Heuristics vs. Zero-Shot LLM vs. World-Model Planner.

Evaluates multiple spatial decision strategies across a standardized suite of
urban disaster scenarios (both Real San Francisco OSM+30m DEM and Riverfront
Metropolis, across Flood and Earthquake regimes with varied storm tracks & epicenters):

  1. `no_action`: Unmitigated disaster baseline.
  2. `hazard_greedy`: Local hazard-intensity heuristic (deploys pumps/crews to
     deepest water or highest PGA without cross-layer cascading reasoning).
  3. `graph_centrality`: Pure topological degree-centrality heuristic (ignores physics).
  4. `zero_shot_llm` (optional, via `--llm-url`): Queries an OpenAI-compatible LLM
     directly from city graph + hazard text without world-model rollouts.
  5. `world_model_planner`: `resilience-wm` simulator-in-the-loop beam search.

Reports Mean Resilience R(t), Mean Resilience Gain ΔR, Mean Rescued Population,
and Wasted/Hallucinated Action Rate (fraction of deployed actions that yield <= 0 gain).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw

from .city import UrbanTwin, build_riverfront_city
from .osm import build_real_sf_city
from .physics import (
    ActionKind,
    EarthquakeForcing,
    FloodForcing,
    HazardType,
    InterventionAction,
    simulate_earthquake_rollout,
    simulate_flood_rollout,
)
from .planner import CounterfactualPlanner, ResourceBudget
from .resilience import RolloutEvaluation, evaluate_rollout
from .viz import _load_font


@dataclass(frozen=True)
class BenchmarkScenario:
    """One deterministic benchmark task instance."""

    scenario_id: str
    city_key: str
    hazard: HazardType
    flood_forcing: FloodForcing | None = None
    earthquake_forcing: EarthquakeForcing | None = None
    budget: ResourceBudget | None = None


@dataclass(frozen=True)
class StrategySummary:
    """Aggregated benchmark metrics for one decision strategy."""

    strategy: str
    scenarios_evaluated: int
    mean_resilience: float
    mean_resilience_gain: float
    mean_rescued_population: float
    wasted_action_rate: (
        float  # fraction of chosen actions with <= 1e-4 standalone or invalid effect
    )
    avg_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "scenarios_evaluated": self.scenarios_evaluated,
            "mean_resilience": round(self.mean_resilience, 4),
            "mean_resilience_gain": round(self.mean_resilience_gain, 4),
            "mean_rescued_population": round(self.mean_rescued_population, 1),
            "wasted_action_rate": round(self.wasted_action_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


def build_default_benchmark_scenarios() -> list[BenchmarkScenario]:
    """Build 12 diverse scenarios spanning Real San Francisco & Riverfront (Flood + Earthquake)."""
    return [
        # Real San Francisco Flood (3 storm footprints)
        BenchmarkScenario(
            "sf_flood_core", "sf", "flood", flood_forcing=FloodForcing(total_rain_mm=340.0)
        ),
        BenchmarkScenario(
            "sf_flood_bayfront",
            "sf",
            "flood",
            flood_forcing=FloodForcing(total_rain_mm=360.0, storm_center=(28.0, 74.0)),
        ),
        BenchmarkScenario(
            "sf_flood_mission",
            "sf",
            "flood",
            flood_forcing=FloodForcing(total_rain_mm=350.0, storm_center=(42.0, 62.0)),
        ),
        # Real San Francisco Earthquake (3 Mw / epicenter combinations)
        BenchmarkScenario(
            "sf_eq_bayfront",
            "sf",
            "earthquake",
            earthquake_forcing=EarthquakeForcing(magnitude_mw=6.8),
        ),
        BenchmarkScenario(
            "sf_eq_hayward",
            "sf",
            "earthquake",
            earthquake_forcing=EarthquakeForcing(magnitude_mw=7.0, epicenter=(32.0, 82.0)),
        ),
        BenchmarkScenario(
            "sf_eq_san_andreas",
            "sf",
            "earthquake",
            earthquake_forcing=EarthquakeForcing(magnitude_mw=6.9, epicenter=(44.0, 32.0)),
        ),
        # Riverfront Metropolis Flood (3 storm tracks)
        BenchmarkScenario(
            "rf_flood_core", "riverfront", "flood", flood_forcing=FloodForcing(total_rain_mm=340.0)
        ),
        BenchmarkScenario(
            "rf_flood_west",
            "riverfront",
            "flood",
            flood_forcing=FloodForcing(total_rain_mm=360.0, storm_center=(36.0, 28.0)),
        ),
        BenchmarkScenario(
            "rf_flood_east",
            "riverfront",
            "flood",
            flood_forcing=FloodForcing(total_rain_mm=360.0, storm_center=(38.0, 62.0)),
        ),
        # Riverfront Metropolis Earthquake (3 epicenters)
        BenchmarkScenario(
            "rf_eq_core",
            "riverfront",
            "earthquake",
            earthquake_forcing=EarthquakeForcing(magnitude_mw=6.8),
        ),
        BenchmarkScenario(
            "rf_eq_north",
            "riverfront",
            "earthquake",
            earthquake_forcing=EarthquakeForcing(magnitude_mw=6.9, epicenter=(20.0, 48.0)),
        ),
        BenchmarkScenario(
            "rf_eq_south",
            "riverfront",
            "earthquake",
            earthquake_forcing=EarthquakeForcing(magnitude_mw=6.9, epicenter=(48.0, 46.0)),
        ),
    ]


def _eval_plan(
    city: UrbanTwin, scenario: BenchmarkScenario, actions: list[InterventionAction]
) -> RolloutEvaluation:
    if scenario.hazard == "flood":
        frames = simulate_flood_rollout(city, forcing=scenario.flood_forcing, actions=actions)
    else:
        frames = simulate_earthquake_rollout(
            city, forcing=scenario.earthquake_forcing, actions=actions
        )
    return evaluate_rollout(city, frames, hazard=scenario.hazard)


def _hazard_greedy_actions(
    city: UrbanTwin, baseline: RolloutEvaluation, budget: ResourceBudget
) -> list[InterventionAction]:
    """Baseline heuristic 1: place resources on deepest water / highest PGA without graph logic."""
    worst = baseline.frames[baseline.worst_frame_index]
    actions: list[InterventionAction] = []
    used: set[str] = set()
    for kind, count in budget.counts.items():
        if count <= 0:
            continue
        if kind == "mobile_generator":
            cands = sorted(
                (n.node_id for n in city.nodes.values() if n.kind == "substation"),
                key=lambda nid: worst.node_hazard.get(nid, 0.0),
                reverse=True,
            )
        elif kind in ("mobile_pump", "flood_barrier"):
            cands = sorted(
                city.nodes.keys(),
                key=lambda nid: worst.node_hazard.get(nid, 0.0),
                reverse=True,
            )
        elif kind == "temp_bridge":
            cands = sorted(
                (e.edge_id for e in city.edges.values() if e.kind in ("bridge", "tunnel")),
                key=lambda eid: worst.edge_hazard.get(eid, 0.0),
                reverse=True,
            )
        else:
            cands = sorted(
                city.edges.keys(),
                key=lambda eid: worst.edge_hazard.get(eid, 0.0),
                reverse=True,
            )
        picked = 0
        for tid in cands:
            if tid in used:
                continue
            used.add(tid)
            actions.append(InterventionAction(kind=kind, target_id=tid))
            picked += 1
            if picked >= count:
                break
    return actions


def _centrality_heuristic_actions(
    city: UrbanTwin, budget: ResourceBudget
) -> list[InterventionAction]:
    """Baseline heuristic 2: pure road-graph degree centrality (ignores physical hazard field)."""
    deg: dict[str, int] = {nid: 0 for nid in city.nodes}
    for e in city.edges.values():
        deg[e.u] = deg.get(e.u, 0) + 1
        deg[e.v] = deg.get(e.v, 0) + 1

    actions: list[InterventionAction] = []
    used: set[str] = set()
    for kind, count in budget.counts.items():
        if count <= 0:
            continue
        if kind == "mobile_generator":
            cands = sorted(
                (n.node_id for n in city.nodes.values() if n.kind == "substation"),
                key=lambda nid: deg.get(nid, 0),
                reverse=True,
            )
        elif kind in ("mobile_pump", "flood_barrier"):
            cands = sorted(city.nodes.keys(), key=lambda nid: deg.get(nid, 0), reverse=True)
        elif kind == "temp_bridge":
            cands = sorted(
                (e.edge_id for e in city.edges.values() if e.kind in ("bridge", "tunnel")),
                key=lambda eid: deg[city.edges[eid].u] + deg[city.edges[eid].v],
                reverse=True,
            )
        else:
            cands = sorted(
                city.edges.keys(),
                key=lambda eid: deg[city.edges[eid].u] + deg[city.edges[eid].v],
                reverse=True,
            )
        picked = 0
        for tid in cands:
            if tid in used:
                continue
            used.add(tid)
            actions.append(InterventionAction(kind=kind, target_id=tid))
            picked += 1
            if picked >= count:
                break
    return actions


def _zero_shot_llm_actions(
    city: UrbanTwin,
    scenario: BenchmarkScenario,
    baseline: RolloutEvaluation,
    budget: ResourceBudget,
    *,
    llm_url: str,
    model: str,
    api_key: str = "EMPTY",
) -> tuple[list[InterventionAction], int]:
    """Query an OpenAI-compatible LLM directly (without World-Model rollouts)."""
    wm = baseline.metrics[baseline.worst_frame_index]
    prompt = (
        f"City: {city.name}. Hazard: {scenario.hazard}. "
        f"Offline substations: {list(wm.offline_substations)}. "
        f"Isolated districts: {list(wm.isolated_districts)}. "
        f"Blocked edges: {list(wm.blocked_edges[:12])}. "
        f"Budget: {dict(budget.counts)}. "
        f"Valid nodes: {sorted(city.nodes.keys())}. Valid edges: {sorted(city.edges.keys())}. "
        'Return ONLY a JSON list of {"kind": str, "target_id": str} within budget.'
    )
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=45.0) as client:
        resp = client.post(f"{llm_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        text = str(resp.json()["choices"][0]["message"]["content"])

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return [], sum(budget.counts.values())
    try:
        items: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [], sum(budget.counts.values())

    valid: list[InterventionAction] = []
    invalid_count = 0
    rem = dict(budget.counts)
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            invalid_count += 1
            continue
        k: ActionKind = it.get("kind", "")
        tid = str(it.get("target_id", ""))
        if rem.get(k, 0) > 0 and (tid in city.nodes or tid in city.edges):
            rem[k] -= 1
            valid.append(InterventionAction(kind=k, target_id=tid))
        else:
            invalid_count += 1
    return valid, invalid_count


def run_benchmark(
    *,
    scenarios: list[BenchmarkScenario] | None = None,
    beam_width: int = 2,
    llm_url: str | None = None,
    llm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
    api_key: str = "EMPTY",
) -> list[StrategySummary]:
    """Run the Spatial-Resilience Benchmark across all scenarios and strategies."""
    sc_list = scenarios or build_default_benchmark_scenarios()
    cities = {"sf": build_real_sf_city(), "riverfront": build_riverfront_city()}

    strat_names = ["no_action", "graph_centrality", "hazard_greedy", "world_model_planner"]
    if llm_url:
        strat_names.insert(3, "zero_shot_llm")

    acc: dict[str, dict[str, float]] = {
        s: {"r": 0.0, "gain": 0.0, "rescued": 0.0, "wasted": 0.0, "total_act": 0.0, "ms": 0.0}
        for s in strat_names
    }

    for sc in sc_list:
        city = cities[sc.city_key]
        budget = sc.budget or ResourceBudget.default_for(sc.hazard)

        t0 = time.perf_counter()
        base = _eval_plan(city, sc, [])
        base_ms = (time.perf_counter() - t0) * 1000.0
        acc["no_action"]["r"] += base.mean_resilience
        acc["no_action"]["ms"] += base_ms

        for sname in strat_names:
            if sname == "no_action":
                continue
            t_start = time.perf_counter()
            extra_invalid = 0
            if sname == "hazard_greedy":
                acts = _hazard_greedy_actions(city, base, budget)
                ev = _eval_plan(city, sc, acts)
            elif sname == "graph_centrality":
                acts = _centrality_heuristic_actions(city, budget)
                ev = _eval_plan(city, sc, acts)
            elif sname == "zero_shot_llm" and llm_url:
                acts, extra_invalid = _zero_shot_llm_actions(
                    city, sc, base, budget, llm_url=llm_url, model=llm_model, api_key=api_key
                )
                ev = _eval_plan(city, sc, acts)
            else:
                planner = CounterfactualPlanner(
                    city,
                    hazard=sc.hazard,
                    flood_forcing=sc.flood_forcing,
                    earthquake_forcing=sc.earthquake_forcing,
                )
                cmp = planner.plan(budget=budget, beam_width=beam_width)
                acts = cmp.selected_actions
                ev = cmp.optimal

            dt_ms = (time.perf_counter() - t_start) * 1000.0

            # Measure wasted/ineffective action rate via single-action marginal check
            wasted = extra_invalid
            for a in acts:
                solo = _eval_plan(city, sc, [a])
                if solo.mean_resilience <= base.mean_resilience + 1e-4 and (
                    solo.peak_isolated_population >= base.peak_isolated_population
                ):
                    wasted += 1

            acc[sname]["r"] += ev.mean_resilience
            acc[sname]["gain"] += ev.mean_resilience - base.mean_resilience
            acc[sname]["rescued"] += max(
                0, base.peak_isolated_population - ev.peak_isolated_population
            )
            acc[sname]["wasted"] += wasted
            acc[sname]["total_act"] += max(1, len(acts) + extra_invalid)
            acc[sname]["ms"] += dt_ms

    n = max(1, len(sc_list))
    summaries: list[StrategySummary] = []
    for sname in strat_names:
        d = acc[sname]
        wasted_rate = d["wasted"] / d["total_act"] if d["total_act"] > 0 else 0.0
        summaries.append(
            StrategySummary(
                strategy=sname,
                scenarios_evaluated=len(sc_list),
                mean_resilience=d["r"] / n,
                mean_resilience_gain=d["gain"] / n,
                mean_rescued_population=d["rescued"] / n,
                wasted_action_rate=wasted_rate,
                avg_latency_ms=d["ms"] / n,
            )
        )
    return summaries


def render_benchmark_chart(summaries: list[StrategySummary], output_path: str | Path) -> Path:
    """Render a crisp publication-grade dark-mode Spatial-Resilience Benchmark chart."""
    W, H = 1280, 560
    img = Image.new("RGB", (W, H), (11, 17, 32))
    draw = ImageDraw.Draw(img)
    h1 = _load_font(24)
    sub = _load_font(15)
    mono = _load_font(14, mono=True)
    small = _load_font(13, mono=True)

    n_sc = summaries[0].scenarios_evaluated if summaries else 0
    draw.text(
        (28, 20),
        f"Spatial-Resilience Benchmark ({n_sc} Multi-City Flood & Earthquake Scenarios)",
        fill=(241, 245, 249),
        font=h1,
    )
    draw.text(
        (28, 52),
        (
            "Real San Francisco (OSM + 30m DEM) & Riverfront Metropolis  |  "
            "LISFLOOD-FP Shallow Water + HAZUS Fragility"
        ),
        fill=(148, 163, 184),
        font=sub,
    )

    # Left chart: Mean Urban Resilience R(t)
    panels = [
        ("Mean Urban Resilience R(t) (Higher is Better)", 28, 96, 616, 520, "r"),
        ("Mean Residents Rescued & Wasted-Action Rate", 648, 96, 1252, 520, "pop"),
    ]
    pretty = {
        "no_action": "No-Action Baseline",
        "graph_centrality": "Graph Centrality (No Physics)",
        "hazard_greedy": "Local Hazard Greedy (No Graph)",
        "zero_shot_llm": "Zero-Shot LLM (No World Model)",
        "world_model_planner": "resilience-wm World Model",
    }
    colors = {
        "no_action": (239, 68, 68),
        "graph_centrality": (148, 163, 184),
        "hazard_greedy": (245, 158, 11),
        "zero_shot_llm": (168, 85, 247),
        "world_model_planner": (16, 185, 129),
    }

    max_pop = max((s.mean_rescued_population for s in summaries), default=1.0)
    max_pop = max(max_pop, 10_000.0)

    for title, x1, y1, x2, y2, mode in panels:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=(15, 23, 42), outline=(51, 65, 85))
        draw.text((x1 + 18, y1 + 14), title, fill=(226, 232, 240), font=mono)

        bar_y = y1 + 64
        bar_h = 44
        gap = 36
        bar_x0 = x1 + 22
        bar_w = (x2 - x1) - 44

        for s in summaries:
            col = colors.get(s.strategy, (56, 189, 248))
            lbl = pretty.get(s.strategy, s.strategy)
            draw.text((bar_x0, bar_y - 20), lbl, fill=(226, 232, 240), font=small)

            # Track background
            draw.rounded_rectangle(
                (bar_x0, bar_y, bar_x0 + bar_w, bar_y + bar_h),
                radius=6,
                fill=(30, 41, 59),
            )
            if mode == "r":
                frac = max(0.02, min(1.0, s.mean_resilience))
                fw = int(bar_w * frac)
                draw.rounded_rectangle(
                    (bar_x0, bar_y, bar_x0 + fw, bar_y + bar_h), radius=6, fill=col
                )
                val_txt = f"R={s.mean_resilience:.3f}  (dR={s.mean_resilience_gain:+.3f})"
            else:
                frac = max(0.02, min(1.0, s.mean_rescued_population / (max_pop * 1.15)))
                fw = int(bar_w * frac)
                draw.rounded_rectangle(
                    (bar_x0, bar_y, bar_x0 + fw, bar_y + bar_h), radius=6, fill=col
                )
                val_txt = (
                    f"+{int(round(s.mean_rescued_population)):,} rescued  |  "
                    f"Wasted={s.wasted_action_rate * 100:.0f}%"
                )

            # High-contrast label right-aligned on the bar track so it never clips a bar edge
            tb = draw.textbbox((0, 0), val_txt, font=small)
            tw = tb[2] - tb[0]
            tx = bar_x0 + bar_w - tw - 12
            draw.rounded_rectangle(
                (tx - 8, bar_y + 8, bar_x0 + bar_w - 6, bar_y + bar_h - 8),
                radius=4,
                fill=(15, 23, 42),
            )
            draw.text((tx, bar_y + 14), val_txt, fill=(248, 250, 252), font=small)
            bar_y += bar_h + gap

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
