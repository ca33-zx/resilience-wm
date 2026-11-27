"""Comprehensive v0.2.0 tests: Real SF twin, LISFLOOD-FP, Monte Carlo, Env, MCP & Benchmark."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from resilience_wm.benchmark import (
    BenchmarkScenario,
    render_benchmark_chart,
    run_benchmark,
)
from resilience_wm.env import ResilienceEnv
from resilience_wm.mcp_server import handle_jsonrpc_request, serve_stdio
from resilience_wm.osm import build_real_sf_city
from resilience_wm.physics import (
    EarthquakeForcing,
    FloodForcing,
    _hazus_damage_ratio,
    _lisflood_inertial_step,
)
from resilience_wm.planner import CounterfactualPlanner
from resilience_wm.resilience import evaluate_monte_carlo
from resilience_wm.viz import render_multimodal_pipeline_banner


def test_real_sf_city_osm_and_30m_dem() -> None:
    sf = build_real_sf_city()
    assert sf.shape == (72, 96)
    assert sf.bbox is not None
    assert sf.total_population() == 820_000
    assert "H_ZSFG" in sf.nodes
    assert "H_MB" in sf.nodes
    assert "S_EMB" in sf.nodes
    assert "S_MIS" in sf.nodes
    # Real SF elevation spans Bay sea level (0m) up to Twin Peaks / Mt Davidson (> 200m)
    assert float(sf.dem.min()) == 0.0
    assert float(sf.dem.max()) > 200.0
    assert int(sf.river_mask.sum()) > 1000
    assert sf.manning_n is not None


def test_lisflood_inertial_strict_mass_conservation() -> None:
    dem = np.zeros((28, 28), dtype=np.float32)
    dem[:, 14:] = 3.0
    h0 = np.zeros((28, 28), dtype=np.float32)
    h0[6:22, 16:24] = 0.75
    qx = np.zeros((28, 27), dtype=np.float32)
    qy = np.zeros((27, 28), dtype=np.float32)
    manning_n = np.full((28, 28), 0.035, dtype=np.float32)

    h1, qx, qy = _lisflood_inertial_step(h0, qx, qy, dem, manning_n, dx=100.0, dt_total=150.0)
    # Water must flow downhill onto the lower left half while conserving volume to float32 precision
    assert float(h1[:, :14].sum()) > 0.0
    rel_err = abs(float(h1.sum() - h0.sum())) / float(h0.sum())
    assert rel_err < 1e-5


def test_hazus_fragility_and_monte_carlo_confidence_bands() -> None:
    assert 0.0 <= _hazus_damage_ratio(0.05, 0.35) < _hazus_damage_ratio(0.35, 0.35)
    assert _hazus_damage_ratio(0.35, 0.35) < _hazus_damage_ratio(0.85, 0.35) <= 1.0

    sf = build_real_sf_city()
    mc = evaluate_monte_carlo(
        sf,
        hazard="earthquake",
        earthquake_forcing=EarthquakeForcing(magnitude_mw=6.8),
        n_samples=10,
        seed=7,
    )
    assert mc.n_samples == 10
    assert len(mc.mean_r) == 7
    for lo, mid, hi in zip(mc.p05_r, mc.mean_r, mc.p95_r, strict=True):
        assert lo <= mid + 1e-4
        assert mid <= hi + 1e-4
    assert 0.0 <= mc.cvar10_mean_r <= 1.0


def test_gymnasium_compatible_resilience_env() -> None:
    env = ResilienceEnv(city="sf", hazard="earthquake", randomize=False)
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert info["valid_action_mask"].any()
    assert "San Francisco" in info["state_text"]

    # Take a valid action
    valid_indices = np.flatnonzero(info["valid_action_mask"])
    first_idx = int(valid_indices[0])
    obs2, reward, done, trunc, info2 = env.step(first_idx)
    assert obs2.shape == obs.shape
    assert trunc is False
    assert info2["invalid_reason"] is None

    # Repeating the exact same target triggers duplicate hallucination penalty (-0.05)
    _, pen_reward, _, _, info_dup = env.step(first_idx)
    assert pen_reward == -0.05
    assert info_dup["invalid_reason"] is not None


def test_mcp_jsonrpc_server_roundtrip() -> None:
    init_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "resilience-wm"

    tools_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert {
        "list_cities",
        "inspect_city_graph",
        "simulate_disaster",
        "optimize_response_plan",
    } <= names

    # Full stdio transport test
    req_lines = "\n".join(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "list_cities", "arguments": {}},
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "simulate_disaster",
                        "arguments": {"city": "sf", "hazard": "earthquake", "mc_samples": 4},
                    },
                }
            ),
        ]
    )
    out_buf = io.StringIO()
    assert serve_stdio(io.StringIO(req_lines + "\n"), out_buf) == 0
    responses = [json.loads(line) for line in out_buf.getvalue().strip().splitlines()]
    assert len(responses) == 2
    cities_payload = json.loads(responses[0]["result"]["content"][0]["text"])
    assert len(cities_payload["cities"]) == 2
    sim_payload = json.loads(responses[1]["result"]["content"][0]["text"])
    assert "monte_carlo" in sim_payload


def test_spatial_resilience_benchmark_and_multimodal_banner(tmp_path: Path) -> None:
    mini_scenarios = [
        BenchmarkScenario(
            "sf_eq", "sf", "earthquake", earthquake_forcing=EarthquakeForcing(magnitude_mw=6.8)
        ),
        BenchmarkScenario(
            "rf_fl", "riverfront", "flood", flood_forcing=FloodForcing(total_rain_mm=340.0)
        ),
    ]
    summaries = run_benchmark(scenarios=mini_scenarios, beam_width=2)
    by_name = {s.strategy: s for s in summaries}
    assert by_name["world_model_planner"].mean_resilience > by_name["no_action"].mean_resilience
    assert (
        by_name["world_model_planner"].mean_resilience >= by_name["hazard_greedy"].mean_resilience
    )
    assert (
        by_name["world_model_planner"].mean_resilience
        >= by_name["graph_centrality"].mean_resilience
    )

    chart = render_benchmark_chart(summaries, tmp_path / "bench.png")
    assert chart.exists()

    # Verify multimodal 3-stage pipeline banner
    sat_tmp = tmp_path / "sat.jpg"
    Image.new("RGB", (512, 512), (40, 70, 90)).save(sat_tmp)
    sf = build_real_sf_city()
    cmp = CounterfactualPlanner(sf, hazard="earthquake").plan(beam_width=1, mc_samples=4)
    banner = render_multimodal_pipeline_banner(sf, cmp, sat_tmp, tmp_path / "pipe.png")
    assert banner.exists()
    with Image.open(banner) as im:
        assert im.size == (1600, 680)
