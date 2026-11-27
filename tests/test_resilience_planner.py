"""Tests for cross-layer cascading failure propagation and counterfactual planning."""

from __future__ import annotations

import numpy as np

from resilience_wm.city import build_riverfront_city
from resilience_wm.physics import WorldFrame
from resilience_wm.planner import CounterfactualPlanner, ResourceBudget
from resilience_wm.resilience import evaluate_frame


def test_cascading_power_outage_degrades_dependent_hospitals() -> None:
    city = build_riverfront_city()
    zero_field = np.zeros(city.shape, dtype=np.float32)
    zero_nodes = {nid: 0.0 for nid in city.nodes}
    zero_edges = {eid: 0.0 for eid in city.edges}

    f_ok = WorldFrame(
        hour=1.0,
        hazard_field=zero_field,
        node_hazard=zero_nodes,
        edge_hazard=zero_edges,
        powered_substations={"S1": True, "S2": True},
    )
    f_out = WorldFrame(
        hour=1.0,
        hazard_field=zero_field,
        node_hazard=zero_nodes,
        edge_hazard=zero_edges,
        powered_substations={"S1": False, "S2": True},
    )

    m_ok = evaluate_frame(city, f_ok, hazard="flood")
    m_out = evaluate_frame(city, f_out, hazard="flood")

    assert m_ok.resilience_index > 0.95
    assert m_ok.hospital_capacity["H1"] == 1.0
    # S1 outage cascades to H1 & H2 (backup diesel = 45%) while H3 (on S2) stays 100%
    assert m_out.hospital_capacity["H1"] == 0.45
    assert m_out.hospital_capacity["H2"] == 0.45
    assert m_out.hospital_capacity["H3"] == 1.0
    assert m_out.power_coverage_ratio < m_ok.power_coverage_ratio
    assert m_out.resilience_index < m_ok.resilience_index


def test_severed_roads_isolate_residential_district() -> None:
    city = build_riverfront_city()
    zero_field = np.zeros(city.shape, dtype=np.float32)
    zero_nodes = {nid: 0.0 for nid in city.nodes}
    # Block every edge touching R4 (E_W13, E_W14, E_W15)
    edge_h = {eid: 0.0 for eid in city.edges}
    for eid in ("E_W13", "E_W14", "E_W15"):
        edge_h[eid] = 0.85

    frame = WorldFrame(
        hour=2.0,
        hazard_field=zero_field,
        node_hazard=zero_nodes,
        edge_hazard=edge_h,
        powered_substations={"S1": True, "S2": True},
    )
    metrics = evaluate_frame(city, frame, hazard="flood")
    assert "R4" in metrics.isolated_districts
    assert metrics.isolated_population >= city.nodes["R4"].population
    assert metrics.road_connectivity_ratio < 1.0


def test_counterfactual_planner_improves_flood_resilience() -> None:
    city = build_riverfront_city()
    planner = CounterfactualPlanner(city, hazard="flood")
    cmp = planner.plan(budget=ResourceBudget.default_for("flood"), beam_width=2)

    assert cmp.rollouts_evaluated > 10
    assert cmp.baseline.mean_resilience < 0.85
    assert cmp.optimal.mean_resilience > cmp.baseline.mean_resilience
    assert cmp.resilience_gain > 0.05
    assert len(cmp.selected_actions) >= 1
    assert len(cmp.attributions) == len(cmp.selected_actions)
    assert any(a.marginal_resilience_gain > 0.0 for a in cmp.attributions)

    payload = cmp.to_dict()
    assert payload["hazard"] == "flood"
    assert payload["optimal_mean_resilience"] > payload["baseline_mean_resilience"]


def test_counterfactual_planner_improves_earthquake_resilience() -> None:
    city = build_riverfront_city()
    planner = CounterfactualPlanner(city, hazard="earthquake")
    cmp = planner.plan(budget=ResourceBudget.default_for("earthquake"), beam_width=2)

    assert cmp.rollouts_evaluated > 10
    assert cmp.baseline.mean_resilience < 0.85
    assert cmp.optimal.mean_resilience > cmp.baseline.mean_resilience
    assert cmp.resilience_gain > 0.05
    assert len(cmp.selected_actions) >= 1
    assert len(cmp.attributions) == len(cmp.selected_actions)
