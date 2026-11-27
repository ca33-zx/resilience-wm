"""Tests for UrbanTwin digital twin construction, GeoJSON loading, and dual-hazard physics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from resilience_wm.city import build_riverfront_city, load_city_from_geojson
from resilience_wm.physics import (
    EarthquakeForcing,
    FloodForcing,
    InterventionAction,
    ObservedDamage,
    simulate_earthquake_rollout,
    simulate_flood_rollout,
)


def test_build_riverfront_city_structure() -> None:
    city = build_riverfront_city()
    assert city.shape == (64, 96)
    assert len(city.nodes) == 24
    assert len(city.edges) == 35
    assert city.total_population() == 620_000
    assert int(city.river_mask.sum()) > 200
    assert city.nearest_node(41, 38, kinds={"substation"}).node_id == "S1"
    assert city.nearest_edge(33, 50).edge_id == "E_B2"


def test_load_city_from_geojson_and_dem(tmp_path: Path) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [120.10, 30.25]},
                "properties": {"id": "S1", "name": "Grid Sub", "kind": "substation"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [120.14, 30.28]},
                "properties": {
                    "id": "H1",
                    "name": "City Hospital",
                    "kind": "hospital",
                    "power_source": "S1",
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [120.18, 30.22]},
                "properties": {
                    "id": "R1",
                    "name": "South Ward",
                    "kind": "residential",
                    "population": 50_000,
                    "power_source": "S1",
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[120.14, 30.28], [120.18, 30.22]],
                },
                "properties": {
                    "id": "E1",
                    "u": "H1",
                    "v": "R1",
                    "kind": "bridge",
                    "length_km": 2.0,
                },
            },
        ],
    }
    gj_path = tmp_path / "mini_city.geojson"
    gj_path.write_text(json.dumps(geojson), encoding="utf-8")

    dem = np.linspace(5.0, 15.0, 32 * 48, dtype=np.float32).reshape(32, 48)
    dem_path = tmp_path / "dem.npy"
    np.save(dem_path, dem)

    twin = load_city_from_geojson(gj_path, dem_npy_path=dem_path)
    assert twin.shape == (32, 48)
    assert set(twin.nodes) == {"S1", "H1", "R1"}
    assert set(twin.edges) == {"E1"}
    assert twin.total_population() == 50_000


def test_load_city_from_geojson_rejects_empty(tmp_path: Path) -> None:
    empty_gj = tmp_path / "empty.geojson"
    empty_gj.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="at least one Point"):
        load_city_from_geojson(empty_gj)


def test_flood_rollout_ponding_and_interventions() -> None:
    city = build_riverfront_city()
    base_frames = simulate_flood_rollout(city, forcing=FloodForcing())
    assert len(base_frames) == 7  # 0h..6h inclusive
    assert base_frames[0].hour == 0.0
    assert base_frames[-1].hour == 6.0
    assert np.all(base_frames[-1].hazard_field >= 0.0)

    # Peak depth in low-lying Old-Town underpass basin (U1) must exceed 0.25m under cloudburst
    peak_u1_base = max(fr.node_hazard["U1"] for fr in base_frames)
    assert peak_u1_base > 0.25

    # Mobile pump + flood barrier at U1 must materially lower U1 flood depth
    mitigated = simulate_flood_rollout(
        city,
        forcing=FloodForcing(),
        actions=[
            InterventionAction("mobile_pump", "U1"),
            InterventionAction("flood_barrier", "U1"),
        ],
    )
    peak_u1_mit = max(fr.node_hazard["U1"] for fr in mitigated)
    assert peak_u1_mit < peak_u1_base * 0.5

    # Emergency generator keeps S1 powered even when flooded
    gen_frames = simulate_flood_rollout(
        city,
        forcing=FloodForcing(),
        actions=[InterventionAction("mobile_generator", "S1")],
    )
    assert all(fr.powered_substations["S1"] for fr in gen_frames)


def test_earthquake_rollout_attenuation_and_recovery() -> None:
    city = build_riverfront_city()
    base_frames = simulate_earthquake_rollout(city, forcing=EarthquakeForcing())
    assert len(base_frames) == 7

    # Near-epicenter central bridge E_B2 should sustain much higher damage than distant E_E15
    dmg_b2 = base_frames[0].edge_hazard["E_B2"]
    dmg_far = base_frames[0].edge_hazard["E_E15"]
    assert dmg_b2 > dmg_far

    # Deploying a temp bridge on E_B2 and generator on S1 restores them by hour 2
    rec_frames = simulate_earthquake_rollout(
        city,
        forcing=EarthquakeForcing(),
        actions=[
            InterventionAction("temp_bridge", "E_B2", deploy_hour=0.0),
            InterventionAction("mobile_generator", "S1", deploy_hour=0.0),
        ],
        observations=[ObservedDamage("E_B2", "damaged_edge", severity=0.95)],
    )
    assert rec_frames[0].edge_hazard["E_B2"] >= 0.95
    assert rec_frames[2].edge_hazard["E_B2"] < 0.15
    assert "E_B2" in rec_frames[2].repaired_edges
    assert rec_frames[2].powered_substations["S1"] is True
