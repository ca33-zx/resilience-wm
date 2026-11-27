"""Spatial digital twin data structures: raster terrain + multi-layer coupled urban graph.

Includes a realistic built-in riverfront metropolis archetype (winding river, 3 bridges,
low-lying underpasses, 2 substations, 3 hospitals, 3 highland shelters, 10 residential
districts totaling 620,000 residents) AND loaders for real GeoJSON road graphs + NumPy DEMs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

NodeKind = Literal["residential", "hospital", "substation", "shelter", "underpass", "hub"]
EdgeKind = Literal["arterial", "bridge", "tunnel", "local"]


@dataclass(frozen=True)
class UrbanNode:
    """One spatial node in the multi-layer urban resilience graph."""

    node_id: str
    name: str
    kind: NodeKind
    row: int  # raster row (0 = north, H-1 = south)
    col: int  # raster col (0 = west, W-1 = east)
    population: int = 0
    flood_threshold_m: float = 0.45  # depth above which node degrades/fails
    pga_threshold_g: float = 0.35  # seismic PGA (g) median fragility threshold
    power_source: str | None = None  # substation node_id supplying primary power
    capacity: float = 1.0


@dataclass(frozen=True)
class UrbanEdge:
    """Undirected road, bridge, or tunnel segment linking two urban nodes."""

    edge_id: str
    u: str
    v: str
    kind: EdgeKind = "arterial"
    length_km: float = 1.5
    speed_kmh: float = 45.0
    flood_block_m: float = 0.38  # water depth (m) that severs traffic
    pga_threshold_g: float = 0.38  # seismic PGA (g) damaging the span

    @property
    def base_travel_min(self) -> float:
        return (self.length_km / max(self.speed_kmh, 1.0)) * 60.0


@dataclass
class UrbanTwin:
    """Coupled raster-terrain + multi-layer infrastructure digital twin of a city."""

    name: str
    dem: np.ndarray  # (H, W) elevation in meters
    imperviousness: np.ndarray  # (H, W) in [0.1, 1.0]
    drainage_mm_hr: np.ndarray  # (H, W) storm drainage capacity (mm/hr)
    river_mask: np.ndarray  # (H, W) bool mask for river channel
    cell_size_m: float
    nodes: dict[str, UrbanNode] = field(default_factory=dict)
    edges: dict[str, UrbanEdge] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.dem.shape[0]), int(self.dem.shape[1])

    def edge_sample_cells(self, edge: UrbanEdge, samples: int = 9) -> list[tuple[int, int]]:
        """Sample raster `(row, col)` cells along the straight path of `edge`."""
        nu = self.nodes[edge.u]
        nv = self.nodes[edge.v]
        h, w = self.shape
        pts: list[tuple[int, int]] = []
        for i in range(samples):
            t = i / max(1, samples - 1)
            r = int(round(nu.row + t * (nv.row - nu.row)))
            c = int(round(nu.col + t * (nv.col - nu.col)))
            r = max(0, min(h - 1, r))
            c = max(0, min(w - 1, c))
            if not pts or pts[-1] != (r, c):
                pts.append((r, c))
        return pts

    def nearest_node(
        self,
        row: float,
        col: float,
        *,
        kinds: set[NodeKind] | None = None,
    ) -> UrbanNode:
        """Snap a spatial coordinate `(row, col)` to the nearest graph node."""
        candidates = [n for n in self.nodes.values() if kinds is None or n.kind in kinds]
        if not candidates:
            raise ValueError(f"No nodes match kinds={kinds!r}.")
        return min(candidates, key=lambda n: (n.row - row) ** 2 + (n.col - col) ** 2)

    def nearest_edge(self, row: float, col: float) -> UrbanEdge:
        """Snap a spatial coordinate `(row, col)` to the nearest road/bridge edge."""
        best_edge: UrbanEdge | None = None
        best_d2 = float("inf")
        for edge in self.edges.values():
            for r, c in self.edge_sample_cells(edge, samples=7):
                d2 = (r - row) ** 2 + (c - col) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_edge = edge
        assert best_edge is not None
        return best_edge

    def total_population(self) -> int:
        return sum(n.population for n in self.nodes.values() if n.kind == "residential")


def build_riverfront_city(height: int = 64, width: int = 96) -> UrbanTwin:
    """Build a realistic 12.8km x 19.2km riverfront metropolis digital twin.

    Geography:
      - Winding north-south river channel around col ~ 46..54 dividing West Old Town
        and East Riverside / High-Tech districts.
      - Western low-lying basin around Underpass U1 (col ~ 26, row ~ 32) and
        Riverside Substation S1 (col ~ 38, row ~ 41) prone to flash-flood ponding.
      - Eastern hilly Highland Ridge (col > 72, row < 28) hosting Highland Substation S2
        and Trauma Hospital H3 on stiff bedrock.
      - Three critical river crossings (North Bridge B1, Central Bridge B2, South Tunnel U2)
        creating genuine spatial bottlenecks and cascading failure dynamics.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    # Winding river centerline col(row)
    river_col = 48.0 + 5.5 * np.sin((yy / height) * 2.0 * math.pi - 0.5)
    dist_river = np.abs(xx - river_col)
    river_mask = dist_river <= 2.4

    # Base elevation: gentle valley rising toward west ridge & northeast highland
    dem = 8.0 + 0.16 * np.minimum(dist_river, 32.0) ** 1.25
    # Northeast highland ridge (safe bedrock refuge)
    dem += 9.5 * np.exp(-(((xx - 80.0) / 18.0) ** 2 + ((yy - 14.0) / 16.0) ** 2))
    # Northwest foothill
    dem += 5.5 * np.exp(-(((xx - 10.0) / 16.0) ** 2 + ((yy - 10.0) / 15.0) ** 2))

    # Realistic urban flood depressions (ponding basins!)
    # 1) West Old-Town Underpass basin (U1)
    dem -= 5.6 * np.exp(-(((xx - 26.0) / 8.5) ** 2 + ((yy - 32.0) / 7.5) ** 2))
    # 2) Riverside Substation & Civic Basin (S1)
    dem -= 5.2 * np.exp(-(((xx - 38.0) / 7.5) ** 2 + ((yy - 41.0) / 7.5) ** 2))
    # 3) South River Tunnel approach (U2)
    dem -= 5.2 * np.exp(-(((xx - 50.0) / 8.0) ** 2 + ((yy - 50.0) / 6.5) ** 2))
    # 4) Eastbank Lowland Hub (around Eastbank Medical H2 & Riverside Tech Park)
    dem -= 5.0 * np.exp(-(((xx - 64.0) / 8.5) ** 2 + ((yy - 36.0) / 8.0) ** 2))

    # Carve river bed slightly below surrounding floodplains
    dem = np.where(river_mask, np.minimum(dem, 4.2), dem)
    dem = np.clip(dem, 3.0, 30.0).astype(np.float32)

    # Urban imperviousness: dense concrete core (0.85-0.95), greener outskirts (0.45-0.65)
    core_dist = np.sqrt(((xx - 48.0) / 36.0) ** 2 + ((yy - 35.0) / 26.0) ** 2)
    imperviousness = np.clip(0.94 - 0.34 * core_dist, 0.45, 0.96).astype(np.float32)
    imperviousness[river_mask] = 1.0

    # Storm drainage capacity (mm/hr): older lowlands have weaker drainage (18 mm/hr)
    drainage = np.full((height, width), 28.0, dtype=np.float32)
    lowland_core = (xx > 14) & (xx < 70) & (yy > 22) & (yy < 56)
    drainage[lowland_core] = 18.0
    drainage[xx > 72] = 36.0
    drainage[river_mask] = 55.0  # river carries runoff away faster until overtopped

    node_list = [
        # Power Substations (Layer 1)
        UrbanNode(
            "S1",
            "Riverside Substation",
            "substation",
            41,
            38,
            flood_threshold_m=0.28,
            pga_threshold_g=0.32,
        ),
        UrbanNode(
            "S2",
            "Highland Substation",
            "substation",
            16,
            78,
            flood_threshold_m=0.55,
            pga_threshold_g=0.48,
        ),
        # Hospitals (Layer 2 — dependent on primary substations!)
        UrbanNode(
            "H1",
            "Central General Hospital",
            "hospital",
            24,
            34,
            flood_threshold_m=0.50,
            pga_threshold_g=0.42,
            power_source="S1",
        ),
        UrbanNode(
            "H2",
            "Eastbank Medical Center",
            "hospital",
            36,
            65,
            flood_threshold_m=0.36,
            pga_threshold_g=0.34,
            power_source="S1",
        ),
        UrbanNode(
            "H3",
            "Highland Trauma Hospital",
            "hospital",
            14,
            82,
            flood_threshold_m=0.60,
            pga_threshold_g=0.52,
            power_source="S2",
        ),
        # Emergency Shelters (Layer 2)
        UrbanNode(
            "SH1",
            "West Hill Stadium Shelter",
            "shelter",
            12,
            14,
            flood_threshold_m=0.65,
            power_source="S1",
        ),
        UrbanNode(
            "SH2",
            "Highland Civic Shelter",
            "shelter",
            20,
            84,
            flood_threshold_m=0.65,
            power_source="S2",
        ),
        UrbanNode(
            "SH3",
            "South Park Shelter",
            "shelter",
            56,
            74,
            flood_threshold_m=0.50,
            power_source="S2",
        ),
        # Critical Transport Chokepoints / Underpasses / Bridge Hubs
        UrbanNode(
            "U1",
            "Old-Town Underpass",
            "underpass",
            32,
            26,
            flood_threshold_m=0.30,
            pga_threshold_g=0.34,
        ),
        UrbanNode(
            "U2",
            "South River Tunnel",
            "underpass",
            50,
            50,
            flood_threshold_m=0.28,
            pga_threshold_g=0.30,
        ),
        UrbanNode(
            "B1W", "North Bridge West", "hub", 14, 42, flood_threshold_m=0.55, pga_threshold_g=0.40
        ),
        UrbanNode(
            "B1E", "North Bridge East", "hub", 14, 56, flood_threshold_m=0.55, pga_threshold_g=0.40
        ),
        UrbanNode(
            "B2W",
            "Central Bridge West",
            "hub",
            33,
            44,
            flood_threshold_m=0.36,
            pga_threshold_g=0.30,
        ),
        UrbanNode(
            "B2E",
            "Central Bridge East",
            "hub",
            34,
            56,
            flood_threshold_m=0.36,
            pga_threshold_g=0.30,
        ),
        # 10 Residential Districts (620,000 total residents)
        UrbanNode(
            "R1", "NW Foothill Ward", "residential", 12, 24, population=48_000, power_source="S1"
        ),
        UrbanNode(
            "R2", "Old-Town North", "residential", 24, 18, population=72_000, power_source="S1"
        ),
        UrbanNode(
            "R3", "Old-Town Core", "residential", 36, 16, population=84_000, power_source="S1"
        ),
        UrbanNode(
            "R4", "SW Canal District", "residential", 50, 20, population=66_000, power_source="S1"
        ),
        UrbanNode(
            "R5", "South Wharf Ward", "residential", 54, 36, population=62_000, power_source="S1"
        ),
        UrbanNode(
            "R6",
            "North Bay Residences",
            "residential",
            12,
            66,
            population=54_000,
            power_source="S2",
        ),
        UrbanNode(
            "R7", "Eastbank Midtown", "residential", 28, 66, population=78_000, power_source="S1"
        ),
        UrbanNode(
            "R8", "Riverside Tech Park", "residential", 44, 62, population=68_000, power_source="S1"
        ),
        UrbanNode(
            "R9", "SE Garden Suburb", "residential", 54, 66, population=46_000, power_source="S2"
        ),
        UrbanNode(
            "R10",
            "East Highland Estates",
            "residential",
            32,
            82,
            population=42_000,
            power_source="S2",
        ),
    ]
    nodes = {n.node_id: n for n in node_list}

    # Realistic 12.8km x 19.2km urban arterial distances (km) & wet/disaster speeds (km/h)
    edge_specs: list[tuple[str, str, str, EdgeKind, float, float, float, float]] = [
        # Three River Crossings (critical spatial bridges!)
        ("E_B1", "B1W", "B1E", "bridge", 3.2, 42.0, 0.55, 0.42),
        ("E_B2", "B2W", "B2E", "bridge", 2.8, 40.0, 0.34, 0.30),
        ("E_U2W", "R5", "U2", "tunnel", 3.0, 36.0, 0.28, 0.28),
        ("E_U2E", "U2", "R9", "tunnel", 3.2, 36.0, 0.28, 0.28),
        # West Bank Road Network
        ("E_W1", "SH1", "R1", "arterial", 2.4, 38.0, 0.45, 0.48),
        ("E_W2", "R1", "B1W", "arterial", 3.8, 40.0, 0.45, 0.44),
        ("E_W3", "R1", "H1", "arterial", 3.2, 38.0, 0.42, 0.44),
        ("E_W4", "R1", "R2", "arterial", 2.8, 36.0, 0.42, 0.42),
        ("E_W5", "R2", "U1", "arterial", 2.4, 34.0, 0.30, 0.34),
        ("E_W6", "R2", "R3", "arterial", 2.8, 34.0, 0.36, 0.38),
        ("E_W7", "R3", "U1", "arterial", 2.4, 34.0, 0.30, 0.34),
        ("E_W8", "U1", "H1", "arterial", 2.6, 36.0, 0.30, 0.34),
        ("E_W9", "U1", "S1", "arterial", 3.0, 34.0, 0.32, 0.32),
        ("E_W10", "H1", "B1W", "arterial", 2.8, 38.0, 0.45, 0.42),
        ("E_W11", "H1", "B2W", "arterial", 2.8, 38.0, 0.36, 0.36),
        ("E_W12", "S1", "B2W", "arterial", 2.2, 36.0, 0.32, 0.30),
        ("E_W13", "R3", "R4", "arterial", 3.2, 36.0, 0.38, 0.38),
        ("E_W14", "R4", "S1", "arterial", 4.0, 36.0, 0.34, 0.34),
        ("E_W15", "R4", "R5", "arterial", 3.4, 36.0, 0.36, 0.36),
        ("E_W16", "S1", "R5", "arterial", 2.8, 36.0, 0.34, 0.32),
        # East Bank Road Network
        ("E_E1", "B1E", "R6", "arterial", 2.4, 40.0, 0.48, 0.45),
        ("E_E2", "R6", "S2", "arterial", 2.6, 40.0, 0.50, 0.48),
        ("E_E3", "S2", "H3", "arterial", 1.8, 42.0, 0.52, 0.50),
        ("E_E4", "H3", "SH2", "arterial", 1.6, 40.0, 0.52, 0.50),
        ("E_E5", "R6", "R7", "arterial", 3.4, 38.0, 0.44, 0.42),
        ("E_E6", "B2E", "R7", "arterial", 2.6, 38.0, 0.36, 0.34),
        ("E_E7", "B2E", "H2", "arterial", 2.2, 36.0, 0.32, 0.32),
        ("E_E8", "R7", "H2", "arterial", 2.0, 36.0, 0.34, 0.34),
        ("E_E9", "R7", "R10", "arterial", 3.6, 40.0, 0.48, 0.44),
        ("E_E10", "SH2", "R10", "arterial", 2.8, 40.0, 0.50, 0.46),
        ("E_E11", "H2", "R8", "arterial", 2.2, 36.0, 0.32, 0.32),
        ("E_E12", "H2", "R10", "arterial", 3.8, 38.0, 0.40, 0.40),
        ("E_E13", "R8", "R9", "arterial", 2.4, 36.0, 0.34, 0.34),
        ("E_E14", "R9", "SH3", "arterial", 2.0, 38.0, 0.44, 0.42),
        ("E_E15", "R10", "SH3", "arterial", 5.2, 40.0, 0.48, 0.46),
    ]
    edges = {
        eid: UrbanEdge(
            edge_id=eid,
            u=u,
            v=v,
            kind=kind,
            length_km=length_km,
            speed_kmh=speed_kmh,
            flood_block_m=flood_block_m,
            pga_threshold_g=pga_threshold_g,
        )
        for eid, u, v, kind, length_km, speed_kmh, flood_block_m, pga_threshold_g in edge_specs
    }

    return UrbanTwin(
        name="Riverfront Metropolis (12.8km x 19.2km)",
        dem=dem,
        imperviousness=imperviousness,
        drainage_mm_hr=drainage,
        river_mask=river_mask,
        cell_size_m=200.0,
        nodes=nodes,
        edges=edges,
    )


def load_city_from_geojson(
    geojson_path: str | Path,
    *,
    dem_npy_path: str | Path | None = None,
    grid_shape: tuple[int, int] = (64, 96),
    name: str | None = None,
) -> UrbanTwin:
    """Import a custom city twin from a GeoJSON FeatureCollection (+ optional `.npy` DEM).

    Point features become `UrbanNode`s (`properties.kind`, `properties.population`,
    `properties.power_source`), and LineString features become `UrbanEdge`s
    (`properties.u`, `properties.v`, `properties.kind`, `properties.length_km`).
    """
    payload = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    features = payload.get("features", [])

    xs: list[float] = []
    ys: list[float] = []
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Point":
            x, y = geom["coordinates"][:2]
            xs.append(float(x))
            ys.append(float(y))

    if not xs:
        raise ValueError(f"GeoJSON {geojson_path} must contain at least one Point feature.")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    h, w = grid_shape

    if dem_npy_path is not None:
        dem = np.load(dem_npy_path).astype(np.float32)
        h, w = int(dem.shape[0]), int(dem.shape[1])
    else:
        dem = np.full((h, w), 10.0, dtype=np.float32)

    nodes: dict[str, UrbanNode] = {}
    edges: dict[str, UrbanEdge] = {}
    for idx, feat in enumerate(features):
        geom = feat.get("geometry") or {}
        props: dict[str, Any] = feat.get("properties") or {}
        gtype = geom.get("type")
        if gtype == "Point":
            x, y = float(geom["coordinates"][0]), float(geom["coordinates"][1])
            col = int(round((x - min_x) / span_x * (w - 5))) + 2
            row = int(round((max_y - y) / span_y * (h - 5))) + 2
            nid = str(props.get("id") or f"N{idx}")
            nodes[nid] = UrbanNode(
                node_id=nid,
                name=str(props.get("name") or nid),
                kind=props.get("kind", "residential"),
                row=max(0, min(h - 1, row)),
                col=max(0, min(w - 1, col)),
                population=int(props.get("population", 0)),
                flood_threshold_m=float(props.get("flood_threshold_m", 0.45)),
                pga_threshold_g=float(props.get("pga_threshold_g", 0.35)),
                power_source=props.get("power_source"),
            )
        elif gtype == "LineString":
            u = str(props["u"])
            v = str(props["v"])
            eid = str(props.get("id") or f"E{idx}")
            edges[eid] = UrbanEdge(
                edge_id=eid,
                u=u,
                v=v,
                kind=props.get("kind", "arterial"),
                length_km=float(props.get("length_km", 1.5)),
                speed_kmh=float(props.get("speed_kmh", 45.0)),
                flood_block_m=float(props.get("flood_block_m", 0.38)),
                pga_threshold_g=float(props.get("pga_threshold_g", 0.38)),
            )

    return UrbanTwin(
        name=name or Path(geojson_path).stem,
        dem=dem,
        imperviousness=np.full((h, w), 0.78, dtype=np.float32),
        drainage_mm_hr=np.full((h, w), 25.0, dtype=np.float32),
        river_mask=np.zeros((h, w), dtype=bool),
        cell_size_m=100.0,
        nodes=nodes,
        edges=edges,
    )
