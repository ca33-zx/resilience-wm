"""Real-world city digital twin loaders: bundled San Francisco (OSM + 30m DEM) & live fetcher.

Supports:
  1. `build_real_sf_city()`: Loads the bundled real San Francisco Peninsula digital twin
     (real AWS Mapzen Terrarium 30m DEM elevation + verified OSM/PG&E/DPH infrastructure).
  2. `fetch_terrarium_dem(bbox)`: Downloads real 30m elevation tiles from AWS Terrain Tiles
     (`s3.amazonaws.com/elevation-tiles-prod/terrarium`) for ANY lat/lon bounding box on Earth.
  3. `fetch_city_from_osm(bbox)`: Queries Overpass API for real hospitals, power substations,
     and arterial links and fuses them with real AWS 30m DEM into an `UrbanTwin`.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from .city import UrbanEdge, UrbanNode, UrbanTwin

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _latlon_to_tile_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n * 256.0
    lat_r = math.radians(max(-85.0, min(85.0, lat)))
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n * 256.0
    return x, y


def fetch_terrarium_dem(
    bbox: tuple[float, float, float, float],
    *,
    grid_shape: tuple[int, int] = (72, 96),
    zoom: int = 12,
    timeout: float = 20.0,
) -> np.ndarray:
    """Download and resample real 30m AWS Terrarium DEM for `(south, west, north, east)`."""
    south, west, north, east = bbox
    x0, y0 = _latlon_to_tile_px(north, west, zoom)
    x1, y1 = _latlon_to_tile_px(south, east, zoom)
    tx0, tx1 = int(x0 // 256), int(x1 // 256)
    ty0, ty1 = int(y0 // 256), int(y1 // 256)

    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), dtype=np.float32)
    headers = {"User-Agent": "resilience-wm/0.2.0 (https://github.com/ca33-zx/resilience-wm)"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                url = (
                    f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{tx}/{ty}.png"
                )
                resp = client.get(url)
                resp.raise_for_status()
                arr = np.asarray(
                    Image.open(io.BytesIO(resp.content)).convert("RGB"), dtype=np.float32
                )
                elev = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
                ry, rx = (ty - ty0) * 256, (tx - tx0) * 256
                mosaic[ry : ry + 256, rx : rx + 256] = elev

    cx0 = max(0, int(round(x0 - tx0 * 256)))
    cy0 = max(0, int(round(y0 - ty0 * 256)))
    cx1 = max(cx0 + 2, int(round(x1 - tx0 * 256)))
    cy1 = max(cy0 + 2, int(round(y1 - ty0 * 256)))
    cropped = mosaic[cy0:cy1, cx0:cx1]
    h, w = grid_shape
    dem_img = Image.fromarray(cropped, mode="F").resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(dem_img, dtype=np.float32)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def build_real_sf_city() -> UrbanTwin:
    """Load the bundled real San Francisco Peninsula digital twin (OSM + AWS 30m DEM).

    Covers `37.708..37.812 N, -122.516..-122.376 W` (~11.6 km x 12.4 km, 820,000 residents):
      - Real 30m elevation DEM from AWS Mapzen Terrarium (SF Bay & Pacific coast <= 0.5m,
        low-lying Bay-fill basins in Embarcadero / Mission Bay / SOMA / Marina, rising to
        255m Twin Peaks / Mount Davidson bedrock hills).
      - Real verified SF hospitals (Zuckerberg SF General, UCSF Mission Bay, CPMC Van Ness,
        UCSF Parnassus), PG&E substations (Embarcadero, Mission, Parnassus), critical US-101 /
        I-280 / Bay Bridge approaches, and 10 census neighborhoods.
    """
    gj_path = _DATA_DIR / "sf_real_city.geojson"
    dem_path = _DATA_DIR / "sf_real_dem.npy"
    payload = json.loads(gj_path.read_text(encoding="utf-8"))
    raw_dem = np.load(dem_path).astype(np.float32)
    h, w = int(raw_dem.shape[0]), int(raw_dem.shape[1])

    bbox = payload["metadata"]["bbox"]
    south, north = float(bbox["south"]), float(bbox["north"])
    west, east = float(bbox["west"]), float(bbox["east"])

    # Real Pacific Ocean & San Francisco Bay water mask (elevation <= 0.5m)
    water_mask = raw_dem <= 0.5
    # Clip seabed to sea-level 0.0m for overland shallow-water routing
    dem = np.where(water_mask, 0.0, np.maximum(0.5, raw_dem)).astype(np.float32)

    # Urban imperviousness & drainage derived from real SF elevation (dense flat Bay-fill vs hills)
    imperviousness = np.clip(0.92 - 0.0022 * np.maximum(0.0, dem - 12.0), 0.52, 0.95).astype(
        np.float32
    )
    imperviousness[water_mask] = 1.0

    drainage = np.where(dem < 12.0, 17.0, 28.0).astype(np.float32)
    drainage[water_mask] = 120.0  # ocean/bay acts as infinite hydraulic sink

    # Manning roughness n(r, c): water=0.028, flat dense urban=0.032, hilly residential=0.055
    manning_n = np.where(water_mask, 0.028, np.where(dem < 20.0, 0.035, 0.055)).astype(np.float32)

    nodes: dict[str, UrbanNode] = {}
    edges: dict[str, UrbanEdge] = {}
    for feat in payload["features"]:
        geom = feat["geometry"]
        props: dict[str, Any] = feat["properties"]
        if geom["type"] == "Point":
            lon, lat = float(geom["coordinates"][0]), float(geom["coordinates"][1])
            col = int(round((lon - west) / (east - west) * (w - 1)))
            row = int(round((north - lat) / (north - south) * (h - 1)))
            nid = str(props["id"])
            nodes[nid] = UrbanNode(
                node_id=nid,
                name=str(props["name"]),
                kind=props["kind"],
                row=max(0, min(h - 1, row)),
                col=max(0, min(w - 1, col)),
                population=int(props.get("population", 0)),
                flood_threshold_m=float(props.get("flood_threshold_m", 0.40)),
                pga_threshold_g=float(props.get("pga_threshold_g", 0.35)),
                power_source=props.get("power_source"),
            )
        elif geom["type"] == "LineString":
            eid = str(props["id"])
            edges[eid] = UrbanEdge(
                edge_id=eid,
                u=str(props["u"]),
                v=str(props["v"]),
                kind=props.get("kind", "arterial"),
                length_km=float(props.get("length_km", 2.4)),
                speed_kmh=float(props.get("speed_kmh", 36.0)),
                flood_block_m=float(props.get("flood_block_m", 0.34)),
                pga_threshold_g=float(props.get("pga_threshold_g", 0.34)),
            )

    return UrbanTwin(
        name="San Francisco Real Twin (OSM + 30m DEM)",
        dem=dem,
        imperviousness=imperviousness,
        drainage_mm_hr=drainage,
        river_mask=water_mask,
        cell_size_m=150.0,
        nodes=nodes,
        edges=edges,
        manning_n=manning_n,
        bbox=(south, west, north, east),
    )


def fetch_city_from_osm(
    bbox: tuple[float, float, float, float],
    *,
    name: str = "Custom OSM City",
    grid_shape: tuple[int, int] = (64, 96),
    overpass_url: str = "https://overpass-api.de/api/interpreter",
    timeout: float = 25.0,
) -> UrbanTwin:
    """Build a live `UrbanTwin` for ANY bounding box `(south, west, north, east)`.

    Downloads real 30m AWS Terrarium DEM tiles + queries OpenStreetMap Overpass for
    real hospitals and power substations, then synthesizes a k-NN arterial graph linking
    facilities and sampled residential grid wards.
    """
    south, west, north, east = bbox
    raw_dem = fetch_terrarium_dem(bbox, grid_shape=grid_shape, timeout=timeout)
    h, w = int(raw_dem.shape[0]), int(raw_dem.shape[1])
    water_mask = raw_dem <= 0.5
    dem = np.where(water_mask, 0.0, np.maximum(0.5, raw_dem)).astype(np.float32)

    query = (
        f"[out:json][timeout:{int(timeout)}];"
        f"("
        f'node["amenity"="hospital"]({south},{west},{north},{east});'
        f'way["amenity"="hospital"]({south},{west},{north},{east});'
        f'node["power"="substation"]({south},{west},{north},{east});'
        f'way["power"="substation"]({south},{west},{north},{east});'
        f");out center 30;"
    )
    headers = {"User-Agent": "resilience-wm/0.2.0 (https://github.com/ca33-zx/resilience-wm)"}
    elements: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.post(overpass_url, data={"data": query})
        resp.raise_for_status()
        elements = resp.json().get("elements", [])

    def _to_rc(lat: float, lon: float) -> tuple[int, int]:
        col = int(round((lon - west) / max(east - west, 1e-6) * (w - 1)))
        row = int(round((north - lat) / max(north - south, 1e-6) * (h - 1)))
        return max(0, min(h - 1, row)), max(0, min(w - 1, col))

    nodes: dict[str, UrbanNode] = {}
    coords: dict[str, tuple[float, float]] = {}
    sub_ids: list[str] = []
    hosp_count = 0
    sub_count = 0

    for el in elements:
        tags = el.get("tags") or {}
        center = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if center.get("lat") is None or center.get("lon") is None:
            continue
        lat, lon = float(center["lat"]), float(center["lon"])
        r, c = _to_rc(lat, lon)
        if tags.get("power") == "substation" and sub_count < 4:
            sub_count += 1
            nid = f"S{sub_count}"
            nodes[nid] = UrbanNode(
                nid, str(tags.get("name") or f"Substation {sub_count}"), "substation", r, c
            )
            coords[nid] = (lat, lon)
            sub_ids.append(nid)
        elif tags.get("amenity") == "hospital" and hosp_count < 5:
            hosp_count += 1
            nid = f"H{hosp_count}"
            nodes[nid] = UrbanNode(
                nid, str(tags.get("name") or f"Hospital {hosp_count}"), "hospital", r, c
            )
            coords[nid] = (lat, lon)

    # Ensure at least 1 substation and 1 hospital exist even for sparse OSM bboxes
    if not sub_ids:
        nodes["S1"] = UrbanNode("S1", "Central Substation", "substation", h // 2, w // 2)
        coords["S1"] = ((south + north) / 2, (west + east) / 2)
        sub_ids.append("S1")
    if hosp_count == 0:
        nodes["H1"] = UrbanNode(
            "H1", "Central Hospital", "hospital", h // 3, w // 2, power_source="S1"
        )
        coords["H1"] = (south + 0.66 * (north - south), (west + east) / 2)

    # Assign nearest substation to every hospital
    for nid, node in list(nodes.items()):
        if node.kind == "hospital" and node.power_source is None:
            lat, lon = coords[nid]
            nearest_sub = min(sub_ids, key=lambda sid: _haversine_km(lat, lon, *coords[sid]))
            nodes[nid] = UrbanNode(
                node.node_id,
                node.name,
                node.kind,
                node.row,
                node.col,
                power_source=nearest_sub,
            )

    # Add 8 spatial residential wards on non-water land cells
    ward_idx = 0
    for gr in (0.2, 0.5, 0.8):
        for gc in (0.2, 0.5, 0.8):
            r, c = int(gr * (h - 1)), int(gc * (w - 1))
            if water_mask[r, c]:
                continue
            ward_idx += 1
            lat = north - gr * (north - south)
            lon = west + gc * (east - west)
            nearest_sub = min(sub_ids, key=lambda sid: _haversine_km(lat, lon, *coords[sid]))
            nid = f"R{ward_idx}"
            nodes[nid] = UrbanNode(
                nid,
                f"Ward {ward_idx}",
                "residential",
                r,
                c,
                population=55_000,
                power_source=nearest_sub,
            )
            coords[nid] = (lat, lon)

    # Connect each node to its 3 nearest spatial neighbors to form a connected road graph
    edges: dict[str, UrbanEdge] = {}
    seen_pairs: set[tuple[str, str]] = set()
    all_ids = list(nodes.keys())
    e_idx = 0
    for u in all_ids:
        nbrs = sorted(
            (v for v in all_ids if v != u),
            key=lambda v: _haversine_km(*coords[u], *coords[v]),
        )[:3]
        for v in nbrs:
            pair = (u, v) if u < v else (v, u)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            e_idx += 1
            dist_km = max(0.8, _haversine_km(*coords[u], *coords[v]) * 1.25)
            edges[f"E{e_idx}"] = UrbanEdge(
                edge_id=f"E{e_idx}",
                u=u,
                v=v,
                kind="arterial",
                length_km=round(dist_km, 2),
                speed_kmh=38.0,
            )

    cell_m = (_haversine_km(south, west, north, west) * 1000.0) / max(1, h)
    return UrbanTwin(
        name=name,
        dem=dem,
        imperviousness=np.full((h, w), 0.80, dtype=np.float32),
        drainage_mm_hr=np.where(water_mask, 120.0, 24.0).astype(np.float32),
        river_mask=water_mask,
        cell_size_m=round(cell_m, 1),
        nodes=nodes,
        edges=edges,
        bbox=bbox,
    )
