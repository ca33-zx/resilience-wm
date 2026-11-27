"""Spatial World Model physics engine: s_{t+1} = T_phys(s_t, a_t, xi_t).

Supports two urban disaster physics regimes:
  1. `flood`: Vectorized 2D diffusive overland shallow-water routing over urban DEM,
     imperviousness, and storm drainage, with mobile pump extraction & flood barrier
     elevation interventions.
  2. `earthquake`: Ground-motion PGA attenuation field + lognormal fragility curves
     for bridges/roads/substations, with progressive crew-repair & mobile-generator
     recovery dynamics over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .city import UrbanTwin

HazardType = Literal["flood", "earthquake"]
ActionKind = Literal[
    "mobile_pump",
    "flood_barrier",
    "mobile_generator",
    "repair_crew",
    "temp_bridge",
]


@dataclass(frozen=True)
class InterventionAction:
    """One concrete spatial emergency intervention action deployed at `target_id`."""

    kind: ActionKind
    target_id: str  # node_id (for pump/barrier/generator) or edge_id (for repair_crew/temp_bridge)
    deploy_hour: float = 0.0
    label: str = ""

    @property
    def display_name(self) -> str:
        return self.label or f"{self.kind}@{self.target_id}"


@dataclass(frozen=True)
class FloodForcing:
    """Spatiotemporal extreme rainfall forcing (e.g. 6-hour 340mm cloudburst)."""

    total_rain_mm: float = 340.0
    duration_hours: float = 6.0
    peak_hour: float = 2.2
    storm_center: tuple[float, float] = (38.0, 46.0)  # (row, col)
    storm_radius_cells: float = 36.0
    dt_minutes: float = 10.0


@dataclass(frozen=True)
class EarthquakeForcing:
    """Near-field urban earthquake forcing with PGA attenuation."""

    magnitude_mw: float = 6.8
    epicenter: tuple[float, float] = (38.0, 46.0)  # (row, col)
    depth_km: float = 10.0
    duration_hours: float = 6.0


@dataclass(frozen=True)
class ObservedDamage:
    """Real-time multimodal / sensor observation snapped onto the city twin."""

    target_id: str
    kind: Literal["flooded_node", "flooded_edge", "damaged_edge", "substation_outage"]
    severity: float = 1.0  # flood depth (m) or damage factor in [0, 1]
    source: str = "vlm_perception"


@dataclass
class WorldFrame:
    """State snapshot `s_t` at hour `hour` during a world-model rollout."""

    hour: float
    hazard_field: np.ndarray  # (H, W) flood depth (m) OR seismic PGA (g)
    node_hazard: dict[str, float]  # local flood depth (m) or damage ratio [0, 1]
    edge_hazard: dict[str, float]  # peak flood depth (m) or damage ratio [0, 1]
    powered_substations: dict[str, bool] = field(default_factory=dict)
    repaired_edges: set[str] = field(default_factory=set)
    active_actions: tuple[str, ...] = ()


def _make_radial_mask(
    shape: tuple[int, int], center: tuple[float, float], radius: float
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    dist2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return np.exp(-dist2 / (2.0 * max(radius, 1.0) ** 2)).astype(np.float32)


def _route_water_step(
    depth: np.ndarray,
    eff_dem: np.ndarray,
    *,
    diffusion_rate: float = 0.22,
) -> np.ndarray:
    """Mass-conserving vectorized 2D diffusive overland routing toward lower water surface."""
    surface = eff_dem + depth
    # Pad edges with reflective boundary so water doesn't teleport out except via river drainage
    s_pad = np.pad(surface, 1, mode="edge")
    # 4-neighbor + diagonal downhill head drops (D8 gradient routing)
    offsets = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 0.707),
        (-1, 1, 0.707),
        (1, -1, 0.707),
        (1, 1, 0.707),
    ]
    h, w = depth.shape
    drops: list[np.ndarray] = []
    total_drop = np.zeros_like(depth)
    for dr, dc, weight in offsets:
        neigh = s_pad[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        drop = np.maximum(0.0, surface - neigh) * weight
        drops.append(drop)
        total_drop += drop

    # Cap outflow per cell to available water depth * diffusion_rate for numerical stability
    max_out = np.minimum(depth * 0.45, total_drop * diffusion_rate)
    denom = np.where(total_drop > 1e-6, total_drop, 1.0)
    scale = np.where(total_drop > 1e-6, max_out / denom, 0.0)

    next_depth = depth - max_out
    for (dr, dc, _), drop in zip(offsets, drops, strict=True):
        outflow = drop * scale
        # Shift outflow into neighbor (r + dr, c + dc)
        src_r0 = max(0, -dr)
        src_r1 = h - max(0, dr)
        src_c0 = max(0, -dc)
        src_c1 = w - max(0, dc)
        dst_r0 = max(0, dr)
        dst_r1 = h - max(0, -dr)
        dst_c0 = max(0, dc)
        dst_c1 = w - max(0, -dc)
        next_depth[dst_r0:dst_r1, dst_c0:dst_c1] += outflow[src_r0:src_r1, src_c0:src_c1]

    return np.maximum(0.0, next_depth)


def simulate_flood_rollout(
    city: UrbanTwin,
    *,
    forcing: FloodForcing | None = None,
    actions: list[InterventionAction] | None = None,
    observations: list[ObservedDamage] | None = None,
) -> list[WorldFrame]:
    """Roll out 2D urban flood depth evolution from `t=0h` to `t=duration_hours`."""
    f = forcing or FloodForcing()
    acts = actions or []
    obs = observations or []
    h, w = city.shape

    dt_hr = f.dt_minutes / 60.0
    n_steps = max(1, int(round(f.duration_hours / dt_hr)))
    steps_per_hour = max(1, int(round(1.0 / dt_hr)))

    # Spatial rainfall footprint (bell-shaped storm core + background regional rain)
    storm_weight = 0.55 + 0.90 * _make_radial_mask((h, w), f.storm_center, f.storm_radius_cells)
    storm_weight = storm_weight / float(storm_weight.mean())

    # Temporal rainfall hyetograph (Gaussian cloudburst peaking at peak_hour)
    step_hours = np.arange(1, n_steps + 1, dtype=np.float32) * dt_hr
    hyeto = np.exp(-0.5 * ((step_hours - f.peak_hour) / 1.35) ** 2)
    hyeto = hyeto / float(hyeto.sum()) * f.total_rain_mm  # mm per step

    # Build effective DEM (flood barriers locally raise berms around protected nodes)
    eff_dem = city.dem.copy()
    pump_masks: list[tuple[float, np.ndarray]] = []
    barrier_nodes: dict[str, float] = {}
    generator_nodes: dict[str, float] = {}

    for act in acts:
        if act.kind == "mobile_pump" and act.target_id in city.nodes:
            node = city.nodes[act.target_id]
            mask = _make_radial_mask((h, w), (float(node.row), float(node.col)), radius=5.0)
            pump_masks.append((act.deploy_hour, mask))
        elif act.kind == "flood_barrier" and act.target_id in city.nodes:
            node = city.nodes[act.target_id]
            barrier_nodes[act.target_id] = act.deploy_hour
            # Raise local protective berm ring around node (and protect node directly)
            yy, xx = np.mgrid[0:h, 0:w]
            dist = np.sqrt((yy - node.row) ** 2 + (xx - node.col) ** 2)
            ring = (dist >= 1.5) & (dist <= 3.5)
            eff_dem[ring] += 1.6
        elif act.kind == "mobile_generator" and act.target_id in city.nodes:
            generator_nodes[act.target_id] = act.deploy_hour

    # Initial water depth (plus any VLM/sensor observed flood pockets!)
    depth = np.zeros((h, w), dtype=np.float32)
    forced_outages: set[str] = set()
    for ob in obs:
        if ob.kind == "flooded_node" and ob.target_id in city.nodes:
            n = city.nodes[ob.target_id]
            depth += float(ob.severity) * _make_radial_mask(
                (h, w), (float(n.row), float(n.col)), 3.2
            )
        elif ob.kind == "flooded_edge" and ob.target_id in city.edges:
            cells = city.edge_sample_cells(city.edges[ob.target_id], samples=5)
            mr, mc = cells[len(cells) // 2]
            depth += float(ob.severity) * _make_radial_mask((h, w), (float(mr), float(mc)), 3.0)
        elif ob.kind == "substation_outage":
            forced_outages.add(ob.target_id)

    def _snapshot(hour: float, cur_depth: np.ndarray) -> WorldFrame:
        node_h: dict[str, float] = {}
        for nid, node in city.nodes.items():
            # Local 3x3 mean around node for smooth measurement
            r0, r1 = max(0, node.row - 1), min(h, node.row + 2)
            c0, c1 = max(0, node.col - 1), min(w, node.col + 2)
            d_val = float(cur_depth[r0:r1, c0:c1].mean())
            if nid in barrier_nodes and hour >= barrier_nodes[nid]:
                d_val *= 0.25  # flood barrier shields critical equipment/ramps
            node_h[nid] = d_val

        edge_h: dict[str, float] = {}
        for eid, edge in city.edges.items():
            cells = city.edge_sample_cells(edge, samples=9)
            # Bridges span above the river channel, so ignore pure river_mask cells for bridges
            vals = [
                float(cur_depth[r, c])
                for r, c in cells
                if not (edge.kind == "bridge" and city.river_mask[r, c])
            ]
            peak = max(vals) if vals else 0.0
            # Flood barriers at either endpoint shield underpass/bridge approaches
            if (edge.u in barrier_nodes and hour >= barrier_nodes[edge.u]) or (
                edge.v in barrier_nodes and hour >= barrier_nodes[edge.v]
            ):
                peak *= 0.35
            edge_h[eid] = peak

        powered: dict[str, bool] = {}
        for nid, node in city.nodes.items():
            if node.kind != "substation":
                continue
            has_gen = nid in generator_nodes and hour >= generator_nodes[nid]
            flooded_out = node_h[nid] >= node.flood_threshold_m or nid in forced_outages
            powered[nid] = True if has_gen else (not flooded_out)

        active = tuple(a.display_name for a in acts if hour >= a.deploy_hour)
        return WorldFrame(
            hour=round(hour, 2),
            hazard_field=cur_depth.copy(),
            node_hazard=node_h,
            edge_hazard=edge_h,
            powered_substations=powered,
            active_actions=active,
        )

    frames: list[WorldFrame] = [_snapshot(0.0, depth)]
    drain_m_per_step = (city.drainage_mm_hr * dt_hr) / 1000.0

    for step_idx in range(n_steps):
        cur_hour = (step_idx + 1) * dt_hr
        rain_m = (float(hyeto[step_idx]) / 1000.0) * storm_weight * city.imperviousness
        depth = np.maximum(0.0, depth + rain_m - drain_m_per_step)

        # Apply active mobile pump extraction (high-capacity emergency dewatering over local basin)
        for deploy_hr, pmask in pump_masks:
            if cur_hour >= deploy_hr:
                depth = np.maximum(0.0, depth - (0.075 * pmask))

        # Run 3 sub-iterations of 2D diffusive routing per 10-min step for crisp valley ponding
        for _ in range(3):
            depth = _route_water_step(depth, eff_dem, diffusion_rate=0.25)

        # River channel carries excess water downstream
        depth[city.river_mask] *= 0.72

        if (step_idx + 1) % steps_per_hour == 0 or step_idx == n_steps - 1:
            frames.append(_snapshot(cur_hour, depth))

    return frames


def simulate_earthquake_rollout(
    city: UrbanTwin,
    *,
    forcing: EarthquakeForcing | None = None,
    actions: list[InterventionAction] | None = None,
    observations: list[ObservedDamage] | None = None,
) -> list[WorldFrame]:
    """Roll out seismic PGA damage + progressive emergency repair recovery over `0h..6h`."""
    f = forcing or EarthquakeForcing()
    acts = actions or []
    obs = observations or []
    h, w = city.shape

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_km = np.sqrt((yy - f.epicenter[0]) ** 2 + (xx - f.epicenter[1]) ** 2) * (
        city.cell_size_m / 1000.0
    )
    r_hypo = np.sqrt(dist_km**2 + f.depth_km**2)

    # Near-field GMPE PGA attenuation (g) scaled by Mw + smooth Vs30 alluvium/bedrock site factor
    base_pga = 0.25 * np.exp(0.68 * (f.magnitude_mw - 5.5)) / ((r_hypo / 10.0) ** 1.35 + 0.35)
    site_amp = (0.70 + 0.62 / (1.0 + np.exp((city.dem - 11.2) * 0.85))).astype(np.float32)
    pga_field = np.clip(base_pga * site_amp, 0.02, 0.95).astype(np.float32)

    def _fragility_damage(pga: float, median_g: float) -> float:
        # Smooth logistic fragility mapping PGA/median -> damage index in [0, 1]
        ratio = pga / max(median_g, 0.05)
        return float(1.0 / (1.0 + np.exp(-6.5 * (ratio - 1.0))))

    init_node_dmg: dict[str, float] = {}
    for nid, node in city.nodes.items():
        local_pga = float(pga_field[node.row, node.col])
        init_node_dmg[nid] = _fragility_damage(local_pga, node.pga_threshold_g)

    init_edge_dmg: dict[str, float] = {}
    for eid, edge in city.edges.items():
        cells = city.edge_sample_cells(edge, samples=7)
        peak_pga = max(float(pga_field[r, c]) for r, c in cells)
        init_edge_dmg[eid] = _fragility_damage(peak_pga, edge.pga_threshold_g)

    # Inject multimodal/sensor observed damage overrides
    for ob in obs:
        if ob.kind == "damaged_edge" and ob.target_id in init_edge_dmg:
            init_edge_dmg[ob.target_id] = max(init_edge_dmg[ob.target_id], float(ob.severity))
        elif ob.kind == "substation_outage" and ob.target_id in init_node_dmg:
            init_node_dmg[ob.target_id] = 1.0

    # Parse intervention schedules
    gen_hours = {a.target_id: a.deploy_hour for a in acts if a.kind == "mobile_generator"}
    repair_hours: dict[str, float] = {}
    temp_bridge_hours: dict[str, float] = {}
    for a in acts:
        if a.kind == "repair_crew":
            repair_hours[a.target_id] = a.deploy_hour
        elif a.kind == "temp_bridge":
            temp_bridge_hours[a.target_id] = a.deploy_hour

    n_hours = max(1, int(round(f.duration_hours)))
    frames: list[WorldFrame] = []
    for hr in range(0, n_hours + 1):
        hour = float(hr)
        node_dmg = dict(init_node_dmg)
        edge_dmg = dict(init_edge_dmg)
        repaired: set[str] = set()

        # Repair crews clear a single emergency ambulance lane in 1.0h (dmg <= 0.36) and
        # restore full arterial passability by 2.0h; temp_bridge restores crossings in 1.0h.
        for eid, start_hr in repair_hours.items():
            if eid in edge_dmg and hour >= start_hr:
                progress = min(1.0, (hour - start_hr) / 2.0)
                if progress >= 1.0:
                    edge_dmg[eid] = min(edge_dmg[eid], 0.12)
                    repaired.add(eid)
                elif progress >= 0.5:
                    edge_dmg[eid] = min(edge_dmg[eid], 0.36)
                    repaired.add(eid)
        for eid, start_hr in temp_bridge_hours.items():
            if eid in edge_dmg and hour >= start_hr + 1.0:
                edge_dmg[eid] = 0.08
                repaired.add(eid)

        # Power substation recovery: emergency generator restores power within 1h;
        # otherwise natural grid rerouting only partially helps after hour 5.
        powered: dict[str, bool] = {}
        for nid, node in city.nodes.items():
            if node.kind != "substation":
                continue
            has_gen = nid in gen_hours and hour >= gen_hours[nid] + 1.0
            damaged_out = node_dmg[nid] >= 0.50
            powered[nid] = True if has_gen else (not damaged_out)

        active = tuple(a.display_name for a in acts if hour >= a.deploy_hour)
        frames.append(
            WorldFrame(
                hour=hour,
                hazard_field=pga_field.copy(),
                node_hazard=node_dmg,
                edge_hazard=edge_dmg,
                powered_substations=powered,
                repaired_edges=repaired,
                active_actions=active,
            )
        )
    return frames
