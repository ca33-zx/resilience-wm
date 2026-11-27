"""Spatial World Model physics engine: s_{t+1} = T_phys(s_t, a_t, xi_t).

Supports two urban disaster physics regimes:
  1. `flood`:
     - Default (`solver="lisflood_inertial"`): Mass-conserving 2D Local Inertial
       Shallow-Water Equations (Bates et al., 2010 LISFLOOD-FP formulation) with
       spatially distributed Manning roughness n(r, c), CFL stability control,
       Green-Ampt infiltration on pervious surfaces, mobile pump extraction, and
       protective flood barriers.
     - Fallback (`solver="diffusive_d8"`): Fast D8 diffusive overland routing.
  2. `earthquake`:
     - Near-field GMPE PGA attenuation + continuous Vs30 alluvium/bedrock site
       amplification + HAZUS-style 4-state lognormal fragility curves
       (Slight / Moderate / Extensive / Complete) + progressive crew repair and
       emergency generator recovery dynamics.
     - Optional spatially correlated log-PGA perturbation (`rng` / `sigma_ln_pga`)
       for Monte Carlo uncertainty quantification and CVaR-10% tail risk analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .city import UrbanTwin

HazardType = Literal["flood", "earthquake"]
FloodSolver = Literal["lisflood_inertial", "diffusive_d8"]
ActionKind = Literal[
    "mobile_pump",
    "flood_barrier",
    "mobile_generator",
    "repair_crew",
    "temp_bridge",
]

_GRAVITY = 9.81


@dataclass(frozen=True)
class InterventionAction:
    """One concrete spatial emergency intervention action deployed at `target_id`."""

    kind: ActionKind
    target_id: str  # node_id (pump/barrier/generator) or edge_id (repair_crew/temp_bridge)
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
    storm_center: tuple[float, float] | None = None  # (row, col); defaults to city core
    storm_radius_cells: float = 36.0
    dt_minutes: float = 10.0
    solver: FloodSolver = "lisflood_inertial"
    infiltration_mm_hr: float = 14.0  # Green-Ampt pervious infiltration rate


@dataclass(frozen=True)
class EarthquakeForcing:
    """Near-field urban earthquake forcing with GMPE PGA attenuation & HAZUS fragility."""

    magnitude_mw: float = 6.8
    epicenter: tuple[float, float] | None = None  # (row, col); defaults to urban core
    depth_km: float = 10.0
    duration_hours: float = 6.0
    sigma_ln_pga: float = 0.0  # lognormal ground-motion uncertainty for Monte Carlo


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
    node_hazard: dict[str, float]  # local flood depth (m) or HAZUS damage ratio [0, 1]
    edge_hazard: dict[str, float]  # peak flood depth (m) or HAZUS damage ratio [0, 1]
    powered_substations: dict[str, bool] = field(default_factory=dict)
    repaired_edges: set[str] = field(default_factory=set)
    active_actions: tuple[str, ...] = ()


def _make_radial_mask(
    shape: tuple[int, int], center: tuple[float, float], radius: float
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    dist2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return np.exp(-dist2 / (2.0 * max(radius, 1.0) ** 2)).astype(np.float32)


def _default_hazard_center(city: UrbanTwin) -> tuple[float, float]:
    """Choose the population/infrastructure centroid when forcing center is unspecified."""
    h, w = city.shape
    if "S1" in city.nodes:
        return (38.0, 46.0)
    if "S_EMB" in city.nodes and "S_MIS" in city.nodes:
        # San Francisco eastern Bay-fill core (Mission Bay / SOMA / Embarcadero)
        s1 = city.nodes["S_EMB"]
        s2 = city.nodes["S_MIS"]
        return ((s1.row + s2.row) * 0.55, (s1.col + s2.col) * 0.50)
    if city.nodes:
        r_avg = sum(n.row for n in city.nodes.values()) / len(city.nodes)
        c_avg = sum(n.col for n in city.nodes.values()) / len(city.nodes)
        return (r_avg, c_avg)
    return (h * 0.5, w * 0.5)


def _default_manning_n(city: UrbanTwin) -> np.ndarray:
    if city.manning_n is not None:
        return city.manning_n.astype(np.float32)
    n = np.where(
        city.river_mask,
        0.028,
        0.025 + 0.055 * (1.0 - city.imperviousness),
    )
    return n.astype(np.float32)


def _lisflood_inertial_step(
    depth: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    eff_dem: np.ndarray,
    manning_n: np.ndarray,
    *,
    dx: float,
    dt_total: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mass-conserving Bates et al. (2010) LISFLOOD-FP local inertial 2D shallow-water step.

    Updates face unit discharges `(qx, qy)` with semi-implicit Manning friction and
    adaptive CFL sub-stepping, then applies flux divergence to conserve water volume.
    """
    h = depth.copy()
    max_h = float(h.max())
    if max_h < 1e-5:
        return h, qx * 0.0, qy * 0.0

    # CFL-stable sub-stepping: dt <= 0.42 * dx / sqrt(g * h_max)
    c_wave = math.sqrt(_GRAVITY * max(max_h, 0.25))
    dt_cfl = 0.42 * dx / max(c_wave, 0.5)
    n_sub = max(1, min(12, int(math.ceil(dt_total / dt_cfl))))
    dt = dt_total / n_sub

    # Face-averaged Manning roughness squared
    nx2 = (0.5 * (manning_n[:, :-1] + manning_n[:, 1:])) ** 2
    ny2 = (0.5 * (manning_n[:-1, :] + manning_n[1:, :])) ** 2

    for _ in range(n_sub):
        eta = eff_dem + h

        # X-direction faces (between col c and c+1)
        hf_x = np.maximum(
            0.0, np.maximum(eta[:, :-1], eta[:, 1:]) - np.maximum(eff_dem[:, :-1], eff_dem[:, 1:])
        )
        slope_x = (eta[:, 1:] - eta[:, :-1]) / dx
        active_x = hf_x > 1e-4
        denom_x = 1.0 + _GRAVITY * dt * nx2 * np.abs(qx) / np.maximum(hf_x, 1e-3) ** (7.0 / 3.0)
        qx_new = np.where(active_x, (qx - _GRAVITY * hf_x * dt * slope_x) / denom_x, 0.0)
        # Wave-speed velocity limiter for unconditional wet-dry stability
        q_lim_x = 0.24 * hf_x * dx / max(dt, 1e-3)
        qx = np.clip(qx_new, -q_lim_x, q_lim_x)

        # Y-direction faces (between row r and r+1)
        hf_y = np.maximum(
            0.0, np.maximum(eta[:-1, :], eta[1:, :]) - np.maximum(eff_dem[:-1, :], eff_dem[1:, :])
        )
        slope_y = (eta[1:, :] - eta[:-1, :]) / dx
        active_y = hf_y > 1e-4
        denom_y = 1.0 + _GRAVITY * dt * ny2 * np.abs(qy) / np.maximum(hf_y, 1e-3) ** (7.0 / 3.0)
        qy_new = np.where(active_y, (qy - _GRAVITY * hf_y * dt * slope_y) / denom_y, 0.0)
        q_lim_y = 0.24 * hf_y * dx / max(dt, 1e-3)
        qy = np.clip(qy_new, -q_lim_y, q_lim_y)

        # Enforce strict local mass conservation (outflow from any cell <= available depth h)
        out_x_pos = np.maximum(0.0, qx) * (dt / dx)  # leaving left cell -> right
        out_x_neg = np.maximum(0.0, -qx) * (dt / dx)  # leaving right cell -> left
        out_y_pos = np.maximum(0.0, qy) * (dt / dx)  # leaving top cell -> bottom
        out_y_neg = np.maximum(0.0, -qy) * (dt / dx)  # leaving bottom cell -> top

        total_out = np.zeros_like(h)
        total_out[:, :-1] += out_x_pos
        total_out[:, 1:] += out_x_neg
        total_out[:-1, :] += out_y_pos
        total_out[1:, :] += out_y_neg

        cap = np.where(total_out > h * 0.95, (h * 0.95) / np.maximum(total_out, 1e-9), 1.0)
        qx = np.where(qx >= 0.0, qx * cap[:, :-1], qx * cap[:, 1:])
        qy = np.where(qy >= 0.0, qy * cap[:-1, :], qy * cap[1:, :])

        # Conservative flux divergence update
        dh = np.zeros_like(h)
        dh[:, :-1] -= qx * (dt / dx)
        dh[:, 1:] += qx * (dt / dx)
        dh[:-1, :] -= qy * (dt / dx)
        dh[1:, :] += qy * (dt / dx)
        h = np.maximum(0.0, h + dh)

    return h, qx, qy


def _route_water_step_d8(
    depth: np.ndarray,
    eff_dem: np.ndarray,
    *,
    diffusion_rate: float = 0.22,
) -> np.ndarray:
    """Mass-conserving vectorized 2D D8 diffusive overland routing."""
    surface = eff_dem + depth
    s_pad = np.pad(surface, 1, mode="edge")
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

    max_out = np.minimum(depth * 0.45, total_drop * diffusion_rate)
    denom = np.where(total_drop > 1e-6, total_drop, 1.0)
    scale = np.where(total_drop > 1e-6, max_out / denom, 0.0)

    next_depth = depth - max_out
    for (dr, dc, _), drop in zip(offsets, drops, strict=True):
        outflow = drop * scale
        src_r0, src_r1 = max(0, -dr), h - max(0, dr)
        src_c0, src_c1 = max(0, -dc), w - max(0, dc)
        dst_r0, dst_r1 = max(0, dr), h - max(0, -dr)
        dst_c0, dst_c1 = max(0, dc), w - max(0, -dc)
        next_depth[dst_r0:dst_r1, dst_c0:dst_c1] += outflow[src_r0:src_r1, src_c0:src_c1]

    return np.maximum(0.0, next_depth)


def simulate_flood_rollout(
    city: UrbanTwin,
    *,
    forcing: FloodForcing | None = None,
    actions: list[InterventionAction] | None = None,
    observations: list[ObservedDamage] | None = None,
    rain_scale: float = 1.0,
) -> list[WorldFrame]:
    """Roll out 2D urban flood depth evolution from `t=0h` to `t=duration_hours`."""
    f = forcing or FloodForcing()
    acts = actions or []
    obs = observations or []
    h, w = city.shape

    dt_hr = f.dt_minutes / 60.0
    dt_sec = f.dt_minutes * 60.0
    n_steps = max(1, int(round(f.duration_hours / dt_hr)))
    steps_per_hour = max(1, int(round(1.0 / dt_hr)))
    center = f.storm_center if f.storm_center is not None else _default_hazard_center(city)

    # Spatial rainfall footprint (bell-shaped storm core + background regional rain)
    storm_weight = 0.55 + 0.90 * _make_radial_mask((h, w), center, f.storm_radius_cells)
    storm_weight = storm_weight / float(storm_weight.mean())

    # Temporal rainfall hyetograph (Gaussian cloudburst peaking at peak_hour)
    step_hours = np.arange(1, n_steps + 1, dtype=np.float32) * dt_hr
    hyeto = np.exp(-0.5 * ((step_hours - f.peak_hour) / 1.35) ** 2)
    hyeto = hyeto / float(hyeto.sum()) * (f.total_rain_mm * rain_scale)

    eff_dem = city.dem.copy()
    manning_n = _default_manning_n(city)
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
            yy, xx = np.mgrid[0:h, 0:w]
            dist = np.sqrt((yy - node.row) ** 2 + (xx - node.col) ** 2)
            ring = (dist >= 1.5) & (dist <= 3.5)
            eff_dem[ring] += 1.6
        elif act.kind == "mobile_generator" and act.target_id in city.nodes:
            generator_nodes[act.target_id] = act.deploy_hour

    depth = np.zeros((h, w), dtype=np.float32)
    qx = np.zeros((h, w - 1), dtype=np.float32)
    qy = np.zeros((h - 1, w), dtype=np.float32)
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
            r0, r1 = max(0, node.row - 1), min(h, node.row + 2)
            c0, c1 = max(0, node.col - 1), min(w, node.col + 2)
            d_val = float(cur_depth[r0:r1, c0:c1].mean())
            if nid in barrier_nodes and hour >= barrier_nodes[nid]:
                d_val *= 0.25
            node_h[nid] = d_val

        edge_h: dict[str, float] = {}
        for eid, edge in city.edges.items():
            cells = city.edge_sample_cells(edge, samples=9)
            # Bridges span above open water, so ignore pure river/bay cells for bridges
            vals = [
                float(cur_depth[r, c])
                for r, c in cells
                if not (edge.kind == "bridge" and city.river_mask[r, c])
            ]
            peak = max(vals) if vals else 0.0
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
    # Combined storm sewer drainage + Green-Ampt pervious soil infiltration
    infil_mm_hr = f.infiltration_mm_hr * (1.0 - city.imperviousness)
    sink_m_per_step = ((city.drainage_mm_hr + infil_mm_hr) * dt_hr) / 1000.0

    for step_idx in range(n_steps):
        cur_hour = (step_idx + 1) * dt_hr
        rain_m = (float(hyeto[step_idx]) / 1000.0) * storm_weight
        depth = np.maximum(0.0, depth + rain_m - sink_m_per_step)

        for deploy_hr, pmask in pump_masks:
            if cur_hour >= deploy_hr:
                depth = np.maximum(0.0, depth - (0.075 * pmask))

        if f.solver == "lisflood_inertial":
            # Scale routing dt slightly for crisp urban basin ponding on coarse 150-200m grids
            depth, qx, qy = _lisflood_inertial_step(
                depth,
                qx,
                qy,
                eff_dem,
                manning_n,
                dx=max(city.cell_size_m, 30.0),
                dt_total=min(dt_sec * 0.25, 180.0),
            )
        else:
            for _ in range(3):
                depth = _route_water_step_d8(depth, eff_dem, diffusion_rate=0.25)

        # Open river / bay channels discharge water downstream
        depth[city.river_mask] *= 0.65

        if (step_idx + 1) % steps_per_hour == 0 or step_idx == n_steps - 1:
            frames.append(_snapshot(cur_hour, depth))

    return frames


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _hazus_damage_ratio(pga: float, median_g: float, *, beta: float = 0.45) -> float:
    """HAZUS-style 4-state lognormal fragility expected damage index in [0, 1].

    Computes exceedance probabilities for Slight (0.55x), Moderate (0.85x),
    Extensive (1.15x), and Complete (1.55x) median PGA thresholds and returns
    their expected structural damage ratio.
    """
    p = max(pga, 1e-4)
    m = max(median_g, 0.05)
    p_slight = _norm_cdf(math.log(p / (0.55 * m)) / beta)
    p_mod = _norm_cdf(math.log(p / (0.85 * m)) / beta)
    p_ext = _norm_cdf(math.log(p / (1.15 * m)) / beta)
    p_comp = _norm_cdf(math.log(p / (1.55 * m)) / beta)
    # Expected damage ratio across the 4 HAZUS damage states
    return float(
        np.clip(
            0.12 * p_slight + 0.28 * p_mod + 0.35 * p_ext + 0.25 * p_comp,
            0.0,
            1.0,
        )
    )


def simulate_earthquake_rollout(
    city: UrbanTwin,
    *,
    forcing: EarthquakeForcing | None = None,
    actions: list[InterventionAction] | None = None,
    observations: list[ObservedDamage] | None = None,
    rng: np.random.Generator | None = None,
) -> list[WorldFrame]:
    """Roll out seismic PGA damage (HAZUS fragility) + emergency recovery over `0h..6h`."""
    f = forcing or EarthquakeForcing()
    acts = actions or []
    obs = observations or []
    h, w = city.shape
    epi = f.epicenter if f.epicenter is not None else _default_hazard_center(city)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_km = np.sqrt((yy - epi[0]) ** 2 + (xx - epi[1]) ** 2) * (city.cell_size_m / 1000.0)
    r_hypo = np.sqrt(dist_km**2 + f.depth_km**2)

    # Near-field GMPE PGA attenuation (g) + continuous Vs30 alluvium/bedrock site amplification
    base_pga = 0.26 * np.exp(0.68 * (f.magnitude_mw - 5.5)) / ((r_hypo / 10.0) ** 1.35 + 0.35)
    # Distinguish soft alluvial/Bay-fill lowlands vs stiff highland bedrock smoothly
    elev_ref = (
        float(np.percentile(city.dem[~city.river_mask], 40)) if (~city.river_mask).any() else 11.2
    )
    z_arg = np.clip((city.dem - max(11.0, elev_ref)) * 0.45, -30.0, 30.0)
    site_amp = (0.72 + 0.62 / (1.0 + np.exp(z_arg))).astype(np.float32)
    pga_field = base_pga * site_amp

    # Optional spatially smooth ground-motion uncertainty perturbation for Monte Carlo
    if f.sigma_ln_pga > 0.0 and rng is not None:
        event_term = float(rng.normal(0.0, f.sigma_ln_pga * 0.65))
        # Coarse spatial residual upsampled smoothly via bilinear interpolation
        coarse = rng.normal(0.0, f.sigma_ln_pga * 0.75, size=(8, 12)).astype(np.float32)
        from PIL import Image

        spatial_err = np.asarray(
            Image.fromarray(coarse, mode="F").resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        pga_field = pga_field * np.exp(event_term + spatial_err)

    pga_field = np.clip(pga_field, 0.02, 0.98).astype(np.float32)

    init_node_dmg: dict[str, float] = {}
    for nid, node in city.nodes.items():
        local_pga = float(pga_field[node.row, node.col])
        beta = 0.38 if node.kind == "substation" else 0.45
        init_node_dmg[nid] = _hazus_damage_ratio(local_pga, node.pga_threshold_g, beta=beta)

    init_edge_dmg: dict[str, float] = {}
    for eid, edge in city.edges.items():
        cells = city.edge_sample_cells(edge, samples=7)
        peak_pga = max(float(pga_field[r, c]) for r, c in cells)
        beta = 0.35 if edge.kind in ("bridge", "tunnel") else 0.45
        init_edge_dmg[eid] = _hazus_damage_ratio(peak_pga, edge.pga_threshold_g, beta=beta)

    for ob in obs:
        if ob.kind == "damaged_edge" and ob.target_id in init_edge_dmg:
            init_edge_dmg[ob.target_id] = max(init_edge_dmg[ob.target_id], float(ob.severity))
        elif ob.kind == "substation_outage" and ob.target_id in init_node_dmg:
            init_node_dmg[ob.target_id] = 1.0

    gen_hours = {a.target_id: a.deploy_hour for a in acts if a.kind == "mobile_generator"}
    repair_hours = {a.target_id: a.deploy_hour for a in acts if a.kind == "repair_crew"}
    temp_bridge_hours = {a.target_id: a.deploy_hour for a in acts if a.kind == "temp_bridge"}

    n_hours = max(1, int(round(f.duration_hours)))
    frames: list[WorldFrame] = []
    for hr in range(0, n_hours + 1):
        hour = float(hr)
        node_dmg = dict(init_node_dmg)
        edge_dmg = dict(init_edge_dmg)
        repaired: set[str] = set()

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
