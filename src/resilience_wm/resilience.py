"""Cross-layer cascading failure propagation and spatial graph resilience metrics.

Evaluates every `WorldFrame` across four coupled urban layers:
  Layer 1 (Power): Substation operational status under flood/seismic stress
  Layer 2 (Medical & Shelters): Hospital capacity degrades to 45% if primary power fails
  Layer 3 (Road Network): Flooded/damaged bridges, tunnels, and arterials sever or slow down
  Layer 4 (Population Spatial Reasoning): Multi-source Dijkstra shortest ambulance time,
          15-min medical accessibility, largest connected component, and isolated communities.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .city import UrbanTwin
from .physics import HazardType, WorldFrame


@dataclass
class FrameMetrics:
    """Spatial resilience metrics evaluated at a single rollout timestamp `hour`."""

    hour: float
    resilience_index: float  # composite R(t) in [0, 1]
    medical_access_ratio: float  # population-weighted 15-min hospital reachability
    road_connectivity_ratio: float  # population fraction in largest connected component
    power_coverage_ratio: float  # population fraction with primary grid power
    isolated_population: int  # residents with no reachable operational hospital within 25 min
    isolated_districts: tuple[str, ...]
    blocked_edges: tuple[str, ...]
    degraded_edges: tuple[str, ...]
    offline_substations: tuple[str, ...]
    hospital_capacity: dict[str, float] = field(default_factory=dict)
    district_travel_min: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "hour": self.hour,
            "resilience_index": round(self.resilience_index, 4),
            "medical_access_ratio": round(self.medical_access_ratio, 4),
            "road_connectivity_ratio": round(self.road_connectivity_ratio, 4),
            "power_coverage_ratio": round(self.power_coverage_ratio, 4),
            "isolated_population": self.isolated_population,
            "isolated_districts": list(self.isolated_districts),
            "blocked_edges": list(self.blocked_edges),
            "degraded_edges": list(self.degraded_edges),
            "offline_substations": list(self.offline_substations),
            "hospital_capacity": {k: round(v, 3) for k, v in self.hospital_capacity.items()},
            "district_travel_min": {
                k: (round(v, 1) if v < 1e6 else None) for k, v in self.district_travel_min.items()
            },
        }


@dataclass
class RolloutEvaluation:
    """Complete time-series resilience evaluation of a World Model rollout."""

    hazard: HazardType
    frames: list[WorldFrame]
    metrics: list[FrameMetrics]

    @property
    def mean_resilience(self) -> float:
        if not self.metrics:
            return 1.0
        return sum(m.resilience_index for m in self.metrics) / len(self.metrics)

    @property
    def min_resilience(self) -> float:
        return min((m.resilience_index for m in self.metrics), default=1.0)

    @property
    def resilience_triangle_loss(self) -> float:
        """Normalized Bruneau Resilience Triangle area `mean(1 - R(t))` in [0, 1]."""
        return 1.0 - self.mean_resilience

    @property
    def peak_isolated_population(self) -> int:
        # Evaluate post-mobilization peak (t >= 1h) when multi-step rollout exists so
        # instantaneous t=0h seismic shocks don't mask emergency rescue effectiveness.
        post = self.metrics[1:] if len(self.metrics) > 1 else self.metrics
        return max((m.isolated_population for m in post), default=0)

    @property
    def end_isolated_population(self) -> int:
        return self.metrics[-1].isolated_population if self.metrics else 0

    @property
    def worst_frame_index(self) -> int:
        if not self.metrics:
            return 0
        return min(range(len(self.metrics)), key=lambda i: self.metrics[i].resilience_index)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "hazard": self.hazard,
            "mean_resilience": round(self.mean_resilience, 4),
            "min_resilience": round(self.min_resilience, 4),
            "resilience_triangle_loss": round(self.resilience_triangle_loss, 4),
            "peak_isolated_population": self.peak_isolated_population,
            "timeline": [m.to_dict() for m in self.metrics],
        }


def _dijkstra_from_sources(
    adjacency: dict[str, list[tuple[str, float]]],
    sources: dict[str, float],
) -> dict[str, float]:
    """Multi-source Dijkstra returning shortest travel minutes from any source."""
    dist: dict[str, float] = {}
    heap: list[tuple[float, str]] = []
    for src, penalty in sources.items():
        dist[src] = penalty
        heapq.heappush(heap, (penalty, src))

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adjacency.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _largest_component_nodes(
    all_nodes: list[str],
    adjacency: dict[str, list[tuple[str, float]]],
    populations: dict[str, int],
) -> set[str]:
    """Find the connected component containing the largest residential population."""
    visited: set[str] = set()
    best_comp: set[str] = set()
    best_pop = -1

    for start in all_nodes:
        if start in visited:
            continue
        comp: set[str] = set()
        stack = [start]
        visited.add(start)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt, _ in adjacency.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        pop = sum(populations.get(n, 0) for n in comp)
        if pop > best_pop:
            best_pop = pop
            best_comp = comp
    return best_comp


def evaluate_frame(
    city: UrbanTwin,
    frame: WorldFrame,
    *,
    hazard: HazardType,
) -> FrameMetrics:
    """Compute cross-layer cascading failures and spatial resilience metrics for one frame."""
    # Layer 1: Power substations
    offline_subs = tuple(sorted(nid for nid, ok in frame.powered_substations.items() if not ok))

    # Layer 2: Hospital operational capacity (cascading from Layer 1 power + direct hazard!)
    hospital_cap: dict[str, float] = {}
    for nid, node in city.nodes.items():
        if node.kind != "hospital":
            continue
        h_val = frame.node_hazard.get(nid, 0.0)
        direct_fail = h_val >= node.flood_threshold_m if hazard == "flood" else h_val >= 0.60
        if direct_fail:
            hospital_cap[nid] = 0.0
            continue
        cap = 1.0
        if node.power_source and not frame.powered_substations.get(node.power_source, True):
            cap *= 0.45  # backup diesel only sustains 45% triage throughput
        if hazard == "flood" and h_val >= node.flood_threshold_m * 0.55:
            cap *= 0.70
        elif hazard == "earthquake" and h_val >= 0.30:
            cap *= max(0.35, 1.0 - 0.7 * h_val)
        hospital_cap[nid] = round(cap, 3)

    # Layer 3: Road & bridge passability
    adjacency: dict[str, list[tuple[str, float]]] = {nid: [] for nid in city.nodes}
    blocked: list[str] = []
    degraded: list[str] = []

    for eid, edge in city.edges.items():
        e_val = frame.edge_hazard.get(eid, 0.0)
        if hazard == "flood":
            if e_val >= edge.flood_block_m:
                blocked.append(eid)
                continue
            if e_val >= edge.flood_block_m * 0.45:
                degraded.append(eid)
                slowdown = 2.4
            else:
                slowdown = 1.0
        else:  # earthquake damage index in [0, 1]
            if e_val >= 0.52:
                blocked.append(eid)
                continue
            if e_val >= 0.25:
                degraded.append(eid)
                slowdown = 1.0 + 2.5 * e_val
            else:
                slowdown = 1.0

        travel = edge.base_travel_min * slowdown
        adjacency[edge.u].append((edge.v, travel))
        adjacency[edge.v].append((edge.u, travel))

    # Layer 4: Spatial reasoning over residential districts
    # Hospital triage wait penalty: degraded hospitals add intake delay (up to +10 min)
    hospital_sources: dict[str, float] = {
        hid: (1.0 - cap) * 14.0 for hid, cap in hospital_cap.items() if cap > 0.05
    }
    travel_to_hosp = _dijkstra_from_sources(adjacency, hospital_sources)

    populations = {
        nid: node.population for nid, node in city.nodes.items() if node.kind == "residential"
    }
    total_pop = max(1, sum(populations.values()))

    lcc_nodes = _largest_component_nodes(list(city.nodes.keys()), adjacency, populations)
    lcc_pop = sum(pop for nid, pop in populations.items() if nid in lcc_nodes)
    road_conn_ratio = lcc_pop / total_pop

    med_score_pop = 0.0
    pwr_pop = 0
    isolated_pop = 0
    isolated_districts: list[str] = []
    district_travel: dict[str, float] = {}

    for nid, pop in populations.items():
        node = city.nodes[nid]
        t_min = travel_to_hosp.get(nid, float("inf"))
        district_travel[nid] = t_min

        # Smooth accessibility score: <= 10 min = 1.0, 10..25 min linear decay, > 25 min = isolated
        if t_min <= 10.0:
            acc = 1.0
        elif t_min <= 25.0:
            acc = max(0.0, 1.0 - (t_min - 10.0) / 15.0)
        else:
            acc = 0.0
            isolated_pop += pop
            isolated_districts.append(nid)
        med_score_pop += pop * acc

        has_power = (
            frame.powered_substations.get(node.power_source, True) if node.power_source else True
        )
        if has_power:
            pwr_pop += pop

    med_ratio = med_score_pop / total_pop
    pwr_ratio = pwr_pop / total_pop
    composite = 0.45 * med_ratio + 0.30 * road_conn_ratio + 0.25 * pwr_ratio

    return FrameMetrics(
        hour=frame.hour,
        resilience_index=composite,
        medical_access_ratio=med_ratio,
        road_connectivity_ratio=road_conn_ratio,
        power_coverage_ratio=pwr_ratio,
        isolated_population=isolated_pop,
        isolated_districts=tuple(sorted(isolated_districts)),
        blocked_edges=tuple(sorted(blocked)),
        degraded_edges=tuple(sorted(degraded)),
        offline_substations=offline_subs,
        hospital_capacity=hospital_cap,
        district_travel_min=district_travel,
    )


def evaluate_rollout(
    city: UrbanTwin,
    frames: list[WorldFrame],
    *,
    hazard: HazardType,
) -> RolloutEvaluation:
    """Evaluate spatial resilience metrics across every frame of a rollout."""
    metrics = [evaluate_frame(city, fr, hazard=hazard) for fr in frames]
    return RolloutEvaluation(hazard=hazard, frames=frames, metrics=metrics)
