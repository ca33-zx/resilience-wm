"""Gymnasium-compatible Multi-Step Spatial RL Environment (`ResilienceEnv`).

Enables RL frameworks (Stable-Baselines3, CleanRL, and multi-trajectory LLM RL
frameworks like `verl` GRPO/RLOO) to train sequential urban emergency dispatch
policies directly inside the `resilience-wm` World Model without requiring
`gymnasium` as a mandatory install dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
from .planner import ResourceBudget
from .resilience import RolloutEvaluation, evaluate_rollout


@dataclass(frozen=True)
class DiscreteSpace:
    """Lightweight Gymnasium-API-compatible discrete space descriptor."""

    n: int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        gen = rng or np.random.default_rng()
        return int(gen.integers(0, self.n))

    def contains(self, x: int) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n


@dataclass(frozen=True)
class BoxSpace:
    """Lightweight Gymnasium-API-compatible continuous observation box descriptor."""

    shape: tuple[int, ...]
    low: float = 0.0
    high: float = 1.0
    dtype: str = "float32"


class ResilienceEnv:
    """Sequential resource-allocation RL environment over the Urban Resilience World Model.

    Episode dynamics:
      - `reset(seed=...)` initializes a disaster scenario (randomizing storm center /
        rain intensity or earthquake epicenter / magnitude when `randomize=True`).
      - At each step `t`, the agent selects one discrete action index mapping to an
        `(ActionKind, target_id)` deployment from the remaining resource budget.
      - The world model executes a full forward rollout and returns:
          * `obs`: compact float32 state vector encoding per-node hazard, substation status,
            current resilience R(t), isolated population fraction, and remaining budget.
          * `reward`: marginal urban resilience gain `ΔR_mean + 0.15 * Δrescued_pop_frac`
            minus a small penalty `-0.05` if the agent selects a duplicate/invalid target
            (enabling hallucination-aware RL training!).
          * `terminated`: `True` once all budget slots are deployed.
          * `info`: rich diagnostics including text prompt (`state_text`) for LLM/VLM agents.
    """

    def __init__(
        self,
        city: UrbanTwin | str = "sf",
        *,
        hazard: HazardType = "flood",
        budget: ResourceBudget | None = None,
        randomize: bool = False,
    ) -> None:
        if isinstance(city, str):
            self.city = build_real_sf_city() if city == "sf" else build_riverfront_city()
        else:
            self.city = city
        self.hazard: HazardType = hazard
        self.base_budget = budget or ResourceBudget.default_for(hazard)
        self.randomize = randomize

        # Build deterministic slot order and discrete action catalog
        self._node_ids = sorted(self.city.nodes.keys())
        self._edge_ids = sorted(self.city.edges.keys())
        self._action_catalog: list[tuple[ActionKind, str]] = []
        for kind, count in self.base_budget.counts.items():
            if count <= 0:
                continue
            if kind == "mobile_generator":
                targets = [n for n in self._node_ids if self.city.nodes[n].kind == "substation"]
            elif kind in ("mobile_pump", "flood_barrier"):
                targets = [
                    n
                    for n in self._node_ids
                    if self.city.nodes[n].kind
                    in ("underpass", "substation", "hub", "hospital", "residential")
                ]
            elif kind == "temp_bridge":
                targets = [
                    e for e in self._edge_ids if self.city.edges[e].kind in ("bridge", "tunnel")
                ]
            else:
                targets = list(self._edge_ids)
            for tid in targets:
                self._action_catalog.append((kind, tid))

        self.action_space = DiscreteSpace(n=len(self._action_catalog))
        # Obs: per-node hazard (N) + [mean_R, min_R, iso_frac, progress_frac] + 5 budget counts
        self._obs_dim = len(self._node_ids) + 9
        self.observation_space = BoxSpace(shape=(self._obs_dim,), low=0.0, high=2.0)

        self._rng = np.random.default_rng(0)
        self._flood_forcing = FloodForcing()
        self._eq_forcing = EarthquakeForcing()
        self._remaining: dict[ActionKind, int] = {}
        self._selected: list[InterventionAction] = []
        self._baseline_eval: RolloutEvaluation | None = None
        self._cur_eval: RolloutEvaluation | None = None
        self._step_count = 0
        self._max_steps = sum(max(0, v) for v in self.base_budget.counts.values())

    @property
    def action_catalog(self) -> list[tuple[ActionKind, str]]:
        return list(self._action_catalog)

    def valid_action_mask(self) -> np.ndarray:
        """Return a boolean mask over `action_space` for valid remaining-budget actions."""
        used = {a.target_id for a in self._selected}
        mask = np.zeros(self.action_space.n, dtype=bool)
        for idx, (kind, tid) in enumerate(self._action_catalog):
            if self._remaining.get(kind, 0) > 0 and tid not in used:
                mask[idx] = True
        return mask

    def _run_world_model(self, actions: list[InterventionAction]) -> RolloutEvaluation:
        if self.hazard == "flood":
            frames = simulate_flood_rollout(self.city, forcing=self._flood_forcing, actions=actions)
        else:
            frames = simulate_earthquake_rollout(
                self.city, forcing=self._eq_forcing, actions=actions
            )
        return evaluate_rollout(self.city, frames, hazard=self.hazard)

    def _build_obs(self) -> np.ndarray:
        assert self._cur_eval is not None
        worst = self._cur_eval.frames[self._cur_eval.worst_frame_index]
        node_vals = [
            float(np.clip(worst.node_hazard.get(nid, 0.0), 0.0, 2.0)) for nid in self._node_ids
        ]
        total_pop = max(1, self.city.total_population())
        iso_frac = self._cur_eval.peak_isolated_population / total_pop
        prog = self._step_count / max(1, self._max_steps)
        budget_vec = [
            float(self._remaining.get(k, 0)) / 4.0
            for k in (
                "mobile_generator",
                "flood_barrier",
                "mobile_pump",
                "temp_bridge",
                "repair_crew",
            )
        ]
        vec = node_vals + [
            float(self._cur_eval.mean_resilience),
            float(self._cur_eval.min_resilience),
            float(iso_frac),
            float(prog),
            *budget_vec,
        ]
        return np.asarray(vec, dtype=np.float32)

    def render_state_text(self) -> str:
        """Render a concise spatial-state summary prompt for LLM/VLM policy agents."""
        assert self._cur_eval is not None
        wm = self._cur_eval.metrics[self._cur_eval.worst_frame_index]
        return (
            f"City: {self.city.name} | Hazard: {self.hazard} | "
            f"Mean R(t)={self._cur_eval.mean_resilience:.3f} | "
            f"Isolated={self._cur_eval.peak_isolated_population:,} "
            f"({', '.join(wm.isolated_districts) or 'None'}) | "
            f"Offline Subs={', '.join(wm.offline_substations) or 'None'} | "
            f"Blocked Edges={len(wm.blocked_edges)} | "
            f"Remaining Budget={dict(self._remaining)}"
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        h, w = self.city.shape
        if self.randomize:
            self._flood_forcing = FloodForcing(
                total_rain_mm=float(self._rng.uniform(280.0, 410.0)),
                storm_center=(
                    float(self._rng.uniform(0.25 * h, 0.75 * h)),
                    float(self._rng.uniform(0.25 * w, 0.75 * w)),
                ),
            )
            self._eq_forcing = EarthquakeForcing(
                magnitude_mw=float(self._rng.uniform(6.4, 7.2)),
                epicenter=(
                    float(self._rng.uniform(0.25 * h, 0.75 * h)),
                    float(self._rng.uniform(0.25 * w, 0.75 * w)),
                ),
            )
        else:
            self._flood_forcing = FloodForcing()
            self._eq_forcing = EarthquakeForcing()

        self._remaining = {k: int(v) for k, v in self.base_budget.counts.items() if v > 0}
        self._selected = []
        self._step_count = 0
        self._baseline_eval = self._run_world_model([])
        self._cur_eval = self._baseline_eval

        info = {
            "mean_resilience": self._cur_eval.mean_resilience,
            "peak_isolated": self._cur_eval.peak_isolated_population,
            "valid_action_mask": self.valid_action_mask(),
            "state_text": self.render_state_text(),
        }
        return self._build_obs(), info

    def step(
        self, action: int | tuple[ActionKind, str]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one intervention action and step the spatial world model."""
        if self._cur_eval is None or self._baseline_eval is None:
            raise RuntimeError("Call env.reset() before env.step().")

        prev_eval = self._cur_eval
        self._step_count += 1

        if isinstance(action, tuple):
            kind, tid = action
        else:
            idx = int(action)
            if not self.action_space.contains(idx):
                raise ValueError(f"Action index {idx} out of bounds [0, {self.action_space.n}).")
            kind, tid = self._action_catalog[idx]

        used = {a.target_id for a in self._selected}
        invalid_reason: str | None = None
        if self._remaining.get(kind, 0) <= 0:
            invalid_reason = f"budget_exhausted:{kind}"
        elif tid in used:
            invalid_reason = f"duplicate_target:{tid}"
        elif tid not in self.city.nodes and tid not in self.city.edges:
            invalid_reason = f"unknown_target:{tid}"

        if invalid_reason is not None:
            reward = -0.05
            new_eval = prev_eval
        else:
            self._remaining[kind] -= 1
            self._selected.append(InterventionAction(kind=kind, target_id=tid))
            new_eval = self._run_world_model(self._selected)
            self._cur_eval = new_eval
            delta_r = new_eval.mean_resilience - prev_eval.mean_resilience
            pop_frac = (
                prev_eval.peak_isolated_population - new_eval.peak_isolated_population
            ) / max(1, self.city.total_population())
            reward = float(delta_r + 0.15 * pop_frac)

        terminated = sum(self._remaining.values()) <= 0 or self._step_count >= self._max_steps
        truncated = False
        info = {
            "action": (kind, tid),
            "invalid_reason": invalid_reason,
            "mean_resilience": new_eval.mean_resilience,
            "total_resilience_gain": new_eval.mean_resilience - self._baseline_eval.mean_resilience,
            "peak_isolated": new_eval.peak_isolated_population,
            "valid_action_mask": self.valid_action_mask(),
            "state_text": self.render_state_text(),
        }
        return self._build_obs(), reward, terminated, truncated, info
