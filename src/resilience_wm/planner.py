"""Model-based Counterfactual World-Model Planner with Leave-One-Out Attribution.

Uses the Spatial World Model as a simulator-in-the-loop:
  1. Rolls out the unmitigated Baseline (`a = empty`) to identify cascading bottlenecks.
  2. Enumerates spatially grounded candidate interventions within `ResourceBudget`.
  3. Runs multi-step rollout search (Greedy + Beam Lookahead) to maximize mean urban
     resilience `mean(R(t))` (equivalently minimizing the Resilience Triangle loss).
  4. Runs leave-one-out counterfactual ablations on the chosen plan to report each
     action's exact marginal resilience gain and residents rescued from isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .city import UrbanTwin
from .physics import (
    ActionKind,
    EarthquakeForcing,
    FloodForcing,
    HazardType,
    InterventionAction,
    ObservedDamage,
    simulate_earthquake_rollout,
    simulate_flood_rollout,
)
from .resilience import (
    MonteCarloResult,
    RolloutEvaluation,
    evaluate_monte_carlo,
    evaluate_rollout,
)


@dataclass(frozen=True)
class ResourceBudget:
    """Available emergency resources for spatial allocation."""

    counts: dict[ActionKind, int] = field(default_factory=dict)

    @classmethod
    def default_for(cls, hazard: HazardType) -> ResourceBudget:
        if hazard == "flood":
            return cls(counts={"mobile_pump": 2, "mobile_generator": 1, "flood_barrier": 1})
        return cls(counts={"repair_crew": 2, "temp_bridge": 1, "mobile_generator": 1})


@dataclass(frozen=True)
class ActionAttribution:
    """Leave-one-out counterfactual causal impact of one chosen intervention."""

    action: InterventionAction
    marginal_resilience_gain: float  # R_plan - R_(plan \ {action})
    marginal_isolated_rescued: int  # peak isolated reduction attributed to this action
    rationale: str


@dataclass
class PlanComparison:
    """Side-by-side Baseline vs. World-Model Optimal Counterfactual Plan."""

    hazard: HazardType
    baseline: RolloutEvaluation
    optimal: RolloutEvaluation
    selected_actions: list[InterventionAction]
    attributions: list[ActionAttribution]
    rollouts_evaluated: int
    baseline_mc: MonteCarloResult | None = None
    optimal_mc: MonteCarloResult | None = None

    @property
    def resilience_gain(self) -> float:
        return self.optimal.mean_resilience - self.baseline.mean_resilience

    @property
    def isolated_population_reduced(self) -> int:
        return self.baseline.peak_isolated_population - self.optimal.peak_isolated_population

    def to_dict(self) -> dict[str, object]:
        return {
            "hazard": self.hazard,
            "rollouts_evaluated": self.rollouts_evaluated,
            "baseline_mean_resilience": round(self.baseline.mean_resilience, 4),
            "optimal_mean_resilience": round(self.optimal.mean_resilience, 4),
            "resilience_gain": round(self.resilience_gain, 4),
            "baseline_peak_isolated": self.baseline.peak_isolated_population,
            "optimal_peak_isolated": self.optimal.peak_isolated_population,
            "isolated_population_reduced": self.isolated_population_reduced,
            "selected_actions": [
                {
                    "kind": a.kind,
                    "target_id": a.target_id,
                    "display_name": a.display_name,
                    "marginal_resilience_gain": round(attr.marginal_resilience_gain, 4),
                    "marginal_isolated_rescued": attr.marginal_isolated_rescued,
                    "rationale": attr.rationale,
                }
                for a, attr in zip(self.selected_actions, self.attributions, strict=True)
            ],
            "baseline": self.baseline.to_summary_dict(),
            "optimal": self.optimal.to_summary_dict(),
            **(
                {
                    "baseline_monte_carlo": self.baseline_mc.to_dict(),
                    "optimal_monte_carlo": self.optimal_mc.to_dict(),
                }
                if self.baseline_mc is not None and self.optimal_mc is not None
                else {}
            ),
        }


class CounterfactualPlanner:
    """Simulator-in-the-loop spatial planner over the Urban Resilience World Model."""

    def __init__(
        self,
        city: UrbanTwin,
        *,
        hazard: HazardType = "flood",
        flood_forcing: FloodForcing | None = None,
        earthquake_forcing: EarthquakeForcing | None = None,
        observations: list[ObservedDamage] | None = None,
    ) -> None:
        self.city = city
        self.hazard = hazard
        self.flood_forcing = flood_forcing or FloodForcing()
        self.earthquake_forcing = earthquake_forcing or EarthquakeForcing()
        self.observations = observations or []
        self._rollout_count = 0

    def rollout(self, actions: list[InterventionAction] | None = None) -> RolloutEvaluation:
        """Execute one full forward rollout of the Spatial World Model."""
        self._rollout_count += 1
        if self.hazard == "flood":
            frames = simulate_flood_rollout(
                self.city,
                forcing=self.flood_forcing,
                actions=actions,
                observations=self.observations,
            )
        else:
            frames = simulate_earthquake_rollout(
                self.city,
                forcing=self.earthquake_forcing,
                actions=actions,
                observations=self.observations,
            )
        return evaluate_rollout(self.city, frames, hazard=self.hazard)

    def _candidate_targets(self, kind: ActionKind, baseline: RolloutEvaluation) -> list[str]:
        """Rank high-value spatial candidate targets from baseline stress diagnostics."""
        worst = baseline.frames[baseline.worst_frame_index]
        if kind == "mobile_generator":
            subs = [n.node_id for n in self.city.nodes.values() if n.kind == "substation"]
            return sorted(subs, key=lambda nid: worst.node_hazard.get(nid, 0.0), reverse=True)

        if kind in ("mobile_pump", "flood_barrier"):
            # Prioritize flooded underpasses, substations, hubs, hospitals, and residential basins
            scored = [
                (worst.node_hazard.get(nid, 0.0), nid)
                for nid, n in self.city.nodes.items()
                if n.kind in ("underpass", "substation", "hub", "hospital", "residential")
            ]
            scored.sort(reverse=True)
            return [nid for _, nid in scored[:10]]

        if kind == "temp_bridge":
            bridges = [
                e.edge_id for e in self.city.edges.values() if e.kind in ("bridge", "tunnel")
            ]
            return sorted(bridges, key=lambda eid: worst.edge_hazard.get(eid, 0.0), reverse=True)

        if kind == "repair_crew":
            worst_m = baseline.metrics[baseline.worst_frame_index]
            iso = set(worst_m.isolated_districts)
            scored_e = [
                (
                    1 if (self.city.edges[eid].u in iso or self.city.edges[eid].v in iso) else 0,
                    worst.edge_hazard.get(eid, 0.0),
                    eid,
                )
                for eid in self.city.edges
                if worst.edge_hazard.get(eid, 0.0) >= 0.25
            ]
            scored_e.sort(reverse=True)
            return [eid for _, _, eid in scored_e[:22]]

        return []

    def _action_label(self, kind: ActionKind, target_id: str) -> str:
        if target_id in self.city.nodes:
            tname = self.city.nodes[target_id].name
        elif target_id in self.city.edges:
            e = self.city.edges[target_id]
            tname = f"{e.u}-{e.v} ({e.kind})"
        else:
            tname = target_id
        pretty = {
            "mobile_pump": "Mobile Pump",
            "flood_barrier": "Flood Barrier",
            "mobile_generator": "Emergency Generator",
            "repair_crew": "Road Repair Crew",
            "temp_bridge": "Temporary Bridge",
        }[kind]
        return f"{pretty} @ {target_id} ({tname})"

    def plan(
        self,
        budget: ResourceBudget | None = None,
        *,
        beam_width: int = 3,
        mc_samples: int = 0,
        mc_seed: int = 42,
    ) -> PlanComparison:
        """Search for the highest-resilience spatial intervention portfolio within `budget`."""
        self._rollout_count = 0
        res_budget = budget or ResourceBudget.default_for(self.hazard)
        baseline = self.rollout([])

        # Expand resource slots in priority order (power -> bridges/barriers -> pumps/crews)
        order: list[ActionKind] = [
            "mobile_generator",
            "temp_bridge",
            "flood_barrier",
            "mobile_pump",
            "repair_crew",
        ]
        slots: list[ActionKind] = []
        for k in order:
            slots.extend([k] * max(0, int(res_budget.counts.get(k, 0))))

        # Beam search over world-model rollouts: each state is (score, actions, eval)
        beam: list[tuple[float, list[InterventionAction], RolloutEvaluation]] = [
            (baseline.mean_resilience, [], baseline)
        ]

        for kind in slots:
            candidates = self._candidate_targets(kind, baseline)
            next_beam: list[tuple[float, list[InterventionAction], RolloutEvaluation]] = []
            seen_sets: set[frozenset[tuple[str, str]]] = set()

            for _, cur_actions, cur_eval in beam:
                used_targets = {a.target_id for a in cur_actions}
                # Always keep the "no-op" branch in case adding an action doesn't help
                sig_noop = frozenset((a.kind, a.target_id) for a in cur_actions)
                if sig_noop not in seen_sets:
                    seen_sets.add(sig_noop)
                    next_beam.append((cur_eval.mean_resilience, cur_actions, cur_eval))

                for tid in candidates:
                    if tid in used_targets:
                        continue
                    act = InterventionAction(
                        kind=kind,
                        target_id=tid,
                        deploy_hour=0.0,
                        label=self._action_label(kind, tid),
                    )
                    trial = [*cur_actions, act]
                    sig = frozenset((a.kind, a.target_id) for a in trial)
                    if sig in seen_sets:
                        continue
                    seen_sets.add(sig)
                    ev = self.rollout(trial)
                    # Only keep actions that genuinely improve resilience or rescue residents
                    score = ev.mean_resilience - 1e-7 * ev.peak_isolated_population
                    cur_score = cur_eval.mean_resilience - 1e-7 * cur_eval.peak_isolated_population
                    if score <= cur_score + 1e-5:
                        continue
                    next_beam.append((score, trial, ev))

            next_beam.sort(key=lambda item: item[0], reverse=True)
            beam = next_beam[: max(1, beam_width)]

        _, best_actions, best_eval = beam[0]

        # Counterfactual Leave-One-Out Attribution for every selected action
        attributions: list[ActionAttribution] = []
        for idx, act in enumerate(best_actions):
            ablated = [a for j, a in enumerate(best_actions) if j != idx]
            ab_eval = self.rollout(ablated) if ablated else baseline
            delta_r = best_eval.mean_resilience - ab_eval.mean_resilience
            rescued = ab_eval.peak_isolated_population - best_eval.peak_isolated_population
            rationale = (
                f"Removing {act.display_name} drops mean resilience by "
                f"{delta_r:+.3f} and isolates {max(0, rescued):,} more residents."
            )
            attributions.append(
                ActionAttribution(
                    action=act,
                    marginal_resilience_gain=delta_r,
                    marginal_isolated_rescued=rescued,
                    rationale=rationale,
                )
            )

        base_mc: MonteCarloResult | None = None
        opt_mc: MonteCarloResult | None = None
        if mc_samples > 0:
            base_mc = evaluate_monte_carlo(
                self.city,
                hazard=self.hazard,
                actions=[],
                observations=self.observations,
                flood_forcing=self.flood_forcing,
                earthquake_forcing=self.earthquake_forcing,
                n_samples=mc_samples,
                seed=mc_seed,
            )
            opt_mc = evaluate_monte_carlo(
                self.city,
                hazard=self.hazard,
                actions=best_actions,
                observations=self.observations,
                flood_forcing=self.flood_forcing,
                earthquake_forcing=self.earthquake_forcing,
                n_samples=mc_samples,
                seed=mc_seed,
            )

        return PlanComparison(
            hazard=self.hazard,
            baseline=baseline,
            optimal=best_eval,
            selected_actions=best_actions,
            attributions=attributions,
            rollouts_evaluated=self._rollout_count,
            baseline_mc=base_mc,
            optimal_mc=opt_mc,
        )
