"""resilience-wm: Neuro-Symbolic Spatial World Model for Urban Resilience."""

from .agent import (
    SpatialReasoningAgent,
    VisualDetection,
    ground_visual_detections_to_city,
    synthesize_causal_briefing,
)
from .city import UrbanEdge, UrbanNode, UrbanTwin, build_riverfront_city, load_city_from_geojson
from .physics import (
    ActionKind,
    EarthquakeForcing,
    FloodForcing,
    HazardType,
    InterventionAction,
    ObservedDamage,
    WorldFrame,
    simulate_earthquake_rollout,
    simulate_flood_rollout,
)
from .planner import ActionAttribution, CounterfactualPlanner, PlanComparison, ResourceBudget
from .resilience import FrameMetrics, RolloutEvaluation, evaluate_frame, evaluate_rollout

__version__ = "0.1.0"

__all__ = [
    "ActionAttribution",
    "ActionKind",
    "CounterfactualPlanner",
    "EarthquakeForcing",
    "FloodForcing",
    "FrameMetrics",
    "HazardType",
    "InterventionAction",
    "ObservedDamage",
    "PlanComparison",
    "ResourceBudget",
    "RolloutEvaluation",
    "SpatialReasoningAgent",
    "UrbanEdge",
    "UrbanNode",
    "UrbanTwin",
    "VisualDetection",
    "WorldFrame",
    "__version__",
    "build_riverfront_city",
    "evaluate_frame",
    "evaluate_rollout",
    "ground_visual_detections_to_city",
    "load_city_from_geojson",
    "simulate_earthquake_rollout",
    "simulate_flood_rollout",
    "synthesize_causal_briefing",
]
