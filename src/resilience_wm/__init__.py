"""resilience-wm: Neuro-Symbolic Spatial World Model for Urban Resilience."""

from .agent import (
    SpatialReasoningAgent,
    VisualDetection,
    ground_visual_detections_to_city,
    synthesize_causal_briefing,
)
from .benchmark import (
    BenchmarkScenario,
    StrategySummary,
    build_default_benchmark_scenarios,
    render_benchmark_chart,
    run_benchmark,
)
from .city import UrbanEdge, UrbanNode, UrbanTwin, build_riverfront_city, load_city_from_geojson
from .env import BoxSpace, DiscreteSpace, ResilienceEnv
from .mcp_server import TOOL_DEFINITIONS, call_tool, handle_jsonrpc_request, serve_stdio
from .osm import build_real_sf_city, fetch_city_from_osm, fetch_terrarium_dem
from .physics import (
    ActionKind,
    EarthquakeForcing,
    FloodForcing,
    FloodSolver,
    HazardType,
    InterventionAction,
    ObservedDamage,
    WorldFrame,
    simulate_earthquake_rollout,
    simulate_flood_rollout,
)
from .planner import ActionAttribution, CounterfactualPlanner, PlanComparison, ResourceBudget
from .resilience import (
    FrameMetrics,
    MonteCarloResult,
    RolloutEvaluation,
    evaluate_frame,
    evaluate_monte_carlo,
    evaluate_rollout,
)
from .validation import (
    HindcastResult,
    NumericalVerification,
    ValidationReport,
    render_validation_figure,
    run_validation_suite,
    validate_sf_1989_loma_prieta_hindcast,
    validate_sf_dec2022_flood_hindcast,
    verify_numerical_invariants,
)

__version__ = "0.2.0"

__all__ = [
    "ActionAttribution",
    "ActionKind",
    "BenchmarkScenario",
    "BoxSpace",
    "CounterfactualPlanner",
    "DiscreteSpace",
    "EarthquakeForcing",
    "FloodForcing",
    "FloodSolver",
    "FrameMetrics",
    "HazardType",
    "HindcastResult",
    "InterventionAction",
    "MonteCarloResult",
    "NumericalVerification",
    "ObservedDamage",
    "PlanComparison",
    "ResilienceEnv",
    "ResourceBudget",
    "RolloutEvaluation",
    "SpatialReasoningAgent",
    "StrategySummary",
    "TOOL_DEFINITIONS",
    "UrbanEdge",
    "UrbanNode",
    "UrbanTwin",
    "ValidationReport",
    "VisualDetection",
    "WorldFrame",
    "__version__",
    "build_default_benchmark_scenarios",
    "build_real_sf_city",
    "build_riverfront_city",
    "call_tool",
    "evaluate_frame",
    "evaluate_monte_carlo",
    "evaluate_rollout",
    "fetch_city_from_osm",
    "fetch_terrarium_dem",
    "ground_visual_detections_to_city",
    "handle_jsonrpc_request",
    "load_city_from_geojson",
    "render_benchmark_chart",
    "render_validation_figure",
    "run_benchmark",
    "run_validation_suite",
    "serve_stdio",
    "simulate_earthquake_rollout",
    "simulate_flood_rollout",
    "synthesize_causal_briefing",
    "validate_sf_1989_loma_prieta_hindcast",
    "validate_sf_dec2022_flood_hindcast",
    "verify_numerical_invariants",
]
