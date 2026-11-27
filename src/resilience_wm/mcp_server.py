"""Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 stdio server.

Exposes the `resilience-wm` Spatial World Model as standard MCP tools so coding &
reasoning agents (Claude Code, Cursor, Antigravity, LangGraph, OpenAI Agents) can
directly simulate urban disasters, inspect cascading failures, and run counterfactual
interventions via `resilience-wm mcp`.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .agent import synthesize_causal_briefing
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
from .planner import CounterfactualPlanner, ResourceBudget
from .resilience import evaluate_monte_carlo, evaluate_rollout

MCP_PROTOCOL_VERSION = "2024-11-05"

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_cities",
        "description": (
            "List built-in urban digital twins (real San Francisco OSM+30m DEM and "
            "Riverfront Metropolis) with population, nodes, and hazard support."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "inspect_city_graph",
        "description": (
            "Inspect infrastructure topology (substations, hospitals, bridges, tunnels, "
            "residential districts, and power dependencies) for a city digital twin."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "enum": ["sf", "riverfront"],
                    "default": "sf",
                }
            },
        },
    },
    {
        "name": "simulate_disaster",
        "description": (
            "Run a forward 2D physical World-Model rollout (LISFLOOD-FP flood or HAZUS "
            "earthquake) with optional interventions and Monte Carlo uncertainty bands."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "enum": ["sf", "riverfront"], "default": "sf"},
                "hazard": {"type": "string", "enum": ["flood", "earthquake"], "default": "flood"},
                "rain_mm": {"type": "number", "default": 340.0},
                "magnitude_mw": {"type": "number", "default": 6.8},
                "mc_samples": {"type": "integer", "default": 0},
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string"},
                            "target_id": {"type": "string"},
                        },
                        "required": ["kind", "target_id"],
                    },
                    "default": [],
                },
            },
        },
    },
    {
        "name": "optimize_response_plan",
        "description": (
            "Run simulator-in-the-loop counterfactual beam search over the Urban Resilience "
            "World Model to find the optimal emergency resource allocation + causal attribution."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "enum": ["sf", "riverfront"], "default": "sf"},
                "hazard": {"type": "string", "enum": ["flood", "earthquake"], "default": "flood"},
                "beam_width": {"type": "integer", "default": 2},
                "mc_samples": {"type": "integer", "default": 12},
            },
        },
    },
]


def _get_city(name: str) -> UrbanTwin:
    if name == "sf":
        return build_real_sf_city()
    if name == "riverfront":
        return build_riverfront_city()
    raise ValueError(f"Unknown city {name!r}. Choose 'sf' or 'riverfront'.")


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one MCP tool invocation and return structured JSON payload."""
    args = arguments or {}

    if name == "list_cities":
        sf = build_real_sf_city()
        rf = build_riverfront_city()
        return {
            "cities": [
                {
                    "id": "sf",
                    "name": sf.name,
                    "shape": list(sf.shape),
                    "nodes": len(sf.nodes),
                    "edges": len(sf.edges),
                    "population": sf.total_population(),
                    "bbox": list(sf.bbox) if sf.bbox else None,
                },
                {
                    "id": "riverfront",
                    "name": rf.name,
                    "shape": list(rf.shape),
                    "nodes": len(rf.nodes),
                    "edges": len(rf.edges),
                    "population": rf.total_population(),
                },
            ]
        }

    if name == "inspect_city_graph":
        city = _get_city(str(args.get("city", "sf")))
        return {
            "city": city.name,
            "total_population": city.total_population(),
            "nodes": [
                {
                    "id": n.node_id,
                    "name": n.name,
                    "kind": n.kind,
                    "population": n.population,
                    "power_source": n.power_source,
                }
                for n in city.nodes.values()
            ],
            "edges": [
                {"id": e.edge_id, "u": e.u, "v": e.v, "kind": e.kind, "length_km": e.length_km}
                for e in city.edges.values()
            ],
        }

    if name == "simulate_disaster":
        city = _get_city(str(args.get("city", "sf")))
        hazard: HazardType = "earthquake" if args.get("hazard") == "earthquake" else "flood"
        raw_actions = args.get("actions") or []
        actions = [
            InterventionAction(kind=str(a["kind"]), target_id=str(a["target_id"]))  # type: ignore[arg-type]
            for a in raw_actions
        ]
        ff = FloodForcing(total_rain_mm=float(args.get("rain_mm", 340.0)))
        ef = EarthquakeForcing(magnitude_mw=float(args.get("magnitude_mw", 6.8)))
        frames = (
            simulate_flood_rollout(city, forcing=ff, actions=actions)
            if hazard == "flood"
            else simulate_earthquake_rollout(city, forcing=ef, actions=actions)
        )
        ev = evaluate_rollout(city, frames, hazard=hazard)
        out: dict[str, Any] = ev.to_summary_dict()
        mc_n = int(args.get("mc_samples", 0))
        if mc_n > 0:
            mc = evaluate_monte_carlo(
                city,
                hazard=hazard,
                actions=actions,
                flood_forcing=ff,
                earthquake_forcing=ef,
                n_samples=mc_n,
            )
            out["monte_carlo"] = mc.to_dict()
        return out

    if name == "optimize_response_plan":
        city = _get_city(str(args.get("city", "sf")))
        hazard_opt: HazardType = "earthquake" if args.get("hazard") == "earthquake" else "flood"
        planner = CounterfactualPlanner(city, hazard=hazard_opt)
        cmp = planner.plan(
            budget=ResourceBudget.default_for(hazard_opt),
            beam_width=max(1, min(4, int(args.get("beam_width", 2)))),
            mc_samples=max(0, min(32, int(args.get("mc_samples", 12)))),
        )
        payload = cmp.to_dict()
        payload["briefing"] = synthesize_causal_briefing(city, cmp)
        return payload

    raise ValueError(f"Unknown MCP tool: {name!r}")


def handle_jsonrpc_request(req: dict[str, Any]) -> dict[str, Any] | None:
    """Process a single MCP JSON-RPC 2.0 request dict and return response dict (or None)."""
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "resilience-wm", "version": "0.2.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOL_DEFINITIONS},
        }

    if method == "tools/call":
        tname = str(params.get("name", ""))
        targs = params.get("arguments") or {}
        try:
            result_data = call_tool(tname, targs)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(result_data, ensure_ascii=False)}
                    ]
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    """Run the line-delimited JSON-RPC 2.0 MCP server loop over stdio."""
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            stdout.write(json.dumps(err) + "\n")
            stdout.flush()
            continue

        resp = handle_jsonrpc_request(req)
        if resp is not None:
            stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0


__all__ = ["ActionKind", "TOOL_DEFINITIONS", "call_tool", "handle_jsonrpc_request", "serve_stdio"]
