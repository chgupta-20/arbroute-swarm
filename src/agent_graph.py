"""
ArbRoute-Swarm: Stateful LangGraph Multi-Agent Orchestrator
Connects live Agmarknet, OSRM, and Open-Meteo telemetry to the C++17 pybind11 CVaR solver.
"""
import os
import requests
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

try:
    import arbroute_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


class SwarmState(TypedDict):
    origin: Dict[str, Any]
    cargo_quintals: float
    base_decay_lambda: float
    candidate_mandis: List[Dict[str, Any]]
    enriched_routes: List[Dict[str, Any]]
    ranked_results: List[Dict[str, Any]]
    dispatch_manifest: str


def market_discovery_node(state: SwarmState) -> SwarmState:
    """Polls Data.gov.in Agmarknet API (with deterministic schema normalization)."""
    api_key = os.getenv("DATA_GOV_API_KEY", "")
    candidates = state["candidate_mandis"]
    if api_key:
        try:
            url = (
                "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
                f"?api-key={api_key}&format=json&limit=10"
            )
            resp = requests.get(url, timeout=4)
            if resp.status_code == 200:
                records = resp.json().get("records", [])
                for idx, rec in enumerate(records[: len(candidates)]):
                    modal = float(rec.get("modal_price", candidates[idx]["spot_price"]))
                    candidates[idx]["spot_price"] = modal
        except Exception:
            pass
    state["candidate_mandis"] = candidates
    return state


def climate_and_osrm_node(state: SwarmState) -> SwarmState:
    """Queries live OSRM road geometry and Open-Meteo highway weather feeds."""
    origin = state["origin"]
    enriched = []

    for dest in state["candidate_mandis"]:
        osrm_url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{origin['lon']},{origin['lat']};{dest['lon']},{dest['lat']}"
            f"?overview=full&geometries=geojson"
        )
        r_osrm = requests.get(osrm_url, timeout=5).json()
        route = r_osrm["routes"][0]
        coords = route["geometry"]["coordinates"]
        mid = coords[len(coords) // 2]

        meteo_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={mid[1]:.4f}&longitude={mid[0]:.4f}"
            f"&current=temperature_2m,precipitation"
        )
        r_meteo = requests.get(meteo_url, timeout=5).json()
        curr = r_meteo.get("current", {})

        enriched.append({
            **dest,
            "distance_km": route["distance"] / 1000.0,
            "duration_hrs": route["duration"] / 3600.0,
            "temp_c": curr.get("temperature_2m", 30.0),
            "precip_mm": curr.get("precipitation", 0.0),
        })

    state["enriched_routes"] = enriched
    return state


def cpp_quant_solver_node(state: SwarmState) -> SwarmState:
    """Invokes the compiled C++17 pybind11 Monte Carlo CVaR module."""
    routes = state["enriched_routes"]
    if CPP_AVAILABLE:
        solver = arbroute_cpp.StochasticArbitrageSolver(10000, 0.45)
        cpp_inputs = [
            arbroute_cpp.RouteCandidate(
                r["id"], r["name"], r["spot_price"],
                r["distance_km"], r["duration_hrs"],
                r["temp_c"], r["precip_mm"]
            )
            for r in routes
        ]
        res = solver.evaluate_routes(
            cpp_inputs, state["cargo_quintals"], state["base_decay_lambda"], 0.24
        )
        state["ranked_results"] = [
            {
                "mandi_name": item.mandi_name,
                "expected_profit": item.expected_profit,
                "cvar_95": item.cvar_95,
                "spoilage_pct": item.expected_spoilage_pct,
            }
            for item in res
        ]
    return state


def supervisor_manifest_node(state: SwarmState) -> SwarmState:
    """Synthesizes an auditable dispatch order with source attribution."""
    best = state["ranked_results"][0]
    state["dispatch_manifest"] = (
        f"DISPATCH APPROVED -> {best['mandi_name']} | "
        f"E[Margin]: INR {best['expected_profit']:,.0f} | "
        f"CVaR(95%): INR {best['cvar_95']:,.0f} | "
        f"Expected Spoilage: {best['spoilage_pct']:.1f}%"
    )
    return state


def build_arbroute_graph():
    workflow = StateGraph(SwarmState)
    workflow.add_node("market_discovery", market_discovery_node)
    workflow.add_node("climate_and_osrm", climate_and_osrm_node)
    workflow.add_node("cpp_quant_solver", cpp_quant_solver_node)
    workflow.add_node("supervisor_manifest", supervisor_manifest_node)

    workflow.set_entry_point("market_discovery")
    workflow.add_edge("market_discovery", "climate_and_osrm")
    workflow.add_edge("climate_and_osrm", "cpp_quant_solver")
    workflow.add_edge("cpp_quant_solver", "supervisor_manifest")
    workflow.add_edge("supervisor_manifest", END)
    return workflow.compile()