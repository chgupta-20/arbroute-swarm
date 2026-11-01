// ============================================================================
// ArbRoute-Swarm: C++17 Monte Carlo & Stochastic Graph Solver (pybind11)
// Author: Chaitanya Gupta (IIT Palakkad, CSE)
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>

namespace py = pybind11;

struct RouteCandidate {
    std::string mandi_id;
    std::string mandi_name;
    double spot_price;
    double distance_km;
    double base_duration_hrs;
    double ambient_temp_c;
    double precipitation_mm;
};

struct SimulationResult {
    std::string mandi_id;
    std::string mandi_name;
    double expected_profit;
    double cvar_95;
    double expected_spoilage_pct;
    double risk_adjusted_utility;
};

class StochasticArbitrageSolver {
private:
    int num_rollouts_;
    double risk_aversion_gamma_;

public:
    StochasticArbitrageSolver(int num_rollouts = 10000, double gamma = 0.45)
        : num_rollouts_(num_rollouts), risk_aversion_gamma_(gamma) {}

    std::vector<SimulationResult> evaluate_routes(
        const std::vector<RouteCandidate>& candidates,
        double cargo_quintals,
        double base_decay_lambda,
        double freight_rate_per_km_q
    ) {
        std::vector<SimulationResult> results;
        results.reserve(candidates.size());

        std::mt19937_64 rng(42);
        std::normal_distribution<double> norm_dist(0.0, 1.0);

        for (const auto& route : candidates) {
            std::vector<double> profits(num_rollouts_);
            double total_profit = 0.0;
            double total_spoilage = 0.0;
            const double base_freight = route.distance_km * freight_rate_per_km_q * cargo_quintals;

            for (int i = 0; i < num_rollouts_; ++i) {
                double z1 = norm_dist(rng);
                double z2 = norm_dist(rng);
                double z3 = norm_dist(rng);

                // Stochastic highway temperature & flood delay multiplier
                double sampled_temp = route.ambient_temp_c + 2.2 * z1;
                double sampled_rain = std::max(0.0, route.precipitation_mm * (1.0 + 0.35 * z2));
                double flood_delay_mult = 1.0 + std::pow(sampled_rain / 22.0, 1.65) + std::max(0.0, 0.08 * z1);
                double transit_hrs = route.base_duration_hrs * flood_delay_mult;

                // Exponential Arrhenius thermal spoilage model
                double thermal_accel = std::exp(0.065 * (sampled_temp - 25.0));
                double survival_ratio = std::exp(-base_decay_lambda * thermal_accel * transit_hrs);
                double spoilage_pct = (1.0 - survival_ratio) * 100.0;

                // Arrival spot price volatility
                double arrival_spot = route.spot_price * (1.0 + 0.035 * z3);
                double delay_penalty = std::max(0.0, transit_hrs - route.base_duration_hrs) * 450.0;
                double net_profit = (cargo_quintals * survival_ratio * arrival_spot) - base_freight - delay_penalty;

                profits[i] = net_profit;
                total_profit += net_profit;
                total_spoilage += spoilage_pct;
            }

            // Compute 95% Conditional Value-at-Risk (Expected Shortfall of worst 5% tail)
            int tail_count = std::max(1, static_cast<int>(0.05 * num_rollouts_));
            std::nth_element(profits.begin(), profits.begin() + tail_count, profits.end());
            double tail_sum = std::accumulate(profits.begin(), profits.begin() + tail_count, 0.0);

            double expected_profit = total_profit / num_rollouts_;
            double cvar_95 = tail_sum / tail_count;
            double utility = expected_profit - risk_aversion_gamma_ * std::abs(expected_profit - cvar_95);

            results.push_back({
                route.mandi_id,
                route.mandi_name,
                expected_profit,
                cvar_95,
                total_spoilage / num_rollouts_,
                utility
            });
        }

        std::sort(results.begin(), results.end(), [](const SimulationResult& a, const SimulationResult& b) {
            return a.risk_adjusted_utility > b.risk_adjusted_utility;
        });

        return results;
    }
};

PYBIND11_MODULE(arbroute_cpp, m) {
    m.doc() = "C++17 Stochastic Commodity Arbitrage & CVaR Graph Solver";

    py::class_<RouteCandidate>(m, "RouteCandidate")
        .def(py::init<std::string, std::string, double, double, double, double, double>());

    py::class_<SimulationResult>(m, "SimulationResult")
        .def_readonly("mandi_id", &SimulationResult::mandi_id)
        .def_readonly("mandi_name", &SimulationResult::mandi_name)
        .def_readonly("expected_profit", &SimulationResult::expected_profit)
        .def_readonly("cvar_95", &SimulationResult::cvar_95)
        .def_readonly("expected_spoilage_pct", &SimulationResult::expected_spoilage_pct)
        .def_readonly("risk_adjusted_utility", &SimulationResult::risk_adjusted_utility);

    py::class_<StochasticArbitrageSolver>(m, "StochasticArbitrageSolver")
        .def(py::init<int, double>(), py::arg("num_rollouts") = 10000, py::arg("gamma") = 0.45)
        .def("evaluate_routes", &StochasticArbitrageSolver::evaluate_routes);
}