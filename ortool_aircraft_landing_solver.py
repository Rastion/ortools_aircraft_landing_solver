import math
from ortools.sat.python import cp_model
from qubots.base_optimizer import BaseOptimizer

class ORToolsAircraftLanding(BaseOptimizer):
    """
    An OR-Tools CP-SAT based solver for the Aircraft Landing Problem.
    
    We'll create integer variables for each plane's landing time.
    - T[i] in [earliest_time[i], latest_time[i]].
    
    We'll also create two helper integer variables for each plane, E[i] and L[i], 
    capturing earliness and lateness, so that:
        T[i] - target_time[i] = L[i] - E[i].
    Then cost[i] = earliness_cost[i] * E[i] + tardiness_cost[i] * L[i].
    
    For separation constraints, we need to ensure that for any two planes i and j:
        T[j] >= T[i] + separation_time[i][j], if i lands before j
        T[i] >= T[j] + separation_time[j][i], if j lands before i

    We can't know in advance who lands first, so we introduce a boolean b[i,j] 
    to indicate i-lands-before-j. Then:
        T[j] >= T[i] + separation_time[i][j] - M*(1 - b[i,j])
        T[i] >= T[j] + separation_time[j][i] - M*b[i,j]
    
    Where M is a sufficiently large constant (big-M).
    The objective is to minimize sum of earliness/tardiness costs for all planes.
    """

    def optimize(self, problem, initial_solution=None, **kwargs):
        model = cp_model.CpModel()

        n = problem.nb_planes
        earliest = problem.earliest_time
        target = problem.target_time
        latest = problem.latest_time
        earliness_costs = problem.earliness_cost
        tardiness_costs = problem.tardiness_cost
        sep = problem.separation_time
        
        # We'll define integer variables T[i] for landing time of plane i
        T = []
        for i in range(n):
            # Force the variable domain to [earliest[i], latest[i]]
            t_var = model.NewIntVar(earliest[i], latest[i], f"T_{i}")
            T.append(t_var)
        
        # For each plane, define earliness E[i] and lateness L[i], non-negative
        # T[i] - target[i] = L[i] - E[i]
        E = []
        L = []
        for i in range(n):
            e_var = model.NewIntVar(0, latest[i] - earliest[i], f"E_{i}")
            l_var = model.NewIntVar(0, latest[i] - earliest[i], f"L_{i}")
            E.append(e_var)
            L.append(l_var)
            # Add linear constraint: T[i] - target[i] + E[i] - L[i] = 0
            # T[i] - target[i] = L[i] - E[i]
            model.Add(T[i] - target[i] + E[i] - L[i] == 0)

        # Introduce boolean variables for ordering between planes
        # b[i][j] = 1 if plane i lands before plane j
        b = {}
        M = max(latest) + max(sep[i][j] for i in range(n) for j in range(n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    b[(i,j)] = model.NewBoolVar(f"b_{i}_{j}")
        
        # Separation constraints
        # T[j] >= T[i] + sep[i][j] - M*(1 - b[i,j])
        # T[i] >= T[j] + sep[j][i] - M*b[i,j]
        for i in range(n):
            for j in range(n):
                if i < j:
                    bij = b[(i,j)]
                    # i before j
                    model.Add(T[j] >= T[i] + sep[i][j] - M*(1 - bij))
                    # j before i
                    model.Add(T[i] >= T[j] + sep[j][i] - M*bij)

        # Objective: minimize sum_{i} ( earliness_cost[i] * E[i] + tardiness_cost[i] * L[i] )
        total_cost = model.NewIntVar(0, 10**9, "total_cost")
        # We'll sum up cost for each plane as a linear expression
        cost_terms = []
        for i in range(n):
            # cost_i = earliness_cost[i]*E[i] + tardiness_cost[i]*L[i]
            # We'll build them in a linear expression
            cost_i = model.NewIntVar(0, 10**9, f"cost_{i}")
            model.Add(cost_i == earliness_costs[i]*E[i] + tardiness_costs[i]*L[i])
            cost_terms.append(cost_i)
        model.Add(total_cost == sum(cost_terms))
        model.Minimize(total_cost)

        # Solve
        solver = cp_model.CpSolver()
        solver_status = solver.Solve(model)

        if solver_status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Extract solution
            landing_times = [solver.Value(T[i]) for i in range(n)]
            # We'll sort planes by their landing time to form the order
            planes_sorted = sorted(range(n), key=lambda i: landing_times[i])
            best_solution = {
                "landing_order": planes_sorted,
                "landing_times": [landing_times[i] for i in planes_sorted]
            }
            best_cost = solver.Value(total_cost)
        else:
            # If infeasible, return a fallback
            best_solution = {"landing_order": list(range(n)), "landing_times": [earliest[i] for i in range(n)]}
            best_cost = 1e9

        return best_solution, best_cost
