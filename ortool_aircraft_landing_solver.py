from ortools.sat.python import cp_model
from qubots.base_optimizer import BaseOptimizer

class ORToolsAircraftLanding(BaseOptimizer):
    def optimize(self, problem, initial_solution=None, **kwargs):
        model = cp_model.CpModel()
        scale_factor = 1000  # e.g. multiply float costs by 1000 to make them integers

        n = problem.nb_planes
        earliest = problem.earliest_time
        target   = problem.target_time
        latest   = problem.latest_time
        earliness_costs = problem.earliness_cost
        tardiness_costs = problem.tardiness_cost
        sep      = problem.separation_time

        # 1) Define T[i], the plane landing times.
        T = []
        for i in range(n):
            t_var = model.NewIntVar(earliest[i], latest[i], f"T_{i}")
            T.append(t_var)

        # 2) Define earliness (E[i]) and lateness (L[i]) as non-negative.
        E, L = [], []
        for i in range(n):
            e_var = model.NewIntVar(0, latest[i] - earliest[i], f"E_{i}")
            l_var = model.NewIntVar(0, latest[i] - earliest[i], f"L_{i}")
            E.append(e_var)
            L.append(l_var)
            # T[i] - target[i] + E[i] - L[i] == 0
            model.Add(T[i] - target[i] + E[i] - L[i] == 0)

        # 3) Boolean ordering variables b[i,j].
        b = {}
        M = max(latest) + max(sep[i][j] for i in range(n) for j in range(n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    b[(i,j)] = model.NewBoolVar(f"b_{i}_{j}")

        # 4) Separation constraints with big-M.
        for i in range(n):
            for j in range(n):
                if i < j:
                    bij = b[(i,j)]
                    model.Add(T[j] >= T[i] + sep[i][j] - M*(1 - bij))
                    model.Add(T[i] >= T[j] + sep[j][i] - M*bij)

        # 5) Objective: sum of earliness + lateness, scaled by factor
        total_cost = model.NewIntVar(0, 10**12, "total_cost")  # domain might need to be large
        cost_terms = []
        for i in range(n):
            # Convert float cost rates to integers by scaling
            earliness_int = int(round(earliness_costs[i] * scale_factor))
            tardiness_int = int(round(tardiness_costs[i] * scale_factor))

            cost_i = model.NewIntVar(0, 10**12, f"cost_{i}")
            model.Add(cost_i == earliness_int * E[i] + tardiness_int * L[i])
            cost_terms.append(cost_i)

        model.Add(total_cost == sum(cost_terms))
        model.Minimize(total_cost)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Extract the scaled cost
            raw_cost = solver.Value(total_cost)
            # Convert back to real cost
            best_cost = raw_cost / scale_factor

            # Build a solution with landing times
            landing_times = [solver.Value(T[i]) for i in range(n)]
            planes_sorted = sorted(range(n), key=lambda i: landing_times[i])
            best_solution = {
                "landing_order": planes_sorted,
                "landing_times": [landing_times[i] for i in planes_sorted],
            }
        else:
            best_solution = {
                "landing_order": list(range(n)),
                "landing_times": [earliest[i] for i in range(n)],
            }
            best_cost = 1e9

        return best_solution, best_cost
