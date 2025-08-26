import numpy as np
import csv
from collections import defaultdict
import os
import matplotlib.pyplot as plt

# -----------------------------
# Discretization of state/control
# -----------------------------

def discretize_spaces(x_step, u_step):
    x_min, x_max = 0.0, 1.5
    u_min, u_max = -1.0, 1.0

    x_vals = np.arange(x_min, x_max + 1e-9, x_step)
    u_vals = np.arange(u_min, u_max + 1e-9, u_step)
    return x_vals, u_vals

# -----------------------------
# Time grid (k = 0, 1, ..., N)
# -----------------------------

def make_time_grid(N, T=2.0):
    k_grid = np.arange(N + 1)
    dt = T / N if N > 0 else T
    return k_grid, dt

# -----------------------------
# Problem-specific ingredients
# -----------------------------

def build_x_index(x_vals, tol=1e-9):
    return {round(float(v), 10): i for i, v in enumerate(x_vals)}

def find_bracketing_indices(x_next, x_vals):
    for j in range(len(x_vals) - 1):
        if x_vals[j] < x_next < x_vals[j + 1]:
            return j, j + 1
    return None, None

# Costs
def stage_cost(u_k):
    return 2.0 * (u_k ** 2)

def terminal_step_cost(x_next, u_last):
    return (x_next ** 2) + 2.0 * (u_last ** 2)

# -----------------------------
# DP Solver
# -----------------------------

def solve_dp(x_step=0.5, u_step=0.5, N=2, T=2.0):
    x_vals, u_vals = discretize_spaces(x_step, u_step)
    k_grid, dt = make_time_grid(N, T)

    x_index_map = build_x_index(x_vals)

    J = np.full((N, len(x_vals)), np.inf)
    pi = [[[] for _ in range(len(x_vals))] for _ in range(N)]

    step_details = [defaultdict(list) for _ in range(N)]

    # Last step
    if N >= 1:
        k = N - 1
        for i, x in enumerate(x_vals):
            best_cost = np.inf
            for uj, u in enumerate(u_vals):
                x_next = round(float(x + u), 10)
                if x_next < 0.0 - 1e-12 or x_next > 1.5 + 1e-12:
                    continue

                interpolated = False
                if x_next in x_index_map:
                    next_cost = 0.0
                else:
                    j_low, j_up = find_bracketing_indices(x_next, x_vals)
                    if j_low is None:
                        continue
                    next_cost = 0.5 * (J[k, j_up] + J[k, j_low])
                    interpolated = True

                total = terminal_step_cost(x_next, u)
                step_details[k][i].append({
                    'u': float(u),
                    'x_next': float(x_next),
                    'immediate_cost': float((x_next ** 2) + 2.0 * (u ** 2)),
                    'next_cost': float(0.0),
                    'total_cost': float(total),
                    'interpolated': interpolated
                })
                if total < best_cost - 1e-12:
                    best_cost = total
                    J[k, i] = best_cost
                    pi[k][i] = [uj]
                elif abs(total - best_cost) <= 1e-12:
                    pi[k][i].append(uj)

    # Previous steps
    for k in range(N - 2, -1, -1):
        for i, x in enumerate(x_vals):
            best_cost = np.inf
            for uj, u in enumerate(u_vals):
                x_next = round(float(x + u), 10)
                if x_next < 0.0 - 1e-12 or x_next > 1.5 + 1e-12:
                    continue

                interpolated = False
                if x_next in x_index_map:
                    i_next = x_index_map[x_next]
                    next_c = J[k + 1, i_next]
                else:
                    j_low, j_up = find_bracketing_indices(x_next, x_vals)
                    if j_low is None:
                        continue
                    next_c = J[k + 1, j_low] + 0.5 * (J[k + 1, j_up] - J[k + 1, j_low])
                    interpolated = True

                immediate = stage_cost(u)
                total = immediate + next_c

                step_details[k][i].append({
                    'u': float(u),
                    'x_next': float(x_next),
                    'immediate_cost': float(immediate),
                    'next_cost': float(next_c),
                    'total_cost': float(total),
                    'interpolated': interpolated
                })

                if total < best_cost - 1e-12:
                    best_cost = total
                    J[k, i] = best_cost
                    pi[k][i] = [uj]
                elif abs(total - best_cost) <= 1e-12:
                    pi[k][i].append(uj)

    return {
        'x_vals': x_vals,
        'u_vals': u_vals,
        'N': N,
        'T': T,
        'dt': dt,
        'J': J,
        'pi': pi,
        'details': step_details,
        'x_step': x_step,
        'u_step': u_step,
    }

# -----------------------------
# Export all steps into a single CSV
# -----------------------------

def export_to_single_csv(result):
    x_vals = result['x_vals']
    u_vals = result['u_vals']
    J = result['J']
    pi = result['pi']
    details = result['details']
    N = result['N']
    x_step = result['x_step']
    u_step = result['u_step']

    filename = f"dp_solution_x{x_step}_u{u_step}_N{N}.csv"

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "x", "u", "x_next", "immediate_cost", "next_cost", "total_cost", "j*", "u*", "interpolated"])

        for k in range(N - 1, -1, -1):
            for i, x in enumerate(x_vals):
                opts = details[k].get(i, [])
                jstar = J[k, i]
                ustar_idxs = pi[k][i]
                ustar_vals = [u_vals[idx] for idx in ustar_idxs]

                if not opts:
                    writer.writerow([k, x, None, None, None, None, None, jstar, "|".join(map(str, ustar_vals)), "no"])
                else:
                    for o in opts:
                        optimal_flag = "yes" if abs(o['total_cost'] - jstar) <= 1e-12 else "no"
                        writer.writerow([
                            k,
                            x,
                            o['u'],
                            o['x_next'],
                            o['immediate_cost'],
                            o['next_cost'],
                            o['total_cost'],
                            jstar,
                            o['u'] if optimal_flag == "yes" else "",
                            "yes" if o['interpolated'] else "no"
                        ])
    print(f"All steps exported to {filename}")
    return filename

# -----------------------------
# Load result from CSV
# -----------------------------

def load_result_from_csv(filename):
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    N = max(int(r["Step"]) for r in rows) + 1
    x_vals = sorted(set(float(r["x"]) for r in rows if r["x"] != ''))
    u_vals = sorted(set(float(r["u"]) for r in rows if r["u"] not in (None, '', 'None')))

    return {
        "rows": rows,
        "N": N,
        "x_vals": np.array(x_vals),
        "u_vals": np.array(u_vals)
    }

# -----------------------------
# Reconstruct optimal trajectories
# -----------------------------

def reconstruct_trajectories(result, x0):
    x_vals = result["x_vals"]
    u_vals = result["u_vals"]
    N = result["N"]
    rows = result["rows"]

    x_index_map = build_x_index(x_vals)

    if round(float(x0), 10) not in x_index_map:
        return []

    def recurse(step, x_curr, cost_so_far):
        if step >= N:
            return [([x_curr], [], [cost_so_far])]

        candidates = [r for r in rows if int(r["Step"]) == step and float(r["x"]) == round(float(x_curr), 10)]
        opt_candidates = [r for r in candidates if r["u*"] not in ('', None)]

        if not opt_candidates:
            return []

        trajectories = []
        for r in opt_candidates:
            u = float(r["u*"])
            x_next = float(r["x_next"])
            immediate = float(r["immediate_cost"])
            total_cost = cost_so_far + immediate
            suffixes = recurse(step + 1, x_next, total_cost)
            for xs, us, cs in suffixes:
                trajectories.append(([x_curr] + xs, [u] + us, [cost_so_far] + cs))
        return trajectories

    return recurse(0, x0, 0.0)

# -----------------------------
# Show trajectories in terminal + plots
# -----------------------------

def show_trajectories(result, x0):
    trajectories = reconstruct_trajectories(result, x0)

    if not trajectories:
        print(f"No feasible trajectory from x0={x0}")
        return

    print(f"\nOptimal trajectories starting from x0={x0}:\n")

    n_traj = len(trajectories)
    fig, axes = plt.subplots(n_traj, 3, figsize=(12, 4 * n_traj))

    if n_traj == 1:
        axes = np.array([axes])  # ensure axes is 2D

    for idx, (xs, us, cs) in enumerate(trajectories):
        print(f"Trajectory {idx+1}:")
        print(f"  States: {xs}")
        print(f"  Controls: {us}")
        print(f"  Total Cost: {cs[-1]:.6f}")

        steps_x = list(range(len(xs)))
        steps_u = list(range(len(us)))
        steps_j = list(range(len(cs)))

        # X trajectory (step plot)
        axes[idx, 0].step(steps_x, xs, where="post", marker="o")
        axes[idx, 0].set_title(f"Trajectory {idx+1} - States")
        axes[idx, 0].set_xlabel("Step")
        axes[idx, 0].set_ylabel("x")
        axes[idx, 0].grid(True)

        # U trajectory (step plot)
        axes[idx, 1].step(steps_u, us, where="post", marker="o", color="orange")
        axes[idx, 1].set_title(f"Trajectory {idx+1} - Controls")
        axes[idx, 1].set_xlabel("Step")
        axes[idx, 1].set_ylabel("u")
        axes[idx, 1].grid(True)

        # Cumulative cost
        axes[idx, 2].step(steps_j, cs, where="post", marker="o", color="green")
        axes[idx, 2].set_title(f"Trajectory {idx+1} - Cumulative Cost")
        axes[idx, 2].set_xlabel("Step")
        axes[idx, 2].set_ylabel("J")
        axes[idx, 2].grid(True)

    plt.tight_layout()
    plt.show()

# -----------------------------
# CLI demo
# -----------------------------
if __name__ == "__main__":
    x_step = float(input("Enter step size for x (e.g. 0.5): "))
    u_step = float(input("Enter step size for u (e.g. 0.5): "))
    N = int(input("Enter number of steps N (e.g. 4): "))

    filename = f"dp_solution_x{x_step}_u{u_step}_N{N}.csv"
    if os.path.exists(filename):
        print(f"Found cached solution file {filename}, loading DP result from CSV (no recomputation).")
        result = load_result_from_csv(filename)
    else:
        print("No cached solution found, solving DP...")
        result = solve_dp(x_step=x_step, u_step=u_step, N=N, T=2.0)
        export_to_single_csv(result)
        result = load_result_from_csv(filename)

    while True:
        x0_str = input("\nEnter initial state x0 (or 'q' to quit): ")
        if x0_str.lower() == 'q':
            break
        try:
            x0 = float(x0_str)
            show_trajectories(result, x0)
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
