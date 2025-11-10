#!/usr/bin/env python3
"""
adaptive_batch_mdp.py
Automatically selects an algorithm (Value Iteration / Asynchronous VI / Fitted Q)
based on dataset size and solves for an optimal deterministic policy.
"""

import csv, os, argparse
import numpy as np
from collections import defaultdict

# ------------------------------------------------------------
# 1. Small: Standard Value Iteration (model-based)
# ------------------------------------------------------------
def value_iteration(S, A, gamma, counts, r_sum, n_sa, prior=1e-6, max_iter=2000, tol=1e-6):
    R_hat = np.zeros((S, A))
    P_hat = {}
    for s in range(1, S+1):
        for a in range(1, A+1):
            key = (s,a)
            if key in n_sa:
                R_hat[s-1,a-1] = r_sum[key]/n_sa[key]
                succ = counts[key]
                total = sum(succ.values()) + prior
                sp_list = np.array(list(succ.keys()) + [s])
                probs = np.array(list(succ.values()) + [prior]) / total
                P_hat[(s-1,a-1)] = (sp_list-1, probs)
            else:
                P_hat[(s-1,a-1)] = (np.array([s-1]), np.array([1.0]))

    V = np.zeros(S)
    for it in range(max_iter):
        V_new = np.zeros_like(V)
        for s in range(S):
            V_new[s] = max(R_hat[s,a] + gamma * np.dot(P_hat[(s,a)][1], V[P_hat[(s,a)][0]]) for a in range(A))
        if np.max(np.abs(V_new - V)) < tol:
            print(f"Value Iteration converged in {it+1} iterations.")
            break
        V = V_new

    Pi = np.ones(S, dtype=int)
    for s in range(S):
        best_a, best_val = 1, -1e18
        for a in range(A):
            q = R_hat[s,a] + gamma * np.dot(P_hat[(s,a)][1], V[P_hat[(s,a)][0]])
            if q > best_val:
                best_val, best_a = q, a+1
        Pi[s] = best_a
    return Pi

# ------------------------------------------------------------
# 2. Medium: Asynchronous Value Iteration (in-place updates)
# ------------------------------------------------------------
def async_value_iteration(S, A, gamma, counts, r_sum, n_sa, prior=1e-6, max_iter=2000, tol=1e-6):
    R_hat = np.zeros((S, A))
    P_hat = {}
    for s in range(1, S+1):
        for a in range(1, A+1):
            key = (s,a)
            if key in n_sa:
                R_hat[s-1,a-1] = r_sum[key]/n_sa[key]
                succ = counts[key]
                total = sum(succ.values()) + prior
                sp_list = np.array(list(succ.keys()) + [s])
                probs = np.array(list(succ.values()) + [prior]) / total
                P_hat[(s-1,a-1)] = (sp_list-1, probs)
            else:
                P_hat[(s-1,a-1)] = (np.array([s-1]), np.array([1.0]))

    V = np.zeros(S)
    for it in range(max_iter):
        delta = 0.0
        for s in range(S):
            old = V[s]
            V[s] = max(R_hat[s,a] + gamma * np.dot(P_hat[(s,a)][1], V[P_hat[(s,a)][0]]) for a in range(A))
            delta = max(delta, abs(old - V[s]))
        if delta < tol:
            print(f"Asynchronous VI converged in {it+1} iterations.")
            break

    Pi = np.ones(S, dtype=int)
    for s in range(S):
        best_a, best_val = 1, -1e18
        for a in range(A):
            q = R_hat[s,a] + gamma * np.dot(P_hat[(s,a)][1], V[P_hat[(s,a)][0]])
            if q > best_val:
                best_val, best_a = q, a+1
        Pi[s] = best_a
    return Pi

# ------------------------------------------------------------
# 3. Large: Fitted Q Iteration (sample-based)
# ------------------------------------------------------------
def fitted_q_iteration(S, A, gamma, transitions, max_iter=30):
    """
    Performs a lightweight sample-based fitted Q iteration.
    transitions = [(s,a,r,sp), ...] using integer IDs (1-based).
    """
    Q = defaultdict(float)
    for it in range(max_iter):
        td_sum = 0.0
        for (s,a,r,sp) in transitions:
            # Estimate target
            max_next = max(Q[(sp,a2)] for a2 in range(1,A+1))
            target = r + gamma * max_next
            td_error = target - Q[(s,a)]
            Q[(s,a)] += 0.1 * td_error  # simple learning rate
            td_sum += abs(td_error)
        if td_sum / len(transitions) < 1e-4:
            print(f"Fitted Q Iteration converged in {it+1} iterations.")
            break

    # Extract greedy deterministic policy
    Pi = np.ones(S, dtype=int)
    for s in range(1,S+1):
        best_a, best_val = 1, -1e18
        for a in range(1,A+1):
            q = Q[(s,a)]
            if q > best_val:
                best_val, best_a = q, a
        Pi[s-1] = best_a
    return Pi

# ------------------------------------------------------------
# 4. Algorithm selector and main
# ------------------------------------------------------------
def choose_and_run(csv_path, gamma=0.95):
    # --- Read transitions and infer S, A ---
    counts = defaultdict(lambda: defaultdict(int))
    r_sum, n_sa = defaultdict(float), defaultdict(int)
    transitions, states_seen, actions_seen = [], set(), set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s,a,r,sp = int(row["s"]), int(row["a"]), float(row["r"]), int(row["sp"])
            counts[(s,a)][sp] += 1
            r_sum[(s,a)] += r
            n_sa[(s,a)] += 1
            transitions.append((s,a,r,sp))
            states_seen.update([s,sp])
            actions_seen.add(a)

    S, A = max(states_seen), max(actions_seen)
    print(f"Detected {S} states, {A} actions. gamma={gamma}")

    # --- Choose algorithm based on size ---
    if S <= 5000:
        print("Using standard Value Iteration.")
        Pi = value_iteration(S,A,gamma,counts,r_sum,n_sa)
    elif S <= 100000:
        print("Using Asynchronous Value Iteration.")
        Pi = async_value_iteration(S,A,gamma,counts,r_sum,n_sa)
    else:
        print("Using Fitted Q Iteration (sample-based).")
        Pi = fitted_q_iteration(S,A,gamma,transitions)

    # --- Write policy ---
    out_file = os.path.splitext(csv_path)[0] + ".POLICY"
    with open(out_file,"w") as f:
        for a in Pi: f.write(f"{a}\n")
    print(f"Wrote {out_file} ({len(Pi)} lines)")
    return Pi

def main():
    parser = argparse.ArgumentParser(description="Adaptive MDP Solver: chooses best algorithm by dataset size.")
    parser.add_argument("csv", help="Input CSV file with s,a,r,sp")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor (default 0.95)")
    args = parser.parse_args()
    choose_and_run(args.csv, gamma=args.gamma)

if __name__ == "__main__":
    main()
