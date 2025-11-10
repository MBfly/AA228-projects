#!/usr/bin/env python3

import argparse, csv, os, sys, time
from collections import defaultdict, Counter
import numpy as np

# parse
def parse_problem(csv_path, n_states_override=None, n_actions_override=None, gamma_override=None):
    samples = []
    reward_sums = defaultdict(float)
    counts_sa = defaultdict(int)
    next_counts = defaultdict(Counter)
    action_reward_sum = defaultdict(float)
    action_count = defaultdict(int)

    max_state = -1
    min_state = 1 << 60
    max_action = -1
    min_action = 1 << 60

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"s", "a", "r", "sp"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"CSV must have header exactly {required}")

        for row in reader:
            s = int(row["s"])
            a = int(row["a"])
            r = float(row["r"])
            sp = int(row["sp"])

            max_state = max(max_state, s, sp)
            min_state = min(min_state, s, sp)
            max_action = max(max_action, a)
            min_action = min(min_action, a)

            samples.append((s, a, r, sp))
            reward_sums[(s, a)] += r
            counts_sa[(s, a)] += 1
            next_counts[(s, a)][sp] += 1
            action_reward_sum[a] += r
            action_count[a] += 1

    if min_state == 1:
        print("converting to 0-based.", file=sys.stderr)

        def shift_state_dict(d):
            return {(s - 1, a): v for (s, a), v in d.items()}

        def shift_next_counts(nd):
            return {
                (s - 1, a): Counter({sp - 1: c for sp, c in sub.items()})
                for (s, a), sub in nd.items()
            }

        samples = [(s - 1, a, r, sp - 1) for (s, a, r, sp) in samples]
        reward_sums = shift_state_dict(reward_sums)
        counts_sa = shift_state_dict(counts_sa)
        next_counts = shift_next_counts(next_counts)
        max_state -= 1
        min_state = 0

    inferred_states = max_state + 1
    inferred_actions = max_action if min_action >= 1 else max_action + 1
    gamma = 0.95

    name = os.path.basename(csv_path).lower()
    if "small" in name:
        inferred_states, inferred_actions, gamma = 100, max(inferred_actions, 4), 0.95
    elif "medium" in name:
        inferred_states, inferred_actions, gamma = 50_000, max(inferred_actions, 7), 0.95
    elif "large" in name:
        inferred_states, inferred_actions, gamma = 302_020, max(inferred_actions, 9), 0.95

    n_states = n_states_override or inferred_states
    n_actions = n_actions_override or inferred_actions
    gamma = gamma_override or gamma

    avg_reward_by_action = {}
    for a in range(1, n_actions + 1):
        c = action_count.get(a, 0)
        avg_reward_by_action[a] = (action_reward_sum[a] / c) if c > 0 else 0.0

    return n_states, n_actions, gamma, samples, reward_sums, counts_sa, next_counts, avg_reward_by_action


# Small
def solve_small(n_states, n_actions, gamma, reward_sums, counts_sa, next_counts, avg_reward_by_action,
                max_iters=500, tol=1e-8):
    R_hat, P_hat = {}, {}
    for (s, a), cnt in counts_sa.items():
        R_hat[(s, a)] = reward_sums[(s, a)] / max(cnt, 1)
        ncnts = next_counts[(s, a)]
        total = sum(ncnts.values())
        P_hat[(s, a)] = ({sp: c / total for sp, c in ncnts.items()} if total > 0 else {})

    V = np.zeros(n_states)
    for _ in range(max_iters):
        delta = 0.0
        V_new = np.zeros_like(V)
        for s in range(n_states):
            best = -1e300
            for a in range(1, n_actions + 1):
                r = R_hat.get((s, a), avg_reward_by_action.get(a, 0.0))
                p = P_hat.get((s, a), {})
                q = r + gamma * sum(prob * V[sp] for sp, prob in p.items() if 0 <= sp < n_states)
                best = max(best, q)
            V_new[s] = best
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < tol:
            break

    policy = np.ones(n_states, dtype=int)
    for s in range(n_states):
        best_a, best_q = 1, -1e300
        for a in range(1, n_actions + 1):
            r = R_hat.get((s, a), avg_reward_by_action.get(a, 0.0))
            p = P_hat.get((s, a), {})
            q = r + gamma * sum(prob * V[sp] for sp, prob in p.items() if 0 <= sp < n_states)
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a
    return policy


# Medium
def solve_medium(n_states, n_actions, gamma, samples, avg_reward_by_action):
    passes, alpha0 = 40, 0.3
    Q = defaultdict(float)
    seen = defaultdict(set)
    for (s, a, r, sp) in samples:
        seen[s].add(a)

    def max_Q(sp):
        if sp in seen:
            return max(Q[(sp, ap)] for ap in seen[sp])
        return max(avg_reward_by_action.values()) if avg_reward_by_action else 0.0

    for t in range(1, passes + 1):
        alpha = alpha0 / (1.0 + 0.2 * (t - 1))
        for (s, a, r, sp) in samples:
            target = r + gamma * max_Q(sp)
            Q[(s, a)] += alpha * (target - Q[(s, a)])

    policy = np.ones(n_states, dtype=int)
    best_global = max(avg_reward_by_action, key=avg_reward_by_action.get, default=1)
    for s in range(n_states):
        if s not in seen:
            policy[s] = best_global
            continue
        policy[s] = max(seen[s], key=lambda a: Q[(s, a)])
    return policy


# Large
def fitted_q_iteration(n_states, n_actions, gamma, transitions, max_iter=30):
    Q = defaultdict(float)
    for it in range(max_iter):
        td_sum = 0.0
        for (s, a, r, sp) in transitions:
            max_next = max(Q[(sp, a2)] for a2 in range(1, n_actions + 1))
            target = r + gamma * max_next
            td_error = target - Q[(s, a)]
            Q[(s, a)] += 0.1 * td_error
            td_sum += abs(td_error)
        if td_sum / len(transitions) < 1e-4:
            print(f"Fitted Q Iteration converged in {it+1} iterations.", file=sys.stderr)
            break

    policy = np.ones(n_states, dtype=int)
    for s in range(1, n_states + 1):
        best_a, best_val = 1, -1e18
        for a in range(1, n_actions + 1):
            q = Q[(s, a)]
            if q > best_val:
                best_val, best_a = q, a
        policy[s - 1] = best_a
    return policy


# select solver
def main():
    start_time = time.time()   # --- START TIMER ---

    parser = argparse.ArgumentParser(description="MDP Solver")
    parser.add_argument("csv_path", help="Input CSV file (s,a,r,sp)")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--n_states", type=int, default=None)
    parser.add_argument("--n_actions", type=int, default=None)
    args = parser.parse_args()

    n_states, n_actions, gamma, samples, reward_sums, counts_sa, next_counts, avg_reward_by_action = \
        parse_problem(args.csv_path, args.n_states, args.n_actions, args.gamma)

    if n_states <= 1000:
        print("tabular value iteration", file=sys.stderr)
        policy = solve_small(n_states, n_actions, gamma,
                             reward_sums, counts_sa, next_counts, avg_reward_by_action)
    elif n_states <= 120_000:
        print("batch Q-iteration", file=sys.stderr)
        policy = solve_medium(n_states, n_actions, gamma,
                              samples, avg_reward_by_action)
    else:
        print("Fitted Q Iteration", file=sys.stderr)
        transitions_1b = [(s+1, a, r, sp+1) for (s,a,r,sp) in samples]
        policy = fitted_q_iteration(n_states, n_actions, gamma, transitions_1b)

    policy = np.clip(policy, 1, n_actions)
    if len(policy) != n_states:
        fixed = np.ones(n_states, dtype=int)
        m = min(len(policy), n_states)
        fixed[:m] = policy[:m]
        fixed[m:] = fixed[m - 1] if m > 0 else 1
        policy = fixed

    base, _ = os.path.splitext(args.csv_path)
    out_path = f"{base}.policy"
    with open(out_path, "w") as f:
        for a in policy:
            f.write(f"{int(a)}\n")

    elapsed = time.time() - start_time
    print(f"\nRuntime: {elapsed:.3f} seconds", file=sys.stderr)   


if __name__ == "__main__":
    main()
