import numpy as np


def run_10_experiments(env, policy, n_experiments=10):
    """
    Run n_experiments simulations and return averages.
    """
    results = []
    print(f"\n[info] Running {n_experiments} experiments - policy: {policy.policy_id}")
    print("-" * 55)
    seed = 0
    for _ in range(n_experiments):
        obs, info = env.reset(seed=seed)
        policy.reset(seed=seed) 
        done = False

        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        total_reward  = round(env.unwrapped._episode_reward, 2)
        total_travel  = env.unwrapped._total_travel
        total_cleaned = env.unwrapped._total_cleaned
        nbr_rooms     = env.unwrapped.nbr_rooms
        dirty_left    = info['dirty_spots']
        success       = (nbr_rooms - dirty_left) == nbr_rooms

        results.append({
            'seed':    seed,
            'reward':  total_reward,
            'travel':  total_travel,
            'cleaned': total_cleaned,
            'success': success,
        })

        status = "OK" if success else "FAIL"
        print(f"  [{status}] seed={seed}  reward={total_reward:>8.1f}"
              f"  travel={total_travel:>4}  cleaned={total_cleaned}")

    avg_reward   = round(np.mean([r['reward']  for r in results]), 2)
    avg_travel   = round(np.mean([r['travel']  for r in results]), 1)
    avg_cleaned  = round(np.mean([r['cleaned'] for r in results]), 1)
    success_rate = sum(1 for r in results if r['success']) / n_experiments * 100

    print("-" * 55)
    print(f"  avg_reward   = {avg_reward}")
    print(f"  avg_travel   = {avg_travel}")
    print(f"  avg_cleaned  = {avg_cleaned}")
    print(f"  success_rate = {success_rate:.0f}%")

    return {
        'avg_reward':   avg_reward,
        'avg_travel':   avg_travel,
        'avg_cleaned':  avg_cleaned,
        'success_rate': success_rate,
        'all_results':  results,
    }


def compare_policies(env, ql_policy, gr_policy, n_experiments=10):
    """
    Compare Q-learning vs Greedy-Random and print results table.
    """
    print("\n" + "=" * 55)
    print("  POLICY COMPARISON")
    print("=" * 55)

    ql_stats = run_10_experiments(env, ql_policy, n_experiments)
    gr_stats = run_10_experiments(env, gr_policy, n_experiments)

    print("\n" + "=" * 55)
    print(f"  {'Metric':<18} {'Q-Learning':>12} {'Greedy-Random':>14}")
    print("-" * 55)
    print(f"  {'avg_reward':<18} {ql_stats['avg_reward']:>12.1f} {gr_stats['avg_reward']:>14.1f}")
    print(f"  {'avg_travel':<18} {ql_stats['avg_travel']:>12.1f} {gr_stats['avg_travel']:>14.1f}")
    print(f"  {'avg_cleaned':<18} {ql_stats['avg_cleaned']:>12.1f} {gr_stats['avg_cleaned']:>14.1f}")
    print(f"  {'success_rate':<18} {ql_stats['success_rate']:>11.0f}% {gr_stats['success_rate']:>13.0f}%")
    print("=" * 55)

    return {'q-learning': ql_stats, 'greedy-random': gr_stats}