import numpy as np
import gymnasium as gym
import sys

from vacuum.world import VacuumCleanerWorldEnv
from vacuum.maps import Map
from vacuum.policy.helpers import make_policy

N_RUNS    = 10
MAX_STEPS = 300

POLICY_ID_QL1 = "q-learning_1"
POLICY_ID_QL2 = "q-learning_2"

MAPS = [
    "vacuum-5rooms-v0", "vacuum-5rooms-v1", "vacuum-5rooms-v2", "vacuum-5rooms-v3", "vacuum-5rooms-v4",
    "vacuum-6rooms-v0", "vacuum-6rooms-v1", "vacuum-6rooms-v2",
    "vacuum-7rooms-v0",
    "vacuum-8rooms-v0", "vacuum-8rooms-v1",
    "vacuum-9rooms-v0",
    "vacuum-12rooms-v0",
    "vacuum-5x5-v0",
    "vacuum-6x6-v0",
]


def make_env(map_id):
    if "VacuumCleanerWorld-v0" in gym.envs.registry:
        del gym.envs.registry["VacuumCleanerWorld-v0"]
    world_map = Map.load_map(map_id)
    wmap = np.array(world_map, dtype='<U1')
    gym.register(
        id="VacuumCleanerWorld-v0",
        entry_point="vacuum.world:VacuumCleanerWorldEnv",
        kwargs={
            'grid': None, 'dirt_comeback': False, 'dirt_proba': 0.09,
            'murphy_proba': 0.11, 'location_sensor': True,
            'episode_max_steps': MAX_STEPS, 'render_mode': None,
        }
    )
    env = gym.make('VacuumCleanerWorld-v0', grid=wmap)
    env.unwrapped.set_map_name(map_id)
    return env


def run_tests(env, policy, label, n=N_RUNS):
    results = []
    print(f"  [{label}] testing {n} seeds...")
    seed = 0
    for _ in range(n):
        obs, info = env.reset(seed=seed)
        policy.reset(seed=seed) 
        done = False
        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        nbr_rooms = env.unwrapped.nbr_rooms
        dirty_left = info['dirty_spots']
        results.append({
            'reward':  round(env.unwrapped._episode_reward, 2),
            'travel':  env.unwrapped._total_travel,
            'cleaned': env.unwrapped._total_cleaned,
            'success': dirty_left == 0,
        })
        status = "OK  " if results[-1]['success'] else "FAIL"
        print(f"    [{status}] seed={seed}"
              f"  reward={results[-1]['reward']:>8.1f}"
              f"  travel={results[-1]['travel']:>4}"
              f"  cleaned={results[-1]['cleaned']}/{nbr_rooms}")

    avg_reward   = round(np.mean([r['reward']  for r in results]), 1)
    avg_travel   = round(np.mean([r['travel']  for r in results]), 1)
    avg_cleaned  = round(np.mean([r['cleaned'] for r in results]), 1)
    success_rate = sum(1 for r in results if r['success']) / n * 100
    print(f"  -> avg: reward={avg_reward}  travel={avg_travel}"
          f"  cleaned={avg_cleaned}  success={success_rate:.0f}%")
    return {'avg_reward': avg_reward, 'avg_travel': avg_travel,
            'avg_cleaned': avg_cleaned, 'success_rate': success_rate}


def compare_one_map(map_id):
    print(f"\n{'='*60}")
    print(f"  MAP: {map_id}")
    print(f"{'='*60}")

    results = []
    for policy_id, label in [
        (POLICY_ID_QL1, "QL1 "),
        (POLICY_ID_QL2, "QL2 "),
    ]:
        env = make_env(map_id)
        try:
            policy = make_policy(policy_id, map_id, env, eco_mode=False)
        except Exception as e:
            print(f"  [!] Failed to create policy {policy_id}: {e}")
            env.close()
            results.append(None)
            continue

        policy.reset(seed=0)

        if not policy.load_qtable():
            if policy_id == POLICY_ID_QL1:
                print(f"  [!] Q-table ({label}) not found for {map_id}")
                print(f"      Train first: python main.py [map_n] 3 -e 1 300 -cl")
            else:
                print(f"  [!] Q-table ({label}) not found for {map_id}")
                print(f"      Train first: python main.py [map_n] 4 -e 1 300 -cl")
            env.close()
            results.append(None)
            continue

        stats = run_tests(env, policy, label)
        env.close()
        results.append(stats)

    if results[0] is None or results[1] is None:
        return None
    return results[0], results[1]


def print_table(all_results):
    print("\n\n" + "="*78)
    print(f"  {'MAP':<22} {'VERSION':<18}"
          f" {'Reward':>8} {'Travel':>8} {'Cleaned':>9} {'Success':>8}")
    print("="*78)
    for map_id, stats in all_results.items():
        if stats is None:
            print(f"  {map_id:<22} [Q-table missing -- skipped]")
            print("-"*78)
            continue
        s1, s2 = stats
        print(f"  {map_id:<22} {'QL1':<18}"
              f" {s1['avg_reward']:>8.1f} {s1['avg_travel']:>8.1f}"
              f" {s1['avg_cleaned']:>9.1f} {s1['success_rate']:>7.0f}%")
        print(f"  {'':22} {'QL2':<18}"
              f" {s2['avg_reward']:>8.1f} {s2['avg_travel']:>8.1f}"
              f" {s2['avg_cleaned']:>9.1f} {s2['success_rate']:>7.0f}%")
        d_travel  = round(s2['avg_travel']  - s1['avg_travel'],  1)
        d_success = round(s2['success_rate'] - s1['success_rate'], 0)
        d_reward  = round(s2['avg_reward']  - s1['avg_reward'],  1)
        sign_t = "+" if d_travel  >= 0 else ""
        sign_s = "+" if d_success >= 0 else ""
        sign_r = "+" if d_reward  >= 0 else ""
        print("-"*78)
    print("="*78)
    print(f"  * {N_RUNS} test runs | murphy_proba=0.11 | dirt_comeback=False")
    print(f"  * QL1 = qlearning_1.py| QL2 = qlearning_2.py")
    print("="*78)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        selected = []
        for arg in sys.argv[1:]:
            matches = [m for m in MAPS if arg in m]
            selected.extend(matches)
        maps_to_run = selected if selected else MAPS
    else:
        maps_to_run = MAPS

    print(f"Comparing QL1 vs QL2  on {len(maps_to_run)} maps...")
    all_results = {}
    for map_id in maps_to_run:
        all_results[map_id] = compare_one_map(map_id)
    print_table(all_results)