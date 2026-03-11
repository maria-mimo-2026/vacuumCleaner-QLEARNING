import numpy as np
import gymnasium as gym
from vacuum.world import VacuumCleanerWorldEnv
from vacuum.maps import Map
from vacuum.policy.qlearning_2  import QLearnPolicy
from vacuum.policy.greedyrandom import GreedyRandomPolicy

N_RUNS    = 10     
MAX_STEPS = 300   

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
            'grid':             None,
            'dirt_comeback':    False,  
            'dirt_proba':       0.09,
            'murphy_proba':     0.11,    
            'location_sensor':  True,
            'episode_max_steps': MAX_STEPS,
            'render_mode':      None,    
        }
    )
    env = gym.make('VacuumCleanerWorld-v0', grid=wmap)
    env.unwrapped.set_map_name(map_id)
    return env
def run_experiments(env, policy, n=N_RUNS):
   
    results = []
    print(f"  [{policy.policy_id}] running {n} experiments...")

    seed = 0
    for _ in range(n):
        obs, info = env.reset(seed=seed)
        policy.reset(seed=seed) 
        done = False

        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        dirty_left = info['dirty_spots']
        nbr_rooms  = env.unwrapped.nbr_rooms
        success    = (dirty_left == 0)

        results.append({
            'reward':  round(env.unwrapped._episode_reward, 2),
            'travel':  env.unwrapped._total_travel,
            'cleaned': env.unwrapped._total_cleaned,
            'success': success,
        })

        status = "OK  " if success else "FAIL"
        print(f"    [{status}] seed={seed}"
              f"  reward={results[-1]['reward']:>8.1f}"
              f"  travel={results[-1]['travel']:>4}"
              f"  cleaned={results[-1]['cleaned']}/{nbr_rooms}")

    avg_reward   = round(np.mean([r['reward']  for r in results]), 1)
    avg_travel   = round(np.mean([r['travel']  for r in results]), 1)
    avg_cleaned  = round(np.mean([r['cleaned'] for r in results]), 1)
    success_rate = sum(1 for r in results if r['success']) / n * 100

    print(f"  → avg: reward={avg_reward}  travel={avg_travel}"
          f"  cleaned={avg_cleaned}  success={success_rate:.0f}%")

    return {
        'avg_reward':   avg_reward,
        'avg_travel':   avg_travel,
        'avg_cleaned':  avg_cleaned,
        'success_rate': success_rate,
    }


def compare_one_map(map_id):
    print(f"\n{'='*60}")
    print(f"  MAP: {map_id}")
    print(f"{'='*60}")

    # ── Q-Learning ──
    env_ql = make_env(map_id)
    ql = QLearnPolicy(map_id, env_ql)
    if not ql.load_qtable():
        print(f"  [!] Q-table Not available for{map_id}")
        print(f"  start first: python main.py [map_n] 3 -e 1 300 -cl")
        env_ql.close()
        return None
    ql_stats = run_experiments(env_ql, ql)
    env_ql.close()

    # ── Greedy-Random ──
    env_gr = make_env(map_id)
    gr = GreedyRandomPolicy(map_id, env_gr, eco=True)
    gr_stats = run_experiments(env_gr, gr)
    env_gr.close()

    return ql_stats, gr_stats


def print_table(all_results):
    print("\n\n" + "="*75)
    print(f"  {'MAP':<22} {'POLICY':<16}"
          f" {'Reward':>8} {'Travel':>8} {'Cleaned':>9} {'Success':>8}")
    print("="*75)

    for map_id, stats in all_results.items():
        if stats is None:
            print(f"  {map_id:<22} [Q-table missing — skipped]")
            print("-"*75)
            continue

        ql_stats, gr_stats = stats

        print(f"  {map_id:<22} {'Q-Learning':<16}"
              f" {ql_stats['avg_reward']:>8.1f}"
              f" {ql_stats['avg_travel']:>8.1f}"
              f" {ql_stats['avg_cleaned']:>9.1f}"
              f" {ql_stats['success_rate']:>7.0f}%")

        print(f"  {'':22} {'Greedy-Random':<16}"
              f" {gr_stats['avg_reward']:>8.1f}"
              f" {gr_stats['avg_travel']:>8.1f}"
              f" {gr_stats['avg_cleaned']:>9.1f}"
              f" {gr_stats['success_rate']:>7.0f}%")

        print("-"*75)

    print("="*75)
    print(f"  * {N_RUNS} runs per policy | murphy_proba=0.11 | dirt_comeback=False")
    print("="*75)


#  MAIN 
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        selected = []
        for arg in sys.argv[1:]:
            matches = [m for m in MAPS if arg in m]
            selected.extend(matches)
        maps_to_run = selected if selected else MAPS
    else:
        maps_to_run = MAPS

    print(f"Running comparison on {len(maps_to_run)} maps, {N_RUNS} runs each...")
    print(f"Maps: {maps_to_run}")

    all_results = {}
    for map_id in maps_to_run:
        all_results[map_id] = compare_one_map(map_id)

    print_table(all_results)