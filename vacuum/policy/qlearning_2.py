from vacuum.policy.base import CleanPolicy
import gymnasium as gym
import os, pickle
import numpy as np
from collections import deque
from tqdm import tqdm
from constants import VISITS_CAP
from .registry import register_policy

# ============================================================
TRAIN_EPISODES = 15000
EPISODE_STEPS  = 300

ALPHA         = 0.4
GAMMA         = 0.97
EPSILON       = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.9994

R_CLEAN = 8.0    
R_NEW   = 0.5    
R_DONE  = 10.0  
P_SEEN  = -0.3  
P_LOOP  = -2.0   
LOOP_WIN = 6
# ============================================================

# world.py: agent[0]=X, agent[1]=Y, map[Y, X]
_MOVE_DXDY = {2:(0,1), 3:(1,0), 4:(0,-1), 5:(-1,0)}
_MOVE_LIST  = [(2,0,1),(3,1,0),(4,0,-1),(5,-1,0)]  # (action, dx, dy)


def _bfs_to_unvisited(sx, sy, visited, wmap, n):
   
    if not visited[sy, sx]:
        return None 

    q    = deque()
    seen = set()
    seen.add((sx, sy))

    for act, dx, dy in _MOVE_LIST:
        nx, ny = sx+dx, sy+dy
        if not (0 <= nx < n and 0 <= ny < n): continue
        if wmap[ny, nx] == '#': continue
        if not visited[ny, nx]:
            return act          
        if (nx, ny) not in seen:
            seen.add((nx, ny))
            q.append((nx, ny, act)) 
    while q:
        x, y, first_act = q.popleft()
        for _, dx, dy in _MOVE_LIST:
            nx, ny = x+dx, y+dy
            if not (0 <= nx < n and 0 <= ny < n): continue
            if wmap[ny, nx] == '#': continue
            if (nx, ny) in seen: continue
            seen.add((nx, ny))
            if not visited[ny, nx]:
                return first_act    
            q.append((nx, ny, first_act))

    return None  


@register_policy("q-learning_2")
class QLearnPolicy(CleanPolicy):

    def __init__(self, world_id, env):
        super().__init__("q-learning_2", world_id, env)
        self.trained       = False
        self.q_table       = None
        self._rng          = np.random.default_rng()
        self._n            = self.env.unwrapped.map_size
        self.map_dimension = self._n
        self._vcache       = {}  
        self._wmap         = None 

        n = self._n
        self.visits      = np.zeros((n, n), dtype=np.int16)
        self.bfs_visited = np.zeros((n, n), dtype=bool)
        self.pos_history = []
        self.steps_stuck = 0
        self._last_action = None   
        self._last_pos    = None   
    # ──────────────────────────────────────────────────────
    def _build_vcache(self):
        n = self._n
        try:
            self._wmap = self.env.unwrapped.init_map
        except Exception:
            self._wmap = None

        for ax in range(n):
            for ay in range(n):
                valid = []
                for act, dx, dy in _MOVE_LIST:
                    nx, ny = ax+dx, ay+dy
                    if not (0 <= nx < n and 0 <= ny < n): continue
                    if self._wmap is not None:
                        try:
                            if self._wmap[ny, nx] == '#': continue
                        except Exception:
                            pass
                    valid.append(act)
                self._vcache[(ax, ay)] = valid if valid else [1]

    def _enc(self, x, y, dirt):
        v = int(min(self.visits[x, y], VISITS_CAP - 1))
        return ((x * self._n + y) * VISITS_CAP + v) * 2 + dirt

    def _encode_state(self, s):
        x, y = int(s['agent'][0]), int(s['agent'][1])
        return self._enc(x, y, int(s['dirt']))
    def get_train_config(self):
        return TRAIN_EPISODES, EPISODE_STEPS

    # ──────────────────────────────────────────────────────
    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.visits.fill(0)
        self.bfs_visited.fill(False)
        self.pos_history = []
        self.steps_stuck = 0
        self._last_action = None
        self._last_pos    = None

    # ──────────────────────────────────────────────────────
    def load_qtable(self):
        fname = f"data/qlearning_table_map_{self.world_id}_v2.pkl"
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data.get('q', data) if isinstance(data, dict) else data
            self.trained = True
            if not self._vcache:
                self._build_vcache()
            print("[info] Q-table loaded")
        return self.trained

    def _save_qtable(self):
        os.makedirs("data", exist_ok=True)
        with open(f"data/qlearning_table_map_{self.world_id}_v2.pkl", 'wb') as f:
            pickle.dump(self.q_table, f)
        print("[info] Q-table saved")

    # ──────────────────────────────────────────────────────
    def select_action(self, state):
        if self.q_table is None:
            return self.env.action_space.sample()

        x, y = int(state['agent'][0]), int(state['agent'][1])
        dirt = int(state['dirt'])

        if (self._last_action is not None and
                self._last_pos is not None and
                self._last_pos == (x, y) and
                self._last_action in (2, 3, 4, 5)):
            self._last_pos = (x, y)
            return self._last_action

        self.visits[x, y]      = min(self.visits[x, y] + 1, VISITS_CAP - 1)
        self.bfs_visited[y, x] = True  

        if dirt:
            self._last_action = 1
            self._last_pos    = (x, y)
            return 1

        try:
            if self.env.unwrapped._n_dirty == 0:
                self._last_action = 0
                self._last_pos    = (x, y)
                return 0
        except Exception:
            pass

        n    = self._n
        wmap = self._wmap if self._wmap is not None else self.env.unwrapped.init_map

        bfs_act = _bfs_to_unvisited(x, y, self.bfs_visited, wmap, n)
        if bfs_act is not None:
            self._last_action = bfs_act
            self._last_pos    = (x, y)
            return bfs_act

        sidx  = self._enc(x, y, 0)
        valid = self._vcache.get((x, y), [3, 5, 2, 4])
        q     = self.q_table[sidx]
        act   = max(valid, key=lambda a: q[a])
        self._last_action = act
        self._last_pos    = (x, y)
        return act

    # ──────────────────────────────────────────────────────
    def train_q_learning(self, env, episodes=TRAIN_EPISODES):
        self._build_vcache()

        orig_murphy = env.unwrapped.murphy_proba
        env.unwrapped.murphy_proba = None   

        env_tl    = gym.wrappers.TimeLimit(env, max_episode_steps=EPISODE_STEPS)
        n_actions = env_tl.action_space.n
        n         = self._n
        vcap      = VISITS_CAP
        n_states  = n * n * vcap * 2

        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        self.q_table[:, 0] = -10.0 

        Q      = self.q_table
        eps    = float(EPSILON)
        alpha  = float(ALPHA)
        gamma  = float(GAMMA)
        one_a  = 1.0 - alpha
        rng    = self._rng
        vc     = self._vcache
        n_     = n
        vcap_  = vcap
        wmap   = self._wmap

        rbuf = np.zeros(300, dtype=np.float32)
        ri   = 0
        reward_log = []
        clean_log  = []
        travel_log = []
        epsilon_log = []
        ep_travel  = 0
        ep_clean   = 0

        for ep in tqdm(range(episodes), desc="Training", unit="ep"):
            obs, _ = env_tl.reset()
            visits      = self.visits
            bfs_visited = self.bfs_visited
            visits.fill(0)
            bfs_visited.fill(False)
            ph  = []
            stk = 0

            x, y = int(obs['agent'][0]), int(obs['agent'][1])
            d    = int(obs['dirt'])
            v    = int(min(visits[x, y], vcap_-1))
            si   = ((x*n_+y)*vcap_+v)*2+d
            ep_r = 0.0
            done = False

            while not done:
                valid = vc.get((x, y), [3,5,2,4])

                if d:
                    act = 1  
                elif rng.random() < eps:
                    unvis = [a for a,(dx,dy) in _MOVE_DXDY.items()
                             if a in valid and
                             0<=x+dx<n_ and 0<=y+dy<n_ and
                             not bfs_visited[y+dy, x+dx]]
                    pool = unvis if unvis else valid
                    act  = pool[int(rng.integers(len(pool)))]
                else:
                    qr  = Q[si]
                    act = valid[0]; bv = qr[valid[0]]
                    for a in valid[1:]:
                        if qr[a] > bv: bv = qr[a]; act = a
                # ────────────────────────────────────────

                pd = d
                obs, rew, term, trunc, _ = env_tl.step(act)
                done = term or trunc

                nx, ny = int(obs['agent'][0]), int(obs['agent'][1])
                d      = int(obs['dirt'])

                vp = int(visits[nx, ny])
                if vp < vcap_-1: visits[nx, ny] = vp+1
                bfs_visited[ny, nx] = True
                if act in (2, 3, 4, 5): ep_travel += 1
                if act == 1 and pd and not d: ep_clean += 1

                r = float(rew)
                if act == 1 and pd and not d: r += R_CLEAN
                r += R_NEW if vp == 0 else P_SEEN * min(vp, 4)

                ph.append((nx, ny))
                if len(ph) > LOOP_WIN:
                    ph.pop(0)
                    if len(set(ph)) <= 2: r += P_LOOP; stk += 1
                    else: stk = 0
                if term:
                  steps_used = len(ph)
                  efficiency_bonus = 0.3 * max(0, 300 - steps_used)
                  r += R_DONE + efficiency_bonus
                # ────────────────────────────────────────

                nv = int(min(visits[nx, ny], vcap_-1))
                ni = ((nx*n_+ny)*vcap_+nv)*2+d
                bq = float(Q[ni].max())
                Q[si, act] = one_a*Q[si, act] + alpha*(r + gamma*bq)

                si = ni; x, y = nx, ny
                ep_r += r

            eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
            epsilon_log.append(eps)
            reward_log.append(ep_r)
            clean_log.append(ep_clean)
            travel_log.append(ep_travel)
            ep_travel = 0
            ep_clean  = 0 
            rbuf[ri % 300] = ep_r; ri += 1

            if (ep+1) % 5000 == 0:
                avg = float(rbuf.mean()) if ri >= 300 else float(rbuf[:ri].mean())
                print(f"  Ep {ep+1:>6} | ε={eps:.4f} | avg_r={avg:.1f}")

        env.unwrapped.murphy_proba = orig_murphy
        from tools import Tools
        Tools.save_training_results(
     self.world_id,
     "q-learning_2",
    {'epsilon': epsilon_log ,'reward': reward_log, 'cleaned': clean_log, 'travel': travel_log }
     )
        from tools import Tools
        Tools.plot_epsilon(episodes, epsilon_log)
        self._save_qtable()  
        self.trained = True
        print("\n✅ Training finished successfully!")