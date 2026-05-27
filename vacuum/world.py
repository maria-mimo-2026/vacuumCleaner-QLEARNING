"""
world.py
________
"""

from vacuum.maps import Map
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame
import numpy as np
import collections
import logging
import time

GYM_NAME = "VacuumCleanerWorld-v0"
LOG_FILENAME = 'logfile.log'

WRONG_PROBA     = 0.17
INIT_DIRT_PROBA = 0.5

LIGHT_BROWN = (225, 193, 110)
LIGHT_GREY  = (211, 211, 211)
BLUE        = (0, 0, 255)
BLUE2       = (51, 51, 255)
RENDER_FPS  = 2
WINDOW_SIZE = 512


class VacuumCleanerWorldEnv(gym.Env):
    GYM_NAME = "VacuumCleanerWorld-v0"
    metadata = {"render_modes": ["human", "console"], "render_fps": RENDER_FPS}

    def __init__(self, grid, dirt_comeback, dirt_proba, murphy_proba,
                 location_sensor, episode_max_steps, render_mode):
        super(VacuumCleanerWorldEnv, self).__init__()
        self.map             = None
        self.init_map        = grid
        self.map_name        = "house"
        self.agent_name      = "vaccum cleaner robot"
        self.map_size        = grid.shape[0]
        self.dirt_comeback   = dirt_comeback
        self.dirt_proba      = dirt_proba
        self.murphy_proba    = murphy_proba
        self.location_sensor = location_sensor
        self.episode_max_steps = episode_max_steps
        self.render_mode     = render_mode

        self._n_dirty        = 0   # fast counter for termination check (O(1))

        self.nbr_rooms       = self.count_rooms()
        self._episode        = None
        self._step           = None
        self._episode_reward = None
        self._action_dict    = self.get_actions()
        self._current_action = None
        self._action_success = None
        self._suck_outcome   = None

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, self.map_size, shape=(2,), dtype=np.int16),
            "dirt":  spaces.Discrete(2),
        })
        self.action_space = spaces.Discrete(6)

        self._action_to_direction = {
            2: np.array([0,  1], dtype=np.int16),   # down
            3: np.array([1,  0], dtype=np.int16),   # right
            4: np.array([0, -1], dtype=np.int16),   # up
            5: np.array([-1, 0], dtype=np.int16),   # left
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = WINDOW_SIZE
        self.window      = None
        self.clock       = None

        logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    # ──────────────────────────────────────────────────────────────
    # setters
    # ──────────────────────────────────────────────────────────────
    def set_agent_name(self, name):  self.agent_name = name
    def set_map_name(self, name):    self.map_name   = name
    def set_frame_rate(self, fps):   self.metadata["render_fps"] = fps

    # ──────────────────────────────────────────────────────────────
    def _get_obs(self):
        l = self._agent_location
        dirty = self.map[l[1], l[0]] == 'x'
        self.logger.info("obs: {},{}".format(l, dirty))
        return {"agent": l, "dirt": dirty}

    def _get_info(self):
        return {
            'action_success': self._action_success,
            'dirty_spots':    self._n_dirty,
            'step':           self._step,
        }

    # ──────────────────────────────────────────────────────────────
    def get_rewards(self):
        return {
            'clean':   2.0,
            'cleaned': 10.0,   
            'dirty':   -2.0,
            'suck':    -0.5,
            'move':    -0.2,
            'throw':   -5.0,
            'none':    0.0,
        }

    def get_actions(self):
        return {0: "none", 1: "suck", 2: "down", 3: "right", 4: "up", 5: "left"}

    def get_episode_reward(self):
        return self._episode_reward

    # ──────────────────────────────────────────────────────────────
    def count_rooms(self, clean=None):
        world    = self.map if self.map is not None else self.init_map
        counters = collections.Counter(world.flatten())
        if clean is None:
            return counters['.'] + counters['x']
        if clean:
            return counters['.']
        return counters['x']

    # ──────────────────────────────────────────────────────────────
    def sample_location(self):
        while True:
            location = self.np_random.integers(0, self.map_size, size=2, dtype=np.int16)
            x, y = location
            if self.map[y, x] != '#':
                break
        return location

    def sample_dirt(self, proba=None):
        if proba is None:
            proba = self.dirt_proba
        row, col = self.map.shape
        for i in range(row):
            for j in range(col):
                if self.map[i, j] != '.':
                    continue
                if self.np_random.random() < proba:
                    self.map[i, j] = 'x'
                    self._total_dirty += 1
                    self._n_dirty     += 1   # update fast counter

    def simulate_action(self, action):
        if self.murphy_proba is not None:
            if self.np_random.random() < self.murphy_proba:
                self.logger.info(f"action '{self._action_dict[action]}' failed!")
                self._failures += 1
                return False
        return True

    # ──────────────────────────────────────────────────────────────
    # reset
    # ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._episode is None:
            self._episode = 0
        else:
            self._episode += 1

        self._step           = 0
        self._episode_reward = 0
        self._n_dirty        = 0
        # FIX: copy avoids mutating init_map across episodes
        self.map             = self.init_map.copy()
        self._total_dirty    = 0
        self._total_cleaned  = 0
        self._total_messed   = 0
        self._total_travel   = 0
        self._failures       = 0

        self.sample_dirt(proba=INIT_DIRT_PROBA)   # updates _n_dirty internally

        self.nbr_rooms    = self.count_rooms()
        clean             = self.count_rooms(clean=True)
        dirty             = self.nbr_rooms - clean
        self._total_dirty = dirty
        self._n_dirty     = dirty   # sync after initial dirt sample

        self.logger.info("env reset: rooms {}, clean {}, dirty {}".format(
            self.nbr_rooms, clean, dirty))

        self._agent_location = self.sample_location()
        observation = self._get_obs()
        info        = self._get_info()

        if self.render_mode == "human":     self._render_frame()
        elif self.render_mode == "console": self._render_console()
        return observation, info

    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        self._current_action = action
        self._suck_outcome   = None
        reward               = 0

        if action == 0:                          # none
            reward               = self.get_rewards()['none']
            self._action_success = True

        elif action == 1:                        # suck
            penalty = self.get_rewards()['suck']
            x, y    = self._agent_location
            if self.map[y, x] == 'x':
                self._action_success = self.simulate_action(action)
                if self._action_success:
                    self.map[y, x]      = '.'
                    self._suck_outcome  = 'cleaned'
                    self._total_cleaned += 1
                    self._n_dirty       -= 1   # fast counter update
                    reward = penalty + self.get_rewards()['cleaned']
                    self.logger.info("room {} cleaned!".format((x, y)))
                else:
                    self._suck_outcome = 'nothing'
            else:
                if self.map[y, x] != '.':
                    self._action_success = False
                    self._suck_outcome   = 'nothing'
                else:
                    self._action_success = self.simulate_action(action)
                    if not self._action_success:
                        self.map[y, x]      = 'x'
                        self._total_dirty  += 1
                        self._total_messed += 1
                        self._n_dirty      += 1   # fast counter update
                        self._suck_outcome  = 'messed'
                        reward = penalty + self.get_rewards()['throw']
                        self.logger.info("agent throws dirt in {}!".format((x, y)))
                    else:
                        self._suck_outcome = 'nothing'

        else:                                    # movement
            assert action in (2, 3, 4, 5)
            reward               = self.get_rewards()["move"]
            self._action_success = self.simulate_action(action)
            if self._action_success:
                self._total_travel += 1
                direction    = self._action_to_direction[action]
                new_location = np.clip(
                    self._agent_location + direction, 0, self.map_size - 1)
                if not np.array_equal(self._agent_location, new_location):
                    if self.map[new_location[1], new_location[0]] != '#':
                        self._agent_location = new_location
                    else:
                        self._action_success = False
                else:
                    self._action_success = False

        # ── dense reward each step ──────────────────────────────
        terminated = False
        self._step += 1
        truncated  = (self._step == self.episode_max_steps)

        nbr_clean = self.nbr_rooms - self._n_dirty   # O(1) using counter
        reward = reward + nbr_clean * self.get_rewards()['clean'] \
                        + self._n_dirty * self.get_rewards()['dirty']
        self._episode_reward = round(self._episode_reward + reward, 2)

        if self.dirt_comeback:
            self.sample_dirt()

        observation = self._get_obs()
        info        = self._get_info()

        if self.render_mode == "human":     self._render_frame()
        elif self.render_mode == "console": self._render_console()

        return observation, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode == "console": self._render_console()
        elif self.render_mode == "human": return self._render_frame()

    def _render_console(self):
        l = self._agent_location
        print("agent room: ({})".format((l[0], l[1])))
        print("world map: \n ", self.map)

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.upbar_size   = 70
            self.downbar_size = 110
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + self.upbar_size + self.downbar_size))
            pygame.display.set_caption("Vacuum Cleaner World-v0 (OpenAI Gym)")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas   = pygame.Surface((self.window_size, self.window_size))
        bg_color = (255, 255, 255)
        canvas.fill(bg_color)
        self.window.fill(bg_color)
        font0_name = "Arial"

        font            = pygame.font.SysFont(font0_name, 18, True)
        text_surface1_1 = font.render('agent',  True, "black")
        text_surface1_2 = font.render('reward', True, "black")
        font            = pygame.font.SysFont(font0_name, 27, False)
        text_surface1_3 = font.render(f'{self.agent_name}',      True, "black")
        text_surface1_4 = font.render(f'{self._episode_reward}', True, "black")
        rect_1 = text_surface1_1.get_rect(topleft=(10, 5))
        rect_2 = text_surface1_2.get_rect(topright=(self.window_size - 10, 5))
        rect_3 = text_surface1_3.get_rect(topleft=tuple(map(sum, zip(rect_1.bottomleft, (0, 5)))))
        rect_4 = text_surface1_4.get_rect(topright=tuple(map(sum, zip(rect_2.bottomright, (0, 5)))))

        font            = pygame.font.SysFont(font0_name, 16, True)
        text_surface2_1 = font.render(
            'Episode {}, Step {}:'.format(self._episode, self._step), True, "black")
        text_surface3_1 = font.render(f"Map '{self.map_name}':", True, "black")
        font            = pygame.font.SysFont(font0_name, 18, False)
        text_surface2_2 = font.render(
            f'Dirt {self._total_dirty}  Clean {self._total_cleaned}'
            f'  Mess {self._total_messed}  Travel {self._total_travel}'
            f'  Fail {self._failures}', True, "black")
        font = pygame.font.SysFont(font0_name, 16, False)
        cfg2 = (f'dirt_comeback:{self.dirt_comeback}  P_dirt={self.dirt_proba}'
                if self.dirt_comeback else f"dirt_comeback:{self.dirt_comeback}")
        cfg2 += f'  P_wrong={self.murphy_proba}  loc_sensor:{self.location_sensor}'
        text_surface3_2 = font.render(cfg2, True, "black")

        pix = self.window_size / self.map_size

        for loc in self._map_locations('x'):
            pygame.draw.rect(canvas, LIGHT_BROWN,
                             pygame.Rect(pix * loc, (pix, pix)))

        center   = (self._agent_location + 0.5) * pix
        radius   = pix / 4
        ag_color = BLUE2
        if self._current_action == 1:
            if self._suck_outcome == 'cleaned':  ag_color = LIGHT_BROWN
            elif self._suck_outcome == 'messed': ag_color = "white"
        elif self._current_action not in (0, None):
            ag_color = BLUE2 if self._action_success else "red"
        pygame.draw.circle(canvas, ag_color, center, radius)

        if self._current_action in (2, 3, 4, 5):
            x, y = center
            p1, p2, p3 = self.get_triangle_points(self._current_action, x, y, radius)
            pygame.draw.polygon(canvas, "yellow", (p1, p2, p3))

        for loc in self._map_locations('#'):
            pygame.draw.rect(canvas, (0, 0, 0),
                             pygame.Rect(pix * loc, (pix, pix)))

        for i in range(self.map_size + 1):
            pygame.draw.line(canvas, 0, (0, pix*i), (self.window_size, pix*i), width=3)
            pygame.draw.line(canvas, 0, (pix*i, 0), (pix*i, self.window_size), width=3)

        self.window.blit(text_surface1_1, rect_1)
        self.window.blit(text_surface1_2, rect_2)
        self.window.blit(text_surface1_3, rect_3)
        self.window.blit(text_surface1_4, rect_4)
        self.window.blit(canvas, (0, self.upbar_size))
        y = self.window_size + self.upbar_size + 10
        self.window.blit(text_surface2_1, (10, y)); y += text_surface2_1.get_height() + 5
        self.window.blit(text_surface2_2, (10, y)); y += text_surface2_2.get_height() + 10
        self.window.blit(text_surface3_1, (10, y)); y += text_surface3_1.get_height() + 5
        self.window.blit(text_surface3_2, (10, y))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit(); pygame.quit(); exit()

    def get_triangle_points(self, action, x, y, r):
        match action:
            case 2: return (x, y+r), (x-r, y), (x+r, y)
            case 3: return (x, y-r), (x, y+r), (x+r, y)
            case 4: return (x-r, y), (x, y-r), (x+r, y)
            case 5: return (x-r, y), (x, y+r), (x, y-r)

    def _map_locations(self, symbol):
        l = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map[i, j] == symbol:
                    l.append(np.array([j, i], dtype=np.int16))
        return l

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()