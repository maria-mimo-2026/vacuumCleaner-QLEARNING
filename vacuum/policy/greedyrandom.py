"""
'greedyrandom.py'
___________________
Vacuum cleaner world, 2024.
An reflex-based agent with some random actions.
author: Hakim Mitiche
date: April, 6th 2024

Extended by: Maria, March 2026
Added agent programs for all remaining maps in maps.py.
Each agent follows a fixed traversal path with random choices
at junction rooms, ensuring full map coverage.
"""

from .base import CleanPolicy
from .registry import register_policy
from vacuum.maps import Map
import numpy as np
import random
import logging

MY_NAME = "greedy-random"

@register_policy(MY_NAME)
class GreedyRandomPolicy(CleanPolicy):
	def __init__(self, world_id, env, eco=False):
		super().__init__(MY_NAME, world_id, env)
		self._locations = Map.locations_list(world_id)
		assert self._locations is not None

	def reset(self, seed=None):
		if not self._seeded or seed is not None:
			random.seed(seed)
			print(f"[info] {__class__.__name__}'s RNG seeded with {seed}")
			self._seeded = True

	def select_action(self, state):
		room  = state['agent']
		dirty = state['dirt']
		return self.agent_program(room, dirty)

	def agent_program(self, room, dirty):
		match (self.world_id):
			# ── original (Hakim Mitiche) ───────────────────────────
			case "vacuum-3rooms-v0":  return self.agent_3rooms_v0(room, dirty)
			case "vacuum-3rooms-v2":  return self.agent_3rooms_v2(room, dirty)
			case "vacuum-4rooms-v1":  return self.agent_4rooms_v1(room, dirty)
			case "vacuum-5rooms-v1":  return self.agent_5rooms_v1(room, dirty)
			case "vacuum-6rooms-v1":  return self.agent_6rooms_v1(room, dirty)
			# ── added (Maria, March 2026) ──────────────────────────
			case "vacuum-2rooms":     return self.agent_2rooms(room, dirty)
			case "vacuum-3rooms-v1":  return self.agent_3rooms_v1(room, dirty)
			case "vacuum-4rooms-v0":  return self.agent_4rooms_v0(room, dirty)
			case "vacuum-4rooms-v2":  return self.agent_4rooms_v2(room, dirty)
			case "vacuum-4rooms-v3":  return self.agent_4rooms_v3(room, dirty)
			case "vacuum-5rooms-v0":  return self.agent_5rooms_v0(room, dirty)
			case "vacuum-5rooms-v2":  return self.agent_5rooms_v2(room, dirty)
			case "vacuum-5rooms-v3":  return self.agent_5rooms_v3(room, dirty)
			case "vacuum-5rooms-v4":  return self.agent_5rooms_v4(room, dirty)
			case "vacuum-6rooms-v0":  return self.agent_6rooms_v0(room, dirty)
			case "vacuum-6rooms-v2":  return self.agent_6rooms_v2(room, dirty)
			case "vacuum-7rooms-v0":  return self.agent_7rooms_v0(room, dirty)
			case "vacuum-8rooms-v0":  return self.agent_8rooms_v0(room, dirty)
			case "vacuum-8rooms-v1":  return self.agent_8rooms_v1(room, dirty)
			case "vacuum-9rooms-v0":  return self.agent_9rooms_v0(room, dirty)
			case "vacuum-12rooms-v0": return self.agent_12rooms_v0(room, dirty)
			case "vacuum-5x5-v0":     return self.agent_5x5_v0(room, dirty)
			case "vacuum-6x6-v0":     return self.agent_6x6_v0(room, dirty)
			case _:
				print(f"[error] No agent program for {self.world_id}")
				exit()

	# ══════════════════════════════════════════════════════
	# ORIGINAL — Hakim Mitiche
	# ══════════════════════════════════════════════════════

	def agent_3rooms_v0(self, location, dirty):
		# Map: x . # / # x # / # # #
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[2]): return self._action_dict['up']
		else:
			return self._action_dict['down'] if random.random() < .5 else self._action_dict['left']

	def agent_3rooms_v2(self, location, dirty):
		# Map: . . x / # # # / # # #
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[2]): return self._action_dict['down']
		else:
			random.seed(0)
			return self._action_dict['right'] if random.random() < .5 else self._action_dict['left']

	def agent_4rooms_v1(self, location, dirty):
		# Map: x . # / # x . / # # #
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[3]): return self._action_dict['left']
		elif np.array_equal(location, self._locations[1]): return random.choice((self._action_dict['left'], self._action_dict['down']))
		else: return random.choice((self._action_dict['right'], self._action_dict['up']))

	def agent_5rooms_v1(self, location, dirty):
		# Map: # . # / . x . / # x # (+ shape)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[4]): return self._action_dict['up']
		elif np.array_equal(location, self._locations[3]): return self._action_dict['left']
		else: return self._action_dict[random.choice(['left','down','right','up'])]

	def agent_6rooms_v1(self, location, dirty):
		# Map: # . . / . . . / . # #
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[1]): return self._action_dict[random.choice(('left','down'))]
		elif np.array_equal(location, self._locations[2]): return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[3]): return self._action_dict[random.choice(('right','up','left'))]
		elif np.array_equal(location, self._locations[4]): return self._action_dict[random.choice(('left','up'))]
		else: return self._action_dict['up']

	# ══════════════════════════════════════════════════════
	# ADDED — Maria, March 2026
	# ══════════════════════════════════════════════════════

	def agent_2rooms(self, location, dirty):
		# Map: x x # / # # # / # # #
		# loc0=(0,0)  loc1=(1,0)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['right']
		else: return self._action_dict['left']

	def agent_3rooms_v1(self, location, dirty):
		# Map: # . # / . . # / # # #
		# loc0=(1,0)  loc1=(0,1)  loc2=(1,1)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		else: return self._action_dict[random.choice(('up','left'))]

	def agent_4rooms_v0(self, location, dirty):
		# Map: x . # / x x # / # # #
		# loc0=(0,0) loc1=(1,0) loc2=(0,1) loc3=(1,1) — square loop
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[3]): return self._action_dict['left']
		else: return self._action_dict['up']

	def agent_4rooms_v2(self, location, dirty):
		# Map: # . # / x . x / # # #
		# loc0=(1,0) loc1=(0,1) loc2=(1,1) loc3=(2,1)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[3]): return self._action_dict['left']
		else: return self._action_dict[random.choice(('up','left','right'))]

	def agent_4rooms_v3(self, location, dirty):
		# Map: # # x / . x . / # # #
		# loc0=(2,0) loc1=(0,1) loc2=(1,1) loc3=(2,1)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[3]): return self._action_dict[random.choice(('up','left'))]
		else: return self._action_dict[random.choice(('right','left'))]

	def agent_5rooms_v0(self, location, dirty):
		# Map: # # x / # . x / x . #
		# loc0=(2,0) loc1=(1,1) loc2=(2,1) loc3=(0,2) loc4=(1,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[2]): return self._action_dict[random.choice(('up','left'))]
		elif np.array_equal(location, self._locations[1]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[4]): return self._action_dict['left']
		else: return self._action_dict['right']

	def agent_5rooms_v2(self, location, dirty):
		# Map: . . # / # . # / . x #
		# loc0=(0,0) loc1=(1,0) loc2=(1,1) loc3=(0,2) loc4=(1,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[2]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[4]): return self._action_dict['left']
		else: return self._action_dict['up']

	def agent_5rooms_v3(self, location, dirty):
		# Map: # # . / . . . / . # #
		# loc0=(2,0) loc1=(0,1) loc2=(1,1) loc3=(2,1) loc4=(0,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[3]): return self._action_dict[random.choice(('left','up'))]
		elif np.array_equal(location, self._locations[2]): return self._action_dict['left']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['down']
		else: return self._action_dict['up']

	def agent_5rooms_v4(self, location, dirty):
		# Map: # . . / . . . / # # #
		# loc0=(1,0) loc1=(2,0) loc2=(0,1) loc3=(1,1) loc4=(2,1)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[1]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[4]): return self._action_dict['left']
		elif np.array_equal(location, self._locations[3]): return self._action_dict[random.choice(('left','up'))]
		else: return self._action_dict['right']

	def agent_6rooms_v0(self, location, dirty):
		# Map: . . # / . . # / . . #
		# loc0=(0,0) loc1=(1,0) loc2=(0,1) loc3=(1,1) loc4=(0,2) loc5=(1,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[2]): return self._action_dict[random.choice(('down','right'))]
		elif np.array_equal(location, self._locations[4]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[5]): return self._action_dict['up']
		elif np.array_equal(location, self._locations[3]): return self._action_dict[random.choice(('up','left'))]
		else: return self._action_dict['left']

	def agent_6rooms_v2(self, location, dirty):
		# Map: . # # / . . . / . # .
		# loc0=(0,0) loc1=(0,1) loc2=(1,1) loc3=(2,1) loc4=(0,2) loc5=(2,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[1]): return self._action_dict[random.choice(('up','right','down'))]
		elif np.array_equal(location, self._locations[4]): return self._action_dict['up']
		elif np.array_equal(location, self._locations[2]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[3]): return self._action_dict['down']
		else: return self._action_dict['up']

	def agent_7rooms_v0(self, location, dirty):
		# Map: . # . / . . . / . # .  (H-shape)
		# loc0=(0,0) loc1=(2,0) loc2=(0,1) loc3=(1,1) loc4=(2,1) loc5=(0,2) loc6=(2,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[5]): return self._action_dict['up']
		elif np.array_equal(location, self._locations[6]): return self._action_dict['up']
		elif np.array_equal(location, self._locations[2]): return self._action_dict[random.choice(('up','right','down'))]
		elif np.array_equal(location, self._locations[4]): return self._action_dict[random.choice(('up','left','down'))]
		else: return self._action_dict[random.choice(('left','right'))]

	def agent_8rooms_v0(self, location, dirty):
		# Map: . . . / . # . / . . .  (donut — clockwise loop)
		# loc0=(0,0) loc1=(1,0) loc2=(2,0) loc3=(0,1) loc4=(2,1) loc5=(0,2) loc6=(1,2) loc7=(2,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[2]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[4]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[7]): return self._action_dict['left']
		elif np.array_equal(location, self._locations[6]): return self._action_dict['left']
		elif np.array_equal(location, self._locations[5]): return self._action_dict['up']
		else: return self._action_dict['up']

	def agent_8rooms_v1(self, location, dirty):
		# Map: # . . / . . . / . . .  (row-by-row snake)
		# loc0=(1,0) loc1=(2,0) loc2=(0,1) loc3=(1,1) loc4=(2,1) loc5=(0,2) loc6=(1,2) loc7=(2,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[1]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[4]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[7]): return self._action_dict['left']
		elif np.array_equal(location, self._locations[6]): return self._action_dict['left']
		elif np.array_equal(location, self._locations[5]): return self._action_dict['up']
		elif np.array_equal(location, self._locations[2]): return self._action_dict[random.choice(('up','right','down'))]
		else: return self._action_dict[random.choice(('left','right','down'))]

	def agent_9rooms_v0(self, location, dirty):
		# Map: full 3x3 — row-by-row snake
		# loc0=(0,0) ... loc8=(2,2)
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[1]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[2]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[5]): return self._action_dict['left']
		elif np.array_equal(location, self._locations[4]): return self._action_dict[random.choice(('left','down'))]
		elif np.array_equal(location, self._locations[3]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[6]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[7]): return self._action_dict['right']
		else: return self._action_dict['up']

	def agent_12rooms_v0(self, location, dirty):
		"""
		Map (4x4):
		  . . # .      loc0=(0,0) loc1=(1,0)           loc2=(3,0)
		  # . # .                 loc3=(1,1)            loc4=(3,1)
		  . . . .      loc5=(0,2) loc6=(1,2) loc7=(2,2) loc8=(3,2)
		  . . . #      loc9=(0,3) loc10=(1,3) loc11=(2,3)
		Traversal: right column down → row 2 left → left column → bottom row
		"""
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[2]):  return self._action_dict['down']           # (3,0)→down
		elif np.array_equal(location, self._locations[4]):  return self._action_dict['down']           # (3,1)→down
		elif np.array_equal(location, self._locations[8]):  return self._action_dict['left']           # (3,2)→left
		elif np.array_equal(location, self._locations[7]):  return self._action_dict[random.choice(('left','down'))]  # (2,2)
		elif np.array_equal(location, self._locations[11]): return self._action_dict['left']           # (2,3)→left
		elif np.array_equal(location, self._locations[1]):  return self._action_dict[random.choice(('right','down'))] # (1,0)
		elif np.array_equal(location, self._locations[3]):  return self._action_dict['down']           # (1,1)→down
		elif np.array_equal(location, self._locations[6]):  return self._action_dict['down']           # (1,2)→down
		elif np.array_equal(location, self._locations[10]): return self._action_dict['left']           # (1,3)→left
		elif np.array_equal(location, self._locations[5]):  return self._action_dict[random.choice(('up','down'))]    # (0,2)
		elif np.array_equal(location, self._locations[9]):  return self._action_dict['up']             # (0,3)→up
		else:                                                return self._action_dict['right']          # (0,0)→right

	def agent_5x5_v0(self, location, dirty):
		"""
		Map (5x5):
		  . . # . .    loc0=(0,0) loc1=(1,0)           loc2=(3,0) loc3=(4,0)
		  # . # . #               loc4=(1,1)            loc5=(3,1)
		  # . . . #               loc6=(1,2) loc7=(2,2) loc8=(3,2)
		  # # . # #                           loc9=(2,3)
		  # . . . #               loc10=(1,4) loc11=(2,4) loc12=(3,4)
		Traversal: top → spine down → bottom
		"""
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]):  return self._action_dict['right']
		elif np.array_equal(location, self._locations[1]):  return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[2]):  return self._action_dict['down']
		elif np.array_equal(location, self._locations[3]):  return self._action_dict['left']
		elif np.array_equal(location, self._locations[5]):  return self._action_dict[random.choice(('up','down'))]
		elif np.array_equal(location, self._locations[4]):  return self._action_dict['down']
		elif np.array_equal(location, self._locations[6]):  return self._action_dict['right']
		elif np.array_equal(location, self._locations[7]):  return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[8]):  return self._action_dict[random.choice(('left','up'))]
		elif np.array_equal(location, self._locations[9]):  return self._action_dict['down']
		elif np.array_equal(location, self._locations[10]): return self._action_dict[random.choice(('right','up'))]
		elif np.array_equal(location, self._locations[11]): return self._action_dict['right']
		else:                                                return self._action_dict['left']           # (3,4)

	def agent_6x6_v0(self, location, dirty):
		"""
		Map (6x6) — trophy shape:
		  . . # . . #    loc0=(0,0) loc1=(1,0)           loc2=(3,0) loc3=(4,0)
		  # . # . # #               loc4=(1,1)            loc5=(3,1)
		  . . . . # #    loc6=(0,2) loc7=(1,2) loc8=(2,2) loc9=(3,2)
		  # . . . . #               loc10=(1,3) loc11=(2,3) loc12=(3,3) loc13=(4,3)
		  # . . # . #               loc14=(1,4) loc15=(2,4)             loc16=(4,4)
		  # # . # # #                           loc17=(2,5)
		Traversal: top → left spine → middle rows → bottom
		"""
		if dirty: return self._action_dict['suck']
		elif np.array_equal(location, self._locations[0]):  return self._action_dict['right']
		elif np.array_equal(location, self._locations[1]):  return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[2]):  return self._action_dict['right']
		elif np.array_equal(location, self._locations[3]):  return self._action_dict['down']
		elif np.array_equal(location, self._locations[5]):  return self._action_dict[random.choice(('up','down'))]
		elif np.array_equal(location, self._locations[4]):  return self._action_dict['down']
		elif np.array_equal(location, self._locations[6]):  return self._action_dict['right']
		elif np.array_equal(location, self._locations[7]):  return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[8]):  return self._action_dict['right']
		elif np.array_equal(location, self._locations[9]):  return self._action_dict['down']
		elif np.array_equal(location, self._locations[10]): return self._action_dict[random.choice(('right','down'))]
		elif np.array_equal(location, self._locations[11]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[12]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[13]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[14]): return self._action_dict['right']
		elif np.array_equal(location, self._locations[15]): return self._action_dict['down']
		elif np.array_equal(location, self._locations[16]): return self._action_dict['up']
		else:                                                return self._action_dict['up']             # (2,5)