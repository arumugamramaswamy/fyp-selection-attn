from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gym import spaces

from .scenario import CustomScenario

import numpy as np


class CustomSimpleEnv(SimpleEnv):
    """
    Main changes here are:
      - observation_spaces modified to return dict_spaces
      - remove the call to astype at the end of observe and state
    """

    def __init__(self, N, scenario, world, max_cycles, continuous_actions, local_ratio):
        super().__init__(
            scenario,
            world,
            max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )

        self.landmark_triggered = False
        self.landmark_steps = 5

        self.observation_spaces = {}

        for agent in self.possible_agents:
            self.observation_spaces[agent] = spaces.Dict(
                {
                    "my_pos": spaces.Box(-np.inf, np.inf, (world.dim_p,)),
                    "my_vel": spaces.Box(-np.inf, np.inf, (world.dim_p,)),
                    "entity_pos": spaces.Box(-np.inf, np.inf, (N, world.dim_p)),
                    "other_pos": spaces.Box(-np.inf, np.inf, (N - 1, world.dim_p)),
                    "comm": spaces.Box(-np.inf, np.inf, (N - 1, world.dim_c)),
                }
            )

    def reset(self):
        super().reset()
        self.landmark_triggered = False
        self.landmark_steps = 5


    def _execute_world_step(self):
        super()._execute_world_step()
        if self.landmark_triggered:
            self.landmark_steps -= 1
            if self.landmark_steps == 0:

                for a in self.agents:
                    self.dones[a] = True

                unoccupied = len(self.agents)
                for l in self.world.landmakrs:
                    for a in self.world.agents:
                        if self.scenario.is_collision(l, a):
                            unoccupied -= 1
                            break
                            
                final_reward = unoccupied * -20
                for agent in self.world.agents:
                    self.rewards[agent.name] += final_reward * (1 - self.local_ratio)

        else:
            for landmark in self.world.landmarks:
                for agent in self.world.agents:
                    if self.scenario.is_collision(landmark, agent):
                        self.landmark_triggered = True
                        return

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        )

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            )
            for agent in self.possible_agents
        )
        return states


class raw_env(CustomSimpleEnv):
    """
    Main Changes:
      - replace scenario with custom scenario
      - add N to constructor
    """

    def __init__(
        self,
        N=3,
        shuffle=False,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        reward_only_single_agent=False,
        reward_agent_for_closest_landmark=False,
        k=None,
    ):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = CustomScenario(
            shuffle=shuffle,
            reward_only_single_agent=reward_only_single_agent,
            reward_agent_for_closest_landmark=reward_agent_for_closest_landmark,
            k=k,
        )
        world = scenario.make_world(N)
        super().__init__(
            N, scenario, world, max_cycles, continuous_actions, local_ratio
        )
        self.metadata["name"] = "simple_spread_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
