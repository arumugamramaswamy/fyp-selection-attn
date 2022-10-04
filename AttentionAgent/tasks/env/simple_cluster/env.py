import time
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gym import spaces

from .scenario import Scenario

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

        self.observation_spaces = {}

        for agent in self.possible_agents:
            self.observation_spaces[agent] = spaces.Dict(
                {
                    "my_pos": spaces.Box(-np.inf, np.inf, (world.dim_p,)),
                    "my_vel": spaces.Box(-np.inf, np.inf, (world.dim_p,)),
                    "other_pos": spaces.Box(-np.inf, np.inf, (N - 1, world.dim_p)),
                }
            )

    def render(self, mode="human"):
        from pettingzoo.mpe._mpe_utils import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color[:3], alpha=0.5)
                else:
                    geom.set_color(*entity.color[:3])
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            outer_bound = rendering.make_polygon([(-1, 1), (1,1),(1,-1),(-1,-1)], False)
            outer_bound.set_color(0,0,0)
            outer_xform = rendering.Transform()
            outer_bound.add_attr(outer_xform)
            self.render_geoms.append(outer_bound)
            self.render_geoms_xform.append(outer_xform)

            inner_bound = rendering.make_polygon([(-0.9, 0.9), (0.9,0.9),(0.9,-0.9),(-0.9,-0.9)], False)
            inner_bound.set_color(0,255,0)
            inner_xform = rendering.Transform()
            inner_bound.add_attr(inner_xform)
            self.render_geoms.append(inner_bound)
            self.render_geoms_xform.append(inner_xform)

            far_bound = rendering.make_polygon([(-10, 10), (10,10),(10,-10),(-10,-10)], False)
            far_bound.set_color(255,0,0)
            far_xform = rendering.Transform()
            far_bound.add_attr(far_xform)
            self.render_geoms.append(far_bound)
            self.render_geoms_xform.append(far_xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    tline = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(tline)
                    idx += 1

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for idx, other in enumerate(self.world.agents):
            if other.silent:
                continue
            if np.all(other.state.c == 0):
                word = '_'
            elif self.continuous_actions:
                word = '[' + ",".join([f"{comm:.2f}" for comm in other.state.c]) + "]"
            else:
                word = alphabet[np.argmax(other.state.c)]

            message = (other.name + ' sends ' + word + '   ')

            self.viewer.text_lines[idx].set_text(message)

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        time.sleep(0.05)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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
        K=2,
        shuffle=False,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
    ):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario(
            shuffle=shuffle,
        )
        world = scenario.make_world(N, K)
        super().__init__(
            N, scenario, world, max_cycles, continuous_actions, local_ratio
        )
        self.metadata["name"] = "simple_cluster"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
