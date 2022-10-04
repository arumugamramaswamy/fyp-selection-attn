import numpy as np

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(
        self,
        shuffle=False,
    ) -> None:
        super().__init__()
        self.shuffle = shuffle

    def make_world(self, N=3, K=2):
        world = World()
        # set any world properties first
        world.dim_c = 0
        self.K = K
        num_agents = N
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if a == agent:
                    continue
                if self.is_collision(a, agent):
                    # collision proportional to distance from nearest agent
                    rew -= 1
        # for x in agent.state.p_pos:
        #    rew -= bound(x)
        return rew

    def global_reward(self, world):
        rew = 0
        for a_ in world.agents:
            # Distance to k agents instead of to all agents
            # Discrepency between 1 and 2 -> isoceles + Distance to closest agent
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - a_.state.p_pos))) for a in world.agents]
            dists = sorted(dists)
            dists = dists[:self.K + 1]
            rew -= sum(dists)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        other_pos = np.array(other_pos, dtype=np.float32)
        if self.shuffle:
            np.random.shuffle(other_pos)

        return {
            "my_pos": np.array(agent.state.p_pos, dtype=np.float32),
            "my_vel": np.array(agent.state.p_vel, dtype=np.float32),
            "other_pos": other_pos,
        }

def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
