import numpy as np
import heapq
from pettingzoo.mpe.scenarios.simple_spread import Scenario


class CustomScenario(Scenario):
    """The main modification here is of the observation method. It is
    changed from returning a numpy array to a dict of numpy arrays.

    The reasoning behind this is: having clear information sources will
    allow for better informaiton fusion.
    """

    def __init__(
        self,
        shuffle=False,
        reward_only_single_agent=False,
        reward_agent_for_closest_landmark=False,
    ) -> None:
        super().__init__()
        self.shuffle = shuffle
        self._reward_only_single_agent = reward_only_single_agent
        self._reward_agent_for_closest_landmark = reward_agent_for_closest_landmark

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        entity_pos = np.array(entity_pos, dtype=np.float32)
        other_pos = np.array(other_pos, dtype=np.float32)
        comm = np.array(comm, dtype=np.float32)

        if self.shuffle:
            np.random.shuffle(entity_pos)
            np.random.shuffle(other_pos)
            np.random.shuffle(comm)

        return {
            "my_pos": np.array(agent.state.p_pos, dtype=np.float32),
            "my_vel": np.array(agent.state.p_vel, dtype=np.float32),
            "entity_pos": entity_pos,
            "other_pos": other_pos,
            "comm": comm,
        }

    def reward(self, agent, world):
        rew = super().reward(agent, world)
        if self._reward_agent_for_closest_landmark:
            dists = [
                np.sqrt(np.sum(np.square(l.state.p_pos - agent.state.p_pos)))
                for l in world.landmarks
            ]
            rew -= min(dists)
        return rew

    def global_reward(self, world):
        if not self._reward_only_single_agent:
            rew = 0
            for l in world.landmarks:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                rew -= min(dists)

            for l in world.landmarks:
                for a in world.agents:
                    if self.is_collision(l, a):
                        break
                else:
                    rew -= 1

            return rew

        priority_queue = []
        for l in world.landmarks:
            for a in world.agents:
                heapq.heappush(
                    priority_queue,
                    (
                        np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))),
                        l.name,
                        a.name,
                    ),
                )

        rew = 0
        used_a = set()
        used_l = set()
        while priority_queue:
            dist, l, a = heapq.heappop(priority_queue)
            if l in used_l or a in used_a:
                continue
            rew -= dist
            used_a.add(a)
            used_l.add(l)

        return rew
