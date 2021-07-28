from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class WatermazeEnv(MiniGridEnv):
    """
    Environment similar to DMLab-30's Watermaze.
    The agent must find a hidden cell that gives a positive reward.
    The agent then gets reset to a random location and needs to find
    the reward cell again.
    """

    def __init__(
        self,
        size=5,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__(
            grid_size=size,
            max_steps=1000,
        )

    def _reset_agent(self):
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate a door on each wall
        self.grid.set(0, height // 2, Door('red'))
        self.grid.set(width-1, height // 2, Door('green'))
        self.grid.set(width // 2, 0, Door('blue'))
        self.grid.set(width // 2, height-1, Door('yellow'))

        # Place Invisible with reward
        obj = Invisible(1)
        self.place_obj(obj)

        # Place agent
        self._reset_agent()

        self.mission = 'find hidden reward'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Check if we found the hidden reward
        curr_cell = self.grid.get(*self.agent_pos)
        if curr_cell and curr_cell.type == 'invisible':
            reward = curr_cell.reward
            self._reset_agent()

        return obs, reward, done, info

class WatermazeEnv9x9(WatermazeEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)

class WatermazeRandomEnv9x9(WatermazeEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, agent_start_pos=None)

register(
    id='MiniGrid-Watermaze-9x9-v0',
    entry_point='gym_minigrid.envs:WatermazeEnv9x9'
)

register(
    id='MiniGrid-Watermaze-Random-9x9-v0',
    entry_point='gym_minigrid.envs:WatermazeRandomEnv9x9'
)

