from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class CollectableBall(Ball):
    def __init__(self, color, reward):
        super().__init__(color)
        self.reward = reward

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True


class GoodObjectEnv(MiniGridEnv):
    """
    Collect good objects without touching bad objects as many as possible
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        numObjs=50,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.numObjs = numObjs
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        objs = []

        # For good objects
        while len(objs) < self.numObjs/2:
            obj = CollectableBall('green', 1)
            self.place_obj(obj)
            objs.append(obj)

        # For bad objects
        while len(objs) < self.numObjs:
            obj = CollectableBall('red', -1)
            self.place_obj(obj)
            objs.append(obj)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "avoid bad objects and get good objects as many as possible"

        self.pos_cnt = 0

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Check if we hit a ball
        front_cell = self.grid.get(*self.agent_pos)
        if front_cell and front_cell.type == 'ball':
            self.grid.grid[self.agent_pos[1] * self.grid.width + self.agent_pos[0]] = None
            reward = front_cell.reward

        if reward > 0:
            self.pos_cnt += 1

        if self.pos_cnt == self.numObjs/2:
            done = True

        return obs, reward, done, info

class GoodObjectEnv16x16(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)

class GoodObjectRandomEnv16x16(GoodObjectEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None)

register(
    id='MiniGrid-GoodObject-16x16-v0',
    entry_point='gym_minigrid.envs:GoodObjectEnv16x16'
)

register(
    id='MiniGrid-GoodObject-Random-16x16-v0',
    entry_point='gym_minigrid.envs:GoodObjectRandomEnv16x16'
)


