import gymnasium as gym
import numpy as np

from gymnasium import spaces
from stars_maps import *
from config import *
from render import Renderer


class GridWorldEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):

        super().__init__()

        self.grid = GRID.copy()

        self.rows, self.cols = self.grid.shape

        # Actions:
        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right
        self.action_space = spaces.Discrete(4)

        # Observation is the flattened grid with the agent's
        # position encoded as 6. Derived from actual grid size
        # instead of hardcoded, so this keeps working if the map
        # in stars_maps.py ever changes shape.
        self.observation_space = spaces.Box(
            low=0,
            high=6,
            shape=(self.rows * self.cols,),
            dtype=np.float32
        )

        # Renderer is created lazily on first render() call, not
        # here. This means training (which never calls render())
        # never opens a pygame window at all — avoids an unused,
        # unresponsive window during train.py runs, and avoids a
        # crash on machines with no display attached.
        self.render_mode = render_mode
        self.renderer = None

        self.reset()

    #######################################################

    def get_state(self):

        obs = self.grid.copy().astype(np.float32)

        # Encode agent position
        obs[self.agent_x][self.agent_y] = 6

        return obs.flatten()

    #######################################################

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.grid = GRID.copy()

        start = np.argwhere(self.grid == START)[0]

        self.agent_x = int(start[0])
        self.agent_y = int(start[1])

        self.steps = 0

        self.remaining_stars = np.sum(self.grid == STAR)

        return self.get_state(), {}

    #######################################################

    def render(self):

        if self.renderer is None:
            self.renderer = Renderer(self.grid)

        self.renderer.render(
            self.grid,
            (self.agent_x, self.agent_y)
        )

    #######################################################

    def step(self, action):

        reward = -0.02

        terminated = False
        truncated = False

        new_x = self.agent_x
        new_y = self.agent_y

        # -------------------------
        # Actions
        # -------------------------

        if action == 0:
            new_x -= 1

        elif action == 1:
            new_x += 1

        elif action == 2:
            new_y -= 1

        elif action == 3:
            new_y += 1

        moved = True

        # -------------------------
        # Boundary
        # -------------------------

        if not (0 <= new_x < self.rows and
                0 <= new_y < self.cols):

            reward -= 2

            new_x = self.agent_x
            new_y = self.agent_y

            moved = False

        else:

            # -------------------------
            # Wall
            # -------------------------

            if self.grid[new_x][new_y] == WALL:

                reward -= 2

                new_x = self.agent_x
                new_y = self.agent_y

                moved = False

        # Move agent

        self.agent_x = new_x
        self.agent_y = new_y

        terrain = self.grid[self.agent_x][self.agent_y]

        # Terrain effects (star/fire/goal) only trigger on a step
        # that actually moved the agent onto this tile. Without
        # this guard, bumping a wall or the boundary while already
        # standing on the goal (with stars remaining) re-applies
        # the goal penalty every such bump, even though the agent
        # never moved.
        if moved:

            # -------------------------
            # Star
            # -------------------------

            if terrain == STAR:

                reward += 100


                self.grid[self.agent_x][self.agent_y] = NORMAL

                self.remaining_stars -= 1

                if self.remaining_stars == 0:

                    reward += 100

            # -------------------------
            # Fire
            # -------------------------

            elif terrain == FIRE:

                reward = -100

                terminated = True

            # -------------------------
            # Goal
            # -------------------------

            elif terrain == GOAL:

                if self.remaining_stars == 0:

                    reward += 500

                    terminated = True

                else:

                    reward -= 20

        # -------------------------

        self.steps += 1

        if self.steps >= MAX_EPISODE_STEPS:

            truncated = True

        info = {

            "stars_left": self.remaining_stars,

            "steps": self.steps,

            "terrain": int(terrain)

        }

        return (

            self.get_state(),

            reward,

            terminated,

            truncated,

            info

        )

    #######################################################

    def close(self):

        if self.renderer is not None:

            self.renderer.close()