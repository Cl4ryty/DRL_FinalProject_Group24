# coding=utf-8
# Copyright 2022 The Google Research Authors.
# Modifications copyright (C) 2022 Cl4ryty, goody139
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An environment which is built by a learning adversary.

Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.
"""
import random

import gym
import gym_minigrid.minigrid as minigrid
import numpy as np

from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid import register

from gym_minigrid.minigrid import COLORS
from gym_minigrid.rendering import fill_coords, point_in_rect
from social_rl.gym_multigrid.multigrid import WorldObj

# # # # # # # # # # # # # # # # # # # # # # code added start # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Unknown(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color="red"):
        super().__init__("unknown", color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Map(multigrid.Grid):
    def __init__(self, initial_size, agent_view_size, initialize_as_type=Unknown):
        super().__init__(initial_size, initial_size)

        # initialize map to be unknown
        self.init_as_floor(initialize_as_type)

        self.num_found_tiles = 0

        # number of walkable tiles connected to unknown tiles
        self.number_explorable = 0
        self.all_unknown = True

        self.size = initial_size
        self.agent_view_size = agent_view_size
        self.connected_unknowns = np.full((self.size, self.size), -1)

        self.agent_pos_in_map = None
        self.previous_agent_pos = None

        self.visibility_mask = np.zeros((self.size, self.size), bool)

    def update(self, agent_pos, agent_dir, agent_forward, agent_right, agent_view_grid, visibility_mask):
        new_tiles = 0

        # update position in map according to difference between previous and provided pos
        if self.previous_agent_pos is None:
            self.agent_pos_in_map = (self.size // 2, self.size // 2)
            self.previous_agent_pos = agent_pos

        else:
            # get the movement / difference between the previous and current pos
            difference = agent_pos - self.previous_agent_pos

            self.agent_pos_in_map = self.agent_pos_in_map + difference
            self.previous_agent_pos = agent_pos

        # check if the map needs to be enlarged to be able to place the agent inside
        if -1 in self.agent_pos_in_map or self.agent_pos_in_map[0] > self.size or self.agent_pos_in_map[1] > self.size:
            # agent is out of bounds, need to enlarge by how much the agent is outside the map + his view size
            max_dist_to_map = max(np.max(np.array(self.agent_pos_in_map)) - self.size,
                                  np.max(np.array(self.agent_pos_in_map) * (-1)))
            self.__enlarge(max_dist_to_map + self.agent_view_size)

        # check if the agent's view requires the map to be enlarged
        self.check_if_enlargement_necessary(self.agent_pos_in_map, agent_dir, self.agent_view_size)

        # Compute the world coordinates (map coordinates) of the bottom-left corner
        # of the agent's view area
        top_left = (
                self.agent_pos_in_map
                + agent_forward * (self.agent_view_size - 1)
                - agent_right * (self.agent_view_size // 2)
        )

        # we have to go through the agent's view twice, first to set the tiles to the new values,
        # then to update the unknown counts → can't be done in one go, as the unknown counts would not be accurate
        # if not all tiles had been set

        # store visibility mask for the map
        self.visibility_mask = np.zeros((self.size, self.size), bool)

        # For each cell in the visibility mask (the agent's view)
        for grid_j in range(0, self.agent_view_size):
            for grid_i in range(0, self.agent_view_size):
                # If this cell is not visible, ignore it
                if not visibility_mask[grid_i, grid_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (agent_forward * grid_j) + (agent_right * grid_i)

                # set the visibility mask to true for the visible value
                self.visibility_mask[abs_i, abs_j] = True

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Update the tile
                # get current content of that map tile to see if this is new information
                current_tile = self.get(abs_i, abs_j)
                if isinstance(current_tile, Unknown):
                    new_tiles += 1
                self.set(abs_i, abs_j, agent_view_grid.get(grid_i, grid_j))

        for grid_j in range(0, self.agent_view_size):
            for grid_i in range(0, self.agent_view_size):
                # If this cell is not visible, ignore it
                if not visibility_mask[grid_i, grid_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (agent_forward * grid_j) + (agent_right * grid_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                tile_location = [abs_i, abs_j]

                # update the unknown count of this and its neighboring tiles
                # due to the limited actions the agent can take, only the 4-neighborhood is important
                count = 0
                neighborhood = [[-1, 0], [1, 0], [0, -1], [0, 1]]
                for offset in neighborhood:
                    location = tile_location + offset
                    if not -1 in location and not np.any(np.greater_equal(location, self.size)):
                        # update the number of connected unknown tiles for this tile as well
                        n_count = self.count_object_type_around_tile(location, Unknown)
                        self.connected_unknowns[location] = n_count

                        if isinstance(self.get(location[0], location[1]), Unknown):
                            count += 1

                self.connected_unknowns[tile_location] = count

        # figure out which parts are unexplored
        # → if there are tiles that are walkable and connected to unknown we're not done
        self.num_found_tiles += new_tiles
        done = not np.any(self.connected_unknowns > 0)

        if self.num_found_tiles == 0:
            done = False

        return done, new_tiles

    def check_if_enlargement_necessary(self, agent_position_in_map, agent_dir, agent_view_size):
        # depending on the agent forward, check if the agent's view would extend beyond the current map

        # get the absolute coordinates (with respect to the map) of the corners of the agent's view
        agent_view_size = agent_view_size or self.agent_view_size

        # Facing right
        if agent_dir == 0:
            topX = agent_position_in_map[0]
            topY = agent_position_in_map[1] - agent_view_size // 2
        # Facing down
        elif agent_dir == 1:
            topX = agent_position_in_map[0] - agent_view_size // 2
            topY = agent_position_in_map[1]
        # Facing left
        elif agent_dir == 2:
            topX = agent_position_in_map[0] - agent_view_size + 1
            topY = agent_position_in_map[1] - agent_view_size // 2
        # Facing up
        elif agent_dir == 3:
            topX = agent_position_in_map[0] - agent_view_size // 2
            topY = agent_position_in_map[1] - agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        corner_coordinates = np.array((topX, topY, botX, botY))
        # if any of the corners is outside the map, it needs to be enlarged
        if np.any(corner_coordinates < 0) or np.any(corner_coordinates > self.size):
            # enlarge the map by at least the max size of the agent's view to ensure that the map will be large enough
            self.__enlarge(self.agent_view_size)

    def count_object_type_around_tile(self, tile_location, object_type):
        # due to the limited actions the agent can take, only the 4-neighborhood is important
        neighborhood = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        count = 0

        for offset in neighborhood:
            location = tile_location + offset

            if not -1 in location and not np.any(np.greater_equal(location, self.size)):
                if isinstance(self.get(location[0], location[1]), object_type):
                    count += 1

        return count

    def __enlarge(self, enlarge_by=5):
        # add padding to all sides of the map
        numpy_grid = np.array(self.grid).reshape((self.size, self.size))
        numpy_grid = np.pad(numpy_grid, enlarge_by, mode='constant', constant_values=Unknown())

        # also add padding to the array containing the connected unknowns
        self.connected_unknowns = np.pad(self.connected_unknowns, enlarge_by, mode='constant', constant_values=-1)

        self.size = numpy_grid.shape[0]
        # update agent position to account for the added padding
        self.agent_pos_in_map += enlarge_by

        # flatten and store as grid again, also adjust height and width
        self.grid = list(numpy_grid.flatten())
        self.height = self.size
        self.width = self.size

        # update number_explorable accordingly to account for the added unknown tiles

        # update all tiles that were on the edge of the map before,
        # as these are the ones where the count could have changed

        # get indices of all previous edge tiles
        indices = np.indices(self.connected_unknowns.shape)
        indices_list = indices[:, enlarge_by, enlarge_by:(self.connected_unknowns.shape[1] - enlarge_by)].T
        indices_list = np.vstack((indices_list, indices[:, (self.connected_unknowns.shape[1] - enlarge_by),
                                                enlarge_by:(self.connected_unknowns.shape[1] - enlarge_by)].T))
        indices_list = np.vstack(
            (indices_list, indices[:, enlarge_by:(self.connected_unknowns.shape[1] - enlarge_by), enlarge_by].T))
        indices_list = np.vstack((indices_list, indices[:, enlarge_by:(self.connected_unknowns.shape[1] - enlarge_by),
                                                (self.connected_unknowns.shape[1] - enlarge_by)].T))

        for tile_indices in indices_list:
            self.connected_unknowns[tile_indices] = self.count_object_type_around_tile(tile_indices, Unknown)

    def render(self,
               highlight=True,
               tile_size=minigrid.TILE_PIXELS):
        """Render the whole-grid human view."""

        if highlight:
            highlight_mask = self.visibility_mask
        else:
            highlight_mask = None

        # Render the whole grid
        img = super(Map, self).render(tile_size, highlight_mask=highlight_mask)

        # plt.imshow(img)
        # plt.show()
        return img

# # # # # # # # # # # # # # # # # # # # # # code added end # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # code added/modified start # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class AdversarialEnv(multigrid.MultiGridEnv):
    """Grid world where an adversary build the environment the agent plays.

  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.
  """

    def __init__(self, n_clutter=50, size=15, agent_view_size=5, max_steps=250,
                 random_z_dim=50):
        """Initializes environment in which adversary places goal, agent, obstacles.

    Args:
      n_clutter: The maximum number of obstacles the adversary can place.
      size: The number of tiles across one side of the grid; i.e. make a
        size x size grid.
      agent_view_size: The number of tiles in one side of the agent's partially
        observed view of the grid.
      max_steps: The maximum number of steps that can be taken before the
        episode terminates.
      random_z_dim: The environment generates a random vector z to condition the
        adversary. This gives the dimension of that vector.
    """
        self.agent_start_pos = None
        self.n_clutter = n_clutter
        self.random_z_dim = random_z_dim

        self.num_found_tiles = 0
        self.findable_tiles = -1
        self.tiles_to_map_completion = -1

        # Add one action for placing the agent.
        self.adversary_max_steps = self.n_clutter + 1

        super().__init__(
            n_agents=1,
            minigrid_mode=True,
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=False,  # Set this to True for maximum speed
            competitive=True,
        )

        # generate map for each agent
        self.map = [Map(initial_size=size, agent_view_size=agent_view_size)] * self.n_agents

        # Metrics
        self.reset_metrics()

        # Create spaces for adversary agent's specs.
        self.adversary_action_dim = (size - 2) ** 2
        self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
        self.adversary_image_obs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.width, self.height, 3),
            dtype='uint8')

        # Adversary observations are dictionaries containing an encoding of the
        # grid, the current time step, and a randomly generated vector used to
        # condition generation (as in a GAN).
        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})


    def _gen_grid(self, width, height):
        """Grid is initially empty, because adversary will create it."""
        # Create an empty grid
        self.grid = multigrid.Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)


    def calculate_findable_tiles(self):
        # start at agent position
        if self.agent_start_pos is None:
            return

        # flood the map
        findable_tiles = self.flood_walkable_and_adjacent(self.agent_start_pos, [])
        self.findable_tiles = findable_tiles
        self.tiles_to_map_completion = findable_tiles - self.num_found_tiles
        return findable_tiles

    def flood_walkable_and_adjacent(self, pos, flooded=[]):
        neighborhood = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        pos_x, pos_y = pos
        n = 1
        flooded.append((pos_x, pos_y))
        if isinstance(self.grid.get(pos_x, pos_y), multigrid.Agent) or self.grid.get(pos_x, pos_y) is None or self.grid.get(pos_x, pos_y).can_overlap():
            # flood the neighboring tiles
            for [x_offset, y_offset] in neighborhood:
                p = (pos_x + x_offset, pos_y + y_offset)
                if p not in flooded:
                    n += self.flood_walkable_and_adjacent(p, flooded)
        return n

    def render_env_and_map(self, tile_size=minigrid.TILE_PIXELS, norm_map_to_tiles=40):
        map_image = self.map[0].render(tile_size=tile_size)
        environment_image = self.render(tile_size=tile_size)

        if map_image.shape[0] > norm_map_to_tiles * tile_size:
            # reduce size by taking only the middle
            d = norm_map_to_tiles * tile_size
            margin0 = (map_image.shape[0] - d) // 2
            margin1 = (map_image.shape[1] - d) // 2
            map_image = map_image[margin0:margin0 + d, margin1:margin1 + d, :]

        if map_image.shape[0] < norm_map_to_tiles * tile_size:
            # pad to the correct size
            d = norm_map_to_tiles * tile_size
            pad = (d - map_image.shape[0]) // 2
            uneven_padding = (d - map_image.shape[0]) % 2
            img = map_image
            i = np.r_[img, np.zeros((pad, img.shape[0], 3), dtype=int)]
            i = np.r_[np.zeros((pad + uneven_padding, img.shape[0], 3), dtype=int), i]
            i = np.concatenate((i, np.zeros((i.shape[0], pad, 3), dtype=int)), axis=1)
            i = np.concatenate((np.zeros((i.shape[0], pad + uneven_padding, 3), dtype=int), i), axis=1)
            map_image = i

        return (map_image, environment_image)

    def step(self, actions):
        actions, collect_video = actions
        # Maintain backwards compatibility with MiniGrid when there is one agent
        if not isinstance(actions, list) and self.n_agents == 1:
            actions = [actions]

        self.step_count += 1

        rewards = [0] * self.n_agents
        dones = [False] * self.n_agents

        # Randomize order in which agents act for fairness
        agent_ordering = np.arange(self.n_agents)
        np.random.shuffle(agent_ordering)

        # Step each agent
        for a in agent_ordering:
            rewards[a], dones[a] = self.step_one_agent(actions[a], a)

        obs = self.gen_obs()

        # Backwards compatibility
        if self.minigrid_mode:
            rewards = rewards[0]

        collective_done = False
        # In competitive version, if one agent finishes the episode is over.
        if self.competitive:
            collective_done = np.sum(dones) >= 1

        # Running out of time applies to all agents
        if self.step_count >= self.max_steps:
            collective_done = True

        return obs, rewards, collective_done, ({}, self.render_env_and_map())

    def step_one_agent(self, action, agent_id):
        reward = 0

        # Get the position in front of the agent
        fwd_pos = self.front_pos[agent_id]
        successful_forward = False

        # Rotate left
        if action == self.actions.left:
            self.agent_dir[agent_id] -= 1
            if self.agent_dir[agent_id] < 0:
                self.agent_dir[agent_id] += 4
            self.rotate_agent(agent_id)

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir[agent_id] = (self.agent_dir[agent_id] + 1) % 4
            self.rotate_agent(agent_id)

        # Move forward
        elif action == self.actions.forward:
            successful_forward = self._forward(agent_id, fwd_pos)
            fwd_cell = self.grid.get(*fwd_pos)

        # Pick up an object
        elif action == self.actions.pickup:
            self._pickup(agent_id, fwd_pos)

        # Drop an object
        elif action == self.actions.drop:
            self._drop(agent_id, fwd_pos)

        # Toggle/activate an object
        elif action == self.actions.toggle:
            self._toggle(agent_id, fwd_pos)

        # Done action -- by default acts as no-op.
        elif action == self.actions.done:
            pass

        else:
            assert False, 'unknown action'

        agent_view_grid, visibility_mask = self.gen_obs_grid(agent_id)

        # # testing: print map
        # self.map[agent_id].render()
        # update map and calculate reward
        done, new_tiles = self.map[agent_id].update(fwd_pos, self.agent_dir[agent_id], self.dir_vec[agent_id],
                                                    self.right_vec[agent_id], agent_view_grid, visibility_mask)

        self.num_found_tiles += new_tiles
        if self.tiles_to_map_completion != -1:
            self.tiles_to_map_completion -= new_tiles
        # # testing: print map
        # self.map[agent_id].render()

        hit_something = not successful_forward

        # todo: calculate reward
        reward = hit_something * -1 + new_tiles * 0.01 + 10 * done

        return reward, done

    def reset_metrics(self):
        self.num_found_tiles = 0
        # generate map for each agent
        self.map = [Map(initial_size=self.height, agent_view_size=self.agent_view_size)] * self.n_agents
        self.tiles_to_map_completion = self.findable_tiles - self.num_found_tiles
        self.n_clutter_placed = 0
        self.deliberate_agent_placement = -1

    def reset(self):
        """Fully resets the environment to an empty grid with no agent or goal."""

        self.step_count = 0
        self.adversary_step_count = 0

        self.agent_start_dir = self._rand_int(0, 4)

        # Current position and direction of the agent
        self.reset_agent_status()

        self.agent_start_pos = None

        # Extra metrics
        self.reset_metrics()

        # Generate the grid. Will be random by default, or same environment if
        # 'fixed_environment' is True.
        self._gen_grid(self.width, self.height)

        image = self.grid.encode()
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs

    def reset_agent_status(self):
        """Reset the agent's position, direction, done, and carrying status."""
        self.agent_pos = [None] * self.n_agents
        self.agent_dir = [self.agent_start_dir] * self.n_agents
        self.done = [False] * self.n_agents
        self.carrying = [None] * self.n_agents
        self.num_found_tiles = 0
        # generate empty map for each agent
        self.map = [Map(initial_size=self.height, agent_view_size=self.agent_view_size)] * self.n_agents
        self.tiles_to_map_completion = self.findable_tiles - self.num_found_tiles

    def reset_agent(self):
        """Resets the agent's start position, but leaves goal and walls."""
        # Remove the previous agents from the world
        for a in range(self.n_agents):
            if self.agent_pos[a] is not None:
                self.grid.set(self.agent_pos[a][0], self.agent_pos[a][1], None)

        # Current position and direction of the agent
        self.reset_agent_status()

        if self.agent_start_pos is None:
            raise ValueError('Trying to place agent at empty start position.')
        else:
            self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)

        for a in range(self.n_agents):
            assert self.agent_pos[a] is not None
            assert self.agent_dir[a] is not None

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos[a])
            if not (start_cell.type == 'agent' or
                    start_cell is None or start_cell.can_overlap()):
                raise ValueError('Wrong object in agent start position.')

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        return obs

    def remove_wall(self, x, y):
        obj = self.grid.get(x, y)
        if obj is not None and obj.type == 'wall':
            self.grid.set(x, y, None)

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def step_adversary(self, loc):
        """The adversary gets n_clutter + 1 move to place the agent, blocks.

    The action space is the number of possible squares in the grid. The squares
    are numbered from left to right, top to bottom.

    Args:
      loc: An integer specifying the location to place the next object which
        must be decoded into x, y coordinates.

    Returns:
      Standard RL observation, reward (always 0), done, and info
    """
        if loc >= self.adversary_action_dim:
            raise ValueError('Position passed to step_adversary is outside the grid.')

        # Add offset of 1 for outside walls
        x = int(loc % (self.width - 2)) + 1
        y = int(loc / (self.width - 2)) + 1
        done = False

        should_choose_agent = self.adversary_step_count == 0

        # Place the agent
        if should_choose_agent:
            self.remove_wall(x, y)  # Remove any walls that might be in this loc

            # Goal has already been placed here
            if self.grid.get(x, y) is not None:
                # Place agent randomly
                self.agent_start_pos = self.place_one_agent(0, rand_dir=False)
                self.deliberate_agent_placement = 0
            else:
                self.agent_start_pos = np.array([x, y])
                self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)
                self.deliberate_agent_placement = 1

        # Place wall
        elif self.adversary_step_count < self.adversary_max_steps:
            # If there is already an object there, action does nothing
            if self.grid.get(x, y) is None:
                self.put_obj(minigrid.Wall(), x, y)
                self.n_clutter_placed += 1

        self.adversary_step_count += 1

        # End of episode
        if self.adversary_step_count >= self.adversary_max_steps:
            done = True
            self.calculate_findable_tiles()

        image = self.grid.encode()
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs, 0, done, {}


# # # # # # # # # # # # # # # # # # # code added/modified end # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class MiniAdversarialEnv(AdversarialEnv):

    def __init__(self):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50)


class NoisyAdversarialEnv(AdversarialEnv):

    def __init__(self):
        super().__init__(goal_noise=0.3)


class MediumAdversarialEnv(AdversarialEnv):

    def __init__(self):
        super().__init__(n_clutter=30, size=10, agent_view_size=5, max_steps=200)


class GoalLastAdversarialEnv(AdversarialEnv):

    def __init__(self):
        super().__init__(choose_goal_last=True)


class MiniGoalLastAdversarialEnv(AdversarialEnv):

    def __init__(self):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50,
                         choose_goal_last=True)


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-Adversarial-v0',
    entry_point=module_path + ':AdversarialEnv'
)

# register.register(
#     env_id='MultiGrid-ReparameterizedAdversarial-v0',
#     entry_point=module_path + ':ReparameterizedAdversarialEnv'
# )

register.register(
    env_id='MultiGrid-MiniAdversarial-v0',
    entry_point=module_path + ':MiniAdversarialEnv'
)

register.register(
    env_id='MultiGrid-MiniReparameterizedAdversarial-v0',
    entry_point=module_path + ':MiniReparameterizedAdversarialEnv'
)

register.register(
    env_id='MultiGrid-NoisyAdversarial-v0',
    entry_point=module_path + ':NoisyAdversarialEnv'
)

register.register(
    env_id='MultiGrid-MediumAdversarial-v0',
    entry_point=module_path + ':MediumAdversarialEnv'
)

register.register(
    env_id='MultiGrid-GoalLastAdversarial-v0',
    entry_point=module_path + ':GoalLastAdversarialEnv'
)

register.register(
    env_id='MultiGrid-MiniGoalLastAdversarial-v0',
    entry_point=module_path + ':MiniGoalLastAdversarialEnv'
)
