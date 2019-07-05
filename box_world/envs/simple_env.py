import colorsys
import enum
import numpy as np
import gym

_WALL_COLOR = 0.0
_SPACE_COLOR = 0.75
_AGENT_COLOR = 0.5
_GEM_COLOR = 1.0

class BoxWorld(gym.Env):

    '''A simple implementation of Deepmind Box-World. '''

    def __init__(self, max_length=4, max_branch_num=4, branch_length=1):
        self.__max_length = max_length
        self.__max_branch_num = max_branch_num
        self.__branch_length = branch_length

        self.__room = None
        self.__branches = None
        self.__colormap = None

        self.__agent_pos = None
        self.__agent_hold = None

    def step(self, action):
        agent_pos = self.__agent_pos.copy()
        next_pos = self.__agent_pos.copy()
        reward = 0
        done = False

        if action == Action.U:
            next_pos[0] -= 1
        elif action == Action.L:
            next_pos[1] -= 1
        elif action == Action.D:
            next_pos[0] += 1
        elif action == Action.R:
            next_pos[1] += 1

        def valid(r, c):
            if self.__isWall(r, c):
                return False
            elif self.__isLockedKey(r, c):
                return False
            elif self.__isLock(r, c):
                return self.__agent_hold is not None and np.array_equal(self.__room[r, c], self.__agent_hold)

            return True


        if valid(*next_pos):
            if self.__isLock(*next_pos):
                self.__agent_hold = None

                for i, (root, branch) in enumerate(self.__branches):
                    color = tuple(self.__room[next_pos[0], next_pos[1]-1])
                    if self.__colormap[color] in branch:
                        if i == 0:
                            reward = 1
                        else:
                            reward = -1
                            done = True

                        break
            elif self.__isGem(*next_pos):
                reward = 10
                done = True
            elif self.__isLooseKey(*next_pos):
                self.__agent_hold = np.copy(self.__room[next_pos[0], next_pos[1]])

            self.__room[agent_pos[0], agent_pos[1]] = _SPACE_COLOR * 255.
            self.__room[next_pos[0], next_pos[1]] = _AGENT_COLOR * 255.
            self.__agent_pos = next_pos

        return np.copy(self.__room), reward, done, None

    def reset(self):
        room_size = 14
        channel = 3

        # initialize room
        room = np.ones((room_size, room_size, 3)) * _SPACE_COLOR
        room[0], room[-1], room[:, 0], room[:, -1] = _WALL_COLOR, _WALL_COLOR, _WALL_COLOR, _WALL_COLOR

        def gen_pos():
            while True:
                real_size = room_size - 2
                idx = np.random.randint(real_size ** 2)
                row, col = idx // real_size + 1, idx % real_size + 1

                if np.all(room[row, col] == _SPACE_COLOR):
                    return row, col

        # set agent position
        row, col = gen_pos()
        room[row, col] = _AGENT_COLOR
        self.__agent_pos = [row, col]

        # generate graph
        branches, node_num = self.__graph_gen()

        # generate colors
        colors = np.arange(node_num) * (1.0 / node_num)
        colors = [colorsys.hsv_to_rgb(h, 0.6, 1.0) for h in colors]

        # set gem color
        colors[branches[0][1][-1]] = (_GEM_COLOR, _GEM_COLOR, _GEM_COLOR)

        # set loose key
        row, col = gen_pos()
        room[row, col] = colors[0]

        # generate pair
        for root, branch in branches:
            for i, node in enumerate(branch):
                while True:
                    row, col = gen_pos()

                    # check left side is space
                    if np.any(room[row, col-1] != _SPACE_COLOR):
                        continue
                    # check left side of pair is space or wall
                    if np.any(room[row, col-2] != _SPACE_COLOR) and np.any(room[row, col-2] != _WALL_COLOR):
                        continue
                    # check right side of pair is space or wall
                    if np.any(room[row, col+1] != _SPACE_COLOR) and np.any(room[row, col+1] != _WALL_COLOR):
                        continue

                    break

                # set key in pair
                room[row, col-1] = colors[node]

                if i == 0:
                    room[row, col] = colors[root]
                else:
                    room[row, col] = colors[branch[i-1]]

        self.__room = room * 255
        self.__branches = branches

        colors = [(r * 255., g * 255., b * 255.) for r, g, b in colors]

        self.__colormap = dict([(c, i) for i, c in enumerate(colors)])

        return np.copy(self.__room)

    def __graph_gen(self):
        length = np.random.randint(self.__max_length) + 1
        branch_num = np.random.randint(self.__max_branch_num + 1)
        branch_pos = np.random.choice(length, branch_num)
        branch_pos.sort()
        branches = []

        # add master branch
        branches.append((0, np.arange(length) + 1))

        cnt = length + 1

        # add sub branches
        for pos in branch_pos:
            branches.append((pos, np.arange(self.__branch_length) + cnt))
            cnt += self.__branch_length

        return branches, cnt

    def __isWall(self, r, c):
        return np.all(self.__room[r, c] == _WALL_COLOR * 255)

    def __isSpace(self, r, c):
        return np.all(self.__room[r, c] == _SPACE_COLOR * 255)

    def __isAgent(self, r, c):
        return np.all(self.__room[r, c] == _AGENT_COLOR * 255)

    def __isGem(self, r, c):
        return np.all(self.__room[r, c] == _GEM_COLOR * 255)

    def __isLockOrKey(self, r, c):
        if self.__isWall(r, c):
            return False
        elif self.__isSpace(r, c):
            return False
        elif self.__isAgent(r, c):
            return False
        else:
            return True

    def __isLock(self, r, c):
        return self.__isLockOrKey(r, c) and self.__isLockOrKey(r, c - 1)

    def __isLockedKey(self, r, c):
        return self.__isLockOrKey(r, c) and self.__isLockOrKey(r, c + 1)

    def __isLooseKey(self, r, c):
        return self.__isLockOrKey(r, c) and not self.__isLockOrKey(r, c - 1) and not self.__isLockOrKey(r, c + 1)

class Action(enum.IntEnum):
    U = 0
    L = 1
    D = 2
    R = 3
