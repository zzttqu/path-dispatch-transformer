import heapq
from typing import Optional  # 引入heapq模块，用于堆操作
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt  # 引入matplotlib模块，用于绘制地图
import numpy as np  # 引入numpy模块，用于矩阵操作


class State:
    def __init__(self, x, y):
        self.x = x  # 状态的x坐标
        self.y = y  # 状态的y坐标
        self.g = float("inf")  # 状态的g值，初始化为无穷大
        self.rhs = float("inf")  # 状态的rhs值，初始化为无穷大
        self.key = (float("inf"), float("inf"))  # 状态的键值，用于优先级队列
        self.previous: Optional[State] = None  # 状态的上一次状态

    def __lt__(self, other):
        return self.key < other.key  # 定义小于运算符，用于堆比较


class DStarLite:
    def __init__(self, start, goal, grid):
        self.start = State(*start)  # 起点状态
        self.goal = State(*goal)  # 目标状态
        self.grid: np.ndarray = grid  # 网格地图
        self.open_list = []  # 优先级队列
        self.states = {}  # 存储所有状态的字典
        self.k_m = 0  # 跟踪启发式变化的累积量
        self.init_state(self.start)  # 初始化起点状态
        self.init_state(self.goal)  # 初始化目标状态
        self.goal.rhs = 0  # 目标状态的rhs值设为0
        self.update_vertex(self.goal)  # 更新目标状态
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                self.init_state(self.get_state(x, y))  # 初始化状态
        self.max_attempts = 1000
        self.attempts = 0

    def init_state(self, state):
        if (state.x, state.y) not in self.states:
            self.states[(state.x, state.y)] = state  # 如果状态不在字典中，则添加

    def get_state(self, x, y):
        if (x, y) not in self.states:
            self.states[(x, y)] = State(x, y)  # 如果状态不在字典中，则创建并添加
        return self.states[(x, y)]  # 返回状态

    def heuristic(self, state1, state2):
        # 改为使用欧氏距离，可以减少转角

        return np.linalg.norm(
            (abs(state1.x - state2.x), abs(state1.y - state2.y))
        )  # 曼哈顿距离启发函数

    def calculate_key(self, state: State):
        # 计算状态的键值
        return (
            min(state.g, state.rhs) + self.heuristic(self.start, state) + self.k_m,
            min(state.g, state.rhs),
        )

    def update_vertex(self, u: State):
        if u != self.goal:
            min_rhs = float("inf")
            # 更新u的rhs值为其邻居中cost(u, v) + v.g的最小值
            for neighbor in self.get_neighbors(u):
                min_rhs = min(min_rhs, neighbor.g + self.cost(u, neighbor))
            # u.rhs, u.previous = min(
            #    (self.cost(u, v) + v.g, v) for v in self.get_neighbors(u)
            # )
            u.rhs = min_rhs  # type: ignore
        # 如果u在开放列表中，将其移除
        if any(u in item for item in self.open_list):
            self.open_list = [item for item in self.open_list if item[1] != u]
            heapq.heapify(self.open_list)  # 重新堆化
        # 如果u的g值和rhs值不同，将其加入开放列表
        if u.g != u.rhs:
            u.key = self.calculate_key(u)  # type: ignore # 计算u的键值
            heapq.heappush(self.open_list, (u.key, u))  # 将键值和状态作为元组加入堆

    def is_turn(self, u: State, v: State):
        if u.previous is None:
            return False
        dx1, dy1 = u.x - u.previous.x, u.y - u.previous.y
        dx2, dy2 = v.x - u.x, v.y - u.y
        return (dx1 != dx2) or (dy1 != dy2)  # 检查前后两个方向是否不同

    def cost(self, u, v):
        if self.grid[v.x][v.y] or self.grid[u.x][u.y] == 1:
            return float("inf")  # 如果目标位置不可达，返回无穷大
        # turn_cost = 0 if self.is_turn(u, v) else 0  # 如果u和v之间是拐点，则增加转弯代价
        else:
            return self.heuristic(u, v)  # 否则返回距离

    def get_neighbors(self, state) -> list[State]:
        neighbors = []
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ]:  # 可能的八个方向
            x, y = state.x + dx, state.y + dy
            if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                neighbors.append(self.get_state(x, y))  # 添加合法的邻居状态
        return neighbors  # 返回邻居列表

    def compute_shortest_path(self):
        self.attempts = 0  # 初始化尝试计数器
        while self.open_list and (
            self.open_list[0][0] < self.calculate_key(self.start)
            or self.start.rhs != self.start.g
        ):
            # print(self.open_list[0][0] < self.calculate_key(self.start))
            # print(self.open_list[0][0], self.calculate_key(self.start))
            if self.attempts > self.max_attempts:  # 检查尝试次数是否超过最大值
                print("Reached maximum attempts limit.")
                break
            _, u = heapq.heappop(self.open_list)  # 从堆中弹出键值最小的状态
            self.attempts += 1  # 增加尝试计数器
            if u.g > u.rhs:
                u.g = u.rhs  # 更新u的g值为rhs值
                for s in self.get_neighbors(u):
                    # s.previous = u  # 设置s的前驱为u
                    self.update_vertex(s)  # 更新所有邻居
            else:
                u.g = float("inf")  # 设置u的g值为无穷大
                self.update_vertex(u)  # 更新u
                for s in self.get_neighbors(u):
                    # s.previous = u  # 设置s的前驱为u
                    self.update_vertex(s)  # 更新所有邻居
        if not self.open_list:
            print("Open list is empty. Path might not be found.")
        if self.start.rhs == float("inf"):
            print("Start state has infinite rhs. Path not found.")

    def update_start(self, new_start: Optional[State] = None):
        self.attempts = 0
        if new_start is None:
            self.start = self.get_state(self.start.x, self.start.y)
            # self.k_m += 1
        else:
            self.k_m += self.heuristic(self.start, new_start)  # 更新启发式变化的累积量
            self.start = self.get_state(new_start.x, new_start.y)  # 更新起点状态
        self.update_vertex(self.start)  # 更新起点状态

    def update_goal(self, new_goal=None):
        self.attempts = 0
        if new_goal is not None:
            self.goal = self.get_state(new_goal[0], new_goal[1])  # 更新目标状态
            self.goal.rhs = 0  # 目标状态的rhs值设为0
        self.update_vertex(self.goal)  # 更新目标状态

    def update_grid(self, new_grid, obstacles_pos=None, compute=True, update=True):
        self.grid = new_grid  # 更新网格地图
        need_to_reset = set()
        self.k_m += self.heuristic(self.start, self.goal)
        if obstacles_pos is not None:
            for x, y in obstacles_pos:
                state = self.get_state(x, y)
                state.g = float("inf")
                state.rhs = float("inf")

                if update:
                    self.update_vertex(state)
                # self.update_vertex(state)
                """ for neighbor in self.get_neighbors(state):
                    self.update_vertex(neighbor) """
        if compute:
            self.compute_shortest_path()
        # heapq.heapify(self.open_list)

        """ if obstacles_pos is not None:
            for x, y in obstacles_pos:
                self.states[(x, y)].g = float("inf")
                self.states[(x, y)].rhs = float("inf")
                self.states[(x, y)].prev = None

        need_to_reset = set()
        if obstacles_pos is not None:
            for x, y in obstacles_pos:
                self.states[(x, y)].g = float("inf")
                self.states[(x, y)].rhs = float("inf")
                self.states[(x, y)].prev = None
                for neighbor in self.get_neighbors(self.states[(x, y)]):
                    need_to_reset.add(neighbor)
            for state in need_to_reset:
                self.update_vertex(state)
            heapq.heapify(self.open_list)
        self.compute_shortest_path() """
        self.attempts = 0

    def find_path(self, need_compute=True) -> tuple[np.ndarray, float, int, int]:
        if need_compute:
            self.compute_shortest_path()  # 计算最短路径
        path = []
        if self.attempts > self.max_attempts:
            return np.array(path), 0, 0, 2
        state = self.start
        while state != self.goal:
            self.attempts += 1
            if self.start.g == float("inf"):
                print("Start state has infinite rhs. Path not found.")
                return np.array(path), 0, 0, 1
            if self.attempts > self.max_attempts:
                return np.array(path), 0, 0, 2
            self.attempts += 1
            path.append([state.x, state.y])  # 将状态添加到路径
            neighbors = self.get_neighbors(state)
            """ state = min(
                neighbors, key=lambda s: s.g + self.cost(state, s)
            )  # 选择下一个状态 """
            # 在选择路径的时候再考虑拐点，三次内不能出现两次拐弯，不能在计算rhs的时候计算
            next_state = min(
                neighbors,
                key=lambda s: s.g + self.cost(state, s),
            )
            next_state.previous = state
            state = next_state
        path.append([self.goal.x, self.goal.y])  # 添加目标状态
        path = np.array(path)
        distance = 0.0
        turn_num = 0
        # print(path.shape[0])
        # 假如初始点和终点就距离1个单位
        if path.shape[0] < 3:
            return np.array(path), 1, 0, 0  # 返回路径
        for point_n in range(0, path.shape[0] - 2, 1):
            v1 = path[point_n + 1] - path[point_n]
            v2 = path[point_n + 2] - path[point_n + 1]
            _dot = np.dot(v1, v2)
            _dis1 = np.linalg.norm(v1)
            _dis2 = np.linalg.norm(v2)
            _cos = _dot / _dis1 / _dis2
            if _cos < 0.9999:
                turn_num += 1
            distance = distance + _dis1

        return np.array(path), float(distance), turn_num, 0  # 返回路径

    """ def turn_penalty(self, u, v):
        if u.previous is None:
            return 0
        return 1 if self.is_turn(u, v) else 0 """

    def visualize(self):
        fig, ax = plt.subplots()
        grid = np.array(self.grid)
        ax.matshow(grid.T, origin="lower")

        for (x, y), state in self.states.items():
            ax.text(
                x,
                y,
                f"g={state.g:.1f}\nr={state.rhs:.1f}",
                va="center",
                ha="center",
                color="black",
            )

        plt.show()


def generate_maze(rows, cols):
    maze = np.zeros((rows, cols), dtype=np.int32)  # 创建一个全是墙的矩阵

    # 设置起点
    start = (np.random.randint(rows), np.random.randint(cols))
    maze[start] = 0

    # 随机化深度优先搜索
    """ stack = [start]
    while stack:
        current = stack[-1]
        neighbors = []
        for i, j in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            x, y = current[0] + i, current[1] + j
            if 0 < x < 2 * rows and 0 < y < 2 * cols and maze[x, y] == 1:
                neighbors.append((x, y))
        if neighbors:
            next_cell = neighbors[np.random.randint(len(neighbors))]
            maze[next_cell] = 0
            maze[
                current[0] + (next_cell[0] - current[0]) // 2,
                current[1] + (next_cell[1] - current[1]) // 2,
            ] = 0
            stack.append(next_cell)
        else:
            stack.pop() """
    for i in range(1, rows, 1):
        for j in range(1, cols, 1):
            if np.random.rand() < 0.2:
                maze[i, j] = 1
    start = (np.random.randint(1, rows), np.random.randint(1, cols))
    end = (np.random.randint(1, rows), np.random.randint(1, cols))
    maze[start] = 0
    maze[end] = 0
    return maze, start, end


def draw_maze(maze, start, end):
    rows, cols = maze.shape
    expanded_maze = np.zeros((rows * 2, cols * 2))

    # 膨胀障碍物
    for i in range(rows):
        for j in range(cols):
            if maze[i, j] == 1:
                expanded_maze[i : i * 2 + 2, j * 2 : j * 2 + 2] = 1


if __name__ == "__main__":
    # 示例：生成一个5x5的迷宫
    n = 20
    m = 20
    np.random.seed(32)
    maze, start, end = generate_maze(n, m)

    # 示例用法
    # start = (1, 1)  # 起点坐标
    # goal = (2 * n - 1, 2 * n - 1)  # 目标坐标
    goal = end
    # 第一个坐标是y，第二个坐标是x
    print(f"出发点坐标:{start}")
    print(f"终点坐标:{end}")
    # 网格地图，1表示障碍，0表示可通行

    # print(grid_np)
    dstar = DStarLite(start, goal, maze)  # 创建DStarLite实例
    path, _, _ = dstar.find_path()  # 计算路径
    print("路径:", path)  # 输出路径
    fig, ax = plt.subplots()
    ax.imshow(maze.T, cmap="gray_r", origin="lower", interpolation="nearest")
    start_scatter = ax.scatter(start[0], start[1], color="red", marker="o", s=100)
    goal_scatter = ax.scatter(goal[0], goal[1], color="red", marker="*", s=100)
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, c="red", linewidth=2)
    else:
        print(maze.T)
    """ plt.xlim(-1, n + 1)
    plt.ylim(-1, m + 1)
    plt.xticks(np.arange(-1,n + 1))
    plt.yticks(np.arange(-1,m + 1))
    plt.xlabel("X-Axis Label")
    plt.ylabel("Y-Axis Label") """
    ax.grid(color="black")
    dstar.visualize()

    # start_scatter.set_offsets(np.array([start[1], start[0]]))
    # goal_scatter.set_offsets(np.array([goal[1], goal[0]]))

    # ax.set_xticks(np.arange(5))
    # ax.set_yticks(np.arange(4))
    # ax.set_xticklabels(np.arange(5))
    # ax.set_yticklabels(np.arange(4))
    count = 0
    plt.show()
    raise SystemExit

""" 
import heapq


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float("inf")  # 到达该细胞的成本
        self.rhs = float("inf")  # 启发式成本
        self.parent = None

    def __lt__(self, other):
        return (self.g, self.rhs) < (other.g, other.rhs)


class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.km = 0  # 估算的最短路径长度
        self.cells = {}  # 存储细胞信息
        self.open_list = []  # 存储待扩展的细胞

    def heuristic(self, cell):
        # 启发式函数，例如使用欧几里得距离
        return abs(cell.x - self.goal[0]) + abs(cell.y - self.goal[1])

    def initialize(self):
        # 初始化网格
        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                if self.grid[x][y] != 1:
                    self.cells[(x, y)] = Cell(x, y)
        self.cells[self.goal].rhs = 0
        heapq.heappush(self.open_list, self.cells[self.goal])

    def update_vertex(self, u):
        # 更新细胞u的启发式成本
        if u != self.goal:
            min_rhs = min(
                (
                    self.cells[(x, y)].g + self.cost(u, (x, y))
                    for x, y in self.neighbors(u)
                )
            )
            self.cells[u].rhs = min_rhs
        if self.cells[u] in self.open_list:
            self.open_list.remove(self.cells[u])
        if self.cells[u].g != self.cells[u].rhs:
            heapq.heappush(self.open_list, self.cells[u])

    def compute_shortest_path(self):
        while self.open_list and (
            self.open_list[0].g < self.heuristic(self.start)
            or self.cells[self.start].rhs != self.cells[self.start].g
        ):
            u = heapq.heappop(self.open_list)
            if u.g > u.rhs:
                u.g = u.rhs
                for pred in self.predecessors(u):
                    self.update_vertex(pred)
            else:
                u.g = float("inf")
                self.update_vertex(u)
                for pred in self.predecessors(u):
                    self.update_vertex(pred)

    def predecessors(self, cell):
        # 获取细胞的前驱细胞
        x, y = cell.x, cell.y
        preds = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [p for p in preds if p in self.cells]

    def neighbors(self, cell):
        # 获取细胞的邻居细胞
        x, y = cell.x, cell.y
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [n for n in neighbors if n in self.cells and self.grid[n[0]][n[1]] != 1]

    def cost(self, u, v):
        # 计算细胞u到细胞v的成本
        return 1 if self.grid[v[0]][v[1]] != 1 else float("inf")

    def replan(self):
        # 重新规划路径
        while self.start != self.goal:
            self.compute_shortest_path()
            min_rhs = min(
                (
                    self.cells[(x, y)].g + self.cost(self.start, (x, y))
                    for x, y in self.neighbors(self.start)
                )
            )
            self.km += self.heuristic(self.start)
            self.start = min(
                self.neighbors(self.start),
                key=lambda n: self.cells[n].g + self.cost(self.start, n),
            )
            if min_rhs == float("inf"):
                break


# 示例用法
grid = [[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]

start = (0, 0)
goal = (3, 4)

d_star_lite = DStarLite(grid, start, goal)
d_star_lite.initialize()
d_star_lite.compute_shortest_path()
d_star_lite.replan()
 """
