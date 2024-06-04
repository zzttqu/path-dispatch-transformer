from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from dstarlite import DStarLite, State


class SimEnv:
    def __init__(self, cols, rows, obstacle_radio, agv_num, time=16):
        self.cols = cols
        self.rows = rows
        self.obstacle_radio = obstacle_radio
        self.grid_map = np.zeros((agv_num, rows, cols), dtype=np.int8)
        self._grid_map = np.zeros((rows, cols), dtype=np.int8)
        self.grid_map_orgin = np.zeros((rows, cols), dtype=np.int8)
        self.start = np.zeros((agv_num, 2), dtype=np.int8)
        self.goal = np.zeros((agv_num, 2), dtype=np.int8)
        self.position = []
        # 50个agv的路线，最长100个节点
        self.max_path_length = 100
        self.dstar_path: np.ndarray = np.full((agv_num, self.max_path_length, 2), -1)
        self.path_mask = np.ones((agv_num, self.max_path_length), dtype=np.int8)
        self.distance = np.zeros(agv_num)
        self.turn_num = np.zeros(agv_num, dtype=np.int8)
        self.AGVs: list[DStarLite] = []
        self.agv_num = agv_num
        self.episode_step = 0
        self.max_episode_step = 100
        self.Dstar_reward = 0
        self.reward = 0

    def generate_maze(self):
        # 设置起点
        for i in range(1, self.rows, 1):
            for j in range(1, self.cols, 1):
                if np.random.rand() < 0.2:
                    self.grid_map[:, i, j] = 1
                    self.grid_map_orgin[i, j] = 1
        for i in range(self.agv_num):
            start = [
                np.random.randint(1, self.rows - 2),
                np.random.randint(1, self.cols - 2),
            ]
            goal = [
                np.random.randint(1, self.rows - 2),
                np.random.randint(1, self.cols - 2),
            ]
            self.start[i] = start
            self.goal[i] = goal
            self.position.append(start)
            self.grid_map[:, start] = 0
            self.grid_map[:, goal] = 0
            self.grid_map_orgin[start] = 0
            self.grid_map_orgin[goal] = 0
        # 设立边界
        self.grid_map[:, self.rows - 1, :] = 1
        self.grid_map[:, :, self.cols - 1] = 1
        self.grid_map[:, 0, :] = 1
        self.grid_map[:, :, 0] = 1
        self.grid_map_orgin[self.rows - 1, :] = 1
        self.grid_map_orgin[:, self.cols - 1] = 1
        self.grid_map_orgin[0, :] = 1
        self.grid_map_orgin[:, 0] = 1

    def init(self):
        self.generate_maze()
        for i in range(self.agv_num):
            self.AGVs.append(DStarLite(self.start[i], self.goal[i], self.grid_map[i]))
        self.first_step()

    def first_step(self):
        _, _, distance, turn_num = self.get_DStar_Path()

        self.Dstar_reward = sum(distance) + sum(turn_num) / 10

    def get_DStar_Path(self):
        """
        返回路径，掩膜，距离，转向次数
        """
        for i in range(self.agv_num):
            path, distance, turn_num, err = self.AGVs[i].find_path()
            if err == 0:
                self.distance[i] = distance
                if path.shape[0] > self.max_path_length:
                    path = path[: self.max_path_length]
                elif path.shape[0] == 0:
                    path = np.array([self.AGVs[i].start.x, self.AGVs[i].start.y])
                self.dstar_path[i, : path.shape[0], :] = path
                # 填充路径的mask
                self.path_mask[i, : path.shape[0]] = 0
                # 填充位置信息
                # self.dstar_path[i, path.shape[0] :, :] = path[-1, :]
                self.turn_num[i] = turn_num
            elif err == 2:
                # 不更新路径
                # 路径规划失败超出次数
                self.distance[i] = 1000
            elif err == 1:
                # 不更新路径
                # 路径规划失败，无法抵达终点
                # 重置地图
                self.grid_map[i] = self.grid_map_orgin
                self.distance[i] = 1000
        return self.dstar_path, self.path_mask, self.distance, self.turn_num

    def update(self, action: np.ndarray):
        """
        arg 动作

        返回障碍物位置
        """
        self.episode_step += 1
        """ action = np.array(
            [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]],
            dtype=np.int8,
        ) """
        obstacles_pos = []
        # print("33", action[3], ~self.path_mask[3, : action[3].shape[0]])
        # print(np.where((action[3] == 1) & (~self.path_mask[3, : action[3].shape[0]]))[0])
        # self.AGVs[2].visualize()
        self.grid_map[:] = self.grid_map_orgin
        for i in range(action.shape[0]):
            # i = tick_num
            tmp: np.ndarray = self.dstar_path[i][
                np.where(
                    (action[i] == 1)
                    & (np.logical_not(self.path_mask[i, : action[i].shape[0]]))
                )[0]
            ]
            tmp = tmp.astype(np.int8)
            if tmp.shape[0] > 0:
                mask = np.any(tmp != self.start[i], axis=1) & np.any(
                    tmp != self.goal[i], axis=1
                )
                tmp = tmp[mask]
                """ mask = np.any(tmp != self.goal[i], axis=1)
                tmp = tmp[mask] """

            """ for row in tmp:
                if np.array_equal(row, self.start[i]):
                    np.delete(tmp, row)
                    print("shabi2")
                if np.array_equal(row, self.goal[i]):
                    np.delete(tmp, row)
                    print("shabi1") """
            self.grid_map[i][tmp[:, 0], tmp[:, 1]] = 1
            obstacles_pos.append(tmp)
            # print(i)
            """ self.AGVs[tick_num].update_grid(self.grid_map[tick_num], obstacles_pos)
            self.AGVs[tick_num].visualize()
            self.AGVs[tick_num].update_grid(
                self.grid_map[tick_num], obstacles_pos, update=False
            )
            self.AGVs[tick_num].visualize()
            break
 """
            # 重新初始化
            self.AGVs[i] = DStarLite(self.start[i], self.goal[i], self.grid_map[i])
            # self.AGVs[i].update_grid(self.grid_map[i], tmp, compute=False)
            # self.AGVs[i].update_grid(self.grid_map[i], obstacles_pos, compute=False)
            # print(i)
            # self.AGVs[i].visualize()
            ########path, distance, turn_num = self.AGVs[i].find_path()
            ########self.dstar_path[i, : path.shape[0], :] = path
        # self.AGVs[tick_num].visualize()
        # path, distance, turn_num = self.AGVs[tick_num].find_path()
        # self.dstar_path[tick_num, : path.shape[0], :] = path
        # self.distance[0] = distance

        # print(path)
        # 填充路径的mask
        # self.path_mask[0, : path.shape[0]] = False
        #
        # self.get_DStar_Path()
        # self.grid_map[i][tmp] = 1
        #
        return obstacles_pos
        # 遇到障碍物就停下不更新位置
        """ if self.grid_map[action[0]] == 1:
            return
        self.position[0] = action[0] """

    def get_state(self):
        """
        返回D*路线，掩膜，是否结束，奖励
        """
        path_list, mask_list, distance, turn_num = self.get_DStar_Path()
        assert isinstance(path_list, np.ndarray), "必须返回numpy数组"
        # 第i个AGV的路径有这么多步
        # p = np.broadcast_to(aa, (path_length, aa.shape[0], aa.shape[1]))
        # aa = self.grid_map_orgin.copy()
        aa = np.expand_dims(self.grid_map_orgin, axis=0)
        path_length = self.max_path_length - mask_list[0].sum()
        p = np.tile(aa, (path_length, 1, 1))
        indice = (
            np.arange(path_length),
            path_list[0, :path_length, 0],
            path_list[0, :path_length, 1],
        )
        p[indice] = 1
        print(p)
        print(self.max_path_length - mask_list[1].sum())
        reward = self.get_reward(distance, turn_num)
        return path_list, mask_list, self.done, reward

    def get_reward(self, distance, turn_num):
        self.done = 0
        self.episode_step += 1
        if self.episode_step >= self.max_episode_step:
            self.done = 1
        # distance = 0
        reward = self.Dstar_reward - sum(distance) - sum(turn_num) / 10
        if reward > -0.50:
            reward = (1 + reward) * 10
        return reward

    def show(self, agv_num, obstacles_pos=[]):
        # plt.close("all")
        fig, axs = plt.subplots(agv_num // 3 + 1, 3, figsize=(16, 9), dpi=100)
        plt.tight_layout()
        # plt.clf()
        axs[-1, -1].imshow(
            self.grid_map_orgin.T,
            cmap="gray_r",
            origin="lower",
            interpolation="nearest",
        )
        for i in range(agv_num):
            # 障碍物
            axs[i // 3, i % 3].imshow(
                self.grid_map[i].T,
                cmap="gray_r",
                origin="lower",
                interpolation="nearest",
            )
            # 起点
            axs[i // 3, i % 3].scatter(
                self.start[i][0], self.start[i][1], color="red", marker="o", s=100
            )
            # 终点
            axs[i // 3, i % 3].scatter(
                self.goal[i][0], self.goal[i][1], color="red", marker="*", s=100
            )
            # 障碍物变化
            if obstacles_pos and i < len(obstacles_pos) and len(obstacles_pos[i]) != 0:
                for obs_p in obstacles_pos[i]:
                    axs[i // 3, i % 3].scatter(
                        obs_p[0], obs_p[1], color="red", marker="x", s=100
                    )
            axs[i // 3, i % 3].text(self.start[i][0], self.start[i][1], i)
            axs[i // 3, i % 3].text(self.goal[i][0], self.goal[i][1], i)
            path_x, path_y = zip(*self.dstar_path[i])
            axs[i // 3, i % 3].plot(path_x, path_y, c="red", linewidth=2)

    # plt.show()
    # plt.pause(5)


if __name__ == "__main__":
    tick_num = 3
    np.random.seed(52)
    agv_num = 10
    env = SimEnv(5, 5, 0.3, agv_num)
    env.init()
    env.get_DStar_Path()
    print(env.get_state()[-1])
    # env.show(agv_num)
    # print(env.dstar_path)
    # print(env.distance)
    # print(env.time)
    # print(env.path_mask)
    # obstacles_pos = env.update()
    # env.get_DStar_Path()
    # print(env.get_state()[-1])
    # show(agv_num, env, obstacles_pos)
    # print(obstacles_pos)
    # plt.show()
    # print(env.get_reward())
    plt.tight_layout()
    plt.savefig("./env_test.png", dpi=200)
