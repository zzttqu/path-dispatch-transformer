from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from dstarlite import DStarLite, State
from loguru import logger


class SimEnv:
    def __init__(self, cols, rows, obstacle_radio, agv_num, max_frame_num=128):
        self.cols = cols
        self.rows = rows
        self.obstacle_radio = obstacle_radio
        self.agv_num = agv_num
        self.grid_map = np.zeros((agv_num, rows, cols), dtype=np.int8)
        self.grid_map_orgin = np.zeros((rows, cols), dtype=np.int8)
        self.start = np.zeros((agv_num, 2), dtype=np.int8)
        self.goal = np.zeros((agv_num, 2), dtype=np.int8)
        self.position = []
        # 50个agv的路线，最长100个节点
        self.path_length_limit = 128
        self.dstar_path: np.ndarray = np.full((agv_num, self.path_length_limit, 2), -1)
        self.path_length: np.ndarray = np.zeros((agv_num), dtype=np.int8)
        self.path_mask = np.ones((agv_num, self.path_length_limit), dtype=np.int8)
        self.distance = np.zeros(agv_num)
        self.turn_num = np.zeros(agv_num, dtype=np.int8)
        self.AGVs: list[DStarLite] = []
        # 视频帧数
        self.max_frame_num = max_frame_num
        self.grid_video = np.zeros(
            (
                self.agv_num,
                1,
                max_frame_num,
                self.grid_map_orgin.shape[0],
                self.grid_map_orgin.shape[1],
            )
        )

        self.episode_step = 0
        self.max_episode_step = 100
        self.Dstar_reward = 0
        self.reward = 0
        self.done = 0

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
        self.update()
        _, _, reward = self.get_state()
        self.Dstar_reward = reward

    def get_DStar_Path(self):
        """
        返回路径，掩膜，距离，转向次数
        """
        max_path_length = 0
        for i in range(self.agv_num):
            path, distance, turn_num, err = self.AGVs[i].find_path()
            if err == 0:
                path_length = path.shape[0]
                self.path_length[i] = path_length
                if path_length > max_path_length:
                    max_path_length = path_length
                self.distance[i] = distance
                if path.shape[0] > self.path_length_limit:
                    path = path[: self.path_length_limit]
                elif path.shape[0] == 0:
                    path = np.array([self.AGVs[i].start.x, self.AGVs[i].start.y])
                self.dstar_path[i, :path_length, :] = path
                # 把最后停止位置保持到最后一位
                self.dstar_path[i, path_length:, :] = path[-1]
                # 填充路径的mask
                self.path_mask[i, :path_length] = 0
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

        return (self.dstar_path, self.distance, self.turn_num, max_path_length)

    def update(self, action: Optional[np.ndarray] = None):
        """
        arg 动作

        返回障碍物位置
        """
        self.episode_step += 1
        if action is None:
            action = np.zeros((self.agv_num, self.max_frame_num))
            self.episode_step -= 1
        obstacles_pos = []
        stop_pos = []
        tmp_stop = np.zeros((self.agv_num, self.max_frame_num), dtype=np.int8)
        # print("33", action[3], ~self.path_mask[3, : action[3].shape[0]])
        # print(np.where((action[3] == 1) & (~self.path_mask[3, : action[3].shape[0]]))[0])
        # self.AGVs[2].visualize()
        self.grid_map[:] = self.grid_map_orgin
        for i in range(self.agv_num):
            # i = tick_num
            tmp_obstacle: np.ndarray = self.dstar_path[i][
                np.where((action[i, : self.path_length[i]] == 1))[0]
            ]
            _tmp_stop: np.ndarray = np.where(action[i, : self.path_length[i]] == 2)[0]

            # _full_size_tmp_obs = np.full(self.max_frame_num, tmp_obstacle[-1])
            #  找到哪些时刻停下了
            _tmp_stop = _tmp_stop.astype(np.int8)
            group_number = 0
            for frame in range(self.max_frame_num):
                if frame in _tmp_stop:
                    group_number += 1
                    # 把停止位置的坐标设置为group_number
                # 0的时候减一就会报错，所以0位置保持不变
                if frame == 0:
                    continue
                tmp_stop[i, frame] = group_number
            tmp_obstacle = tmp_obstacle.astype(np.int8)
            # 筛选出所有的障碍物坐标，排除起点和终点
            if tmp_obstacle.shape[0] > 0:
                mask = np.any(tmp_obstacle != self.start[i], axis=1) & np.any(
                    tmp_obstacle != self.goal[i], axis=1
                )
                tmp_obstacle = tmp_obstacle[mask]
            self.grid_map[i][tmp_obstacle[:, 0], tmp_obstacle[:, 1]] = 1
            obstacles_pos.append(tmp_obstacle)
            stop_pos.append(self.dstar_path[i][_tmp_stop])
            # print(i)
            # 重新初始化
            self.AGVs[i] = DStarLite(self.start[i], self.goal[i], self.grid_map[i])
            # self.AGVs[i].update_grid(self.grid_map[i], tmp, compute=False)
            # self.AGVs[i].update_grid(self.grid_map[i], obstacles_pos, compute=False)
            # print(i)
            # self.AGVs[i].visualize()
        path_list, distance, turn_num, max_path_length = self.get_DStar_Path()
        assert isinstance(path_list, np.ndarray), "必须返回numpy数组"
        # logger.info(tmp_stop)
        for step in range(max_path_length):
            for agv in range(self.agv_num):
                aa = self.grid_map_orgin[:].astype(np.float32)
                for _agv in range(self.agv_num):
                    # 当这个agv等于他自己的时候，要设置为0.5，突出特点
                    if _agv == agv:
                        aa[
                            path_list[_agv, step - tmp_stop[_agv, step], 0],
                            path_list[_agv, step - tmp_stop[_agv, step], 1],
                        ] += 0.5
                    else:
                        aa[
                            path_list[_agv, step - tmp_stop[_agv, step], 0],
                            path_list[_agv, step - tmp_stop[_agv, step], 1],
                        ] += 0.8
                # 大于1.0就说明两车处于同一位置，就说明相撞了
                self.grid_video[agv, 0, step] = aa
        if action is None:
            self.Dstar_reward: float = self.get_reward(
                distance, self.grid_video, turn_num
            )
        else:
            self.reward: float = self.get_reward(distance, self.grid_video, turn_num)
        # for agv in range(self.agv_num):
        #     aa = np.expand_dims(self.grid_map_orgin, axis=0).astype(np.float32)
        #     # path_length = self.max_path_length - mask_list[agv].sum()
        #     p = np.tile(aa, (self.max_frame_num, 1, 1))
        #     # p[:, path_coords[:, 0], path_coords[:, 1]] = 0.5
        #     for step in range(max_path_length):
        #         logger.info(f"{tmp_stop[agv,step]}")
        #         for p in tmp_stop:
        #             pass
        #         for _agv in range(self.agv_num):
        #             # 当这个agv等于他自己的时候，要设置为0.5，突出特点
        #             if _agv == agv:
        #                 p[
        #                     step, path_list[_agv, step, 0], path_list[_agv, step, 1]
        #                 ] += 0.5
        #             else:
        #                 p[
        #                     step, path_list[_agv, step, 0], path_list[_agv, step, 1]
        #                 ] += 0.8
        #     # 大于1.0就说明两车处于同一位置，就说明相撞了
        #     self.grid_video[agv, 0] = p
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
        return obstacles_pos, stop_pos
        # 遇到障碍物就停下不更新位置
        """ if self.grid_map[action[0]] == 1:
            return
        self.position[0] = action[0] """

    def get_state(self):
        """
        返回D*路线，掩膜，是否结束，奖励
        """

        return self.grid_video, self.done, self.reward

    def get_reward(self, video, distance, turn_num) -> float:
        self.done = 0
        self.episode_step += 1
        if self.episode_step >= self.max_episode_step:
            self.done = 1
        # distance = 0
        reward = -np.sum(distance) - np.sum(turn_num) / 10
        # TODO 这里可以优化为numpy
        conflict_num = []
        for frame in self.grid_video:
            filtered_elements = frame[(frame > 1.1)]
            conflict_num.append(np.sum(filtered_elements))
        # 出现路径冲突就扣分
        reward -= max(conflict_num)
        logger.warning(f"冲突次数:{max(conflict_num)/1.3:.2f}")
        # 因为这两个都是负数
        reward = reward - self.Dstar_reward
        if reward > -0.50:
            reward = (1 + reward) * 5

        return reward

    def show(self, axs=None, obstacles_pos=[], stop_pos=[]):
        agv_num = self.agv_num
        # plt.close("all")
        if axs is None:
            fig, axs = plt.subplots(agv_num // 3 + 1, 3, figsize=(16, 9), dpi=90)
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
            if stop_pos and i < len(stop_pos) and len(stop_pos[i]) != 0:
                for stop_p in stop_pos[i]:
                    axs[i // 3, i % 3].scatter(
                        stop_p[0], stop_p[1], color="blue", marker="x", s=100
                    )
            axs[i // 3, i % 3].text(
                self.start[i][0],
                self.start[i][1],
                f"{i}开始{self.start[i][0]},{self.start[i][1]}",
                ha="center",
                va="center",
            )
            axs[i // 3, i % 3].text(
                self.goal[i][0],
                self.goal[i][1],
                f"{i}目标{self.goal[i][0]},{self.goal[i][1]}",
                ha="center",
                va="center",
            )
            path_x, path_y = zip(*self.dstar_path[i])
            axs[i // 3, i % 3].plot(path_x, path_y, c="red", linewidth=2)

    # plt.show()
    # plt.pause(5)


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    tick_num = 2
    np.random.seed(52)
    agv_num = 3
    env = SimEnv(7, 7, 0.3, agv_num)
    env.init()
    env.get_DStar_Path()
    print(env.get_state()[-1])
    print(env.get_state()[-1])
    env.show()
    # print(env.dstar_path)
    # print(env.distance)
    # print(env.time)
    # print(env.path_mask)
    # obstacles_pos = env.update()
    # env.get_DStar_Path()
    # print(env.get_state()[-1])
    # show(agv_num, env, obstacles_pos)
    # print(obstacles_pos)
    plt.show()
    # print(env.get_reward())
    plt.tight_layout()
    plt.savefig("./env_test.png", dpi=90)
