import os
import datetime
from loguru import logger
from typing import Optional, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SequentialSampler

from torch.utils.tensorboard.writer import SummaryWriter
from vit_pytorch.vit_3d import ViT
from vit_pytorch.simple_vit_3d import SimpleViT
from vit_pytorch.vivit import ViT as VIVIT
from env_test import SimEnv
import torch._dynamo

# from transformers import VivitModel,VivitConfig
# torch._dynamo.config.suppress_errors = True
# vit=VivitModel(VivitConfig(),)
# output=vit(torch.randn(2,32,3,224,224))
# logger.info(output.last_hidden_state)
# logger.info(output.last_hidden_state.shape)


# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(-2)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class TransFormerMy(nn.Module):
    def __init__(self, image_size: int, frames: int):
        super(TransFormerMy, self).__init__()
        frame_patch_size = 8
        output = frames * 8
        self.frames = frames
        self.vivit = VIVIT(
            image_size=image_size,  # image size
            frames=frames,  # number of frames
            channels=1,
            image_patch_size=16,  # image patch size
            frame_patch_size=frame_patch_size,  # frame patch size
            num_classes=output,
            dim=512,
            spatial_depth=4,  # depth of the spatial transformer
            temporal_depth=6,  # depth of the temporal transformer
            heads=4,
            mlp_dim=512,
        )
        """ self.simple_vit = SimpleViT(
            image_size=image_size,  # image size
            frames=frames,  # number of frames
            channels=1,
            image_patch_size=8,  # image patch size
            frame_patch_size=1,  # frame patch size
            num_classes=24,
            dim=512,
            depth=4,  # depth of the spatial transformer
            heads=4,
            mlp_dim=1024,
        ) """
        self.fc1 = nn.Linear(output // frame_patch_size, 2)
        self.fc2 = nn.Linear(output // frame_patch_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # logger.info(x.shape)
        x = self.vivit(x)
        # logger.info(self.frames)
        # 需要折叠张量使得连续的空间信息可以被处理，变成我需要的原来的frame长度
        x = x.view(x.shape[0], self.frames, -1)
        # logger.info(x.shape)
        action: torch.Tensor = self.fc1(x)
        action: torch.Tensor = torch.softmax(action, dim=1)
        value: torch.Tensor = torch.mean(self.fc2(x))
        return action, value


class PPOMemory:
    def __init__(
        self,
        batch_size: int,
        agv_num: int,
        device: torch.device,
        image_size,
        max_path_length=128,
    ):
        self.device = device
        # 每个的路径
        self.paths = np.zeros(
            (batch_size, agv_num, 1, max_path_length, image_size, image_size)
        )
        self.maps = np.zeros((batch_size, max_path_length, 128, 128))
        self.actions = np.zeros((batch_size, agv_num, max_path_length), dtype=np.int8)
        self.action_log_probs = np.zeros((batch_size, agv_num, max_path_length))
        self.values = np.zeros(batch_size)
        self.rewards = np.zeros(batch_size)
        self.dones = np.zeros(batch_size, dtype=np.int8)
        self.count = 0

    def remember(self, state, action, action_log_prob, value, reward, done):
        # print("jiyi",self.count)
        self.paths[self.count] = state
        self.actions[self.count] = action
        self.action_log_probs[self.count] = action_log_prob
        # print(type(value), value.shape)
        self.values[self.count] = value
        self.rewards[self.count] = reward
        self.dones[self.count] = done
        self.count += 1

    def generate_batches(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        self.count = 0
        return (
            torch.from_numpy(self.paths).float().to(self.device),
            torch.from_numpy(self.actions).float().to(self.device),
            torch.from_numpy(self.action_log_probs).float().to(self.device),
            torch.from_numpy(self.values).float().to(self.device),
            torch.from_numpy(self.rewards).float().to(self.device),
            torch.from_numpy(self.dones).to(self.device),
        )


class PPOAgent:
    def __init__(
        self,
        model: torch.nn.Module,
        memory: PPOMemory,
        device: torch.device,
        batch_size,
        lr=0.001,
        gamma=0.99,
        eps_clip=0.2,
    ):
        self.actor_lr = lr
        self.critic_lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.batch_size = batch_size
        self.mini_batch_size = 1
        self.epochs = 8
        # 第二个参数是agv数量
        self.memory = memory
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.actor_lr)

    def select_action(
        self,
        state: Union[torch.Tensor, np.ndarray],
        path_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """选择动作

        Args:
            state (Union[torch.Tensor, np.ndarray]): 当前状态，由栅格地图视频组成
            path_mask (Union[torch.Tensor, np.ndarray]): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_ 返回动作，动作对数概率和选择熵
        """
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(self.device)
        # if type(path_mask) == np.ndarray:
        #     path_mask = torch.from_numpy(path_mask).bool().to(self.device)
        # logger.info(f"state形状是:{state.shape}")
        probs, _ = self.model(state)
        action_dists = Categorical(probs)
        actions = action_dists.sample()
        log_prob = action_dists.log_prob(actions)
        entropy = action_dists.entropy()
        return actions, log_prob, entropy

    def get_value(
        self,
        state: Union[torch.Tensor, np.ndarray],
        path_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(self.device)
        if type(path_mask) == np.ndarray:
            path_mask = torch.from_numpy(path_mask).bool().to(self.device)
        _, value = self.model(state)
        return value

    def learn(self, last_state, last_done):
        """rewards = [1, 2]
        values = [1, 2]
        dones = [1, 2]"""
        paths, actions, action_log_probs, values, rewards, dones = (
            self.memory.generate_batches()
        )
        with torch.no_grad():
            last_value = self.get_value(last_state)
            advantages = torch.zeros_like(rewards, dtype=torch.float32)
            for t in reversed(range(self.batch_size)):
                last_gae_lam = 0
                if t == self.batch_size - 1:
                    last_gae_lam = 0
                    if t == self.batch_size - 1:
                        next_nonterminal = 1.0 - last_done
                        next_values = last_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.gamma * next_values * next_nonterminal
                        - values[t]
                    )
                    last_gae_lam = (
                        delta
                        + self.gamma * self.gae_lambda * next_nonterminal * last_gae_lam
                    )
                    advantages[t] = last_gae_lam
            # print(advantages.dtype, values.dtype)
            returns = advantages + values
        # returns = torch.from_numpy(returns)
        for _ in range(self.epochs):
            for index in BatchSampler(
                SequentialSampler(range(self.batch_size)), self.mini_batch_size, False
            ):
                mini_states: torch.Tensor = paths[index][0]
                mini_probs = action_log_probs[index][0]
                # logger.info(mini_states.shape)
                actions, new_probs, entropy = self.select_action(mini_states)
                ratios = torch.mean(torch.exp(new_probs - mini_probs))
                surr1 = ratios * advantages[index]
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages[index]
                )
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                new_value: torch.Tensor = self.get_value(mini_states)
                # print(returns[index].shape,new_value.shape)
                # print(returns[index],new_value.unsqueeze_(0))

                critic_loss = F.mse_loss(returns[index], new_value.unsqueeze_(0))
                total_loss = actor_loss.mean() + 0.5 * critic_loss

                self.optimizer.zero_grad()
                # print(total_loss.dtype)
                total_loss.backward()
                self.optimizer.step()
        torch.save(self.model.state_dict(), "./model/model.pth")


class Train:
    def __init__(self):
        self.writer = SummaryWriter(
            log_dir="./logs/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        self.agv_num = 8
        self.max_frme_num = 64
        self.map_size = 32
        self.env = SimEnv(
            self.map_size,
            self.map_size,
            0.4,
            self.agv_num,
            max_frame_num=self.max_frme_num,
        )
        self.env.init()
        self.paths, self.done, self.reward = self.env.get_state()
        self.batch_size = 16
        self.epochs = 8
        self.model = TransFormerMy(self.map_size, 64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = "./model/model.pth"
        if os.path.exists(name):
            model_param = torch.load("./model/model.pth")
            self.model.load_state_dict(model_param)
        # self.model = torch.compile(self.model)
        self.model.to(self.device)
        self.memory = PPOMemory(
            self.batch_size,
            self.agv_num,
            self.device,
            self.map_size,
            max_path_length=self.max_frme_num,
        )
        self.agent = PPOAgent(self.model, self.memory, self.device, self.batch_size)
        self.step_num = 0

    def step(self, axs):
        self.model.eval()
        with torch.no_grad():
            # print(self.paths, self.path_masks)
            actions, log_prob, entropy = self.agent.select_action(self.paths)
            value = self.agent.get_value(self.paths)
        # logger.info(f"action:形状{actions.shape}")
        pos = self.env.update(actions.cpu().numpy())
        next_paths, done, reward = self.env.get_state()
        # self.env.show(axs, pos)
        # logger.info(next_paths.shape)
        # self.env.show(self.agv_num, pos)
        # plt.show()
        # plt.pause(0.1)
        self.memory.remember(
            self.paths,
            actions.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
            reward,
            done,
        )
        self.paths = next_paths
        self.done = done
        return reward

    def run(self, axs):
        for i in range(0, 1):
            if i % self.batch_size == self.batch_size - 1:
                self.agent.learn(self.paths, self.done)
            reward = self.step(axs)
            self.writer.add_scalar("reward", reward, i)
            self.step_num += 1


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    # region
    vv = ViT(
        image_size=64,  # image size
        frames=64,  # number of frames
        image_patch_size=16,  # image patch size
        frame_patch_size=4,  # frame patch size
        num_classes=3,
        dim=1024,
        heads=4,
        mlp_dim=2048,
        channels=1,
        depth=6,
    ).cuda()
    vvv = SimpleViT(
        image_size=64,  # image size
        frames=64,  # number of frames
        image_patch_size=16,  # image patch size
        frame_patch_size=4,  # frame patch size
        num_classes=3,
        dim=1024,
        heads=4,
        mlp_dim=2048,
        channels=1,
        depth=6,
    ).cuda()
    v = VIVIT(
        image_size=64,  # image size
        frames=64,  # number of frames
        image_patch_size=16,  # image patch size
        frame_patch_size=1,  # frame patch size
        num_classes=3,
        dim=1024,
        spatial_depth=6,  # depth of the spatial transformer
        temporal_depth=6,  # depth of the temporal transformer
        heads=4,
        mlp_dim=2048,
        channels=1,
    ).cuda()

    video = torch.randn(
        4, 1, 8, 64, 64
    ).cuda()  # (batch, channels, frames, height, width)
    now = datetime.datetime.now()
    logger.info(now)
    for _ in range(10):
        # preds = v(video)  # (4, 1000)
        # preds = vv(video)  # (4, 1000)
        preds = vvv(video)  # (4, 1000)
        # print(preds.shape)
    logger.info(f"time:{datetime.datetime.now() - now}")
    logger.info(sum(p.numel() for p in v.parameters()))
    logger.info(sum(p.numel() for p in vv.parameters()))
    logger.info(sum(p.numel() for p in vvv.parameters()))
    # endregion
    # raise SystemExit
    # fig, axs = plt.subplots(8 // 3 + 1, 3, figsize=(16, 9), dpi=90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3407)
    np.random.seed(3410)
    train = Train()
    train.run(None)
    raise SystemExit
    x = torch.randint(0, 10, (3, 10, 2))
    x = x.float()
    x[0, 4:, :] = float(-1)
    x[1, 5:, :] = float(-1)
    src_mask = torch.full((3, 10), False)
    src_mask[0, 4:] = True
    src_mask[1, 5:] = True
    # print(src_mask)
    model = TransFormerMy()
    ppo = PPOAgent(model)
    p, _ = model(x, src_mask)
    # print(p)
    print(p.shape)
    dist = Categorical(p)
    # 三个选项，该位置是绕，停还是不变
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
    print(total_params)
