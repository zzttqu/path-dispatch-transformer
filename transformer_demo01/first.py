import os
import datetime
from typing import Optional, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SequentialSampler
from vit_pytorch import SimpleViT, ViT
from vit_pytorch.vivit import ViT as VIVIT
from env_test import SimEnv
import torch._dynamo

# torch._dynamo.config.suppress_errors = True


# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
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
    def __init__(self,image_size,frames):
        super(TransFormerMy, self).__init__()
        self.vivit=VIVIT(
        image_size=image_size,  # image size
        frames=frames,  # number of frames
        channels=1,
        image_patch_size=16,  # image patch size
        frame_patch_size=2,  # frame patch size
        num_classes=24,
        dim=1024,
        spatial_depth=6,  # depth of the spatial transformer
        temporal_depth=6,  # depth of the temporal transformer
        heads=4,
        mlp_dim=2048,
        )
        self.fc1=nn.Linear(24,3)
        self.fc2=nn.Linear(24,1)

    def forward(self, x: torch.Tensor):
        x=self.vivit(x)
        action = self.fc1(x)
        print(action.shape)
        action = torch.softmax(action, dim=1)
        value: torch.Tensor = torch.mean(self.fc2(x))
        return action, value


class PPOMemory:
    def __init__(self, batch_size: int, agv_num: int, device, max_path_length=100):
        self.device = device
        # 每个的路径
        self.paths = np.zeros((batch_size, agv_num, max_path_length, 2))
        self.path_masks = np.zeros((batch_size, agv_num, max_path_length), dtype=bool)
        self.maps = np.zeros((batch_size, 128, 128))
        self.actions = np.zeros((batch_size, agv_num, max_path_length), dtype=np.int8)
        self.action_log_probs = np.zeros((batch_size, agv_num, max_path_length))
        self.values = np.zeros(batch_size)
        self.rewards = np.zeros(batch_size)
        self.dones = np.zeros(batch_size, dtype=np.int8)
        self.count = 0

    def remember(self, state, masks, action, action_log_prob, value, reward, done):
        # print("jiyi",self.count)
        self.paths[self.count] = state
        self.path_masks[self.count] = masks
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
        torch.Tensor,
    ]:

        self.count = 0
        return (
            torch.from_numpy(self.paths).float().to(self.device),
            torch.from_numpy(self.path_masks).bool().to(self.device),
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
        self.batch_size = 16
        self.mini_batch_size = 1
        self.epochs = 16
        # 第二个参数是agv数量
        self.memory = memory
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.actor_lr)

    def select_action(
        self,
        state: Union[torch.Tensor, np.ndarray],
        path_mask: Union[torch.Tensor, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(self.device)
        if type(path_mask) == np.ndarray:
            path_mask = torch.from_numpy(path_mask).bool().to(self.device)
        probs, _ = self.model(state)
        action_dists = Categorical(probs)
        actions = action_dists.sample()
        log_prob = action_dists.log_prob(actions)
        entropy = action_dists.entropy()
        return actions, log_prob, entropy

    def get_value(
        self,
        state: Union[torch.Tensor, np.ndarray],
        path_mask: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(self.device)
        if type(path_mask) == np.ndarray:
            path_mask = torch.from_numpy(path_mask).bool().to(self.device)
        _, value = self.model(state, path_mask)
        return value

    def learn(self, last_state, last_path_mask, last_done):
        """rewards = [1, 2]
        values = [1, 2]
        dones = [1, 2]"""
        paths, path_masks, actions, action_log_probs, values, rewards, dones = (
            self.memory.generate_batches()
        )
        with torch.no_grad():
            last_value = self.get_value(last_state, last_path_mask)
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
                mini_states: torch.Tensor = paths[index].squeeze()
                mini_masks = path_masks[index].squeeze()
                mini_probs = action_log_probs[index].squeeze()
                actions, new_probs, entropy = self.select_action(
                    mini_states, mini_masks
                )
                ratios = torch.mean(torch.exp(new_probs - mini_probs))
                surr1 = ratios * advantages[index]
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages[index]
                )
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                new_value: torch.Tensor = self.get_value(mini_states, mini_masks)
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
        self.agv_num = 8
        self.env = SimEnv(18, 18, 0.3, self.agv_num)
        self.env.init()
        self.paths, self.path_masks, self.done, self.reward = self.env.get_state()
        self.batch_size = 16
        self.epochs = 10
        self.model = TransFormerMy(32,8)
        
        video = torch.randn(4, 1, 8, 32,32)  # (batch, channels, frames, height, width)
        self.model(video)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = "./model/model.pth"
        if os.path.exists(name):
            model_param = torch.load("./model/model.pth")
            self.model.load_state_dict(model_param)
        # self.model = torch.compile(self.model)
        self.model.to(self.device)
        self.memory = PPOMemory(self.batch_size, self.agv_num, self.device)
        self.agent = PPOAgent(self.model, self.memory, self.device)
        self.step_num = 0

    def step(self):
        self.model.eval()
        with torch.no_grad():
            # print(self.paths, self.path_masks)
            actions, log_prob, entropy = self.agent.select_action(
                self.paths, self.path_masks
            )
            value = self.agent.get_value(self.paths, self.path_masks)
        pos = self.env.update(actions.cpu().numpy())
        next_paths, next_path_masks, done, reward = self.env.get_state()
        # self.env.show(self.agv_num, pos)
        # plt.show()
        # plt.pause(1)
        self.memory.remember(
            self.paths,
            self.path_masks,
            actions.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
            reward,
            done,
        )
        self.paths = next_paths
        self.path_masks = next_path_masks
        self.done = done
        return reward

    def run(self):
        for i in range(0, 2):
            print(i)
            if i % self.batch_size == 15:
                self.agent.learn(self.paths, self.path_masks, self.done)
            reward = self.step()
            print(reward)
            self.step_num += 1


if __name__ == "__main__":
    """ v = ViT(
        image_size=64,  # image size
        frames=8,  # number of frames
        image_patch_size=16,  # image patch size
        frame_patch_size=2,  # frame patch size
        num_classes=3,
        dim=1024,
        spatial_depth=6,  # depth of the spatial transformer
        temporal_depth=6,  # depth of the temporal transformer
        heads=4,
        mlp_dim=2048,
        channels=1,
    )

    video = torch.randn(4, 1, 8, 64, 64)  # (batch, channels, frames, height, width)
    now = datetime.datetime.now()
    print(now)
    for _ in range(10):
        preds = v(video)  # (4, 1000)
        print(preds.shape)
    print("time:", datetime.datetime.now() - now)
    print(sum(p.numel() for p in v.parameters()))
    raise SystemExit """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3407)
    video = torch.randn(4, 1, 8, 64, 64)  # (batch, channels, frames, height, width)
    # np.random.seed(3407)
    train = Train()
    train.run()
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
