from typing import Optional, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SequentialSampler
from vit_pytorch import SimpleViT
from env_test import SimEnv


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
    def __init__(self):
        super(TransFormerMy, self).__init__()

        self.embedding = nn.Linear(2, 50)
        self.pos_encoding = PositionalEncoding(50)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(50, 10, 20, 0.1, batch_first=True), 6
        )
        self.fc1 = nn.Linear(50, 3)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor, src_mask=None):
        x = self.embedding(x)
        if src_mask is not None:
            x.masked_fill_(src_mask.unsqueeze(-1), 0)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_mask)
        action = self.fc1(x)
        action = torch.softmax(action, dim=2)
        value: torch.Tensor = torch.mean(self.fc2(x))
        return action, value


class PPOMemory:
    def __init__(self, batch_size, agv_num: int, max_path_length=100):
        # 每个的路径
        self.paths = np.zeros((batch_size, agv_num, max_path_length, 2))
        self.path_masks = np.zeros((batch_size, agv_num, max_path_length))
        self.maps = np.zeros((batch_size, 128, 128))
        self.actions = np.zeros((batch_size, agv_num, max_path_length))
        self.action_log_probs = np.zeros((batch_size, agv_num, max_path_length))
        self.values = np.zeros(batch_size)
        self.rewards = np.zeros(batch_size)
        self.dones = np.zeros(batch_size)
        self.count = 0

    def remember(self, state, masks, action, action_log_prob, value, reward, done):
        #print("jiyi",self.count)
        self.paths[self.count] = state
        self.path_masks[self.count] = masks
        self.actions[self.count] = action.numpy()
        self.action_log_probs[self.count] = action_log_prob.numpy()
        # print(type(value), value.shape)
        self.values[self.count] = value.numpy()
        self.rewards[self.count] = reward
        self.dones[self.count] = done
        self.count += 1

    def generate_batches(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        np.ndarray,
    ]:
        
        self.count = 0
        return (
            self.paths,
            self.path_masks,
            self.actions,
            torch.from_numpy(self.action_log_probs).float(),
            torch.from_numpy(self.values).float(),
            self.rewards,
            self.dones,
        )


class PPOAgent:
    def __init__(
        self,
        model: torch.nn.Module,
        memory: PPOMemory,
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
        self.epochs = 8
        # 第二个参数是agv数量
        self.memory = memory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.actor_lr)

    def select_action(
        self,
        state: Union[torch.Tensor, np.ndarray],
        path_mask: Union[torch.Tensor, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float()
        if type(path_mask) == np.ndarray:
            path_mask = torch.from_numpy(path_mask).bool()
        probs, _ = self.model(state, path_mask)
        action_dists = Categorical(probs)
        actions = action_dists.sample()
        log_prob = action_dists.log_prob(actions)
        entropy = action_dists.entropy()
        return actions, log_prob, entropy

    def get_value(
        self,
        state: Union[torch.Tensor, np.ndarray],
        path_mask: Union[torch.Tensor, np.ndarray],
    ):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float()
        if type(path_mask) == np.ndarray:
            path_mask = torch.from_numpy(path_mask).bool()
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
            rewards = torch.from_numpy(rewards).float()
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
                mini_states = torch.from_numpy(paths[index]).float().squeeze()
                mini_masks = torch.from_numpy(path_masks[index]).bool().squeeze()
                mini_probs = (action_log_probs[index]).squeeze()
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

                critic_loss = F.mse_loss(returns[index], new_value.mean())
                total_loss = actor_loss.mean() + 0.5 * critic_loss

                self.optimizer.zero_grad()
                # print(total_loss.dtype)
                total_loss.backward()
                self.optimizer.step()


class Train:
    def __init__(self):
        self.agv_num = 4
        self.env = SimEnv(13, 13, 0.2, self.agv_num)
        self.env.init()
        self.paths, self.path_masks, self.done, self.reward = self.env.get_state()
        self.batch_size = 16
        self.epochs = 10
        self.model = TransFormerMy()
        self.memory = PPOMemory(self.batch_size, self.agv_num)
        self.agent = PPOAgent(self.model, self.memory)
        self.step_num = 0

    def step(self):
        self.model.eval()
        with torch.no_grad():
            #print(self.paths, self.path_masks)
            actions, log_prob, entropy = self.agent.select_action(
                self.paths, self.path_masks
            )
            value = self.agent.get_value(self.paths, self.path_masks)
        pos = self.env.update(actions.numpy())
        next_paths, next_path_masks, done, reward = self.env.get_state()
        self.env.show(self.agv_num, pos)
        # plt.show()
        # plt.pause(1)
        self.memory.remember(
            self.paths, self.path_masks, actions, log_prob, value, reward, done
        )
        self.paths = next_paths
        self.path_masks = next_path_masks
        self.done = done

    def run(self):
        for i in range(0, 1000):
            print(i)
            if i % self.batch_size == 1:
                self.agent.learn(self.paths, self.path_masks, self.done)
            self.step()
            self.step_num += 1


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    """ 
    v = SimpleViT(
        channels=1,
        image_size=256,
        patch_size=32,
        num_classes=100,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
    )

    img = torch.randn(1, 1, 256, 256)

    preds = v(img)  # (1, 1000)
    print(preds) """
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
