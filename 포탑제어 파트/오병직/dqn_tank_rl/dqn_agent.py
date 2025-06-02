# dqn_agent.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA

        self.steps_done = 0
        self.eps_start = config.EPS_START
        self.eps_end = config.EPS_END
        self.eps_decay = config.EPS_DECAY

    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
                action_values = self.policy_net(state)
                action_idx = action_values.max(1)[1].item()
                return self.idx_to_action(action_idx)
        else:
            return self.random_action()

    def idx_to_action(self, idx):
        # discrete action을 continuous 값으로 매핑 (예시)
        action_list = [
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],  # 발사
        ]
        return np.array(action_list[idx], dtype=np.float32)

    def random_action(self):
        idx = random.randint(0, 9)
        return self.idx_to_action(idx)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.tensor(batch_state, dtype=torch.float32).to(self.device)
        batch_action = torch.tensor([self.action_to_idx(a) for a in batch_action], dtype=torch.int64).unsqueeze(1).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32).to(self.device)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(batch_state).gather(1, batch_action)
        next_q_values = self.target_net(batch_next_state).max(1)[0].detach().unsqueeze(1)
        expected_q_values = batch_reward + (1 - batch_done) * self.gamma * next_q_values

        loss = nn.functional.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def action_to_idx(self, action):
        # continuous action -> discrete 인덱스 매핑 (간단 비교)
        action_list = [
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],  # 발사
        ]
        diffs = [np.linalg.norm(np.array(action) - np.array(a)) for a in action_list]
        return int(np.argmin(diffs))
