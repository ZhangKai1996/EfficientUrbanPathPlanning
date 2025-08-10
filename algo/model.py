import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, emb_dim=256, hidden_dim=64):
        super(DuelingDQN, self).__init__()
        self.embedding = nn.Embedding(state_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Value 分支
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

        # Advantage 分支
        self.advantage_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.advantage_fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x1 = self.embedding(state[:, 0])
        x2 = self.embedding(state[:, 1])
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)  # [batch_size, 1]
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)  # [batch_size, action_dim]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
