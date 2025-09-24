#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from langchain_core.messages import BaseMessage

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 200):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_agents)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.silu(self.fc1(state))
        x = self.dropout(x)
        x = F.silu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)

class StateEncoder:
    def __init__(self):
        self.feature_dim = 10
        
    def encode_messages(self, messages: List[BaseMessage]) -> torch.Tensor:
        features = torch.zeros(self.feature_dim)
        
        if not messages:
            return features
            
        last_message = messages[-1].content if messages else ""
        
        features[0] = len(messages)
        features[1] = len(last_message)
        features[2] = len([m for m in messages if m.type == "human"])
        features[3] = len([m for m in messages if m.type == "ai"])
        features[4] = len([m for m in messages if "math" in m.content.lower()])
        features[5] = len([m for m in messages if "research" in m.content.lower()])
        features[6] = 1.0 if any(char.isdigit() for char in last_message) else 0.0
        features[7] = 1.0 if "?" in last_message else 0.0
        features[8] = len(last_message.split())
        features[9] = len(messages) / 100.0
        
        return features

class RLOrchestrator:
    def __init__(self, agent_names: List[str], lr: float = 0.001):
        self.agent_names = agent_names
        self.num_agents = len(agent_names)
        self.state_encoder = StateEncoder()
        
        self.policy_net = PolicyNetwork(
            state_dim=self.state_encoder.feature_dim,
            num_agents=self.num_agents
        )
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.episode_log_probs = []
        self.episode_rewards = []
        
    def select_agent(self, messages: List[BaseMessage]) -> str:
        state = self.state_encoder.encode_messages(messages)
        
        with torch.no_grad():
            action_probs = self.policy_net(state)
            
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        self.episode_log_probs.append(action_dist.log_prob(action))
        
        return self.agent_names[action.item()]
    
    def get_action_probs(self, messages: List[BaseMessage]) -> torch.Tensor:
        state = self.state_encoder.encode_messages(messages)
        return self.policy_net(state)
    
    def reset_episode(self):
        self.episode_log_probs = []
        self.episode_rewards = []