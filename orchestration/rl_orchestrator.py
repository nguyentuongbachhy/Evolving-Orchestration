#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import List, Dict
from langchain_core.messages import BaseMessage
from collections import defaultdict

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # Multi-layer architecture for better representation
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism for state features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Agent-specific processing
        self.agent_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Final decision layers
        self.decision_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_agents)
        )
        
        # Temperature parameter for exploration control
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state, return_logits=False):
        batch_size = state.shape[0] if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # State embedding
        embedded_state = self.state_embedding(state)
        
        # Self-attention for feature interaction
        attended_state, _ = self.attention(
            embedded_state.unsqueeze(1),
            embedded_state.unsqueeze(1),
            embedded_state.unsqueeze(1)
        )
        attended_state = attended_state.squeeze(1)
        
        # Agent-specific processing
        agent_features = self.agent_projector(attended_state)
        
        # Final decision
        logits = self.decision_layers(agent_features)
        
        if return_logits:
            return logits
        
        # Temperature-scaled softmax for controlled exploration
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=2.0)
        probabilities = F.softmax(scaled_logits, dim=-1)
        
        if batch_size == 1:
            return probabilities.squeeze(0)
        return probabilities
    
    def get_action_distribution(self, state):
        """Get categorical distribution for action sampling"""
        probs = self.forward(state)
        return torch.distributions.Categorical(probs)
    
    def compute_entropy(self, state):
        """Compute entropy for exploration bonus"""
        probs = self.forward(state)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    
    def update_temperature(self, epoch, total_epochs):
        """Adaptive temperature scheduling"""
        decay_rate = 0.99
        min_temp = 0.1
        initial_temp = 2.0

        target_temp = initial_temp * (decay_rate ** epoch)
        target_temp = max(target_temp, min_temp)
        
        with torch.no_grad():
            self.temperature.data = torch.tensor(target_temp)
    
    def get_agent_importance(self, state):
        """Get importance scores for each agent"""
        logits = self.forward(state, return_logits=True)
        return F.softmax(logits, dim=-1)

class StateEncoder:
    def __init__(self):
        self.feature_dim = 24
        self.math_keywords = {'calculate', 'solve', 'math', 'number', 'equation', '+', '-', '*', '/', '=', '**'}
        self.research_keywords = {'search', 'find', 'research', 'information', 'web', 'look', 'investigate'}
        self.code_keywords = {'code', 'python', 'script', 'program', 'function', 'debug', 'execute', 'run'}
        self.summary_keywords = {'summarize', 'conclude', 'final', 'synthesis', 'overall', 'combine', 'integrate'}
        
    def encode_messages(self, messages: List[BaseMessage]) -> torch.Tensor:
        features = torch.zeros(self.feature_dim)
        
        if not messages:
            return features
            
        last_message = messages[-1].content if messages else ""
        all_content = " ".join([m.content for m in messages])
        
        # Basic conversation features
        features[0] = len(messages)
        features[1] = len(last_message)
        features[2] = len([m for m in messages if m.type == "human"])
        features[3] = len([m for m in messages if m.type == "ai"])
        
        # Task type detection
        features[4] = self._calculate_keyword_density(last_message, self.math_keywords)
        features[5] = self._calculate_keyword_density(last_message, self.research_keywords)
        features[6] = self._calculate_keyword_density(last_message, self.code_keywords)
        features[7] = self._calculate_keyword_density(last_message, self.summary_keywords)
        
        # Content complexity
        features[8] = 1.0 if any(char.isdigit() for char in last_message) else 0.0
        features[9] = len(re.findall(r'[?!.]', last_message))
        features[10] = len(last_message.split())
        features[11] = len(set(last_message.lower().split()))
        
        # Conversation dynamics
        features[12] = self._calculate_turn_taking_pattern(messages)
        features[13] = self._calculate_complexity_progression(messages)
        features[14] = self._detect_task_completion_signals(last_message)
        
        # Agent interaction history
        features[15] = self._count_agent_references(all_content)
        features[16] = self._calculate_reasoning_depth(messages)
        
        # Contextual features
        features[17] = self._detect_error_patterns(last_message)
        features[18] = self._calculate_information_density(last_message)
        features[19] = self._detect_collaborative_signals(last_message)
        
        # Multi-agent orchestration features
        features[20] = self._detect_code_patterns(last_message)
        features[21] = self._detect_synthesis_needs(all_content)
        
        # Normalization features
        features[22] = min(len(messages) / 100.0, 1.0)
        features[23] = min(len(last_message) / 1000.0, 1.0)
        
        return features
    
    def _calculate_keyword_density(self, text: str, keywords: set) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        return sum(1 for word in words if any(kw in word for kw in keywords)) / len(words)
    
    def _calculate_turn_taking_pattern(self, messages: List[BaseMessage]) -> float:
        if len(messages) < 2:
            return 0.0
        alternations = sum(1 for i in range(1, len(messages)) 
                          if messages[i].type != messages[i-1].type)
        return alternations / (len(messages) - 1)
    
    def _calculate_complexity_progression(self, messages: List[BaseMessage]) -> float:
        if len(messages) < 2:
            return 0.0
        complexities = [len(m.content.split()) for m in messages]
        return (complexities[-1] - complexities[0]) / max(complexities[0], 1)
    
    def _detect_task_completion_signals(self, text: str) -> float:
        completion_signals = {'final', 'answer', 'result', 'conclusion', 'done', 'complete'}
        return 1.0 if any(signal in text.lower() for signal in completion_signals) else 0.0
    
    def _count_agent_references(self, text: str) -> float:
        agent_refs = ['agent', 'research', 'math', 'calculate', 'search']
        return sum(text.lower().count(ref) for ref in agent_refs)
    
    def _calculate_reasoning_depth(self, messages: List[BaseMessage]) -> float:
        depth_indicators = ['because', 'therefore', 'however', 'moreover', 'furthermore']
        total_depth = sum(sum(indicator in m.content.lower() for indicator in depth_indicators) 
                         for m in messages)
        return min(total_depth / 10.0, 1.0)
    
    def _detect_error_patterns(self, text: str) -> float:
        error_patterns = ['error', 'wrong', 'incorrect', 'mistake', 'failed']
        return 1.0 if any(pattern in text.lower() for pattern in error_patterns) else 0.0
    
    def _calculate_information_density(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        unique_words = len(set(w.lower() for w in words))
        return unique_words / len(words)
    
    def _detect_collaborative_signals(self, text: str) -> float:
        collab_signals = ['help', 'assist', 'collaborate', 'together', 'team']
        return 1.0 if any(signal in text.lower() for signal in collab_signals) else 0.0
    
    def _detect_code_patterns(self, text: str) -> float:
        code_patterns = ['def ', 'import ', 'print(', 'for ', 'if ', 'class ', '()']
        return 1.0 if any(pattern in text.lower() for pattern in code_patterns) else 0.0
    
    def _detect_synthesis_needs(self, text: str) -> float:
        synthesis_indicators = ['multiple', 'combine', 'together', 'both', 'all', 'overall']
        return 1.0 if any(indicator in text.lower() for indicator in synthesis_indicators) else 0.0

class RLOrchestrator:
    def __init__(self, agent_names: List[str], lr: float = 0.001, max_depth: int = 4, max_width: int = 2):
        self.agent_names = agent_names + ["terminate"]
        self.num_agents = len(self.agent_names)
        self.max_depth = max_depth
        self.max_width = max_width
        
        self.state_encoder = StateEncoder()
        self.policy_net = PolicyNetwork(
            state_dim=self.state_encoder.feature_dim,
            num_agents=self.num_agents
        )
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), 
            lr=lr,
            weight_decay=1e-5
        )
        
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        
        # Dynamic orchestration tracking
        self.agent_usage_history = defaultdict(int)
        self.collaboration_graph = defaultdict(set)
        self.reasoning_depth = 0
        self.parallel_explorations = 0
        
        # Cycle detection for graph topology
        self.agent_sequence = []
        self.transition_counts = defaultdict(int)
        
    def reset_episode(self):
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        self.reasoning_depth = 0
        self.parallel_explorations = 0
        self.agent_sequence = []
        
    def select_agent(self, messages: List[BaseMessage], 
                    exploration_bonus: float = 0.1) -> str:
        
        if self._should_terminate_early():
            return "terminate"
            
        state = self.state_encoder.encode_messages(messages)
        self.episode_states.append(state.clone())
        
        # Get action distribution
        action_dist = self.policy_net.get_action_distribution(state)
        
        # Add exploration bonus for less-used agents
        exploration_weights = self._compute_exploration_weights()
        adjusted_probs = action_dist.probs * (1 + exploration_bonus * exploration_weights)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        # Sample action
        adjusted_dist = torch.distributions.Categorical(adjusted_probs)
        action = adjusted_dist.sample()
        
        # Store for policy gradient
        self.episode_log_probs.append(adjusted_dist.log_prob(action))
        self.episode_actions.append(action.item())
        
        selected_agent = self.agent_names[action.item()]
        
        # Update orchestration tracking
        self._update_orchestration_state(selected_agent)
        
        return selected_agent
    
    def _should_terminate_early(self) -> bool:
        # Terminate if max depth reached
        if self.reasoning_depth >= self.max_depth:
            return True
            
        # Terminate if too many parallel explorations
        if self.parallel_explorations >= self.max_width:
            return True
            
        # Detect cycles and terminate if stuck
        if len(self.agent_sequence) >= 3:
            recent_sequence = tuple(self.agent_sequence[-3:])
            if self.transition_counts[recent_sequence] > 2:
                return True
                
        return False
    
    def _compute_exploration_weights(self) -> torch.Tensor:
        weights = torch.ones(self.num_agents)
        total_usage = sum(self.agent_usage_history.values()) + 1
        
        for i, agent_name in enumerate(self.agent_names):
            usage_count = self.agent_usage_history[agent_name]
            # Less used agents get higher weights
            weights[i] = 1.0 / (1.0 + usage_count / total_usage)
            
        return weights
    
    def _update_orchestration_state(self, selected_agent: str):
        self.agent_usage_history[selected_agent] += 1
        self.agent_sequence.append(selected_agent)
        
        # Update collaboration graph
        if len(self.agent_sequence) >= 2:
            prev_agent = self.agent_sequence[-2]
            self.collaboration_graph[prev_agent].add(selected_agent)
            
        # Update depth and width tracking
        if selected_agent != "terminate":
            self.reasoning_depth += 1
            
        # Track transition patterns for cycle detection
        if len(self.agent_sequence) >= 3:
            recent_sequence = tuple(self.agent_sequence[-3:])
            self.transition_counts[recent_sequence] += 1
    
    def get_action_probs(self, messages: List[BaseMessage]) -> torch.Tensor:
        state = self.state_encoder.encode_messages(messages)
        return self.policy_net(state)
    
    def compute_policy_loss(self, rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
        if not self.episode_log_probs:
            return torch.tensor(0.0, requires_grad=True)
        
        # Compute discounted returns
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss with entropy regularization
        policy_loss = []
        entropy_loss = []
        
        for i, log_prob in enumerate(self.episode_log_probs):
            policy_loss.append(-log_prob * returns[i])
            
        # Add entropy bonus for exploration
        for state in self.episode_states:
            entropy = self.policy_net.compute_entropy(state)
            entropy_loss.append(-0.01 * entropy)
            
        total_loss = torch.stack(policy_loss).sum()
        if entropy_loss:
            total_loss += torch.stack(entropy_loss).sum()
            
        return total_loss
    
    def get_orchestration_metrics(self) -> Dict:
        """Get metrics for analyzing orchestration quality"""
        agent_diversity = len(set(self.agent_sequence)) / len(self.agent_names)
        
        # Compute graph density
        total_possible_edges = len(self.agent_names) * (len(self.agent_names) - 1)
        actual_edges = sum(len(connections) for connections in self.collaboration_graph.values())
        graph_density = actual_edges / max(total_possible_edges, 1)
        
        # Cycle detection
        cycles = self._detect_cycles()
        
        return {
            "agent_diversity": agent_diversity,
            "graph_density": graph_density,
            "reasoning_depth": self.reasoning_depth,
            "cycle_count": len(cycles),
            "agent_usage": dict(self.agent_usage_history),
            "collaboration_edges": {k: list(v) for k, v in self.collaboration_graph.items()}
        }
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles in collaboration graph"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.collaboration_graph.get(node, []):
                dfs(neighbor, path + [neighbor])
                
            rec_stack.remove(node)
        
        for agent in self.agent_names:
            if agent not in visited:
                dfs(agent, [agent])
                
        return cycles
    
    def update_policy(self, loss: torch.Tensor):
        """Update policy with gradient clipping"""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()