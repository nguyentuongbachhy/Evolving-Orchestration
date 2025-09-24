import torch
import numpy as np
from typing import List, Dict, Tuple
from orchestration.rl_orchestrator import RLOrchestrator
from orchestration.reward_system import RewardSystem
from utils.cost_tracker import CostTracker

class TrainingManager:
    def __init__(self, 
                 orchestrator: RLOrchestrator,
                 reward_system: RewardSystem,
                 gamma: float = 0.99):
        self.orchestrator = orchestrator
        self.reward_system = reward_system
        self.gamma = gamma
        
        self.training_history = []
        self.episode_rewards = []
        self.episode_costs = []
        
    def train_episode(self, 
                     task: str, 
                     expected_answer: str,
                     agent_executor_fn) -> Dict:
        
        cost_tracker = CostTracker()
        cost_tracker.start_episode()
        
        self.orchestrator.reset_episode()
        
        result, episode_info = agent_executor_fn(
            task, 
            self.orchestrator, 
            cost_tracker
        )
        
        reward_components = self.reward_system.get_reward_components(
            result, expected_answer, cost_tracker
        )
        
        total_reward = reward_components["total_reward"]
        
        loss = self._compute_policy_loss(total_reward)
        
        self.orchestrator.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.orchestrator.policy_net.parameters(), 1.0)
        self.orchestrator.optimizer.step()
        
        episode_stats = {
            "reward": total_reward,
            "loss": loss.item(),
            "cost_stats": cost_tracker.get_episode_stats(),
            "reward_components": reward_components,
            "result": result
        }
        
        self.training_history.append(episode_stats)
        self.episode_rewards.append(total_reward)
        self.episode_costs.append(cost_tracker.get_total_cost())
        
        return episode_stats
        
    def _compute_policy_loss(self, reward: float) -> torch.Tensor:
        if not self.orchestrator.episode_log_probs:
            return torch.tensor(0.0, requires_grad=True)
            
        policy_loss = []
        for log_prob in self.orchestrator.episode_log_probs:
            policy_loss.append(-log_prob * reward)
            
        return torch.stack(policy_loss).sum()
        
    def train_batch(self, 
                   training_tasks: List[Tuple[str, str]], 
                   agent_executor_fn,
                   num_epochs: int = 1) -> List[Dict]:
        
        batch_results = []
        
        for epoch in range(num_epochs):
            epoch_results = []
            
            for task, expected in training_tasks:
                episode_stats = self.train_episode(
                    task, expected, agent_executor_fn
                )
                epoch_results.append(episode_stats)
                
            batch_results.extend(epoch_results)
            
            if (epoch + 1) % 10 == 0:
                avg_reward = np.mean([r["reward"] for r in epoch_results])
                avg_cost = np.mean([r["cost_stats"]["total_cost"] for r in epoch_results])
                print(f"Epoch {epoch+1}: Avg Reward = {avg_reward:.3f}, Avg Cost = {avg_cost:.3f}")
                
        return batch_results
        
    def get_training_metrics(self, window_size: int = 100) -> Dict:
        if not self.training_history:
            return {}
            
        recent_rewards = self.episode_rewards[-window_size:]
        recent_costs = self.episode_costs[-window_size:]
        
        return {
            "total_episodes": len(self.training_history),
            "avg_reward": np.mean(recent_rewards),
            "reward_std": np.std(recent_rewards),
            "avg_cost": np.mean(recent_costs),
            "cost_std": np.std(recent_costs),
            "reward_trend": self._calculate_trend(recent_rewards),
            "cost_trend": self._calculate_trend(recent_costs)
        }
        
    def _calculate_trend(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        return np.polyfit(x, y, 1)[0]
        
    def save_model(self, filepath: str):
        torch.save({
            'policy_net_state_dict': self.orchestrator.policy_net.state_dict(),
            'optimizer_state_dict': self.orchestrator.optimizer.state_dict(),
            'training_history': self.training_history,
            'episode_rewards': self.episode_rewards,
            'episode_costs': self.episode_costs
        }, filepath)
        
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.orchestrator.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.orchestrator.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_costs = checkpoint['episode_costs']