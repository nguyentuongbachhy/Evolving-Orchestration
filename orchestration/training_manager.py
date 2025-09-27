import torch
import numpy as np
from typing import List, Dict, Tuple, Callable
from orchestration.rl_orchestrator import RLOrchestrator
from orchestration.reward_system import RewardSystem
from utils.cost_tracker import CostTracker

class TrainingManager:
    def __init__(self, 
                 orchestrator: RLOrchestrator,
                 reward_system: RewardSystem,
                 learning_rate: float = 0.001):
        self.orchestrator = orchestrator
        self.reward_system = reward_system
        self.learning_rate = learning_rate
        
        # From paper page 4: γ = 0.99
        self.gamma = reward_system.gamma
        
        self.training_history = []
        self.episode_rewards = []
        self.episode_costs = []
        
    def train_episode(self, 
                     task: str, 
                     expected_answer: str,
                     agent_executor_fn: Callable) -> Dict:
        """Train single episode using REINFORCE (paper page 4)"""
        
        cost_tracker = CostTracker()
        cost_tracker.start_episode()
        
        self.orchestrator.reset_episode()
        
        # Execute task with current policy
        result, episode_info = agent_executor_fn(
            task, 
            self.orchestrator, 
            cost_tracker
        )
        
        step_count = episode_info.get("steps", 1)
        
        # Calculate trajectory rewards using paper's formula
        trajectory_rewards = self.reward_system.calculate_trajectory_rewards(
            result, expected_answer, cost_tracker, step_count
        )
        
        # REINFORCE policy update (paper equation 5)
        loss = self._compute_reinforce_loss(trajectory_rewards)
        
        # Gradient ascent: θ ← θ + α∇θJ(θ)
        self.orchestrator.update_policy(loss)
        
        # Get orchestration metrics for analysis
        orchestration_metrics = self.orchestrator.get_orchestration_metrics()
        
        episode_stats = {
            "reward": trajectory_rewards[-1] if trajectory_rewards else 0.0,
            "loss": loss.item(),
            "cost_stats": cost_tracker.get_episode_stats(),
            "orchestration_metrics": orchestration_metrics,
            "result": result,
            "step_count": step_count,
            "trajectory_rewards": trajectory_rewards
        }
        
        self.training_history.append(episode_stats)
        self.episode_rewards.append(episode_stats["reward"])
        self.episode_costs.append(cost_tracker.get_total_cost())
        
        return episode_stats
        
    def _compute_reinforce_loss(self, trajectory_rewards: List[float]) -> torch.Tensor:
        """REINFORCE loss from paper equation (5): ∇θJ(θ) ≈ (1/N)Σ∇θlog πθ(at|St)·R(τ)"""
        
        if not self.orchestrator.episode_log_probs or not trajectory_rewards:
            return torch.tensor(0.0, requires_grad=True)
        
        # Compute discounted returns
        returns = self._compute_discounted_returns(trajectory_rewards)
        
        # Baseline subtraction for variance reduction (standard REINFORCE practice)
        if len(returns) > 1:
            baseline = returns.mean()
            returns = returns - baseline
        
        # REINFORCE gradient: -log π(a|s) * R
        policy_loss = []
        for i, log_prob in enumerate(self.orchestrator.episode_log_probs):
            if i < len(returns):
                policy_loss.append(-log_prob * returns[i])
                
        return torch.stack(policy_loss).sum() if policy_loss else torch.tensor(0.0, requires_grad=True)
        
    def _compute_discounted_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns with γ from paper"""
        discounted_returns = []
        discounted_sum = 0
        
        # Backward computation for efficiency
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_returns.insert(0, discounted_sum)
            
        return torch.tensor(discounted_returns, dtype=torch.float32)
        
    def train_batch(self, 
                   training_tasks: List[Tuple[str, str]], 
                   agent_executor_fn: Callable,
                   num_epochs: int = 1) -> List[Dict]:
        """Train on batch of tasks"""
        
        batch_results = []
        
        for epoch in range(num_epochs):
            epoch_results = []
            
            # Update policy network temperature for exploration control
            self.orchestrator.policy_net.update_temperature(epoch, num_epochs)
            
            for task, expected in training_tasks:
                episode_stats = self.train_episode(
                    task, expected, agent_executor_fn
                )
                epoch_results.append(episode_stats)
                
            batch_results.extend(epoch_results)
            
            # Log progress periodically
            if (epoch + 1) % 10 == 0:
                avg_reward = np.mean([r["reward"] for r in epoch_results])
                avg_cost = np.mean([r["cost_stats"]["total_cost"] for r in epoch_results])
                avg_steps = np.mean([r["step_count"] for r in epoch_results])
                
                print(f"Epoch {epoch+1}: Avg Reward = {avg_reward:.3f}, "
                      f"Avg Cost = {avg_cost:.3f}, Avg Steps = {avg_steps:.1f}")
                
        return batch_results
        
    def get_training_metrics(self, window_size: int = 100) -> Dict:
        """Get training progress metrics"""
        if not self.training_history:
            return {}
            
        recent_episodes = self.training_history[-window_size:]
        recent_rewards = [ep["reward"] for ep in recent_episodes]
        recent_costs = [ep["cost_stats"]["total_cost"] for ep in recent_episodes]
        recent_steps = [ep["step_count"] for ep in recent_episodes]
        
        # Orchestration quality metrics
        orchestration_data = [ep["orchestration_metrics"] for ep in recent_episodes]
        avg_graph_density = np.mean([om.get("graph_density", 0) for om in orchestration_data])
        avg_agent_diversity = np.mean([om.get("agent_diversity", 0) for om in orchestration_data])
        avg_cycle_count = np.mean([om.get("cycle_count", 0) for om in orchestration_data])
        
        return {
            "total_episodes": len(self.training_history),
            "avg_reward": np.mean(recent_rewards),
            "reward_std": np.std(recent_rewards),
            "avg_cost": np.mean(recent_costs),
            "cost_std": np.std(recent_costs),
            "avg_steps": np.mean(recent_steps),
            "steps_std": np.std(recent_steps),
            "reward_trend": self._calculate_trend(recent_rewards),
            "cost_trend": self._calculate_trend(recent_costs),
            "avg_graph_density": avg_graph_density,
            "avg_agent_diversity": avg_agent_diversity,
            "avg_cycle_count": avg_cycle_count
        }
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Linear trend calculation"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        return np.polyfit(x, y, 1)[0]
        
    def save_model(self, filepath: str):
        """Save training state"""
        torch.save({
            'policy_net_state_dict': self.orchestrator.policy_net.state_dict(),
            'optimizer_state_dict': self.orchestrator.optimizer.state_dict(),
            'training_history': self.training_history,
            'episode_rewards': self.episode_rewards,
            'episode_costs': self.episode_costs,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'lambda_cost': self.reward_system.lambda_cost,
                'phi': self.reward_system.phi,
                'F': self.reward_system.F
            }
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load training state"""
        checkpoint = torch.load(filepath)
        self.orchestrator.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.orchestrator.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_costs = checkpoint['episode_costs']