import time
import math
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AgentCost:
    agent_name: str
    tokens_used: int
    execution_time: float
    calls_count: int

class CostTracker:
    def __init__(self, cost_factor: float = 1.0):
        self.cost_factor = cost_factor
        self.episode_costs: List[AgentCost] = []
        self.total_tokens = 0
        self.total_time = 0
        self.total_calls = 0
        self.start_time = None
        
    def start_episode(self):
        self.episode_costs = []
        self.total_tokens = 0
        self.total_time = 0
        self.total_calls = 0
        self.start_time = time.time()
        
    def log_agent_call(self, agent_name: str, tokens_used: int, execution_time: float):
        cost = AgentCost(
            agent_name=agent_name,
            tokens_used=tokens_used,
            execution_time=execution_time,
            calls_count=1
        )
        
        self.episode_costs.append(cost)
        self.total_tokens += tokens_used
        self.total_time += execution_time
        self.total_calls += 1
        
    def calculate_step_cost(self, tokens_used: int) -> float:
        return self.cost_factor * math.log(1 + tokens_used)
        
    def get_total_cost(self) -> float:
        total_cost = 0
        for cost in self.episode_costs:
            total_cost += self.calculate_step_cost(cost.tokens_used)
        return total_cost
        
    def get_episode_stats(self) -> Dict:
        episode_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            "total_tokens": self.total_tokens,
            "total_time": episode_time,
            "total_calls": self.total_calls,
            "total_cost": self.get_total_cost(),
            "avg_tokens_per_call": self.total_tokens / max(1, self.total_calls),
            "agent_breakdown": self._get_agent_breakdown()
        }
        
    def _get_agent_breakdown(self) -> Dict[str, Dict]:
        breakdown = {}
        for cost in self.episode_costs:
            if cost.agent_name not in breakdown:
                breakdown[cost.agent_name] = {
                    "calls": 0,
                    "tokens": 0,
                    "time": 0
                }
            breakdown[cost.agent_name]["calls"] += cost.calls_count
            breakdown[cost.agent_name]["tokens"] += cost.tokens_used
            breakdown[cost.agent_name]["time"] += cost.execution_time
        return breakdown