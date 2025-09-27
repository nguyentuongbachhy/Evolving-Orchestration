import re
import math
import numpy as np
from typing import Dict, List, Tuple
from utils.cost_tracker import CostTracker

class RewardSystem:
    def __init__(self, lambda_cost: float = 0.1, gamma: float = 0.99, phi: int = 4, F: float = 1.0):
        # From paper page 4
        self.lambda_cost = lambda_cost  # λ = 0.1
        self.gamma = gamma  # γ = 0.99
        self.phi = phi  # φ = episode length = 4 (page 5)
        self.F = F  # F scaling factor (page 4)
        
    def calculate_trajectory_rewards(self, 
                                   task_result: str,
                                   expected_answer: str,
                                   cost_tracker: CostTracker,
                                   step_count: int) -> List[float]:
        """Calculate rewards using paper's formula: Rt = r - λ·CT at terminal, γ·Rt+1 - λ·Ct otherwise"""
        
        terminal_quality = self._evaluate_quality(task_result, expected_answer)
        total_cost = self._calculate_total_cost(cost_tracker, step_count)
        
        # Paper formula from page 4, equation (6)
        trajectory_rewards = []
        
        for t in range(step_count):
            if t == step_count - 1:  # Terminal state (t = T)
                reward = terminal_quality - self.lambda_cost * total_cost
            else:  # Intermediate states (t < T)
                step_cost = self._calculate_step_cost(t)
                reward = -self.lambda_cost * step_cost
                
            trajectory_rewards.append(reward)
            
        return trajectory_rewards
    
    def _calculate_step_cost(self, t: int) -> float:
        """Paper formula: Ct = F · log(1 + t/φ)"""
        return self.F * math.log(1 + t / self.phi)
    
    def _calculate_total_cost(self, cost_tracker: CostTracker, step_count: int) -> float:
        """Sum of all step costs"""
        total = 0.0
        for t in range(step_count):
            total += self._calculate_step_cost(t)
            
        # Add token costs if available
        if hasattr(cost_tracker, 'get_total_cost'):
            total += cost_tracker.get_total_cost()
            
        return total
    
    def _evaluate_quality(self, result: str, expected: str) -> float:
        """Paper mentions r ∈ {0,1} for correctness, r ∈ [0,1] for quality"""
        if not result or not expected:
            return 0.0
            
        result_clean = self._clean_text(result)
        expected_clean = self._clean_text(expected)
        
        if self._is_math_problem(expected):
            return self._evaluate_math_correctness(result_clean, expected_clean)
        else:
            return self._evaluate_text_quality(result_clean, expected_clean)
    
    def _evaluate_math_correctness(self, result: str, expected: str) -> float:
        """Binary correctness for math problems (r ∈ {0,1})"""
        result_numbers = self._extract_numbers(result)
        expected_numbers = self._extract_numbers(expected)
        
        if not result_numbers or not expected_numbers:
            return 1.0 if result.lower().strip() == expected.lower().strip() else 0.0
            
        final_result = result_numbers[-1]
        final_expected = expected_numbers[-1]
        
        # Binary correctness with numerical tolerance
        if abs(final_result - final_expected) < 1e-6:
            return 1.0
        return 0.0
    
    def _evaluate_text_quality(self, result: str, expected: str) -> float:
        """Quality score for open-ended tasks (r ∈ [0,1])"""
        if result == expected:
            return 1.0
            
        result_words = set(result.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
            
        # Jaccard similarity as quality measure
        intersection = result_words.intersection(expected_words)
        union = result_words.union(expected_words)
        return len(intersection) / len(union) if union else 0.0
    
    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip())
        
    def _is_math_problem(self, text: str) -> bool:
        math_indicators = ['calculate', 'solve', '=', '+', '-', '*', '/', '**']
        return any(indicator in text.lower() for indicator in math_indicators)
        
    def _extract_numbers(self, text: str) -> List[float]:
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers if num]
        
    def get_reward_components(self, 
                             task_result: str,
                             expected_answer: str,
                             cost_tracker: CostTracker,
                             step_count: int) -> Dict[str, float]:
        """Breakdown following paper's reward structure"""
        
        quality = self._evaluate_quality(task_result, expected_answer)
        total_cost = self._calculate_total_cost(cost_tracker, step_count)
        total_reward = quality - self.lambda_cost * total_cost
        
        return {
            "quality_reward": quality,  # r term
            "total_cost": total_cost,   # CT term  
            "total_reward": total_reward,  # r - λ·CT
            "lambda_cost": self.lambda_cost,  # λ = 0.1
            "gamma": self.gamma,  # γ = 0.99
            "phi": self.phi,  # φ = 4 (episode length)
            "F": self.F  # F scaling factor
        }