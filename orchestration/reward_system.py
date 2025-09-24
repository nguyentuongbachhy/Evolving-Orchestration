import re
from typing import Dict
from utils.cost_tracker import CostTracker

class RewardSystem:
    def __init__(self, lambda_cost: float = 0.1):
        self.lambda_cost = lambda_cost
        
    def calculate_reward(self, 
                        task_result: str, 
                        expected_answer: str, 
                        cost_tracker: CostTracker) -> float:
        
        quality_reward = self._evaluate_quality(task_result, expected_answer)
        total_cost = cost_tracker.get_total_cost()
        
        reward = quality_reward - self.lambda_cost * total_cost
        return reward
        
    def _evaluate_quality(self, result: str, expected: str) -> float:
        if not result or not expected:
            return 0.0
            
        result_clean = self._clean_text(result)
        expected_clean = self._clean_text(expected)
        
        if self._is_math_problem(expected):
            return self._evaluate_math_quality(result_clean, expected_clean)
        else:
            return self._evaluate_text_quality(result_clean, expected_clean)
            
    def _evaluate_math_quality(self, result: str, expected: str) -> float:
        result_numbers = self._extract_numbers(result)
        expected_numbers = self._extract_numbers(expected)
        
        if not result_numbers or not expected_numbers:
            return 0.0
            
        final_result = result_numbers[-1]
        final_expected = expected_numbers[-1]
        
        if abs(final_result - final_expected) < 1e-6:
            return 1.0
        elif abs(final_result - final_expected) / abs(final_expected) < 0.01:
            return 0.8
        elif abs(final_result - final_expected) / abs(final_expected) < 0.05:
            return 0.5
        else:
            return 0.0
            
    def _evaluate_text_quality(self, result: str, expected: str) -> float:
        if result.lower() == expected.lower():
            return 1.0
            
        words_result = set(result.lower().split())
        words_expected = set(expected.lower().split())
        
        if not words_expected:
            return 0.0
            
        intersection = words_result.intersection(words_expected)
        jaccard_score = len(intersection) / len(words_result.union(words_expected))
        
        return jaccard_score
        
    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip())
        
    def _is_math_problem(self, text: str) -> bool:
        math_indicators = ['=', '+', '-', '*', '/', 'calculate', 'solve', '**']
        return any(indicator in text.lower() for indicator in math_indicators)
        
    def _extract_numbers(self, text: str) -> list:
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers if num]
        
    def get_reward_components(self, 
                             task_result: str, 
                             expected_answer: str, 
                             cost_tracker: CostTracker) -> Dict[str, float]:
        
        quality = self._evaluate_quality(task_result, expected_answer)
        cost = cost_tracker.get_total_cost()
        total_reward = quality - self.lambda_cost * cost
        
        return {
            "quality_reward": quality,
            "cost_penalty": cost,
            "total_reward": total_reward,
            "lambda_cost": self.lambda_cost
        }