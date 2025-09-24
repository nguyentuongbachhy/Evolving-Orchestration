#!/usr/bin/env python
import json
import time
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from original_supervisor import supervisor_with_description
from rl_supervisor import RLSupervisor
from test.testcases import TestCases
from langchain_core.messages import convert_to_messages

@dataclass
class BenchmarkResult:
    task: str
    expected_answer: str
    original_result: str
    rl_result: str
    original_time: float
    rl_time: float
    original_cost: int
    rl_cost: int
    original_agents: List[str]
    rl_agents: List[str]
    original_accuracy: float
    rl_accuracy: float

class BenchmarkPipeline:
    def __init__(self, rl_supervisor: RLSupervisor, output_file: str = "benchmark_results.json"):
        self.rl_supervisor = rl_supervisor
        self.output_file = output_file
        self.results = []
        
    def run_original_supervisor(self, task: str) -> Dict[str, Any]:
        start_time = time.time()
        
        messages_log = []
        agent_calls = []
        final_result = ""
        
        try:
            for chunk in supervisor_with_description.stream(
                {"messages": [{"role": "user", "content": task}]},
                subgraphs=True
            ):
                if isinstance(chunk, tuple):
                    ns, update = chunk
                else:
                    update = chunk
                    
                for node_name, node_update in update.items():
                    if "messages" in node_update:
                        new_messages = convert_to_messages(node_update["messages"])
                        
                        for msg in new_messages:
                            messages_log.append({
                                "role": msg.type,
                                "content": msg.content,
                                "node": node_name
                            })
                            
                        if node_name in ["research_agent", "math_agent"]:
                            agent_calls.append({
                                "agent": node_name,
                                "tokens": len(new_messages[-1].content.split()) if new_messages else 0
                            })
            
            final_result = messages_log[-1]["content"] if messages_log else ""
            
        except Exception as e:
            final_result = f"Error: {str(e)}"
        
        execution_time = time.time() - start_time
        
        return {
            "result": final_result,
            "execution_time": execution_time,
            "total_cost": sum(call["tokens"] for call in agent_calls),
            "agent_sequence": [call["agent"] for call in agent_calls],
            "success": len(agent_calls) > 0
        }
    
    def run_rl_supervisor(self, task: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            result, episode_info = self.rl_supervisor.execute_task(task)
            execution_time = time.time() - start_time
            
            return {
                "result": result,
                "execution_time": execution_time,
                "total_cost": episode_info["cost_stats"]["total_tokens"],
                "agent_sequence": list(episode_info["cost_stats"]["agent_breakdown"].keys()),
                "success": bool(result and len(result.strip()) > 0)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "result": f"Error: {str(e)}",
                "execution_time": execution_time,
                "total_cost": 0,
                "agent_sequence": [],
                "success": False
            }
    
    def evaluate_accuracy(self, result: str, expected: str, task: str) -> float:
        if not result or not expected:
            return 0.0
        
        result_clean = result.lower().strip()
        expected_clean = expected.lower().strip()
        
        if self._is_math_problem(task):
            return self._evaluate_math_accuracy(result_clean, expected_clean)
        else:
            return self._evaluate_text_accuracy(result_clean, expected_clean)
    
    def _is_math_problem(self, task: str) -> bool:
        math_indicators = ['calculate', 'what is', '**', '+', '-', '*', '/', 'solve']
        return any(indicator in task.lower() for indicator in math_indicators)
    
    def _evaluate_math_accuracy(self, result: str, expected: str) -> float:
        import re
        
        result_numbers = re.findall(r'-?\d+\.?\d*', result)
        expected_numbers = re.findall(r'-?\d+\.?\d*', expected)
        
        if not result_numbers or not expected_numbers:
            return 1.0 if result == expected else 0.0
        
        try:
            result_val = float(result_numbers[-1])
            expected_val = float(expected_numbers[-1])
            
            if abs(result_val - expected_val) < 1e-6:
                return 1.0
            elif abs(result_val - expected_val) / abs(expected_val) < 0.01:
                return 0.9
            elif abs(result_val - expected_val) / abs(expected_val) < 0.05:
                return 0.7
            else:
                return 0.0
        except:
            return 1.0 if result == expected else 0.0
    
    def _evaluate_text_accuracy(self, result: str, expected: str) -> float:
        if result == expected:
            return 1.0
        
        result_words = set(result.split())
        expected_words = set(expected.split())
        
        if not expected_words:
            return 0.0
        
        intersection = result_words.intersection(expected_words)
        return len(intersection) / len(expected_words)
    
    def benchmark_single_task(self, task: str, expected_answer: str) -> BenchmarkResult:
        print(f"Benchmarking: {task[:60]}...")
        
        original_result = self.run_original_supervisor(task)
        rl_result = self.run_rl_supervisor(task)
        
        original_accuracy = self.evaluate_accuracy(
            original_result["result"], expected_answer, task
        )
        rl_accuracy = self.evaluate_accuracy(
            rl_result["result"], expected_answer, task
        )
        
        benchmark_result = BenchmarkResult(
            task=task,
            expected_answer=expected_answer,
            original_result=original_result["result"],
            rl_result=rl_result["result"],
            original_time=original_result["execution_time"],
            rl_time=rl_result["execution_time"],
            original_cost=original_result["total_cost"],
            rl_cost=rl_result["total_cost"],
            original_agents=original_result["agent_sequence"],
            rl_agents=rl_result["agent_sequence"],
            original_accuracy=original_accuracy,
            rl_accuracy=rl_accuracy
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def benchmark_test_suite(self, problem_types: List[str] = ["math", "research"]) -> Dict:
        all_test_cases = TestCases.get_all_test_cases()
        
        for problem_type in problem_types:
            if problem_type in all_test_cases:
                print(f"\n=== Benchmarking {problem_type.upper()} Problems ===")
                
                for task, expected in all_test_cases[problem_type]:
                    self.benchmark_single_task(task, expected)
        
        return self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self) -> Dict:
        if not self.results:
            return {"error": "No benchmark results available"}
        
        original_accuracies = [r.original_accuracy for r in self.results]
        rl_accuracies = [r.rl_accuracy for r in self.results]
        original_times = [r.original_time for r in self.results]
        rl_times = [r.rl_time for r in self.results]
        original_costs = [r.original_cost for r in self.results]
        rl_costs = [r.rl_cost for r in self.results]
        
        time_improvements = [
            (orig - rl) / orig if orig > 0 else 0 
            for orig, rl in zip(original_times, rl_times)
        ]
        
        cost_improvements = [
            (orig - rl) / orig if orig > 0 else 0 
            for orig, rl in zip(original_costs, rl_costs)
        ]
        
        report = {
            "summary": {
                "total_tasks": len(self.results),
                "original_avg_accuracy": np.mean(original_accuracies),
                "rl_avg_accuracy": np.mean(rl_accuracies),
                "accuracy_improvement": np.mean(rl_accuracies) - np.mean(original_accuracies),
                "original_avg_time": np.mean(original_times),
                "rl_avg_time": np.mean(rl_times),
                "avg_time_improvement": np.mean(time_improvements),
                "original_avg_cost": np.mean(original_costs),
                "rl_avg_cost": np.mean(rl_costs),
                "avg_cost_improvement": np.mean(cost_improvements)
            },
            "detailed_metrics": {
                "accuracy_std": {
                    "original": np.std(original_accuracies),
                    "rl": np.std(rl_accuracies)
                },
                "time_std": {
                    "original": np.std(original_times),
                    "rl": np.std(rl_times)
                },
                "cost_std": {
                    "original": np.std(original_costs),
                    "rl": np.std(rl_costs)
                }
            },
            "win_rates": {
                "rl_better_accuracy": sum(1 for r in self.results if r.rl_accuracy > r.original_accuracy) / len(self.results),
                "rl_faster": sum(1 for r in self.results if r.rl_time < r.original_time) / len(self.results),
                "rl_cheaper": sum(1 for r in self.results if r.rl_cost < r.original_cost) / len(self.results)
            },
            "agent_usage_comparison": self._analyze_agent_usage(),
            "task_breakdown": self._analyze_by_task_type()
        }
        
        return report
    
    def _analyze_agent_usage(self) -> Dict:
        original_agent_counts = {}
        rl_agent_counts = {}
        
        for result in self.results:
            for agent in result.original_agents:
                original_agent_counts[agent] = original_agent_counts.get(agent, 0) + 1
            for agent in result.rl_agents:
                rl_agent_counts[agent] = rl_agent_counts.get(agent, 0) + 1
        
        return {
            "original_usage": original_agent_counts,
            "rl_usage": rl_agent_counts,
            "usage_efficiency": {
                "original_avg_agents_per_task": np.mean([len(r.original_agents) for r in self.results]),
                "rl_avg_agents_per_task": np.mean([len(r.rl_agents) for r in self.results])
            }
        }
    
    def _analyze_by_task_type(self) -> Dict:
        math_results = [r for r in self.results if self._is_math_problem(r.task)]
        text_results = [r for r in self.results if not self._is_math_problem(r.task)]
        
        def analyze_group(results, name):
            if not results:
                return {name: {"count": 0}}
                
            return {
                name: {
                    "count": len(results),
                    "original_avg_accuracy": np.mean([r.original_accuracy for r in results]),
                    "rl_avg_accuracy": np.mean([r.rl_accuracy for r in results]),
                    "original_avg_cost": np.mean([r.original_cost for r in results]),
                    "rl_avg_cost": np.mean([r.rl_cost for r in results])
                }
            }
        
        breakdown = {}
        breakdown.update(analyze_group(math_results, "math_problems"))
        breakdown.update(analyze_group(text_results, "research_problems"))
        
        return breakdown
    
    def save_results(self):
        results_dict = {
            "benchmark_results": [
                {
                    "task": r.task,
                    "expected_answer": r.expected_answer,
                    "original_result": r.original_result,
                    "rl_result": r.rl_result,
                    "original_time": r.original_time,
                    "rl_time": r.rl_time,
                    "original_cost": r.original_cost,
                    "rl_cost": r.rl_cost,
                    "original_agents": r.original_agents,
                    "rl_agents": r.rl_agents,
                    "original_accuracy": r.original_accuracy,
                    "rl_accuracy": r.rl_accuracy
                }
                for r in self.results
            ],
            "comprehensive_report": self.generate_comprehensive_report()
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    rl_supervisor = RLSupervisor(training_mode=False)
    
    benchmark = BenchmarkPipeline(rl_supervisor)
    report = benchmark.benchmark_test_suite(["math", "research"])
    
    benchmark.save_results()
    
    print("\n=== BENCHMARK REPORT ===")
    print(json.dumps(report["summary"], indent=2))