#!/usr/bin/env python
import json
import time
import torch
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from original_supervisor import supervisor_with_description
from rl_supervisor import agents
from orchestration.rl_orchestrator import RLOrchestrator
from utils.cost_tracker import CostTracker
from test.testcases import TestCases
from langchain_core.messages import convert_to_messages, HumanMessage

@dataclass
class BenchmarkResult:
    task: str
    expected_answer: str
    original_result: str
    enhanced_result: str
    original_time: float
    enhanced_time: float
    original_cost: int
    enhanced_cost: int
    original_agents: List[str]
    enhanced_agents: List[str]
    original_accuracy: float
    enhanced_accuracy: float

class BenchmarkPipeline:
    def __init__(self, output_file: str = "dataset/benchmark_results.json"):
        self.output_file = output_file
        self.results = []
        
        # Setup enhanced orchestrator
        self.enhanced_orchestrator = RLOrchestrator(agent_names=list(agents.keys()))
        self._load_checkpoint()
        
    def _load_checkpoint(self, checkpoint_path: str = "checkpoint/orchestrator.pth"):
        """Load trained orchestrator checkpoint"""
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
            self.enhanced_orchestrator.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            print(f"✅ Loaded trained orchestrator from {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
            print("🔄 Using untrained orchestrator for comparison")
    
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
    
    def run_enhanced_supervisor(self, task: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            cost_tracker = CostTracker()
            cost_tracker.start_episode()
            
            messages = [HumanMessage(content=task)]
            result = ""
            step_count = 0
            max_steps = 5
            
            self.enhanced_orchestrator.reset_episode()
            
            while step_count < max_steps:
                selected_agent = self.enhanced_orchestrator.select_agent(messages)
                
                if selected_agent == "terminate":
                    break
                    
                if selected_agent not in agents:
                    break
                    
                agent_result = agents[selected_agent].invoke({"messages": messages})
                
                if agent_result and "messages" in agent_result:
                    new_messages = agent_result["messages"]
                    if new_messages:
                        last_message = new_messages[-1]
                        result = last_message.content
                        
                        tokens_used = len(result.split()) * 1.3
                        cost_tracker.log_agent_call(
                            selected_agent, 
                            int(tokens_used), 
                            0.1
                        )
                        
                        messages.extend(new_messages)
                        
                step_count += 1
            
            execution_time = time.time() - start_time
            orchestration_metrics = self.enhanced_orchestrator.get_orchestration_metrics()
            
            return {
                "result": result,
                "execution_time": execution_time,
                "total_cost": cost_tracker.get_total_cost(),
                "agent_sequence": list(orchestration_metrics.get("agent_usage", {}).keys()),
                "success": bool(result and len(result.strip()) > 0),
                "orchestration_metrics": orchestration_metrics
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
            elif abs(expected_val) > 0 and abs(result_val - expected_val) / abs(expected_val) < 0.01:
                return 0.9
            elif abs(expected_val) > 0 and abs(result_val - expected_val) / abs(expected_val) < 0.05:
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
        enhanced_result = self.run_enhanced_supervisor(task)
        
        original_accuracy = self.evaluate_accuracy(
            original_result["result"], expected_answer, task
        )
        enhanced_accuracy = self.evaluate_accuracy(
            enhanced_result["result"], expected_answer, task
        )
        
        benchmark_result = BenchmarkResult(
            task=task,
            expected_answer=expected_answer,
            original_result=original_result["result"],
            enhanced_result=enhanced_result["result"],
            original_time=original_result["execution_time"],
            enhanced_time=enhanced_result["execution_time"],
            original_cost=original_result["total_cost"],
            enhanced_cost=int(enhanced_result["total_cost"]),
            original_agents=original_result["agent_sequence"],
            enhanced_agents=enhanced_result["agent_sequence"],
            original_accuracy=original_accuracy,
            enhanced_accuracy=enhanced_accuracy
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
        enhanced_accuracies = [r.enhanced_accuracy for r in self.results]
        original_times = [r.original_time for r in self.results]
        enhanced_times = [r.enhanced_time for r in self.results]
        original_costs = [r.original_cost for r in self.results]
        enhanced_costs = [r.enhanced_cost for r in self.results]
        
        time_improvements = [
            (orig - enh) / orig if orig > 0 else 0 
            for orig, enh in zip(original_times, enhanced_times)
        ]
        
        cost_improvements = [
            (orig - enh) / orig if orig > 0 else 0 
            for orig, enh in zip(original_costs, enhanced_costs)
        ]
        
        report = {
            "summary": {
                "total_tasks": len(self.results),
                "enhanced_avg_accuracy": np.mean(enhanced_accuracies),
                "accuracy_improvement": np.mean(enhanced_accuracies) - np.mean(original_accuracies),
                "original_avg_time": np.mean(original_times),
                "enhanced_avg_time": np.mean(enhanced_times),
                "avg_time_improvement": np.mean(time_improvements),
                "original_avg_cost": np.mean(original_costs),
                "enhanced_avg_cost": np.mean(enhanced_costs),
                "avg_cost_improvement": np.mean(cost_improvements)
            },
            "win_rates": {
                "enhanced_better_accuracy": sum(1 for r in self.results if r.enhanced_accuracy > r.original_accuracy) / len(self.results),
                "enhanced_faster": sum(1 for r in self.results if r.enhanced_time < r.original_time) / len(self.results),
                "enhanced_cheaper": sum(1 for r in self.results if r.enhanced_cost < r.original_cost) / len(self.results)
            },
            "agent_usage_analysis": self._analyze_agent_usage()
        }
        
        return report
    
    def _analyze_agent_usage(self) -> Dict:
        original_agent_counts = {}
        enhanced_agent_counts = {}
        
        for result in self.results:
            for agent in result.original_agents:
                original_agent_counts[agent] = original_agent_counts.get(agent, 0) + 1
            for agent in result.enhanced_agents:
                enhanced_agent_counts[agent] = enhanced_agent_counts.get(agent, 0) + 1
        
        return {
            "original_usage": original_agent_counts,
            "enhanced_usage": enhanced_agent_counts,
            "usage_efficiency": {
                "original_avg_agents_per_task": np.mean([len(r.original_agents) for r in self.results]),
                "enhanced_avg_agents_per_task": np.mean([len(r.enhanced_agents) for r in self.results])
            }
        }
    
    def save_results(self):
        results_dict = {
            "benchmark_results": [
                {
                    "task": r.task,
                    "expected_answer": r.expected_answer,
                    "original_result": r.original_result,
                    "enhanced_result": r.enhanced_result,
                    "original_time": r.original_time,
                    "enhanced_time": r.enhanced_time,
                    "original_cost": r.original_cost,
                    "enhanced_cost": r.enhanced_cost,
                    "original_agents": r.original_agents,
                    "enhanced_agents": r.enhanced_agents,
                    "original_accuracy": r.original_accuracy,
                    "enhanced_accuracy": r.enhanced_accuracy
                }
                for r in self.results
            ],
            "comprehensive_report": self.generate_comprehensive_report()
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    benchmark = BenchmarkPipeline()
    report = benchmark.benchmark_test_suite(["math", "research"])
    
    benchmark.save_results()
    
    print("\n=== BENCHMARK REPORT ===")
    print(json.dumps(report["summary"], indent=2))