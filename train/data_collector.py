#!/usr/bin/env python
import json
import time
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import convert_to_messages
from original_supervisor import supervisor_with_description
from test.testcases import TestCases

class DataCollector:
    def __init__(self, output_file: str = "execution_traces.json"):
        self.output_file = output_file
        self.traces = []
        
    def collect_trace(self, task: str) -> Dict[str, Any]:
        trace = {
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "agent_sequence": [],
            "messages": [],
            "total_cost": 0,
            "execution_time": 0,
            "success": False,
            "final_result": ""
        }
        
        start_time = time.time()
        
        try:
            messages_log = []
            agent_calls = []
            
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
                                "input": new_messages[0].content if new_messages else "",
                                "output": new_messages[-1].content if new_messages else "",
                                "tokens": len(new_messages[-1].content.split()) if new_messages else 0
                            })
            
            trace["agent_sequence"] = [call["agent"] for call in agent_calls]
            trace["messages"] = messages_log
            trace["agent_calls"] = agent_calls
            trace["total_cost"] = sum(call["tokens"] for call in agent_calls)
            trace["execution_time"] = time.time() - start_time
            trace["success"] = len(agent_calls) > 0
            trace["final_result"] = messages_log[-1]["content"] if messages_log else ""
            
        except Exception as e:
            trace["error"] = str(e)
            trace["execution_time"] = time.time() - start_time
            
        return trace
    
    def collect_batch(self, tasks: List[str], save_interval: int = 5) -> List[Dict]:
        batch_traces = []
        
        for i, task in enumerate(tasks):
            print(f"Collecting trace {i+1}/{len(tasks)}: {task[:50]}...")
            
            trace = self.collect_trace(task)
            batch_traces.append(trace)
            self.traces.append(trace)
            
            if (i + 1) % save_interval == 0:
                self.save_traces()
                print(f"Saved {len(self.traces)} traces to {self.output_file}")
                
        self.save_traces()
        return batch_traces
    
    def collect_from_test_cases(self, problem_types: List[str] = ["math", "research", "mixed"]) -> Dict:
        all_test_cases = TestCases.get_all_test_cases()
        
        collection_results = {
            "total_traces": 0,
            "successful_traces": 0,
            "by_type": {}
        }
        
        for problem_type in problem_types:
            if problem_type in all_test_cases:
                tasks = [task for task, _ in all_test_cases[problem_type]]
                
                print(f"\nCollecting {problem_type} problems...")
                type_traces = self.collect_batch(tasks)
                
                successful = sum(1 for trace in type_traces if trace["success"])
                
                collection_results["by_type"][problem_type] = {
                    "total": len(type_traces),
                    "successful": successful,
                    "success_rate": successful / len(type_traces) if type_traces else 0
                }
                
                collection_results["total_traces"] += len(type_traces)
                collection_results["successful_traces"] += successful
        
        collection_results["overall_success_rate"] = (
            collection_results["successful_traces"] / collection_results["total_traces"]
            if collection_results["total_traces"] > 0 else 0
        )
        
        return collection_results
    
    def save_traces(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.traces, f, indent=2)
    
    def load_traces(self, filepath: str = None):
        filepath = filepath or self.output_file
        try:
            with open(filepath, 'r') as f:
                self.traces = json.load(f)
            return self.traces
        except FileNotFoundError:
            return []
    
    def get_statistics(self) -> Dict:
        if not self.traces:
            return {}
            
        successful_traces = [t for t in self.traces if t["success"]]
        
        return {
            "total_traces": len(self.traces),
            "successful_traces": len(successful_traces),
            "success_rate": len(successful_traces) / len(self.traces),
            "avg_execution_time": sum(t["execution_time"] for t in self.traces) / len(self.traces),
            "avg_cost": sum(t["total_cost"] for t in self.traces) / len(self.traces),
            "avg_agents_per_task": sum(len(t["agent_sequence"]) for t in successful_traces) / len(successful_traces) if successful_traces else 0,
            "most_used_agent": self._get_most_used_agent(),
            "agent_usage_stats": self._get_agent_usage_stats()
        }
    
    def _get_most_used_agent(self) -> str:
        agent_counts = {}
        for trace in self.traces:
            for agent in trace.get("agent_sequence", []):
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        return max(agent_counts, key=agent_counts.get) if agent_counts else "none"
    
    def _get_agent_usage_stats(self) -> Dict:
        agent_stats = {}
        for trace in self.traces:
            for agent in trace.get("agent_sequence", []):
                if agent not in agent_stats:
                    agent_stats[agent] = {"count": 0, "tasks": []}
                agent_stats[agent]["count"] += 1
                if trace["task"] not in agent_stats[agent]["tasks"]:
                    agent_stats[agent]["tasks"].append(trace["task"][:30] + "...")
        return agent_stats

if __name__ == "__main__":
    collector = DataCollector("execution_traces.json")
    
    results = collector.collect_from_test_cases(["math", "research"])
    
    print("\nCollection Results:")
    print(json.dumps(results, indent=2))
    
    stats = collector.get_statistics()
    print("\nExecution Statistics:")
    print(json.dumps(stats, indent=2))