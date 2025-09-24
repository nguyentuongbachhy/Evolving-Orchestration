#!/usr/bin/env python
import os
import getpass
import time
from typing import Dict, Tuple
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import convert_to_messages
import math

from orchestration.rl_orchestrator import RLOrchestrator
from orchestration.reward_system import RewardSystem
from orchestration.training_manager import TrainingManager
from utils.cost_tracker import CostTracker

def _define_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_define_if_undefined("OPENAI_API_KEY")
_define_if_undefined("TAVILY_API_KEY")

def calculate_math_expression(expression: str) -> float:
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith('__')}
    allowed.update({"abs": abs, "round": round, "pow": pow, "min": min, "max": max})
    return eval(expression, {"__builtin__": {}}, allowed)

class RLSupervisor:
    def __init__(self, training_mode: bool = False):
        self.training_mode = training_mode
        
        self.web_search = TavilySearch(max_results=3)
        
        self.research_agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[self.web_search],
            prompt=(
                "You are a research agent. "
                "Assist ONLY research-related tasks, DO NOT do any else. "
                "After you're done with your tasks, respond to the supervisor directly. "
                "Respond ONLY with the results of your work, do NOT include ANY other text."
            ),
            name="research_agent"
        )
        
        self.math_agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[calculate_math_expression],
            prompt=(
                "You are a math agent. "
                "Assist ONLY math-related tasks, DO NOT do any else. "
                "After you're done with your tasks, respond to the supervisor directly. "
                "Respond ONLY with the results of your work, do NOT include ANY other text."
            ),
            name="math_agent"
        )
        
        self.agents = {
            "research_agent": self.research_agent,
            "math_agent": self.math_agent
        }
        
        self.orchestrator = RLOrchestrator(
            agent_names=list(self.agents.keys())
        )
        
        self.reward_system = RewardSystem(lambda_cost=0.1)
        
        if training_mode:
            self.training_manager = TrainingManager(
                self.orchestrator, 
                self.reward_system
            )
    
    def execute_task(self, task: str, max_steps: int = 5) -> Tuple[str, Dict]:
        cost_tracker = CostTracker()
        cost_tracker.start_episode()
        
        messages = [{"role": "user", "content": task}]
        state = {"messages": convert_to_messages(messages)}
        
        result = ""
        step_count = 0
        
        while step_count < max_steps:
            selected_agent = self.orchestrator.select_agent(state["messages"])
            
            if selected_agent == "terminate":
                break
                
            start_time = time.time()
            agent_result = self.agents[selected_agent].invoke(state)
            execution_time = time.time() - start_time
            
            if agent_result and "messages" in agent_result:
                new_messages = agent_result["messages"]
                if new_messages:
                    last_message = new_messages[-1]
                    result = last_message.content
                    
                    tokens_used = len(result.split()) * 1.3
                    cost_tracker.log_agent_call(
                        selected_agent, 
                        int(tokens_used), 
                        execution_time
                    )
                    
                    state["messages"].extend(new_messages)
                    
            step_count += 1
            
            if self._should_terminate(result, step_count):
                break
        
        episode_info = {
            "steps": step_count,
            "cost_stats": cost_tracker.get_episode_stats(),
            "final_agent": selected_agent if step_count > 0 else None
        }
        
        return result, episode_info
    
    def _should_terminate(self, result: str, step_count: int) -> bool:
        if not result:
            return False
            
        termination_indicators = [
            "final answer", "conclusion", "result:", 
            "therefore", "answer is", "solution:"
        ]
        
        return any(indicator in result.lower() for indicator in termination_indicators)
    
    def train_on_task(self, task: str, expected_answer: str) -> Dict:
        if not self.training_mode:
            raise ValueError("Supervisor not in training mode")
            
        return self.training_manager.train_episode(
            task, 
            expected_answer, 
            self._agent_executor_wrapper
        )
    
    def _agent_executor_wrapper(self, task: str, orchestrator: RLOrchestrator, cost_tracker: CostTracker):
        result, episode_info = self.execute_task(task)
        return result, episode_info
    
    def train_batch(self, training_data: list, num_epochs: int = 10) -> list:
        if not self.training_mode:
            raise ValueError("Supervisor not in training mode")
            
        return self.training_manager.train_batch(
            training_data, 
            self._agent_executor_wrapper,
            num_epochs
        )
    
    def get_performance_metrics(self) -> Dict:
        if not self.training_mode:
            return {"error": "No training metrics available"}
            
        return self.training_manager.get_training_metrics()
    
    def save_model(self, filepath: str):
        if self.training_mode:
            self.training_manager.save_model(filepath)
    
    def load_model(self, filepath: str):
        if self.training_mode:
            self.training_manager.load_model(filepath)

if __name__ == "__main__":
    supervisor = RLSupervisor(training_mode=False)
    
    result, info = supervisor.execute_task(
        "What is 2**10 + 3 - 2**9?"
    )
    
    print("Result:", result)
    print("Episode Info:", info)