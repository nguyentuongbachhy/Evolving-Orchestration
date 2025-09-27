#!/usr/bin/env python
import time
import torch
from typing import Dict, Tuple
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import math

from orchestration.rl_orchestrator import RLOrchestrator
from utils.cost_tracker import CostTracker

@tool
def calculate_math_expression(expression: str) -> float:
    "Evaluates a math expression given as a string and returns the number."
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith('__')}
    allowed.update({"abs": abs, "round": round, "pow": pow, "min": min, "max": max})
    return eval(expression, {"__builtin__": {}}, allowed)

@tool
def summarize_results(context: str) -> str:
    """Summarize and synthesize results from multiple agents into final answer"""
    return f"Final synthesis: {context}"

web_search = TavilySearch(max_results=3)
python_repl = PythonREPLTool()

research_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search],
    prompt=(
        "You are a research agent. "
        "Assist ONLY research-related tasks, DO NOT do any else. "
        "After you're done with your tasks, respond to the supervisor directly. "
        "Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent"
)

math_agent = create_react_agent(
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

code_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[python_repl],
    prompt=(
        "You are a code agent. "
        "Assist ONLY programming and code-related tasks, DO NOT do any else. "
        "Write and execute Python code to solve problems. "
        "After you're done with your tasks, respond to the supervisor directly. "
        "Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="code_agent"
)

summary_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[summarize_results],
    prompt=(
        "You are a summary agent. "
        "Synthesize and summarize results from other agents into a final answer. "
        "Provide clear, concise, and well-structured final responses. "
        "After you're done with your tasks, respond to the supervisor directly. "
        "Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="summary_agent"
)

agents = {
    "research_agent": research_agent,
    "math_agent": math_agent,
    "code_agent": code_agent,
    "summary_agent": summary_agent
}

orchestrator = RLOrchestrator(agent_names=list(agents.keys()))

def load_checkpoint(checkpoint_path: str = "checkpoint/orchestrator.pth"):
    """Load trained orchestrator checkpoint"""
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
        orchestrator.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        print(f"✅ Loaded trained orchestrator from {checkpoint_path}")
        return True
    except FileNotFoundError:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("🔄 Using untrained orchestrator")
        return False
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("🔄 Using untrained orchestrator")
        return False

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def execute_task(task: str, max_steps: int = 5, use_checkpoint: bool = True) -> Tuple[str, Dict]:
    if use_checkpoint:
        load_checkpoint()
        
    cost_tracker = CostTracker()
    cost_tracker.start_episode()
    
    messages = [HumanMessage(content=task)]
    result = ""
    step_count = 0
    
    orchestrator.reset_episode()
    
    print(f"🎯 Task: {task}")
    print("="*50)
    
    while step_count < max_steps:
        selected_agent = orchestrator.select_agent(messages)
        
        print(f"🎭 Step {step_count + 1}: Enhanced Orchestrator selects → {selected_agent}")
        
        if selected_agent == "terminate":
            print("🏁 Orchestrator decides to terminate")
            break
            
        if selected_agent not in agents:
            print(f"⚠️ Unknown agent: {selected_agent}")
            break
            
        start_time = time.time()
        agent_result = agents[selected_agent].invoke({"messages": messages})
        execution_time = time.time() - start_time
        
        if agent_result and "messages" in agent_result:
            new_messages = agent_result["messages"]
            if new_messages:
                last_message = new_messages[-1]
                result = last_message.content
                
                print(f"📤 {selected_agent} output:")
                pretty_print_message(last_message, indent=True)
                
                tokens_used = len(result.split()) * 1.3
                cost_tracker.log_agent_call(
                    selected_agent, 
                    int(tokens_used), 
                    execution_time
                )
                
                messages.extend(new_messages)
                
        step_count += 1
        print()
        
    episode_info = {
        "steps": step_count,
        "cost_stats": cost_tracker.get_episode_stats(),
        "orchestration_metrics": orchestrator.get_orchestration_metrics()
    }
    
    print("📊 Orchestration Analysis:")
    metrics = episode_info["orchestration_metrics"]
    print(f"   Agent Diversity: {metrics.get('agent_diversity', 0):.2f}")
    print(f"   Graph Density: {metrics.get('graph_density', 0):.2f}")
    print(f"   Reasoning Depth: {metrics.get('reasoning_depth', 0)}")
    print(f"   Cycle Count: {metrics.get('cycle_count', 0)}")
    print(f"   Agent Usage: {metrics.get('agent_usage', {})}")
    
    return result, episode_info

if __name__ == "__main__":
    task = input("Enter your task: ")
    result, info = execute_task(task)
    print(f"\n🎉 Final Result: {result}")
    print(f"📈 Episode Stats: {info['steps']} steps, Cost: {info['cost_stats']['total_cost']:.2f}")