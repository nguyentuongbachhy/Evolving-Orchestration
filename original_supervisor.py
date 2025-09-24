#!/usr/bin/env python
# coding: utf-8
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

web_search = TavilySearch(max_results=3)

research_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTION:\n"
        "- Assist ONLY research-related tasks, DO NOT do any else.\n"
        "- After you're done with your tasks, respond to the supervisor directly.\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent"
)





import math

def calculate_math_expression(expression: str) -> float:
    """
    Calculate a mathematical expresison and return a float number
    Args:
        expression: A mathematical expression given as a string
    """
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith('__')}
    allowed.update({"abs": abs, "round": round, "pow": pow, "min": min, "max": max})
    return eval(expression, {"__builtin__": {}}, allowed)

math_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[calculate_math_expression],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTION:\n"
        "- Assist ONLY math-related tasks, DO NOT do any else.\n"
        "-After you're done with your tasks, respond to the supervisor directly.\n"
        "- Respond ONLY with the results of your woork, do NOT include ANY other text."
    ),
    name="math_agent"
)

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help"

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id
        }
        return Command(
            goto=agent_name,
            update={**state, "messages": state["messages"] + [tool_message]}
        )

    return handoff_tool

assign_to_research = create_handoff_tool(agent_name="research_agent", description="Assign task to a researcher agent.")
assign_to_math = create_handoff_tool(agent_name="math_agent", description="Assign task to a math agent.")

supervisor_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[assign_to_research, assign_to_math],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor"
)

from langgraph.graph import END

supervisor = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent, destinations=["research_agent", "math_agent", END])
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()
)

# IPython display removed for script compatibility

from langgraph.types import Send

def create_task_description_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[str, "Description of what the next agent should do, including all the relevant context."],
        state: Annotated[MessagesState, InjectedState]
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT
        )
    return handoff_tool

assign_to_research_with_description = create_task_description_handoff_tool(agent_name="research_agent", description="Assign task to a researcher agent.")
assign_to_math_with_description = create_task_description_handoff_tool(agent_name="math_agent", description="Assign task to a math agent.")

supervisor_agent_with_description = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[assign_to_research_with_description, assign_to_math_with_description],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this assistant\n"
        "- a math agent. Assign math-related tasks to this assistant\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor"
)

supervisor_with_description = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent_with_description, destinations=["research_agent", "math_agent", END])
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()
)

# IPython display removed for script compatibility

from langchain_core.messages import convert_to_messages

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

# Test code removed - should be run only when needed