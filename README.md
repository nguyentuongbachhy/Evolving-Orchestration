# Multi-Agent RL Orchestrator

A Reinforcement Learning-based orchestrator for multi-agent collaboration, inspired by the **Puppeteer** model from "Multi-Agent Collaboration via Evolving Orchestration" research.

## 🎯 Overview

This project transforms static multi-agent systems into dynamic, learning-based orchestrators that can:

- **Learn optimal agent selection** through reinforcement learning
- **Reduce computational costs** while maintaining accuracy
- **Adapt to different problem types** automatically
- **Evolve collaboration patterns** over time

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐     ┌─────────────────┐
│ RL Orchestrator  ───▶  Policy Network   ────▶ Agent Selection  │
│                 │    │                  │     │                 │
└─────────────────┘    └──────────────────┘     └─────────────────┘
        ▲                       ▲                      │
        │                       │                      ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Reward System  │    │  State Encoder   │    │ Research Agent  │
│                 │    │                  │    │   Math Agent    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites

```bash
Python 3.8+
PyTorch 1.9+
```

### Setup

```bash
# Clone repository
git clone https://github.com/nguyentuongbachhy/Evolving-Orchestration.git

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
```

### Requirements.txt

```txt
torch>=1.9.0
numpy>=1.20.0
langchain>=0.1.0
langchain-openai
langchain-tavily
langgraph>=0.1.0
scikit-learn>=1.0.0
```

## 🚀 Quick Start

### Full Pipeline

```bash
# Run complete experiment
python main.py --name "my_experiment" --problems math research

# Results will be saved in outputs/my_experiment/
```

### Step-by-Step Execution

```bash
# 1. Collect execution traces from original supervisor
python main.py --step 1 --problems math research

# 2. Train RL orchestrator via imitation learning
python main.py --step 2

# 3. Fine-tune with reinforcement learning
python main.py --step 3

# 4. Benchmark performance comparison
python main.py --step 4
```

### Individual Components

```python
from rl_supervisor import RLSupervisor

# Use RL supervisor directly
supervisor = RLSupervisor(training_mode=False)
result, info = supervisor.execute_task("What is 2**10 + 3 - 2**9?")

print(f"Result: {result}")
print(f"Cost: {info['cost_stats']['total_cost']}")
print(f"Agents used: {info['cost_stats']['agent_breakdown'].keys()}")
```

## 🔄 Pipeline Steps

### Step 1: Data Collection

Collects execution traces from the original static supervisor:

- **Input**: Test cases (math, research, mixed problems)
- **Output**: `execution_traces.json` with agent sequences and costs
- **Purpose**: Generate training data for RL orchestrator

### Step 2: Imitation Learning

Pre-trains RL orchestrator to mimic original supervisor:

- **Input**: Execution traces
- **Method**: Supervised learning (cross-entropy loss)
- **Output**: Pre-trained policy network
- **Purpose**: Warm start for RL training

### Step 3: RL Fine-tuning

Optimizes orchestrator using REINFORCE algorithm:

- **Method**: Policy gradient with reward = accuracy - λ×cost
- **Objective**: Maximize task success while minimizing computational cost
- **Output**: Fully trained RL orchestrator
- **Purpose**: Learn optimal agent selection strategies

### Step 4: Benchmarking

Compares original vs RL supervisor performance:

- **Metrics**: Accuracy, execution time, cost, agent utilization
- **Analysis**: Statistical significance, win rates, efficiency gains
- **Output**: Comprehensive performance report
- **Purpose**: Validate improvements and analyze behavior

## 📊 Expected Results

Based on the research paper, you should expect:

- **Similar or better accuracy** (±2%)
- **20-40% cost reduction** through smarter agent selection
- **10-30% faster execution** via early termination
- **More efficient agent utilization** patterns

### Sample Output

```json
{
  "summary": {
    "total_tasks": 30,
    "original_avg_accuracy": 0.85,
    "rl_avg_accuracy": 0.87,
    "accuracy_improvement": 0.02,
    "avg_time_improvement": 0.25,
    "avg_cost_improvement": 0.32
  },
  "win_rates": {
    "rl_better_accuracy": 0.6,
    "rl_faster": 0.73,
    "rl_cheaper": 0.8
  }
}
```

## 📁 Project Structure

```
evolving-orchestration/
├── README.md
├── requirements.txt
├── main.py                     # Main pipeline orchestrator
├── original_supervisor.py      # Your existing static supervisor
├── rl_supervisor.py            # New RL-based supervisor
├── test_cases.py              # Test case definitions
├── data_collector.py          # Step 1: Data collection
├── imitation_trainer.py       # Step 2: Imitation learning
├── benchmark_pipeline.py      # Step 4: Performance comparison
│
├── orchestration/
│   ├── rl_orchestrator.py     # RL orchestrator core
│   ├── reward_system.py       # Reward calculation
│   └── training_manager.py    # RL training management
│
├── utils/
│   ├── cost_tracker.py        # Cost tracking utilities
│   └── logger.py             # Logging utilities
│
└── outputs/                   # Generated results
    └── experiment_name/
        ├── execution_traces.json
        ├── pretrained_orchestrator.pth
        ├── rl_trained_model.pth
        ├── benchmark_results.json
        └── pipeline_summary.json
```

## 🛠️ Configuration

### Hyperparameters

```python
# In main.py or component files
LEARNING_RATE = 0.001          # Policy network learning rate
LAMBDA_COST = 0.1              # Cost-accuracy tradeoff
BATCH_SIZE = 16                # Training batch size
NUM_EPOCHS = 100               # Imitation learning epochs
RL_EPISODES = 20               # RL fine-tuning episodes
```

### Problem Types

```python
# Supported problem categories
PROBLEM_TYPES = ["math", "research", "mixed"]

# Add custom test cases in test_cases.py
TestCases.get_math_problems()     # Mathematical calculations
TestCases.get_research_problems() # Factual questions
TestCases.get_mixed_problems()    # Multi-step tasks
```

## 📈 Monitoring & Analysis

### Training Progress

```bash
# View training logs
tail -f outputs/experiment_name/training.log

# Analyze results
python -c "
import json
with open('outputs/experiment_name/benchmark_results.json') as f:
    results = json.load(f)
print('Accuracy improvement:', results['comprehensive_report']['summary']['accuracy_improvement'])
print('Cost reduction:', results['comprehensive_report']['summary']['avg_cost_improvement'])
"
```

### Visualization

```python
# Plot training curves (optional)
import matplotlib.pyplot as plt
import json

with open('outputs/experiment_name/rl_training_results.json') as f:
    data = json.load(f)

rewards = [ep['reward'] for ep in data['training_results']]
plt.plot(rewards)
plt.title('RL Training Progress')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
```

## 🤝 Contributing

### Adding New Agents

```python
# In rl_supervisor.py
def calculate_custom_function(expression: str) -> str:
    # Your custom agent logic
    return result

custom_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[calculate_custom_function],
    prompt="You are a custom agent...",
    name="custom_agent"
)

# Update agent list
self.agents["custom_agent"] = custom_agent
```

### Adding New Test Cases

```python
# In testcases.py
@staticmethod
def get_custom_problems():
    return [
        ("Your custom task", "expected_answer"),
        # Add more tasks...
    ]
```

## 🐛 Troubleshooting

### Common Issues

**"No traces collected"**

- Ensure API keys are set correctly
- Check original supervisor runs successfully
- Verify test cases are appropriate

**"Low imitation accuracy"**

- Increase training epochs
- Reduce learning rate
- Check trace data quality

**"RL not improving"**

- Adjust lambda_cost parameter
- Verify reward function
- Check training data diversity

**"Benchmark fails"**

- Ensure both supervisors use same API keys
- Check model file paths exist
- Verify test case format

## 📚 References

- [Multi-Agent Collaboration via Evolving Orchestration](link-to-paper)
- [LangChain Documentation](https://python.langchain.com/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues:

1. Check troubleshooting section above
2. Review existing issues in repository
3. Create new issue with detailed error logs
4. Include your environment details and configuration

---

**Happy orchestrating! 🎭**
