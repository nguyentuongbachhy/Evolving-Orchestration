# Multi-Agent RL Orchestrator

Hệ thống điều phối đa tác nhân dựa trên Reinforcement Learning, lấy cảm hứng từ mô hình **Puppeteer** trong nghiên cứu "Multi-Agent Collaboration via Evolving Orchestration".

## 🎯 Tổng quan

Dự án này chuyển đổi các hệ thống đa tác nhân tĩnh thành các orchestrator học tập động có khả năng:

- **Học cách chọn tác nhân tối ưu** thông qua reinforcement learning
- **Giảm chi phí tính toán** trong khi duy trì độ chính xác
- **Thích ứng với các loại bài toán khác nhau** một cách tự động
- **Phát triển các mô hình hợp tác** theo thời gian

## 🏗️ Kiến trúc

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
└─────────────────┘    └──────────────────┘    │   Code Agent    │
                                               │  Summary Agent  │
                                               └─────────────────┘
```

## 📦 Cài đặt

### Yêu cầu hệ thống

```bash
Python 3.8+
PyTorch 2.0+
```

### Thiết lập

```bash
# Clone repository
git clone https://github.com/nguyentuongbachhy/Evolving-Orchestration.git

# Cài đặt dependencies
pip install -r requirements.txt

# Thiết lập biến môi trường
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
```

### Tạo file .env

```bash
cp .env.example .env
# Chỉnh sửa .env với API keys của bạn
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## 🚀 Hướng dẫn sử dụng

### Chạy toàn bộ pipeline

```bash
# Chạy thực nghiệm hoàn chỉnh
python train.py

# Kết quả sẽ được lưu trong dataset/ và checkpoint/
```

### Thực thi từng bước

```bash
# Bước 1: Thu thập dữ liệu từ original supervisor
python train.py --step 1 --problems math research

# Bước 2: Huấn luyện RL orchestrator qua imitation learning
python train.py --step 2

# Bước 3: Fine-tuning với reinforcement learning
python train.py --step 3

# Bước 4: So sánh hiệu suất
python train.py --step 4
```

### Sử dụng RL Supervisor trực tiếp

```python
from rl_supervisor import execute_task

# Sử dụng RL supervisor đã được huấn luyện
result, info = execute_task("What is 2**10 + 3 - 2**9?")

print(f"Kết quả: {result}")
print(f"Chi phí: {info['cost_stats']['total_cost']}")
print(f"Các agent được sử dụng: {info['cost_stats']['agent_breakdown'].keys()}")
```

### Chạy interactive mode

```bash
# Chạy RL supervisor ở chế độ tương tác
python rl_supervisor.py
```

## 🔄 Các bước trong Pipeline

### Bước 1: Thu thập dữ liệu (Data Collection)

Thu thập execution traces từ original static supervisor:

- **Input**: Test cases (toán học, nghiên cứu, lập trình, hỗn hợp)
- **Output**: `expert_traces.json` với chuỗi agents và chi phí
- **Mục đích**: Tạo dữ liệu huấn luyện cho RL orchestrator

### Bước 2: Imitation Learning

Pre-train RL orchestrator để bắt chước original supervisor:

- **Input**: Execution traces
- **Phương pháp**: Supervised learning (cross-entropy loss)
- **Output**: Pre-trained policy network
- **Mục đích**: Khởi tạo cho RL training

### Bước 3: RL Fine-tuning

Tối ưu orchestrator sử dụng thuật toán REINFORCE:

- **Phương pháp**: Policy gradient với reward = accuracy - λ×cost
- **Mục tiêu**: Tối đa hóa thành công task và tối thiểu hóa chi phí tính toán
- **Output**: RL orchestrator được huấn luyện hoàn chỉnh
- **Mục đích**: Học các chiến lược chọn agent tối ưu

### Bước 4: Benchmarking

So sánh hiệu suất original vs RL supervisor:

- **Metrics**: Độ chính xác, thời gian thực thi, chi phí, sử dụng agent
- **Phân tích**: Ý nghĩa thống kê, tỷ lệ thắng, cải thiện hiệu suất
- **Output**: Báo cáo hiệu suất toàn diện
- **Mục đích**: Xác thực cải tiến và phân tích hành vi

## 🤖 Các Agent trong hệ thống

### Research Agent

- **Chức năng**: Xử lý các task nghiên cứu, tìm kiếm thông tin
- **Tools**: TavilySearch (tìm kiếm web)
- **Ứng dụng**: Trả lời câu hỏi sự kiện, tìm kiếm thông tin mới

### Math Agent

- **Chức năng**: Xử lý các phép tính toán học
- **Tools**: Math expression evaluator
- **Ứng dụng**: Giải các bài toán số học, đại số

### Code Agent

- **Chức năng**: Xử lý các task lập trình
- **Tools**: Python REPL
- **Ứng dụng**: Viết và thực thi code Python, debug

### Summary Agent

- **Chức năng**: Tổng hợp kết quả từ các agent khác
- **Tools**: Result synthesizer
- **Ứng dụng**: Tạo câu trả lời cuối cùng, tổng hợp thông tin

## 📊 Kết quả mong đợi

Dựa trên nghiên cứu, bạn có thể mong đợi:

- **Độ chính xác tương đương hoặc tốt hơn** (±2%)
- **Giảm 20-40% chi phí** thông qua việc chọn agent thông minh hơn
- **Nhanh hơn 10-30%** qua việc kết thúc sớm
- **Sử dụng agent hiệu quả hơn**

### Mẫu Output

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
  },
  "orchestration_metrics": {
    "agent_diversity": 0.75,
    "graph_density": 0.6,
    "reasoning_depth": 3,
    "cycle_count": 0
  }
}
```

## 📁 Cấu trúc Project

```
seminar/
├── README.md                   # File hướng dẫn này
├── requirements.txt           # Danh sách dependencies
├── .env.example              # Template cho environment variables
├── langgraph.json            # Cấu hình LangGraph
├── train.py                  # Main training pipeline
├── original_supervisor.py    # Original static supervisor
├── rl_supervisor.py         # RL-based supervisor
│
├── orchestration/           # Core RL orchestration logic
│   ├── rl_orchestrator.py  # RL orchestrator với PolicyNetwork
│   ├── reward_system.py    # Hệ thống tính reward
│   └── training_manager.py # Quản lý RL training
│
├── train/                  # Training modules
│   ├── data_collector.py   # Thu thập dữ liệu từ original supervisor
│   ├── imitation_trainer.py # Imitation learning trainer
│   └── benchmark_pipeline.py # So sánh hiệu suất
│
├── test/                   # Test cases và testing utilities
│   ├── testcases.py       # Định nghĩa các test case
│   └── __init__.py
│
├── utils/                  # Utilities
│   └── cost_tracker.py    # Theo dõi chi phí API calls
│
├── dataset/               # Generated training data
│   ├── expert_traces.json # Expert demonstrations
│   ├── collection_stats.json
│   ├── training_results.json
│   └── benchmark_results.json
│
├── checkpoint/            # Model checkpoints
│   └── orchestrator.pth  # Trained RL orchestrator
│
└── .langgraph_api/       # LangGraph runtime data
    └── store.pckl
```

## 🛠️ Cấu hình

### Hyperparameters chính

```python
# Trong train.py và các component
LEARNING_RATE = 0.001          # Learning rate cho policy network
LAMBDA_COST = 0.1              # Trọng số tradeoff cost-accuracy
BATCH_SIZE = 16                # Batch size cho training
NUM_EPOCHS = 100               # Số epoch cho imitation learning
RL_EPISODES = 20               # Số episode cho RL fine-tuning
HIDDEN_DIM = 256               # Kích thước hidden layer
```

### Loại bài toán hỗ trợ

```python
# Các loại problem được hỗ trợ
PROBLEM_TYPES = ["math", "research", "code", "mixed"]

# Trong test/testcases.py:
TestCases.get_math_test_cases()     # Bài toán tính toán
TestCases.get_research_test_cases() # Câu hỏi sự kiện
TestCases.get_code_test_cases()     # Bài toán lập trình
TestCases.get_mixed_test_cases()    # Bài toán đa bước
```

### Policy Network Architecture

```python
# Multi-layer với attention mechanism
class PolicyNetwork(nn.Module):
    - State embedding layer
    - Multi-head attention
    - Agent-specific projection
    - Temperature-controlled softmax
    - Entropy regularization
```

## 📈 Theo dõi & Phân tích

### Theo dõi quá trình training

```bash
# Xem training logs
tail -f dataset/training_results.json

# Phân tích kết quả
python -c "
import json
with open('dataset/benchmark_results.json') as f:
    results = json.load(f)
print('Cải thiện độ chính xác:', results['comprehensive_report']['summary']['accuracy_improvement'])
print('Giảm chi phí:', results['comprehensive_report']['summary']['avg_cost_improvement'])
"
```

### Metrics theo dõi

```python
# Orchestration metrics được track:
- Agent diversity: Đa dạng trong việc chọn agent
- Graph density: Mật độ kết nối giữa các agent
- Reasoning depth: Độ sâu suy luận
- Cycle count: Số lượng chu kỳ lặp
- Agent usage: Thống kê sử dụng từng agent

# Cost metrics:
- Total cost: Tổng chi phí API
- Agent breakdown: Chi phí theo từng agent
- Time efficiency: Hiệu suất thời gian
```

### Visualization (tùy chọn)

```python
# Vẽ biểu đồ training progress
import matplotlib.pyplot as plt
import json

with open('dataset/training_results.json') as f:
    data = json.load(f)

# Plot reward curve nếu có
if 'rl_training' in data:
    rewards = data['rl_training']['episode_rewards']
    plt.plot(rewards)
    plt.title('RL Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
```

## 🤝 Mở rộng hệ thống

### Thêm Agent mới

```python
# Trong rl_supervisor.py
from langchain_core.tools import tool

@tool
def custom_function(input_data: str) -> str:
    """Mô tả chức năng của tool"""
    # Logic xử lý của bạn
    return result

# Tạo agent mới
custom_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[custom_function],
    prompt=(
        "You are a custom agent. "
        "Assist ONLY custom-related tasks, DO NOT do any else. "
        "After you're done with your tasks, respond to the supervisor directly. "
        "Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="custom_agent"
)

# Cập nhật danh sách agents
agents["custom_agent"] = custom_agent

# Cập nhật orchestrator
orchestrator = RLOrchestrator(agent_names=list(agents.keys()))
```

### Thêm Test Cases mới

```python
# Trong test/testcases.py
@staticmethod
def get_custom_test_cases() -> List[Tuple[str, str]]:
    """Custom test cases for your domain"""
    return [
        ("Your custom task", "expected_answer"),
        ("Another custom task", "another_answer"),
        # Thêm nhiều task hơn...
    ]

# Cập nhật get_all_test_cases()
@staticmethod
def get_all_test_cases() -> Dict[str, List[Tuple[str, str]]]:
    return {
        "math": TestCases.get_math_test_cases(),
        "research": TestCases.get_research_test_cases(),
        "code": TestCases.get_code_test_cases(),
        "mixed": TestCases.get_mixed_test_cases(),
        "custom": TestCases.get_custom_test_cases()  # Thêm dòng này
    }
```

## 🐛 Xử lý sự cố

### Các vấn đề thường gặp

**"Không thu thập được traces"**

- Kiểm tra API keys đã được set đúng trong .env
- Kiểm tra original supervisor chạy thành công
- Xác minh test cases phù hợp và đúng format

**"Độ chính xác imitation learning thấp"**

- Tăng số epoch training
- Giảm learning rate
- Kiểm tra chất lượng dữ liệu trace
- Đảm bảo dữ liệu đủ đa dạng

**"RL không cải thiện"**

- Điều chỉnh tham số lambda_cost
- Kiểm tra reward function
- Kiểm tra đa dạng dữ liệu training
- Tăng số episode training

**"Benchmark thất bại"**

- Đảm bảo cả hai supervisor sử dụng cùng API keys
- Kiểm tra checkpoint file tồn tại
- Xác minh format test case đúng

**"CUDA out of memory"**

- Giảm batch size
- Sử dụng CPU thay vì GPU: `device = "cpu"`
- Giảm hidden_dim của PolicyNetwork

**"API rate limit"**

- Thêm delay giữa các API call
- Sử dụng API key có rate limit cao hơn
- Giảm số test case để test

## � Chi tiết kỹ thuật

### PolicyNetwork Architecture

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 256):
        # Multi-layer architecture với:
        - State embedding với LayerNorm và SiLU activation
        - Multi-head attention mechanism (8 heads)
        - Agent-specific projection layers
        - Temperature-controlled softmax cho exploration
        - Dropout cho regularization
```

### Reward System

```python
# Reward = Task Success - λ * Cost
reward = success_reward - lambda_cost * normalized_cost

# Với:
- success_reward: 1.0 nếu thành công, 0.0 nếu thất bại
- lambda_cost: Trọng số balance accuracy/cost (default: 0.1)
- normalized_cost: Chi phí được chuẩn hóa theo baseline
```

### State Representation

```python
# State bao gồm:
- Task embedding (từ nội dung task)
- History của agent selections
- Current message context
- Agent usage statistics
- Cost accumulation
```

## 📚 Tài liệu tham khảo

- [Multi-Agent Collaboration via Evolving Orchestration](https://arxiv.org/abs/2310.00615)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [REINFORCE Algorithm](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

## 📄 License

MIT License - xem LICENSE file để biết chi tiết.

## 🙋‍♂️ Hỗ trợ

Để được hỗ trợ khi gặp vấn đề:

1. Kiểm tra phần troubleshooting ở trên
2. Xem các issue đã có trong repository
3. Tạo issue mới với error logs chi tiết
4. Bao gồm thông tin môi trường và cấu hình

## 🎯 Roadmap

- [ ] Thêm hỗ trợ cho multi-modal agents
- [ ] Tích hợp với các LLM khác (Claude, Gemini)
- [ ] Web UI để monitor training process
- [ ] Distributed training cho datasets lớn
- [ ] Advanced reward shaping strategies

---

**Chúc bạn orchestrating vui vẻ! 🎭**
