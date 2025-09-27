#!/usr/bin/env python
import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from train.data_collector import DataCollector
from train.imitation_trainer import ImitationTrainer
from train.benchmark_pipeline import BenchmarkPipeline
from orchestration.rl_orchestrator import RLOrchestrator

class ComprehensiveTrainingPipeline:
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.dataset_dir = "dataset"
        self.checkpoint_dir = "checkpoint"
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"🚀 Enhanced Multi-Agent Training Pipeline: {self.experiment_name}")
    
    def step_1_collect_data(self, problem_types: list = ["math", "research"]) -> str:
        """Collect expert demonstrations and create synthetic data"""
        print("\n" + "="*50)
        print("STEP 1: Data Collection")
        print("="*50)
        
        traces_file = os.path.join(self.dataset_dir, "expert_traces.json")
        collector = DataCollector(traces_file)
        
        # Collect from original supervisor
        print("📊 Collecting expert demonstrations from original supervisor...")
        results = collector.collect_from_test_cases(problem_types)
        
        print(f"✅ Collected {results['total_traces']} original traces")
        print(f"   Success rate: {results['overall_success_rate']:.2%}")
        
        # Create synthetic data for 4-agent system
        print("🔧 Creating synthetic data for code/summary agents...")
        self._create_synthetic_traces(traces_file)
        
        # Save collection stats
        stats_file = os.path.join(self.dataset_dir, "collection_stats.json")
        final_stats = collector.get_statistics()
        with open(stats_file, 'w') as f:
            json.dump({
                "collection_results": results,
                "final_statistics": final_stats,
                "experiment_name": self.experiment_name
            }, f, indent=2)
        
        print(f"✅ Enhanced dataset with 4-agent traces")
        print(f"📁 Dataset saved to: {traces_file}")
        
        return traces_file
    
    def _create_synthetic_traces(self, traces_file: str):
        """Add synthetic traces for code_agent and summary_agent"""
        
        with open(traces_file, 'r') as f:
            traces = json.load(f)
        
        synthetic_traces = []
        
        # Code agent traces
        code_tasks = [
            ("Write a Python function to calculate fibonacci numbers", "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
            ("Create a function to check if a number is prime", "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))"),
            ("Write code to sort a list of numbers", "sorted_list = sorted(input_list)"),
            ("Create a lambda function to square numbers", "square = lambda x: x**2"),
            ("Write code to count vowels in a string", "count = sum(1 for char in text.lower() if char in 'aeiou')"),
        ]
        
        for task, expected in code_tasks:
            synthetic_traces.append({
                "task": task,
                "timestamp": datetime.now().isoformat(),
                "agent_sequence": ["code_agent"],
                "messages": [
                    {"role": "user", "content": task, "node": "supervisor"},
                    {"role": "ai", "content": expected, "node": "code_agent"}
                ],
                "success": True,
                "final_result": expected,
                "total_cost": 10,
                "execution_time": 2.0
            })
        
        # Summary agent traces  
        summary_tasks = [
            ("Summarize the research and math results", "Based on research and calculations, the final answer is..."),
            ("Combine all agent outputs into final answer", "Integrating all findings: ..."),
            ("Provide overall conclusion", "In conclusion, considering all aspects: ..."),
            ("Synthesize information from multiple sources", "After analyzing multiple sources: ..."),
            ("Create final report from agent results", "Final comprehensive report: ..."),
        ]
        
        for task, expected in summary_tasks:
            synthetic_traces.append({
                "task": task,
                "timestamp": datetime.now().isoformat(),
                "agent_sequence": ["summary_agent"],
                "messages": [
                    {"role": "user", "content": task, "node": "supervisor"},
                    {"role": "ai", "content": expected, "node": "summary_agent"}
                ],
                "success": True,
                "final_result": expected,
                "total_cost": 8,
                "execution_time": 1.5
            })
        
        # Mixed multi-agent traces
        mixed_tasks = [
            ("Research Python libraries and write code example", ["research_agent", "code_agent", "summary_agent"]),
            ("Calculate fibonacci and implement the function", ["math_agent", "code_agent", "summary_agent"]),
            ("Find sorting algorithms and implement one", ["research_agent", "code_agent", "summary_agent"]),
            ("Calculate prime numbers and write detection code", ["math_agent", "code_agent", "summary_agent"]),
            ("Research machine learning and provide code sample", ["research_agent", "code_agent", "summary_agent"]),
        ]
        
        for task, sequence in mixed_tasks:
            synthetic_traces.append({
                "task": task,
                "timestamp": datetime.now().isoformat(),
                "agent_sequence": sequence,
                "messages": [
                    {"role": "user", "content": task, "node": "supervisor"},
                    {"role": "ai", "content": f"Final result from {sequence[-1]}", "node": sequence[-1]}
                ],
                "success": True,
                "final_result": f"Comprehensive solution for: {task}",
                "total_cost": 25,
                "execution_time": 5.0
            })
        
        all_traces = traces + synthetic_traces
        
        with open(traces_file, 'w') as f:
            json.dump(all_traces, f, indent=2)
            
        print(f"   Added {len(synthetic_traces)} synthetic traces")
    
    def step_2_train_orchestrator(self, traces_file: str) -> str:
        """Train orchestrator using imitation learning"""
        print("\n" + "="*50)
        print("STEP 2: Orchestrator Training")
        print("="*50)
        
        orchestrator = RLOrchestrator(["research_agent", "math_agent", "code_agent", "summary_agent"])
        trainer = ImitationTrainer(orchestrator, learning_rate=0.001)
        
        print("🎓 Starting imitation learning...")
        training_results = trainer.train_imitation(
            traces_file, 
            num_epochs=50,
            batch_size=8,
            validation_split=0.2
        )
        
        checkpoint_path = os.path.join(self.checkpoint_dir, "orchestrator.pth")
        trainer.save_pretrained_model(checkpoint_path)
        
        # Save training results
        results_file = os.path.join(self.dataset_dir, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "training_results": training_results,
                "experiment_name": self.experiment_name,
                "checkpoint_path": checkpoint_path
            }, f, indent=2)
        
        print(f"✅ Training completed!")
        print(f"   Best validation accuracy: {training_results['best_val_accuracy']:.2%}")
        print(f"   Final train loss: {training_results['final_train_loss']:.4f}")
        print(f"💾 Checkpoint saved to: {checkpoint_path}")
        
        return checkpoint_path
    
    def step_3_benchmark_evaluation(self, checkpoint_path: str) -> str:
        """Benchmark trained orchestrator against original supervisor"""
        print("\n" + "="*50)
        print("STEP 3: Benchmark Evaluation")
        print("="*50)
        
        benchmark_file = os.path.join(self.dataset_dir, "benchmark_results.json")
        benchmark = BenchmarkPipeline(benchmark_file)
        
        print("📊 Running comprehensive benchmark...")
        report = benchmark.benchmark_test_suite(["math", "research"])
        benchmark.save_results()
        
        print(f"✅ Benchmarking completed!")
        print(f"   Tasks tested: {report['summary']['total_tasks']}")
        print(f"   Enhanced accuracy: {report['summary']['enhanced_avg_accuracy']:.2%}")
        print(f"   Accuracy improvement: {report['summary']['accuracy_improvement']:.2%}")
        print(f"   Speed improvement: {report['summary']['avg_time_improvement']:.1%}")
        print(f"   Cost improvement: {report['summary']['avg_cost_improvement']:.1%}")
        print(f"📊 Results saved to: {benchmark_file}")
        
        return benchmark_file
    
    def run_full_pipeline(self, 
                         problem_types: list = ["math", "research"], 
                         skip_steps: list = [],
                         run_benchmark: bool = True):
        """Run complete training pipeline"""
        print(f"🚀 Running Full Enhanced Training Pipeline")
        print(f"Experiment: {self.experiment_name}")
        
        results = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "problem_types": problem_types
        }
        
        try:
            # Step 1: Data Collection
            if 1 not in skip_steps:
                traces_file = self.step_1_collect_data(problem_types)
                results["traces_file"] = traces_file
            else:
                traces_file = os.path.join(self.dataset_dir, "expert_traces.json")
            
            # Step 2: Training
            if 2 not in skip_steps:
                checkpoint_path = self.step_2_train_orchestrator(traces_file)
                results["checkpoint_path"] = checkpoint_path
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, "orchestrator.pth")
            
            # Step 3: Benchmark (optional)
            if run_benchmark and 3 not in skip_steps:
                benchmark_file = self.step_3_benchmark_evaluation(checkpoint_path)
                results["benchmark_file"] = benchmark_file
            
            results["status"] = "completed"
            results["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            print(f"❌ Pipeline failed: {e}")
            raise
        
        # Save pipeline summary
        summary_file = os.path.join(self.dataset_dir, "pipeline_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 All results saved in: {self.dataset_dir} and {self.checkpoint_dir}")
        print(f"📋 Summary: {summary_file}")
        print(f"\n💡 To use trained orchestrator:")
        print(f"   python rl_supervisor.py")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent Training Pipeline")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.add_argument("--problems", nargs="+", default=["math", "research"], 
                       help="Problem types to include")
    parser.add_argument("--skip", nargs="+", type=int, default=[], 
                       help="Steps to skip (1-3)")
    parser.add_argument("--no-benchmark", action="store_true", 
                       help="Skip benchmark evaluation")
    parser.add_argument("--step", type=int, help="Run single step only")
    
    args = parser.parse_args()
    
    pipeline = ComprehensiveTrainingPipeline(args.name)
    
    if args.step:
        if args.step == 1:
            pipeline.step_1_collect_data(args.problems)
        elif args.step == 2:
            traces_file = os.path.join(pipeline.dataset_dir, "expert_traces.json")
            pipeline.step_2_train_orchestrator(traces_file)
        elif args.step == 3:
            checkpoint_path = os.path.join(pipeline.checkpoint_dir, "orchestrator.pth")
            pipeline.step_3_benchmark_evaluation(checkpoint_path)
    else:
        pipeline.run_full_pipeline(
            args.problems, 
            args.skip, 
            not args.no_benchmark
        )

if __name__ == "__main__":
    main()