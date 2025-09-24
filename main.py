#!/usr/bin/env python
import os
import json
import torch
import argparse
from datetime import datetime

from train.data_collector import DataCollector
from train.imitation_trainer import ImitationTrainer
from train.benchmark_pipeline import BenchmarkPipeline
from orchestration.rl_orchestrator import RLOrchestrator
from rl_supervisor import RLSupervisor

class MultiAgentRLPipeline:
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = f"outputs/{self.experiment_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Starting Multi-Agent RL Pipeline: {self.experiment_name}")
        print(f"Output directory: {self.output_dir}")
    
    def step_1_collect_data(self, problem_types: list = ["math", "research"]) -> str:
        print("\n" + "="*50)
        print("STEP 1: Data Collection from Original Supervisor")
        print("="*50)
        
        traces_file = os.path.join(self.output_dir, "execution_traces.json")
        collector = DataCollector(traces_file)
        
        collection_results = collector.collect_from_test_cases(problem_types)
        
        stats = collector.get_statistics()
        
        print(f"✅ Data collection completed!")
        print(f"   📊 Total traces: {collection_results['total_traces']}")
        print(f"   ✔️  Success rate: {collection_results['overall_success_rate']:.2%}")
        print(f"   💾 Saved to: {traces_file}")
        
        stats_file = os.path.join(self.output_dir, "collection_stats.json")
        with open(stats_file, 'w') as f:
            json.dump({"collection_results": collection_results, "statistics": stats}, f, indent=2)
        
        return traces_file
    
    def step_2_imitation_learning(self, traces_file: str) -> str:
        print("\n" + "="*50)
        print("STEP 2: Imitation Learning Pre-training")
        print("="*50)
        
        orchestrator = RLOrchestrator(["research_agent", "math_agent"])
        trainer = ImitationTrainer(orchestrator, learning_rate=0.001)
        
        training_results = trainer.train_imitation(
            traces_file, 
            num_epochs=100,
            batch_size=16,
            validation_split=0.2
        )
        
        pretrained_model = os.path.join(self.output_dir, "pretrained_orchestrator.pth")
        trainer.save_pretrained_model(pretrained_model)
        
        eval_results = trainer.evaluate_imitation_quality(traces_file)
        
        print(f"✅ Imitation learning completed!")
        print(f"   🎯 Best validation accuracy: {training_results['best_val_accuracy']:.2%}")
        print(f"   📈 Final train loss: {training_results['final_train_loss']:.4f}")
        print(f"   🧪 Test accuracy: {eval_results.get('test_accuracy', 0):.2%}")
        print(f"   💾 Model saved to: {pretrained_model}")
        
        imitation_results_file = os.path.join(self.output_dir, "imitation_results.json")
        with open(imitation_results_file, 'w') as f:
            json.dump({
                "training_results": training_results,
                "evaluation_results": eval_results
            }, f, indent=2)
        
        return pretrained_model
    
    def step_3_rl_training(self, pretrained_model: str, training_tasks: list = None) -> str:
        print("\n" + "="*50)
        print("STEP 3: Reinforcement Learning Fine-tuning")
        print("="*50)
        
        rl_supervisor = RLSupervisor(training_mode=True)
        
        if os.path.exists(pretrained_model):
            rl_supervisor.training_manager.orchestrator.policy_net.load_state_dict(
                torch.load(pretrained_model)['policy_net_state_dict']
            )
            print("✅ Loaded pretrained model")
        
        if not training_tasks:
            from test.testcases import TestCases
            test_cases = TestCases.get_training_set("math", train_ratio=0.8)
            training_tasks = test_cases["train"]
        
        training_results = rl_supervisor.train_batch(
            training_tasks,
            num_epochs=20
        )
        
        rl_model = os.path.join(self.output_dir, "rl_trained_model.pth")
        rl_supervisor.save_model(rl_model)
        
        metrics = rl_supervisor.get_performance_metrics()
        
        print(f"✅ RL training completed!")
        print(f"   📊 Total episodes: {metrics.get('total_episodes', 0)}")
        print(f"   🎯 Average reward: {metrics.get('avg_reward', 0):.3f}")
        print(f"   💰 Average cost: {metrics.get('avg_cost', 0):.1f}")
        print(f"   💾 Model saved to: {rl_model}")
        
        rl_results_file = os.path.join(self.output_dir, "rl_training_results.json")
        with open(rl_results_file, 'w') as f:
            json.dump({
                "training_results": training_results[-10:],  # Last 10 episodes
                "metrics": metrics
            }, f, indent=2)
        
        return rl_model
    
    def step_4_benchmark(self, rl_model: str = None) -> str:
        print("\n" + "="*50)
        print("STEP 4: Benchmark Comparison")
        print("="*50)
        
        rl_supervisor = RLSupervisor(training_mode=False)
        
        if rl_model and os.path.exists(rl_model):
            try:
                import torch
                checkpoint = torch.load(rl_model)
                rl_supervisor.orchestrator.policy_net.load_state_dict(
                    checkpoint['policy_net_state_dict']
                )
                print("✅ Loaded RL trained model")
            except:
                print("⚠️ Could not load RL model, using default")
        
        benchmark_results_file = os.path.join(self.output_dir, "benchmark_results.json")
        benchmark = BenchmarkPipeline(rl_supervisor, benchmark_results_file)
        
        report = benchmark.benchmark_test_suite(["math", "research"])
        benchmark.save_results()
        
        print(f"✅ Benchmarking completed!")
        print(f"   📊 Tasks tested: {report['summary']['total_tasks']}")
        print(f"   🎯 Original accuracy: {report['summary']['original_avg_accuracy']:.2%}")
        print(f"   🚀 RL accuracy: {report['summary']['rl_avg_accuracy']:.2%}")
        print(f"   ⚡ Time improvement: {report['summary']['avg_time_improvement']:.1%}")
        print(f"   💰 Cost improvement: {report['summary']['avg_cost_improvement']:.1%}")
        print(f"   💾 Results saved to: {benchmark_results_file}")
        
        return benchmark_results_file
    
    def run_full_pipeline(self, problem_types: list = ["math", "research"], skip_steps: list = []):
        print(f"\n🚀 Running Full Multi-Agent RL Pipeline")
        print(f"Experiment: {self.experiment_name}")
        
        results = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "problem_types": problem_types
        }
        
        traces_file = None
        pretrained_model = None
        rl_model = None
        benchmark_file = None
        
        try:
            if 1 not in skip_steps:
                traces_file = self.step_1_collect_data(problem_types)
                results["step_1_traces"] = traces_file
            
            if 2 not in skip_steps and traces_file:
                pretrained_model = self.step_2_imitation_learning(traces_file)
                results["step_2_model"] = pretrained_model
            
            if 3 not in skip_steps:
                rl_model = self.step_3_rl_training(pretrained_model)
                results["step_3_model"] = rl_model
            
            if 4 not in skip_steps:
                benchmark_file = self.step_4_benchmark(rl_model)
                results["step_4_results"] = benchmark_file
            
            results["status"] = "completed"
            results["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            print(f"❌ Pipeline failed: {e}")
            raise
        
        pipeline_summary = os.path.join(self.output_dir, "pipeline_summary.json")
        with open(pipeline_summary, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 All results saved in: {self.output_dir}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RL Pipeline")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.add_argument("--problems", nargs="+", default=["math", "research"], 
                       help="Problem types to include")
    parser.add_argument("--skip", nargs="+", type=int, default=[], 
                       help="Steps to skip (1-4)")
    parser.add_argument("--step", type=int, help="Run single step only")
    
    args = parser.parse_args()
    
    pipeline = MultiAgentRLPipeline(args.name)
    
    if args.step:
        if args.step == 1:
            pipeline.step_1_collect_data(args.problems)
        elif args.step == 2:
            traces_file = "outputs/execution_traces.json"
            pipeline.step_2_imitation_learning(traces_file)
        elif args.step == 3:
            pretrained_model = "outputs/pretrained_orchestrator.pth"
            pipeline.step_3_rl_training(pretrained_model)
        elif args.step == 4:
            rl_model = "outputs/rl_trained_model.pth"
            pipeline.step_4_benchmark(rl_model)
    else:
        pipeline.run_full_pipeline(args.problems, args.skip)

if __name__ == "__main__":
    main()