#!/usr/bin/env python
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from langchain_core.messages import HumanMessage, AIMessage
from orchestration.rl_orchestrator import RLOrchestrator

class ImitationTrainer:
    def __init__(self, orchestrator: RLOrchestrator, learning_rate: float = 0.001):
        self.orchestrator = orchestrator
        self.agent_to_idx = {name: idx for idx, name in enumerate(orchestrator.agent_names)}
        self.idx_to_agent = {idx: name for name, idx in self.agent_to_idx.items()}
        
        self.imitation_optimizer = torch.optim.Adam(
            orchestrator.policy_net.parameters(), 
            lr=learning_rate
        )
        
        self.training_history = []
        
    def load_traces(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r') as f:
            traces = json.load(f)
        return [trace for trace in traces if trace.get("success", False)]
    
    def extract_training_data(self, traces: List[Dict]) -> List[Tuple]:
        training_pairs = []
        
        for trace in traces:
            if not trace.get("agent_sequence") or not trace.get("messages"):
                continue
                
            messages = []
            agent_sequence = trace["agent_sequence"]
            
            for i, agent_name in enumerate(agent_sequence):
                if agent_name in self.agent_to_idx:
                    messages_up_to_step = self._reconstruct_messages_at_step(
                        trace["messages"], i
                    )
                    
                    if messages_up_to_step:
                        training_pairs.append((
                            messages_up_to_step,
                            self.agent_to_idx[agent_name]
                        ))
        
        return training_pairs
    
    def _reconstruct_messages_at_step(self, all_messages: List[Dict], step: int) -> List:
        reconstructed = []
        agent_calls_seen = 0
        
        for msg in all_messages:
            if msg.get("node") in ["research_agent", "math_agent"]:
                if agent_calls_seen >= step:
                    break
                agent_calls_seen += 1
            
            if msg["role"] == "user":
                reconstructed.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                reconstructed.append(AIMessage(content=msg["content"]))
        
        return reconstructed
    
    def train_imitation(self, 
                       traces_filepath: str, 
                       num_epochs: int = 50,
                       batch_size: int = 16,
                       validation_split: float = 0.2) -> Dict:
        
        traces = self.load_traces(traces_filepath)
        print(f"Loaded {len(traces)} successful traces")
        
        training_pairs = self.extract_training_data(traces)
        print(f"Extracted {len(training_pairs)} training pairs")
        
        if not training_pairs:
            raise ValueError("No valid training data extracted")
        
        train_pairs, val_pairs = train_test_split(
            training_pairs, 
            test_size=validation_split,
            random_state=42
        )
        
        best_val_accuracy = 0
        training_results = {
            "train_losses": [],
            "val_accuracies": [],
            "best_val_accuracy": 0,
            "final_train_loss": 0
        }
        
        for epoch in range(num_epochs):
            epoch_train_loss = self._train_epoch(train_pairs, batch_size)
            val_accuracy = self._validate(val_pairs)
            
            training_results["train_losses"].append(epoch_train_loss)
            training_results["val_accuracies"].append(val_accuracy)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                training_results["best_val_accuracy"] = best_val_accuracy
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, "
                      f"Val Accuracy = {val_accuracy:.4f}")
        
        training_results["final_train_loss"] = training_results["train_losses"][-1]
        self.training_history.append(training_results)
        
        return training_results
    
    def _train_epoch(self, train_pairs: List[Tuple], batch_size: int) -> float:
        total_loss = 0
        num_batches = 0
        
        np.random.shuffle(train_pairs)
        
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i + batch_size]
            batch_loss = self._train_batch(batch)
            total_loss += batch_loss
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def _train_batch(self, batch: List[Tuple]) -> float:
        self.imitation_optimizer.zero_grad()
        
        batch_loss = 0
        
        for messages, target_agent_idx in batch:
            state = self.orchestrator.state_encoder.encode_messages(messages)
            action_probs = self.orchestrator.policy_net(state)
            
            target = torch.tensor(target_agent_idx, dtype=torch.long)
            loss = F.cross_entropy(action_probs.unsqueeze(0), target.unsqueeze(0))
            
            batch_loss += loss
        
        if len(batch) > 0:
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.orchestrator.policy_net.parameters(), 1.0)
            self.imitation_optimizer.step()
        
        return batch_loss.item()
    
    def _validate(self, val_pairs: List[Tuple]) -> float:
        if not val_pairs:
            return 0.0
            
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for messages, target_agent_idx in val_pairs:
                state = self.orchestrator.state_encoder.encode_messages(messages)
                action_probs = self.orchestrator.policy_net(state)
                
                predicted_agent_idx = torch.argmax(action_probs).item()
                
                if predicted_agent_idx == target_agent_idx:
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions
    
    def evaluate_imitation_quality(self, test_traces_filepath: str) -> Dict:
        test_traces = self.load_traces(test_traces_filepath)
        test_pairs = self.extract_training_data(test_traces)
        
        if not test_pairs:
            return {"error": "No test data available"}
        
        accuracy = self._validate(test_pairs)
        agent_confusion = self._compute_confusion_matrix(test_pairs)
        
        return {
            "test_accuracy": accuracy,
            "total_test_pairs": len(test_pairs),
            "agent_confusion_matrix": agent_confusion,
            "per_agent_accuracy": self._compute_per_agent_accuracy(test_pairs)
        }
    
    def _compute_confusion_matrix(self, test_pairs: List[Tuple]) -> Dict:
        confusion = {}
        
        with torch.no_grad():
            for messages, target_idx in test_pairs:
                state = self.orchestrator.state_encoder.encode_messages(messages)
                action_probs = self.orchestrator.policy_net(state)
                predicted_idx = torch.argmax(action_probs).item()
                
                target_agent = self.idx_to_agent[target_idx]
                predicted_agent = self.idx_to_agent[predicted_idx]
                
                if target_agent not in confusion:
                    confusion[target_agent] = {}
                if predicted_agent not in confusion[target_agent]:
                    confusion[target_agent][predicted_agent] = 0
                    
                confusion[target_agent][predicted_agent] += 1
        
        return confusion
    
    def _compute_per_agent_accuracy(self, test_pairs: List[Tuple]) -> Dict:
        agent_stats = {agent: {"correct": 0, "total": 0} for agent in self.agent_to_idx.keys()}
        
        with torch.no_grad():
            for messages, target_idx in test_pairs:
                state = self.orchestrator.state_encoder.encode_messages(messages)
                action_probs = self.orchestrator.policy_net(state)
                predicted_idx = torch.argmax(action_probs).item()
                
                target_agent = self.idx_to_agent[target_idx]
                agent_stats[target_agent]["total"] += 1
                
                if predicted_idx == target_idx:
                    agent_stats[target_agent]["correct"] += 1
        
        return {
            agent: stats["correct"] / max(1, stats["total"]) 
            for agent, stats in agent_stats.items()
        }
    
    def save_pretrained_model(self, filepath: str):
        torch.save({
            'policy_net_state_dict': self.orchestrator.policy_net.state_dict(),
            'imitation_optimizer_state_dict': self.imitation_optimizer.state_dict(),
            'training_history': self.training_history,
            'agent_to_idx': self.agent_to_idx
        }, filepath)
        
    def load_pretrained_model(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.orchestrator.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.imitation_optimizer.load_state_dict(checkpoint['imitation_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']

if __name__ == "__main__":
    orchestrator = RLOrchestrator(["research_agent", "math_agent"])
    trainer = ImitationTrainer(orchestrator)
    
    results = trainer.train_imitation("execution_traces.json", num_epochs=100)
    print(f"Training completed. Best validation accuracy: {results['best_val_accuracy']:.4f}")
    
    trainer.save_pretrained_model("pretrained_orchestrator.pth")
    
    eval_results = trainer.evaluate_imitation_quality("execution_traces.json")
    print("Evaluation Results:")
    print(json.dumps(eval_results, indent=2))