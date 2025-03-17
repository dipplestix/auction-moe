import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


class ExpertUtilizationTracker:
    """A dedicated class for tracking expert utilization in MoE layers.
    
    This class handles the collection, analysis, and visualization of 
    expert utilization statistics, keeping these concerns separate from
    the model implementation.
    """
    
    def __init__(self, num_layers, num_experts_per_layer, device=None):
        """Initialize the expert utilization tracker.
        
        Args:
            num_layers: Number of MoE layers to track
            num_experts_per_layer: Number of experts in each layer
            device: Computation device
        """
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.device = device if device else torch.device('cpu')
        
        self.reset()
    
    def reset(self):
        """Reset all tracking statistics."""
        self.expert_counts = [
            torch.zeros(num_experts, dtype=torch.long, device=self.device)
            for num_experts in self.num_experts_per_layer
        ]
        self.expert_gate_values = [
            torch.zeros(num_experts, device=self.device)
            for num_experts in self.num_experts_per_layer
        ]
        self.samples_processed = 0
        self.batches_processed = 0
    
    def update(self, layer_idx, routing_info):
        """Update statistics based on routing information from a forward pass.
        
        Args:
            layer_idx: Index of the MoE layer
            routing_info: Tuple of (gates, expert_indices, batch_size)
                - gates: Tensor of shape [batch_size, num_experts] with routing probabilities
                - expert_indices: Tensor of shape [batch_size, k] with indices of selected experts
                - batch_size: Number of samples in this batch
        """
        gates, expert_indices, batch_size = routing_info
        self.samples_processed += batch_size
        self.batches_processed += 1
        
        # Update expert counts and gate values
        num_experts = self.num_experts_per_layer[layer_idx]
        for expert_idx in range(num_experts):
            # Find which batch samples use this expert (any position in k)
            mask = (expert_indices == expert_idx).any(dim=1)
            count = mask.sum().item()
            
            # Update counts
            self.expert_counts[layer_idx][expert_idx] += count
            
            # Skip if no samples use this expert
            if not mask.any():
                continue
            
            # Update gate values (importance)
            self.expert_gate_values[layer_idx][expert_idx] += gates[mask, expert_idx].sum().item()
            
    def get_stats(self, layer_idx=None):
        """Get expert utilization statistics.
        
        Args:
            layer_idx: Optional index of layer to get stats for. If None, returns stats for all layers.
            
        Returns:
            dict: Dictionary containing expert utilization metrics
        """
        if self.batches_processed == 0:
            return {}
            
        if layer_idx is not None:
            return self._compute_layer_stats(layer_idx)
        
        # Compute stats for all layers
        all_stats = {}
        for i in range(self.num_layers):
            stats = self._compute_layer_stats(i)
            all_stats[f"layer{i+1}"] = stats
            
        return all_stats
    
    def _compute_layer_stats(self, layer_idx):
        """Compute statistics for a specific layer."""
        num_experts = self.num_experts_per_layer[layer_idx]
        expert_counts = self.expert_counts[layer_idx]
        expert_gate_values = self.expert_gate_values[layer_idx]
        
        # Expert frequency (how often each expert is selected)
        expert_freq = expert_counts.float() / max(1, self.samples_processed)
        
        # Compute load balancing metrics
        expert_entropy = -(expert_freq * torch.log(expert_freq + 1e-10)).sum()
        max_entropy = np.log(num_experts)
        load_balance = expert_entropy / max_entropy  # 1.0 = perfectly balanced
        
        # Compute utilization - % of experts that get used at all
        experts_used = (expert_counts > 0).float().mean() * 100.0
        
        # Compute importance - average gate value when expert is selected
        expert_importance = expert_gate_values / (expert_counts.float() + 1e-10)
        
        # Top-5 most used and least used experts
        top_experts = torch.argsort(expert_freq, descending=True)[:5].tolist()
        bottom_experts = torch.argsort(expert_freq)[:5].tolist()
        
        # Compute coefficient of variation (lower = more balanced)
        cv = torch.std(expert_freq) / (torch.mean(expert_freq) + 1e-10)
        
        return {
            "expert_freq": expert_freq.cpu().numpy(),
            "load_balance": load_balance.item(),
            "experts_used_percent": experts_used.item(),
            "top_experts": top_experts,
            "bottom_experts": bottom_experts,
            "expert_importance": expert_importance.cpu().numpy(),
            "expert_cv": cv.item(),
        }
    
    def log_to_wandb(self, epoch, prefix=""):
        """Log expert utilization metrics to wandb."""
        if self.batches_processed == 0:
            return
            
        stats = self.get_stats()
        
        # Log scalar metrics
        scalar_metrics = {}
        for layer_name, layer_stats in stats.items():
            scalar_metrics[f"{prefix}{layer_name}/load_balance"] = layer_stats["load_balance"]
            scalar_metrics[f"{prefix}{layer_name}/experts_used_percent"] = layer_stats["experts_used_percent"]
            scalar_metrics[f"{prefix}{layer_name}/expert_cv"] = layer_stats["expert_cv"]
        
        wandb.log(scalar_metrics, step=epoch)
        
        # Create and log expert frequency charts every 5 epochs
        if epoch % 5 == 0:
            self._log_charts_to_wandb(stats, epoch, prefix)
    
    def _log_charts_to_wandb(self, stats, epoch, prefix=""):
        """Log visualization charts to wandb."""
        for layer_name, layer_stats in stats.items():
            # Expert frequency chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=np.arange(len(layer_stats["expert_freq"])), 
                       y=layer_stats["expert_freq"], ax=ax1)
            ax1.set_title(f"{layer_name}: Expert Selection Frequency (Epoch {epoch})")
            ax1.set_xlabel("Expert ID")
            ax1.set_ylabel("Selection Frequency")
            wandb.log({f"{prefix}{layer_name}/expert_frequency": wandb.Image(fig1)}, step=epoch)
            plt.close(fig1)
            
            # Expert importance chart
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=np.arange(len(layer_stats["expert_importance"])), 
                       y=layer_stats["expert_importance"], ax=ax2)
            ax2.set_title(f"{layer_name}: Expert Importance (Epoch {epoch})")
            ax2.set_xlabel("Expert ID")
            ax2.set_ylabel("Average Gate Value")
            wandb.log({f"{prefix}{layer_name}/expert_importance": wandb.Image(fig2)}, step=epoch)
            plt.close(fig2)
    
    def print_summary(self, layer_idx=None):
        """Print a summary of expert utilization."""
        stats = self.get_stats(layer_idx)
        
        if layer_idx is not None:
            print(f"\nLayer {layer_idx+1} Expert Utilization:")
            layer_stats = stats
            print(f"Load Balance: {layer_stats['load_balance']:.4f} (1.0 = perfectly balanced)")
            print(f"Experts Used: {layer_stats['experts_used_percent']:.1f}%")
            print(f"Top 5 Experts: {layer_stats['top_experts']}")
            print(f"Bottom 5 Experts: {layer_stats['bottom_experts']}")
            print(f"Coefficient of Variation: {layer_stats['expert_cv']:.4f} (lower = more balanced)")
            return
            
        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name} Expert Utilization:")
            print(f"Load Balance: {layer_stats['load_balance']:.4f} (1.0 = perfectly balanced)")
            print(f"Experts Used: {layer_stats['experts_used_percent']:.1f}%")
            print(f"Top 5 Experts: {layer_stats['top_experts']}")
            print(f"Bottom 5 Experts: {layer_stats['bottom_experts']}")
            print(f"Coefficient of Variation: {layer_stats['expert_cv']:.4f} (lower = more balanced)") 