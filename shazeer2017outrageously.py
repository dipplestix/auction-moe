import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Expert(nn.Module):
    """Expert module for Mixture of Experts architecture.
    
    Implements a feed-forward network with gating mechanism as described in
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
    (Shazeer et al., 2017).
    """
    def __init__(self, input_dim, intermediate_dim, output_dim, activation=nn.SiLU):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = activation()

        self.gate_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, output_dim, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)

    def forward(self, x):
        gate = self.activation(self.gate_proj(x))
        proj = self.up_proj(x)
        fuse = proj * gate
        out = self.down_proj(fuse)
        return out
    
class SoftmaxGate(nn.Module):
    def __init__(self, input_dim, num_experts, activation=nn.SiLU):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.activation = activation()

        self.wg = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        gate = self.activation(self.wg(x))
        gate = torch.softmax(gate, dim=-1)
        return gate
    
class NoisyTopKGate(nn.Module):
    """Noisy Top-K Gating as described in the MoE paper.
    
    Adds tunable noise to the routing decisions and selects top-k experts.
    """
    def __init__(self, input_dim, num_experts, k, device=None):
        super().__init__()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.device = device

        self.wg = nn.Linear(input_dim, num_experts, bias=False)
        self.wn = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.wg.weight, std=0.02)
        nn.init.normal_(self.wn.weight, std=0.02)
        
    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, x):
        logits = self.wg(x)
        noise = torch.randn_like(logits) * F.softplus(self.wn(x))
        logits = logits + noise
        
        top_k_values, top_k_indices = torch.topk(logits, self.k, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_values)
        
        gates = torch.softmax(mask, dim=-1)
        experts = top_k_indices
        
        return gates, experts
    
class GatedMoE(nn.Module):
    """Gated Mixture of Experts layer as described in Shazeer et al. (2017).
    
    Combines multiple expert networks with a gating mechanism that selects 
    the top-k experts for each input.
    """
    def __init__(self, input_dim, output_dim, num_experts, k, device=None):
        super().__init__()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.device = device

        self.gate = NoisyTopKGate(input_dim, num_experts, k, device=device)
        self.experts = nn.ModuleList([
            Expert(input_dim, input_dim * 4, output_dim) for _ in range(num_experts)
        ])
        
    def to(self, device):
        super().to(device)
        self.device = device
        self.gate.to(device)
        for expert in self.experts:
            expert.to(device)
        return self

    def forward(self, x, return_routing_info=False):
        """Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            return_routing_info: Whether to return routing information for tracking
            
        Returns:
            If return_routing_info is False:
                Tensor of shape [batch_size, output_dim] - the output of the MoE layer
            If return_routing_info is True:
                Tuple of (output, routing_info), where routing_info is a tuple of
                (gates, expert_indices, batch_size) for expert utilization tracking
        """
        batch_size = x.shape[0]
        
        # Get routing probabilities and expert indices
        # gates: [batch_size, num_experts] - sparse tensor with values only at top-k positions
        # expert_indices: [batch_size, k] - indices of top-k experts per sample
        gates, expert_indices = self.gate(x)
        
        # Initialize output tensor [batch_size, output_dim]
        final_output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which batch samples use this expert (any position in k)
            mask = (expert_indices == expert_idx).any(dim=1)
            
            # Skip if no samples use this expert
            if not mask.any():
                continue
            
            # Get the inputs for this expert as a batch
            expert_inputs = x[mask]
            
            # Process all samples for this expert at once
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # Get the gate values for this expert
            # Only for samples that use this expert
            gate_values = gates[mask, expert_idx].unsqueeze(1)
            
            # Apply the gate values to the expert outputs
            gated_output = gate_values * expert_output
            
            # Add the gated outputs to the final output
            final_output[mask] += gated_output
        
        if return_routing_info:
            routing_info = (gates, expert_indices, batch_size)
            return final_output, routing_info
        
        return final_output
