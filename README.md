# Auction-MoE: Mixture of Experts with Auction-based Routing

This project implements a novel approach to Mixture of Experts (MoE) training by combining auction theory and computational budgeting to create more efficient and balanced expert routing mechanisms.

## Project Overview

Traditional MoE models use top-k routing with learned gates to dispatch inputs to a subset of experts. While effective, this approach can lead to inefficient expert utilization, load imbalance, and instability. This project aims to reframe expert routing as an auction-based mechanism where:

1. **Experts bid** for the right to process certain inputs
2. **Inputs have budgets** for purchasing expert compute
3. **Market-clearing mechanisms** efficiently match experts to inputs

## Current Implementation

- ✅ Core MoE architecture based on "Outrageously Large Neural Networks" (Shazeer et al., 2017)
- ✅ Efficient batched implementation with proper device handling
- ✅ Noisy top-k gating mechanism with tunable noise
- ✅ Comprehensive expert utilization tracking and visualization
- ✅ Training framework with gradient clipping and learning rate scheduling
- ✅ WandB integration for experiment tracking
- ✅ Separation of concerns (model vs. tracking)

## TODO: Auction Mechanism Integration

The following components still need to be implemented:

- [ ] **Expert Bidding System**
  - [ ] Define bid generation networks for experts
  - [ ] Implement bid adjustment mechanisms based on expert capacity
  - [ ] Create differential bid weighting based on expertise areas

- [ ] **Input Budget Allocation**
  - [ ] Design budget allocation strategies for inputs
  - [ ] Implement dynamic budget adjustment based on task difficulty
  - [ ] Create budget constraint mechanisms

- [ ] **Auction Market Clearing**
  - [ ] Implement the auction clearing algorithm
  - [ ] Add price discovery mechanisms
  - [ ] Create priority-based allocation for critical inputs

- [ ] **Load Balancing Constraints**
  - [ ] Implement capacity constraints for experts
  - [ ] Add expert utilization penalties to the auction system
  - [ ] Create adaptive capacity adjustment based on utilization history

## Technical Details

### Current Architecture

- **Expert Network**: FFN with gating mechanism
- **Routing**: Noisy top-k gating with learned routing probabilities
- **Tracking**: External utilization tracker for monitoring expert usage

### Proposed Auction-Based Routing

- **Bidding**: Experts generate bids for each input based on relevance
- **Budgeting**: Inputs receive computational budgets to spend on experts
- **Auction**: Market-clearing algorithm allocates inputs to experts
- **Pricing**: Dynamic price discovery ensures efficient allocation

## Getting Started

1. Install the required packages:
```bash
pip install torch torchvision wandb matplotlib seaborn scikit-learn
```

2. Login to Weights & Biases:
```bash
wandb login
```

3. Run the training script:
```bash
python train_moe.py
```

## Current Issues

Based on training logs, the model shows instability with occasional large spikes in loss values. The expert utilization data shows:

- Layer 1 maintains good balance (load_balance ~1.22)
- Layer 2 shows increasing imbalance over time (expert_cv increasing from 0.7 to 0.93)
- All experts are being used (experts_used_percent = 100%)
- Expert preference patterns are forming with clear favorites (Top 5 experts: [5, 2, 7, 8, 3])

These issues highlight the need for the proposed auction mechanism to improve stability and balance.

## Next Steps

1. Implement the basic bidding mechanism for experts
2. Create the input budget allocation system
3. Develop the market clearing algorithm
4. Integrate the auction system with the existing MoE framework
5. Compare performance against the baseline top-k routing

## References

- Shazeer et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- Fedus et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models"
- Various auction mechanisms in computational resource allocation