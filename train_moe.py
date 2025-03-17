import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from shazeer2017outrageously import GatedMoE
from expert_tracker import ExpertUtilizationTracker

class MoENetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, k, dropout=0.1):
        super().__init__()
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # First MoE layer
        self.moe1 = GatedMoE(
            input_dim=input_dim,
            output_dim=hidden_dim,
            num_experts=num_experts,
            k=k
        )
        
        # Hidden normalization
        self.hidden_norm = nn.BatchNorm1d(hidden_dim)
        
        # ReLU activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Second MoE layer
        self.moe2 = GatedMoE(
            input_dim=hidden_dim,
            output_dim=hidden_dim // 2,
            num_experts=num_experts,
            k=k
        )
        
        # Output layer (simple linear)
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x, return_routing_info=False):
        # Normalize input
        x = self.input_norm(x)
        
        # First MoE block
        if return_routing_info:
            x, routing_info1 = self.moe1(x, return_routing_info=True)
        else:
            x = self.moe1(x)
            
        x = self.hidden_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second MoE block
        if return_routing_info:
            x, routing_info2 = self.moe2(x, return_routing_info=True)
        else:
            x = self.moe2(x)
            
        x = self.relu(x)
        
        # Output projection
        x = self.output_layer(x)
        
        if return_routing_info:
            return x, (routing_info1, routing_info2)
        return x

def train(model, dataloader, criterion, optimizer, device, expert_tracker=None, clip_value=1.0):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Get model output and routing info if we're tracking experts
        if expert_tracker is not None:
            output, routing_infos = model(data, return_routing_info=True)
            
            # Update expert tracker with routing information
            expert_tracker.update(0, routing_infos[0])  # Layer 1
            expert_tracker.update(1, routing_infos[1])  # Layer 2
        else:
            output = model(data)
            
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f'Train Batch: {batch_idx + 1}/{len(dataloader)} | Loss: {running_loss/(batch_idx+1):.5f}')
    
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return False
            
        if val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = val_loss
            self.counter = 0
            return False

def main():
    # Config
    config = {
        'batch_size': 64,          # Smaller batch for better generalization
        'epochs': 100,
        'lr': 0.005,               # Higher starting LR for OneCycleLR
        'weight_decay': 1e-4,      # Stronger weight decay
        'num_experts': 10,
        'k': 3,                    # Increased k to 3 for better redundancy
        'hidden_dim': 128,         # Reduced complexity
        'seed': 42,
        'patience': 15,
        'dropout': 0.2,            # Added dropout
        'clip_value': 0.5          # More aggressive gradient clipping
    }
    
    # Initialize wandb
    wandb.init(project="moe-experiment", config=config)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load California Housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Standardize features and target
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=config['seed']
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    input_dim = X_train.shape[1]  # Features in the California Housing dataset
    output_dim = 1  # Regression task, single output
    
    model = MoENetwork(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        output_dim=output_dim,
        num_experts=config['num_experts'],
        k=config['k'],
        dropout=config['dropout']
    ).to(device)
    
    # Expert utilization tracker
    expert_tracker = ExpertUtilizationTracker(
        num_layers=2,  # Two MoE layers
        num_experts_per_layer=[config['num_experts'], config['num_experts']],
        device=device
    )
    
    # Watch model in wandb
    wandb.watch(model)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler - OneCycleLR instead of ReduceLROnPlateau
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        steps_per_epoch=steps_per_epoch,
        epochs=config['epochs'],
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1000,  # min_lr = initial_lr/1000
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Reset expert statistics for this epoch
        expert_tracker.reset()
        
        # Train
        model.train()
        train_loss = train(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device,
            expert_tracker=expert_tracker,
            clip_value=config['clip_value']
        )
        
        # Evaluate
        test_loss = evaluate(model, test_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate RMSE (in original scale)
        rmse = np.sqrt(test_loss) * scaler_y.scale_[0]
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'rmse': rmse,
            'learning_rate': current_lr
        }, step=epoch + 1)
        
        # Log expert utilization
        expert_tracker.log_to_wandb(epoch + 1)
        
        # Print basic metrics
        print(f"Train Loss: {train_loss:.5f}")
        print(f"Test Loss: {test_loss:.5f}")
        print(f"RMSE: ${rmse:.5f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Print expert utilization summary
        expert_tracker.print_summary()
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "moe_housing_model.pt")
            wandb.save("moe_housing_model.pt")
            print("Model saved!")
        
        # Check early stopping
        if early_stopping(test_loss):
            print("Early stopping triggered!")
            break
    
    # Final evaluation
    model.load_state_dict(torch.load("moe_housing_model.pt"))
    final_test_loss = evaluate(model, test_loader, criterion, device)
    final_rmse = np.sqrt(final_test_loss) * scaler_y.scale_[0]
    
    print(f"\nFinal Test Loss: {final_test_loss:.5f}")
    print(f"Final RMSE: ${final_rmse:.5f}")
    
    # Finish wandb session
    wandb.finish()

if __name__ == "__main__":
    main() 