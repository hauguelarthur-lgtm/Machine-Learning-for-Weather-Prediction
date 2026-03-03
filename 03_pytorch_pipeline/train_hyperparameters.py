import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import optuna
import numpy as np
import os
import xarray as xr

# Import your custom pipeline modules
from dataset import MeteorologicalDataset
from architecture import ResUNet
from loss_functions import LatitudeWeightedMSELoss

import numpy as np

def extract_latitudes(reference_file="/workspace/data/processed/latitudes.npy"):
    """Extracts the definitive latitude array for the spatial loss function."""
    lats = np.load(reference_file)
    return lats

def objective(trial):
    """
    The Optuna mathematical evaluation function.
    Defines the search space and executes a localized training loop.
    """
    # 1. Define the Bayesian Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    
    # 2. Hardware Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(in_channels=102, out_channels=102).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Extract latitudes and instantiate the physics-informed loss
    lats = extract_latitudes()
    criterion = LatitudeWeightedMSELoss(latitudes=lats).to(device)
    
    # 3. Memory-Mapped Dataset & Chronological Splitting
    full_dataset = MeteorologicalDataset(tensor_dir="/workspace/data/processed/tensors/", horizon=1)
    
    # Total steps per year is exactly 1460 (365 days * 4 steps/day) + leap years.
    # We approximate the indices to enforce a strict temporal boundary.
    # 1996-2019 (24 years) ≈ 35064 steps. 
    train_end_idx = 35064 
    val_end_idx = train_end_idx + (2 * 1460) # 2020-2021 (2 years)
    
    train_subset = Subset(full_dataset, range(0, train_end_idx))
    val_subset = Subset(full_dataset, range(train_end_idx, val_end_idx))
    
    # pin_memory=True locks the RAM pages, speeding up the CPU-to-GPU transfer.
    # num_workers must be low (e.g., 2-4) to prevent I/O thrashing on the SSD memmaps.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # 4. Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # 5. Fast Evaluation Loop (3 Epochs per Trial)
    epochs = 3 
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass in float16
            with torch.amp.autocast('cuda'):
                predictions = model(x)
                loss = criterion(predictions, y)
                
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Unscale gradients and apply L2 clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    val_preds = model(x_val)
                    batch_loss = criterion(val_preds, y_val)
                
                val_loss += batch_loss.item()
                
        val_loss /= len(val_loader)
        
        # Report intermediate results to Optuna for algorithmic pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

if __name__ == "__main__":
    print("Initializing Bayesian Hyperparameter Optimization...")
    
    # The MedianPruner terminates unpromising trials early to save GPU time
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    
    # Execute 20 distinct trials
    study.optimize(objective, n_trials=20)
    
    print("\nOptimization Complete.")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")