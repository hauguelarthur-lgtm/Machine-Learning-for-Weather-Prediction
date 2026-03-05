import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import optuna
import numpy as np
import os
import xarray as xr

# Import your custom pipeline modules
from dataset import MeteorologicalDataset_hyper
from architecture import ResUNet
from loss_functions import LatitudeWeightedMSELoss

import numpy as np

def extract_latitudes(reference_file="/workspace/data/processed/latitudes.npy"):
    """Extracts the definitive latitude array for the spatial loss function."""
    lats = np.load(reference_file)
    return lats
def get_objective(train_subset, val_subset):
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
    
        # pin_memory=True locks the RAM pages, speeding up the CPU-to-GPU transfer.
        # num_workers must be low (e.g., 2-4) to prevent I/O thrashing on the SSD memmaps.
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    
        # 5. Fast Evaluation Loop (3 Epochs per Trial)
        epochs = 15 
    
        # Define the scalar to multiply your physical batch size
        # If Optuna selects batch_size=32, effective batch size becomes 128
        accumulation_steps = 4 
    
        for epoch in range(epochs):
            model.train()
            
            # Initialize gradient matrix to zero strictly at the start of the epoch
            optimizer.zero_grad(set_to_none=True)
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
                # Forward pass in float16
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    predictions = model(x)
                
                # Mathematically scale the loss to compute the true mean over the effective batch
                loss = criterion(predictions.float(), y.float()) / accumulation_steps
                
                # Accumulate the native gradients
                loss.backward()
            
                # Execute the hardware update strictly at the accumulation boundary or epoch end
                if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                    optimizer.step()
                    
                    # Flush the gradient matrix for the next accumulation cycle
                    optimizer.zero_grad(set_to_none=True)
            
            # Validation Phase
            model.eval()
            total_val_loss = 0.0
            total_samples = 0
            
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                    
                    # Extract the exact physical batch size (vital for the final remainder batch)
                    current_batch_size = x_val.size(0)
                
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        val_preds = model(x_val)
                        
                    # criterion() returns the mean scalar across the batch domain
                    batch_loss = criterion(val_preds.float(), y_val.float())
                
                    # Reconstruct the absolute sum of errors for this batch and accumulate
                    total_val_loss += batch_loss.item() * current_batch_size
                    total_samples += current_batch_size
                
            # Divide the absolute error sum by the exact global sample count
            true_val_loss = total_val_loss / total_samples
        
            # Report the mathematically rigorous scalar to the TPE algorithm
            trial.report(true_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        return true_val_loss
    return objective

if __name__ == "__main__":
    print("Initializing Bayesian Hyperparameter Optimization...")

    full_dataset = MeteorologicalDataset_hyper(tensor_dir="/workspace/data/processed/tensors/", horizon=1)
    
    # Define exact temporal boundaries to prevent chronological leakage (See Fix #3)
    train_end_idx = 35064
    val_end_idx = train_end_idx + (2 * 1460)
    
    train_subset = Subset(full_dataset, range(0, train_end_idx - full_dataset.horizon))
    val_subset = Subset(full_dataset, range(train_end_idx, val_end_idx - full_dataset.horizon))
    
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Pass the closure to the study
    study.optimize(get_objective(train_subset, val_subset), n_trials=20)
    
    print("\nOptimization Complete.")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")