import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import xarray as xr

from dataset import MeteorologicalDataset
from architecture import ResUNet
from loss_functions import LatitudeWeightedMSELoss
from metrics import WeatherBench2Metrics

def extract_latitudes(reference_file="/workspace/data/processed/latitudes.npy"):
    """Extracts the definitive latitude array for the spatial loss function."""
    lats = np.load(reference_file)
    return lats

def execute_training_pipeline(OPTIMAL_LR, OPTIMAL_WD, BATCH_SIZE, EPOCHS=100):
    # 1. Optimal Hyperparameter Injection
    # Replace these with the exact output from train_hyperparameters.py
    print(OPTIMAL_LR, OPTIMAL_WD, BATCH_SIZE, EPOCHS)
    
    os.makedirs("./models/checkpoints/", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware: {device}")

    # 2. Memory-Mapped Dataset Allocation
    full_dataset = MeteorologicalDataset(tensor_dir="/workspace/data/processed/tensors/", horizon=1)
    
    # Define the accumulation scalar explicitly here
    accumulation_steps = 4
    
    # Chronological Matrix Splitting
    train_end_idx = 35064 
    val_end_idx = train_end_idx + (2 * 1460)
    
    # --- Exact Mathematical Boundary Truncation ---
    # 1. Determine the absolute maximum bounds permitted without chronological leakage
    raw_train_len = train_end_idx - full_dataset.horizon
    
    # 2. Calculate the total number of physical batches this limit supports
    total_physical_batches = raw_train_len // BATCH_SIZE
    
    # 3. Force the physical batch count to a perfect multiple of the accumulation steps
    valid_effective_batches = total_physical_batches // accumulation_steps
    perfect_physical_batches = valid_effective_batches * accumulation_steps
    
    # 4. Project the valid batch count back into an absolute tensor index
    perfect_train_len = perfect_physical_batches * BATCH_SIZE
    # ----------------------------------------------
    
    # Allocate the Subsets using the strictly defragmented boundaries
    train_subset = Subset(full_dataset, range(0, perfect_train_len))
    val_subset = Subset(full_dataset, range(train_end_idx, val_end_idx - full_dataset.horizon))
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, drop_last=False)
    
    # 3. Model & Mathematics Initialization
    model = ResUNet(in_channels=102, out_channels=102).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMAL_LR, weight_decay=OPTIMAL_WD)
    
    # Dynamic LR reduction when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    lats = extract_latitudes()
    criterion = LatitudeWeightedMSELoss(latitudes=lats).to(device)
    evaluator = WeatherBench2Metrics(latitudes=lats)
    evaluator.to(device)
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_rmse = float('inf')

    accumulation_steps = 4  # Enforce the same dynamic scalar
    
    # 4. Global Convergence Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_accum = 0.0
        
        # Flush the gradients before the first batch of the epoch
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Asynchronous VRAM transfer
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                predictions = model(x)
            
            # Scale the loss for backpropagation
            loss = criterion(predictions.float(), y.float()) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            # Boundary check: Update weights when the sub-batch queue is full
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Reset gradients for the next cycle
                optimizer.zero_grad(set_to_none=True)
            
            # Multiply the scaled loss back by accumulation_steps to record the true physical scalar
            train_loss_accum += (loss.item() * accumulation_steps)
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # 5. Validation and Metric Tracking
        model.eval()
        
        # Initialize an accumulator tensor of shape (102,) on the GPU
        global_mse_accum = torch.zeros(102, device=device)
        total_val_samples = 0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                batch_size = x_val.size(0)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    val_preds = model(x_val)
                    
                # Extract batch MSE per channel: shape (102,)
                batch_mse = evaluator.compute_mse(val_preds.float(), y_val.float()) 
                
                # Multiply by batch_size to eliminate batch averaging bias, then sum
                global_mse_accum += batch_mse * batch_size
                total_val_samples += batch_size
                
        # Calculate true global MSE, then apply the square root strictly once
        true_global_mse = global_mse_accum / total_val_samples
        true_global_rmse = torch.sqrt(true_global_mse)
        
        # We track the mean RMSE across all channels for the scalar scheduler
        avg_val_rmse = torch.mean(true_global_rmse).item()
        # Adjust learning rate based on strictly independent validation data
        scheduler.step(avg_val_rmse)
        
        print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4e} | Val Mean RMSE: {avg_val_rmse:.4f}")
        
        # 6. Strict Checkpointing Logic
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            # Define a static path to strictly overwrite the prior state
            save_path = "./models/checkpoints/resunet_optimal.pth"
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_rmse': best_val_rmse
            }
            torch.save(checkpoint, save_path)
            print(f"  -> Physical state improved. Checkpoint serialized to SSD.")

if __name__ == "__main__":
    execute_training_pipeline()