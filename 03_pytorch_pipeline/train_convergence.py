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

def extract_latitudes(reference_file="./data/processed/engineered_1996.nc"):
    """Extracts the exact physical coordinate array for the loss and metrics."""
    ds = xr.open_dataset(reference_file)
    lats = ds['latitude'].values
    ds.close()
    return lats

def execute_training_pipeline(OPTIMAL_LR, OPTIMAL_WD, BATCH_SIZE, EPOCHS=100):
    # 1. Optimal Hyperparameter Injection
    # Replace these with the exact output from train_hyperparameters.py
    
    os.makedirs("./models/checkpoints/", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware: {device}")

    # 2. Memory-Mapped Dataset Allocation
    full_dataset = MeteorologicalDataset(tensor_dir="/workspace/data/processed/tensors/", horizon=1)
    
    # Chronological Matrix Splitting (1996-2019 Train | 2020-2021 Validation)
    train_end_idx = 35064 
    val_end_idx = train_end_idx + (2 * 1460)
    
    train_subset = Subset(full_dataset, range(0, train_end_idx))
    val_subset = Subset(full_dataset, range(train_end_idx, val_end_idx))
    
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

    # 4. Global Convergence Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_accum = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Asynchronous VRAM transfer
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                predictions = model(x)
                loss = criterion(predictions, y)
                
            scaler.scale(loss).backward()
            
            # Unscale and constrain L2 gradient norm to 1.0 to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_accum += loss.item()
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # 5. Validation and Metric Tracking
        model.eval()
        val_rmse_accum = 0.0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    val_preds = model(x_val)
                    
                    # Calculate true WB2 RMSE across all 102 channels for the batch
                    # Returns tensor of shape (102,)
                    batch_rmse = evaluator.compute_rmse(val_preds, y_val) 
                
                # We track the global mean RMSE across all channels for the scheduler
                val_rmse_accum += torch.mean(batch_rmse).item()
                
        avg_val_rmse = val_rmse_accum / len(val_loader)
        
        # Adjust learning rate based on strictly independent validation data
        scheduler.step(avg_val_rmse)
        
        print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4e} | Val Mean RMSE: {avg_val_rmse:.4f}")
        
        # 6. Strict Checkpointing Logic
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            save_path = f"./models/checkpoints/resunet_best_epoch_{epoch}.pth"
            
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