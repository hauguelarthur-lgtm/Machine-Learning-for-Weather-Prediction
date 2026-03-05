import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.utils.checkpoint as checkpoint
import numpy as np
import os

from dataset import MeteorologicalDataset
from architecture import ResUNet
from loss_functions import LatitudeWeightedMSELoss
from metrics import WeatherBench2Metrics

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.mmaps = [np.load(f, mmap_mode='r') for f in dataset.tensor_files]

def extract_latitudes(reference_file="/workspace/data/processed/latitudes.npy"):
    lats = np.load(reference_file)
    return lats

def execute_training_pipeline(OPTIMAL_LR, OPTIMAL_WD, BATCH_SIZE, EPOCHS=100):
    print(OPTIMAL_LR, OPTIMAL_WD, BATCH_SIZE, EPOCHS)
    
    os.makedirs("./models/checkpoints/", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Hardware: {device}")

    rollout_steps = 3
    full_dataset = MeteorologicalDataset(tensor_dir="/workspace/data/processed/tensors/", rollout_steps=rollout_steps)
    
    accumulation_steps = 4
    
    train_end_idx = 35064 
    val_end_idx = train_end_idx + (2 * 1460)
    
    raw_train_len = train_end_idx - full_dataset.rollout_steps
    total_physical_batches = raw_train_len // BATCH_SIZE
    valid_effective_batches = total_physical_batches // accumulation_steps
    perfect_physical_batches = valid_effective_batches * accumulation_steps
    perfect_train_len = perfect_physical_batches * BATCH_SIZE
    
    train_subset = Subset(full_dataset, range(0, perfect_train_len))
    val_subset = Subset(full_dataset, range(train_end_idx, val_end_idx - full_dataset.rollout_steps))
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    
    surface_indices = torch.tensor([3,4,5,6], device=device)

    model = ResUNet(in_channels=102, out_channels=102).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMAL_LR, weight_decay=OPTIMAL_WD)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    lats = extract_latitudes()
    criterion = LatitudeWeightedMSELoss(latitudes=lats).to(device)
    evaluator = WeatherBench2Metrics(latitudes=lats)
    evaluator.to(device)
    
    best_val_rmse = float('inf')

    # Strictly isolated index array for the targeted surface variables. 
    # Assumes MSLP, T2M, U10, V10 correlate to indices [0, 1, 2, 3].
    # You must verify this aligns with your channel_ordering.json.
    surface_indices = torch.tensor([3,4,5,6], device=device)
    gamma = 0.9
    gamma_sum = sum([gamma ** k for k in range(rollout_steps)])
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Calculate curriculum learning decay boundary (force 0.0 at halfway point)
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch / (EPOCHS * 0.5)))
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = 0.0
                current_state = x
                
                for k in range(rollout_steps):
                    target_state = y[:, k]
                    
                    next_state = checkpoint.checkpoint(
                        model, current_state, use_reentrant=False, preserve_rng_state=False
                    )
                    
                    # Compute Composite Differentiable Loss
                    base_loss = criterion(next_state.float(), target_state.float())
                    surface_pred = next_state[:, surface_indices, :, :]
                    surface_targ = target_state[:, surface_indices, :, :]
                    surface_loss = criterion(surface_pred.float(), surface_targ.float())
                    
                    step_loss = base_loss + (15.0 * surface_loss)
                    loss += step_loss * (gamma ** k)
                    
                    # Scheduled Sampling: Probabilistic state pushforward
                    if k < rollout_steps - 1:
                        if torch.rand(1).item() < teacher_forcing_ratio:
                            current_state = target_state # Anchor to true physical manifold
                        else:
                            current_state = next_state # Enforce exposure bias correction
                
                loss = loss / (accumulation_steps * gamma_sum)
                
            loss.backward()
            
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss_accum += (loss.item() * accumulation_steps)
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # -------------------------------------------------------------
        # Validation Phase: Strict Surface Isolation Tracking
        # -------------------------------------------------------------
        model.eval()
        global_mse_accum = torch.zeros(102, device=device)
        total_val_samples = 0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                batch_size = x_val.size(0)
                
                # Evaluate strictly on the t+1 baseline to monitor uncompounded error
                y_val_step_1 = y_val[:, 0]
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    val_preds = model(x_val)
                    
                batch_mse = evaluator.compute_mse(val_preds.float(), y_val_step_1.float()) 
                global_mse_accum += batch_mse * batch_size
                total_val_samples += batch_size
                
        true_global_mse = global_mse_accum / total_val_samples
        true_global_rmse = torch.sqrt(true_global_mse)
        
        # Isolate the validation scalar strictly to the targeted boundary layer physics
        surface_rmse = true_global_rmse[surface_indices]
        avg_val_rmse = torch.mean(surface_rmse).item() 

        print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4e} | Val Surface RMSE: {avg_val_rmse:.4f}")
        
        scheduler.step()
        
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_surface_rmse': avg_val_rmse
        }
        torch.save(checkpoint_dict, "./models/checkpoints/resunet_latest.pth")

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(checkpoint_dict, "./models/checkpoints/resunet_optimal.pth")
            print(f"  -> Targeted boundary layer metrics improved. Checkpoint serialized to SSD.")

if __name__ == "__main__":
    # Example execution with fixed parameters, update to pass Optuna parameters
    execute_training_pipeline(OPTIMAL_LR=4.63e-4, OPTIMAL_WD=1.14e-4, BATCH_SIZE=16, EPOCHS=100)