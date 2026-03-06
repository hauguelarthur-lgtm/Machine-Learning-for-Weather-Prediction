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
    # Strictly unpack the PyTorch Subset object
    base_dataset = worker_info.dataset.dataset
    base_dataset.mmaps = [np.load(f, mmap_mode='r') for f in base_dataset.tensor_files]

def extract_latitudes(reference_file="/workspace/data/processed/latitudes.npy"):
    lats = np.load(reference_file)
    return lats

def execute_training_pipeline(OPTIMAL_LR, OPTIMAL_WD, BATCH_SIZE, EPOCHS=300):

    torch.backends.cudnn.benchmark = True
    print(f"Initializing Convergence Trajectory: LR={OPTIMAL_LR}, WD={OPTIMAL_WD}, BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}")
    
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
                              num_workers=8, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=8, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn, persistent_workers=True)
    
    lats = extract_latitudes()
    criterion = LatitudeWeightedMSELoss(latitudes=lats).to(device)
    evaluator = WeatherBench2Metrics(latitudes=lats)
    evaluator.to(device)
    
    best_val_rmse = float('inf')

    gamma = 0.9
    
    phase_length = EPOCHS // 3
    
    model = ResUNet(in_channels=102, out_channels=102).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMAL_LR, weight_decay=OPTIMAL_WD)
    
    # 2. Strictly bind the momentum restart frequency to the phase duration
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=phase_length, T_mult=1, eta_min=1e-6
    )

    start_epoch = 1
    latest_checkpoint_path = "./models/checkpoints/resunet_latest.pth"
    
    if os.path.exists(latest_checkpoint_path):
        print(f"Detected interrupted execution. Restoring continuous state from: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        # Overwrite initialized tensors with the serialized mathematical state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Shift the temporal integration bound
        start_epoch = checkpoint['epoch'] + 1
        
        # Recover the minimum physical error baseline
        optimal_checkpoint_path = "./models/checkpoints/resunet_optimal.pth"
        if os.path.exists(optimal_checkpoint_path):
            opt_checkpoint = torch.load(optimal_checkpoint_path, map_location=device)
            best_val_rmse = opt_checkpoint['val_surface_rmse']
            
        print(f"Momentum and Parametric topologies restored. Resuming at Epoch {start_epoch}.")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # -------------------------------------------------------------
        # Progressive Temporal Unrolling Curriculum
        # -------------------------------------------------------------
        # Phase 1: 1-Step Advection (Epochs 1 - 50)
        if epoch <= phase_length:
            active_rollout_steps = 1
        elif epoch <= (phase_length * 2):
            active_rollout_steps = 2
        else:
            active_rollout_steps = rollout_steps
            
        # Recompute the integral of the discount scalar for the active sequence length
        active_gamma_sum = sum([gamma ** k for k in range(active_rollout_steps)])
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = 0.0
                current_state = x
                current_state.requires_grad_()
                
                # Execute strictly up to the active_rollout_steps boundary
                for k in range(active_rollout_steps):
                    target_state = y[:, k]
                    
                    next_state = checkpoint.checkpoint(
                        model, current_state, use_reentrant=False, preserve_rng_state=False
                    )
                    
                    base_loss = criterion(next_state.float(), target_state.float())
                    surface_pred = next_state[:, 3:7, :, :]
                    surface_targ = target_state[:, 3:7, :, :]
                    surface_loss = criterion(surface_pred.float(), surface_targ.float())
                    
                    step_loss = base_loss + (15.0 * surface_loss)
                    loss += step_loss * (gamma ** k)
                    
                    # Strict Autoregression (Zero Gradient Bias)
                    if k < active_rollout_steps - 1:
                        current_state = next_state 
                
                # Dynamic normalization by the exact temporal integral to prevent gradient explosion
                loss = loss / (accumulation_steps * active_gamma_sum)
                
            loss.backward()
            
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Continuous fractional step evaluation for the Cosine Manifold
                current_fractional_epoch = epoch - 1 + (batch_idx / len(train_loader))
                scheduler.step(current_fractional_epoch)
                
                optimizer.zero_grad(set_to_none=True)
            
            train_loss_accum += (loss.item() * accumulation_steps * active_gamma_sum)
            
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
                
                current_state = x_val
                rollout_mse = torch.zeros(102, device=device)
                
                # Validation strictly evaluates the full K=3 global trajectory 
                for k in range(rollout_steps):
                    target_state = y_val[:, k]
                    
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        current_state = model(current_state)
                        
                    rollout_mse += evaluator.compute_mse(current_state.float(), target_state.float())
                
                avg_rollout_mse = rollout_mse / rollout_steps
                global_mse_accum += avg_rollout_mse * batch_size
                total_val_samples += batch_size 
                
        true_global_mse = global_mse_accum / total_val_samples
        true_global_rmse = torch.sqrt(true_global_mse)
        
        surface_rmse = true_global_rmse[3:7]
        avg_val_rmse = torch.mean(surface_rmse).item() 

        print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4e} | Val Surface RMSE: {avg_val_rmse:.4f}")
        
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_surface_rmse': avg_val_rmse
        }
        
        # -------------------------------------------------------------
        # I/O State Machine: Atomic Serialization
        # -------------------------------------------------------------
        tmp_latest = "./models/checkpoints/resunet_latest_tmp.pth"
        out_latest = "./models/checkpoints/resunet_latest.pth"
        torch.save(checkpoint_dict, tmp_latest)
        os.replace(tmp_latest, out_latest)

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            tmp_opt = "./models/checkpoints/resunet_optimal_tmp.pth"
            out_opt = "./models/checkpoints/resunet_optimal.pth"
            torch.save(checkpoint_dict, tmp_opt)
            os.replace(tmp_opt, out_opt)
            print(f"  -> Targeted boundary layer metrics improved. Checkpoint serialized atomically to SSD.")

if __name__ == "__main__":
    # Example execution with fixed parameters, update to pass Optuna parameters
    execute_training_pipeline(OPTIMAL_LR=0.000441, OPTIMAL_WD=0.00034, BATCH_SIZE=16, EPOCHS=300)