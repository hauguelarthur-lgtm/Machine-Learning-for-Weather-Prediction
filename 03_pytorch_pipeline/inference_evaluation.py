import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Subset
from datetime import datetime, timedelta
import json
import os

from dataset import MeteorologicalDataset
from architecture import ResUNet

def extract_latitudes(reference_file="/workspace/data/processed/latitudes.npy"):
    lats = np.load(reference_file)
    return lats

def get_climatology_slice(climatology_ds, target_time):
    """
    Extracts the strict 2D spatial baseline using the composite diurnal coordinate.
    """
    doy = target_time.timetuple().tm_yday
    if doy == 366 and 'dayofyear' in climatology_ds.dims and climatology_ds.dayofyear.size == 365:
        doy = 365 
        
    hour = target_time.hour
    doy_hour_str = f"{doy:03d}_{hour:02d}"
    
    c_slice = climatology_ds.sel(doy_hour=doy_hour_str)
    c_tensor = torch.from_numpy(c_slice.to_array().values).float()
    return c_tensor

def execute_wb2_evaluation():
    # OPTIMIZATION 3: Algorithmic Heuristic Benchmarking
    # Forces the NVIDIA driver to select the absolute fastest convolution algorithms
    # for the rigid (1, 102, 64, 64) spatial manifold.
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing strict WB2 Autoregressive Evaluation on {device}...")

    # Instantiate the topological graph in active memory
    model = ResUNet(in_channels=102, out_channels=102, base_filters=128)
    model = model.to(device)    

    checkpoint = torch.load("./models/checkpoints/resunet_latest.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # OPTIMIZATION 2: Static Graph Compilation (Kernel Fusion)
    # Minimizes CUDA kernel launch overhead and GPU memory bandwidth pressure
    print("Compiling network topology via TorchDynamo...")
    model = torch.compile(model, mode="reduce-overhead")

    lats = extract_latitudes()
    lat_rad = np.deg2rad(np.array(lats, dtype=np.float32))
    weights_normalized = np.cos(lat_rad) / np.mean(np.cos(lat_rad))
    W = torch.tensor(weights_normalized, device=device).view(1, 1, len(lats), 1)
    
    # Load Data Engineering Denormalization Tensors
    with open("./data/processed/global_stats.json", "r") as f:
        stats = json.load(f)
        
    with open("./data/processed/tensors/channel_ordering.json", "r") as f:
        channel_names = json.load(f)
        
    mu_list = []
    sigma_list = []
    
    for var in channel_names:
        if var in stats["mean"]:
            mu_list.append(stats["mean"][var])
            sigma_list.append(stats["std"][var])
        else:
            mu_list.append(0.0)
            sigma_list.append(1.0)
            
    mu = torch.tensor(mu_list, device=device).view(1, 102, 1, 1)
    sigma = torch.tensor(sigma_list, device=device).view(1, 102, 1, 1)

    lead_times = 28 # Evaluates exactly 7 continuous days (28 * 6 hours)
    time_step = timedelta(hours=6)
    
    print("Pre-allocating Climatological Baseline into memory...")
    climatology_ds = xr.open_dataset("./data/processed/evaluation/wb2_climatology_baseline.nc").load()
    
    # Memory isolation and coordinate mapping
    eval_vars = ['z_500', 't_850', 't2m', 'u10', 'v10']
    eval_indices = [channel_names.index(var) for var in eval_vars]
    num_eval_vars = len(eval_vars)
    
    # Accumulators strictly allocated for 5 parameters to prevent RAM overflow
    acc_cov = torch.zeros((lead_times, num_eval_vars), device=device)
    acc_var_P = torch.zeros((lead_times, num_eval_vars), device=device)
    acc_var_T = torch.zeros((lead_times, num_eval_vars), device=device)
    
    full_dataset = MeteorologicalDataset(tensor_dir="./data/processed/tensors/", rollout_steps=lead_times)
    
    # OPTIMIZATION 1: Asynchronous Memory Pinning and DMA Transfer
    # Map the exact 120-step deterministic sequence into a strict subset
    eval_indices_list = list(range(35064, len(full_dataset) - lead_times, 120))
    eval_subset = Subset(full_dataset, eval_indices_list)
    
    eval_loader = DataLoader(
        eval_subset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True # Locks memory pages for non-blocking PCI-e transfer
    )
    
    total_sequences = 0
    print("Executing Autoregressive Pushforward Trajectory...")
    
    with torch.no_grad():
        for batch_idx, (P_input, T_targets) in enumerate(eval_loader):
            # Asynchronous DMA projection to GPU hardware
            current_state = P_input.to(device, non_blocking=True)
            T_targets = T_targets.to(device, non_blocking=True)
            
            # Map the dataloader batch back to the absolute temporal origin
            actual_start_idx = eval_indices_list[batch_idx]
            current_time = datetime(2020, 1, 1, 0, 0, 0) + (actual_start_idx - 35064) * time_step
            
            print(f"  -> BPTT Rollout executing on index {actual_start_idx} | Base Date: {current_time}")
            
            for k in range(lead_times):
                target_time = current_time + ((k + 1) * time_step)
                
                # Signal to the Torch compiler that a new independent graph step is beginning
                torch.compiler.cudagraph_mark_step_begin()
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # STRICT CORRECTION: Clone the tensor to break the static memory pointer
                    current_state = model(current_state).clone()
                
                # Executed outside autocast context to maintain strict FP32 metric integrity
                P_pred_phys = (current_state.float() * sigma) + mu
                T_target_phys = (T_targets[:, k].float() * sigma) + mu
                
                P_pred_eval = P_pred_phys[:, eval_indices, :, :]
                T_target_eval = T_target_phys[:, eval_indices, :, :]
                
                C_baseline = get_climatology_slice(climatology_ds, target_time).unsqueeze(0).to(device)
                
                P_prime = P_pred_eval - C_baseline
                T_prime = T_target_eval - C_baseline
                
                acc_cov[k] += torch.sum(W * P_prime * T_prime, dim=(0, 2, 3))
                acc_var_P[k] += torch.sum(W * (P_prime ** 2), dim=(0, 2, 3))
                acc_var_T[k] += torch.sum(W * (T_prime ** 2), dim=(0, 2, 3))
                
            total_sequences += 1

    final_acc = acc_cov / torch.sqrt(acc_var_P * acc_var_T)
    final_acc = final_acc.cpu().numpy()
    
    print("\nStrict WB2 Autoregressive Skill Decay (ACC):")
    z500_idx = eval_vars.index("z_500")
    t850_idx = eval_vars.index("t_850")
    t2m_idx  = eval_vars.index("t2m")
    
    print("Lead Time | Z500 ACC | T850 ACC | T2M ACC")
    print("-" * 45)
    for k in range(lead_times):
        hours_ahead = (k + 1) * 6
        z_acc = final_acc[k, z500_idx]
        t_acc = final_acc[k, t850_idx]
        t2m_acc = final_acc[k, t2m_idx]
        print(f" +{hours_ahead:03d}h    |  {z_acc:.4f}  |  {t_acc:.4f}  |  {t2m_acc:.4f}")

if __name__ == "__main__":
    execute_wb2_evaluation()