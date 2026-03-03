import multiprocessing as mp
import time

# Strictly isolate PyTorch imports to the child processes to prevent the 
# parent process from inadvertently initializing the global CUDA context.

def execute_optimization_phase(result_queue):
    """
    Executes isolated within Child Process A.
    The OS assigns a unique PID and isolated VRAM address space.
    """
    import torch
    import optuna
    from train_hyperparameters import objective

    print(f"[PID: {mp.current_process().pid}] Initializing Optuna Search Space...")
    
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_trial.params
    print(f"[PID: {mp.current_process().pid}] Optimization Complete. Coordinates: {best_params}")
    
    # Transmit the coordinates back to the parent process via IPC
    result_queue.put(best_params)

def execute_convergence_phase(params):
    """
    Executes isolated within Child Process B.
    Initializes onto a perfectly defragmented GPU.
    """
    import torch
    from train_convergence import execute_training_pipeline

    print(f"[PID: {mp.current_process().pid}] Initializing Global Gradient Convergence...")
    
    execute_training_pipeline(
        optimal_lr=params["lr"],
        optimal_wd=params["weight_decay"],
        batch_size=params["batch_size"],
        epochs=100
    )

if __name__ == "__main__":
    # 1. Enforce strict OS-level memory isolation
    mp.set_start_method('spawn')
    
    # Inter-Process Communication (IPC) buffer
    queue = mp.Queue()
    
    print("===========================================================")
    print("PHASE 1: Spawning Parameter Search Process")
    print("===========================================================")
    
    # Instantiate and start Process A
    p_search = mp.Process(target=execute_optimization_phase, args=(queue,))
    p_search.start()
    
    # The parent process blocks execution here until Process A terminates
    p_search.join()
    
    if p_search.exitcode != 0:
        raise RuntimeError("Optimization process failed. Halting pipeline.")

    # Process A is now dead. The Linux kernel has flushed all CUDA allocations.
    
    # Extract the mathematical coordinates from the IPC queue
    optimal_params = queue.get()
    
    print("\n===========================================================")
    print("PHASE 2: VRAM Purged. Spawning Convergence Process")
    print("===========================================================")
    
    # Instantiate and start Process B using the injected coordinates
    p_train = mp.Process(target=execute_convergence_phase, args=(optimal_params,))
    p_train.start()
    
    # Block until Process B completes the 100-epoch minimization
    p_train.join()
    
    if p_train.exitcode != 0:
        raise RuntimeError("Convergence process failed.")

    print("\n===========================================================")
    print("PIPELINE EXECUTION COMPLETE.")
    print("===========================================================")