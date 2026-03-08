import sys
import os
import json
import torch
import dash
import numpy as np
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from functools import lru_cache

dash.register_page(__name__, path="/modelscomparison", name="Models Comparison")

current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'pytorch_pipeline'))
if pipeline_path not in sys.path:
    sys.path.append(pipeline_path)

from architecture import ResUNet
from dataset import MeteorologicalDataset

device = torch.device("cpu")

data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
with open(os.path.join(data_dir, 'processed', 'global_stats.json'), "r") as f:
    stats = json.load(f)
with open(os.path.join(data_dir, 'processed', 'tensors', 'channel_ordering.json'), "r") as f:
    channel_names = json.load(f)

z500_idx = channel_names.index("z_500")
mu_list = [stats["mean"].get(var, 0.0) for var in channel_names]
sigma_list = [stats["std"].get(var, 1.0) for var in channel_names]

mu = torch.tensor(mu_list, device=device).view(1, len(channel_names), 1, 1)
sigma = torch.tensor(sigma_list, device=device).view(1, len(channel_names), 1, 1)

tensor_dir = os.path.join(data_dir, 'processed', 'tensors')
eval_dataset = MeteorologicalDataset(tensor_dir=tensor_dir, rollout_steps=1)
P_initial, _ = eval_dataset[35064]
P_initial = P_initial.unsqueeze(0).to(device)

model_dir = os.path.join(data_dir, 'models', 'checkpoints')
compiled_models = {}

for name, filename in [("optimal", "resunet_optimal.pth"), ("latest", "resunet_latest.pth")]:
    path = os.path.join(model_dir, filename)
    if os.path.exists(path):
        m = ResUNet(in_channels=len(channel_names), out_channels=len(channel_names), base_filters=128).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(checkpoint['model_state_dict'])
        m.eval()
        
        # JIT Compilation mapping
        with torch.no_grad():
            compiled_models[name] = torch.jit.trace(m, P_initial)

MAX_LEAD_TIME = 168
MAX_STEPS = MAX_LEAD_TIME // 6

# Dictionary to hold the pre-calculated 2D spatial manifolds
# Structure: cache[model_name][step] = masked_z500_numpy_array
precomputed_cache = {"optimal": {}, "latest": {}}

for model_name, path in [("optimal", "./data/models/checkpoints/resunet_optimal.pth"), 
                         ("latest", "./data/models/checkpoints/resunet_latest.pth")]:
    if os.path.exists(path):
        # 1. Load model
        m = ResUNet(in_channels=len(channel_names), out_channels=len(channel_names), base_filters=128).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(checkpoint['model_state_dict'])
        m.eval()
        
        # 2. Compile model for faster initialization looping
        with torch.no_grad():
            compiled_model = torch.jit.trace(m, P_initial)
            
            current_state = P_initial
            
            # Store t=0
            P_phys_0 = (current_state.float() * sigma) + mu
            z500_field_0 = P_phys_0[0, z500_idx, :, :].cpu().numpy()
            precomputed_cache[model_name][0] = np.where(mask_matrix, z500_field_0, np.nan)
            
            # 3. Execute the full deterministic sequence
            for step in range(1, MAX_STEPS + 1):
                current_state = compiled_model(current_state)
                
                # Pre-calculate Denormalization
                P_phys = (current_state.float() * sigma) + mu
                
                # Extract Manifold
                z500_field = P_phys[0, z500_idx, :, :].cpu().numpy()
                
                # Pre-calculate Masking
                masked_field = np.where(mask_matrix, z500_field, np.nan)
                
                # Cache the terminal 2D physical array
                precomputed_cache[model_name][step] = masked_field

print("Cache Initialization Complete. Starting Dash Server...")

# Bi-variate recursive memoization (Hash signature includes model invariant)
@lru_cache(maxsize=128)
def get_comparison_state(model_name: str, step: int):
    if step == 0:
        return P_initial
    prev_state = get_comparison_state(model_name, step - 1)
    with torch.no_grad():
        return compiled_models[model_name](prev_state)

layout = html.Div(style={'fontFamily': 'monospace', 'backgroundColor': '#111111', 'color': '#ffffff', 'padding': '20px'}, children=[
    html.H1("ResUNet Autoregressive Surrogate: Multi-Phase Matrix Evaluation"),
    html.Div("Target Variable: Z500 (Geopotential at 500 hPa)"),
    html.Div([
        html.Label("Parameter Matrix:"),
        dcc.Dropdown(
            id='matrix-selector',
            options=[
                {'label': 'Phase 2 (K=2) - Maximum Precision', 'value': 'optimal'},
                {'label': 'Phase 3 (K=3) - Long-Term Stability', 'value': 'latest'}
            ],
            value='latest',
            style={'color': '#000000', 'width': '400px', 'marginTop': '10px'}
        )
    ], style={'marginTop': '20px', 'marginBottom': '20px'}),
    html.Div([
        html.Label("Temporal Rollout Horizon (Hours):"),
        dcc.Slider(
            id='lead-time-slider',
            min=0, max=168, step=6, value=24,
            marks={i: {'label': f'+{i}h', 'style': {'color': '#ffffff'}} for i in range(0, 169, 24)}
        )
    ]),
    dcc.Graph(id='spatial-forecast-plot', style={'height': '700px', 'marginTop': '30px'})
])

@dash.callback(
    Output('spatial-forecast-plot', 'figure'),
    [Input('matrix-selector', 'value'),
     Input('lead-time-slider', 'value')]
)
def compute_pushforward_mapping(selected_matrix, lead_time):
    if selected_matrix not in precomputed_cache:
        return go.Figure().update_layout(title="Error: Model checkpoint not loaded.", template="plotly_dark")
    
    steps = lead_time // 6
    
    # O(1) Memory Retrieval. PyTorch inference is bypassed entirely.
    z500_field_masked = precomputed_cache[selected_matrix][steps]
    
    contour_trace = go.Contour(
        z=z500_field_masked, x=lon_array, y=lat_array,
        colorscale='RdBu_r', contours=dict(showlines=False),
        colorbar=dict(title='Geopotential (m²/s²)')
    )
    current_state = get_comparison_state(selected_matrix, steps)
    
    P_phys = (current_state.float() * sigma) + mu
    z500_field = P_phys[0, z500_idx, :, :].cpu().numpy()
    
    fig = go.Figure(data=go.Contour(
        z=z500_field, colorscale='RdBu_r', contours=dict(showlines=False), colorbar=dict(title='Geopotential (m²/s²)')
    ))
    
    fig.update_layout(
        title=f"Z500 Extrapolation | Checkpoint: {selected_matrix}.pth | Horizon: +{lead_time}h",
        xaxis_title="Longitude Coordinate Index", yaxis_title="Latitude Coordinate Index",
        template="plotly_dark", plot_bgcolor='#111111', paper_bgcolor='#111111'
    )
    return fig