import sys
import os
import json
import torch
import dash
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
    if selected_matrix not in compiled_models:
        return go.Figure().update_layout(title="Error: Model checkpoint not compiled.", template="plotly_dark")
    
    steps = lead_time // 6
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