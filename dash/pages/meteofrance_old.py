import sys
import os
import json
import torch
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from functools import lru_cache

dash.register_page(__name__, path="/meteofrance", name="Meteo Prediction")

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
t2m_idx = channel_names.index("t2m")

mu_list = [stats["mean"].get(var, 0.0) for var in channel_names]
sigma_list = [stats["std"].get(var, 1.0) for var in channel_names]

mu = torch.tensor(mu_list, device=device).view(1, len(channel_names), 1, 1)
sigma = torch.tensor(sigma_list, device=device).view(1, len(channel_names), 1, 1)

tensor_dir = os.path.join(data_dir, 'processed', 'tensors')
eval_dataset = MeteorologicalDataset(tensor_dir=tensor_dir, rollout_steps=1)
P_initial, _ = eval_dataset[35064]
P_initial = P_initial.unsqueeze(0).to(device)

model_path = os.path.join(data_dir, 'models', 'checkpoints', 'resunet_optimal.pth')
model = ResUNet(in_channels=len(channel_names), out_channels=len(channel_names), base_filters=128).to(device)

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

# JIT Compilation of the execution graph
with torch.no_grad():
    compiled_model = torch.jit.trace(model, P_initial)

# Load serialized characteristic matrix and boundaries
H, W = P_initial.shape[-2:]
lat_array = np.linspace(52.0, 41.0, H)
lon_array = np.linspace(-6.0, 10.0, W)

mask_matrix = np.load(os.path.join(tensor_dir, "france_mask.npy"))
with open(os.path.join(tensor_dir, "france_boundaries.json"), "r") as f:
    bounds = json.load(f)
shp_lons, shp_lats = bounds["lons"], bounds["lats"]

# Recursive memoization of the autoregressive mapping
@lru_cache(maxsize=32)
def get_forecast_state(step: int):
    if step == 0:
        return P_initial
    prev_state = get_forecast_state(step - 1)
    with torch.no_grad():
        return compiled_model(prev_state)

layout = html.Div(style={'fontFamily': 'monospace', 'backgroundColor': '#111111', 'color': '#ffffff', 'padding': '20px'}, children=[
    html.H1("Operational Forecast Execution (JIT Compiled)"),
    html.Div([
        html.Label("Target Variable Manifold:"),
        dcc.Dropdown(
            id='forecast-variable-selector',
            options=[
                {'label': 'Z500 (Geopotential 500 hPa)', 'value': z500_idx},
                {'label': 'T2M (Surface Temperature)', 'value': t2m_idx}
            ],
            value=z500_idx,
            style={'color': '#000000', 'width': '300px', 'marginTop': '10px'}
        )
    ], style={'marginTop': '20px', 'marginBottom': '20px'}),
    html.Div([
        html.Label("Autoregressive Horizon (Hours):"),
        dcc.Slider(
            id='forecast-lead-time',
            min=6, max=72, step=6, value=6,
            marks={i: {'label': f'+{i}h', 'style': {'color': '#ffffff'}} for i in range(6, 73, 12)}
        )
    ]),
    dcc.Graph(id='forecast-plot', style={'height': '700px', 'marginTop': '30px'})
])

@dash.callback(
    Output('forecast-plot', 'figure'),
    [Input('forecast-variable-selector', 'value'),
     Input('forecast-lead-time', 'value')]
)
def compute_forecast(var_idx, lead_time):
    steps = lead_time // 6
    current_state = get_forecast_state(steps)
    
    P_phys = (current_state.float() * sigma) + mu
    spatial_field = P_phys[0, var_idx, :, :].cpu().numpy()
    spatial_field_masked = np.where(mask_matrix, spatial_field, np.nan)
    
    colorscale = 'RdBu_r' if var_idx == z500_idx else 'Inferno'
    title = "Z500 Extrapolation" if var_idx == z500_idx else "T2M Extrapolation"
    unit = "m²/s²" if var_idx == z500_idx else "K"
    
    contour_trace = go.Contour(
        z=spatial_field_masked, x=lon_array, y=lat_array,
        colorscale=colorscale, contours=dict(showlines=False), colorbar=dict(title=unit)
    )
    boundary_trace = go.Scatter(
        x=shp_lons, y=shp_lats, mode='lines',
        line=dict(color='white', width=0.8), hoverinfo='skip', showlegend=False
    )
    
    fig = go.Figure(data=[contour_trace, boundary_trace])
    fig.update_layout(
        title=f"{title} | Horizon: +{lead_time}h",
        xaxis_title="Longitude", yaxis_title="Latitude",
        xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(constrain="domain"),
        template="plotly_dark", plot_bgcolor='#111111', paper_bgcolor='#111111'
    )
    return fig