import sys
import os
import json
import torch
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

dash.register_page(__name__, path="/modelcomparison")

# 1. Path Topology: Inject the PyTorch pipeline into the execution context
sys.path.append(os.path.abspath('./pytorch_pipeline'))
from architecture import ResUNet
from dataset import MeteorologicalDataset

# 2. Hardware Allocation Matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing Dash Presentation Layer on {device} (FP32 precision strictly enforced)...")

# 3. Global State Initialization (Pre-allocated in 32GB RAM)
print("Pre-loading Statistical Tensors...")
with open("./data/processed/global_stats.json", "r") as f:
    stats = json.load(f)
with open("./data/processed/tensors/channel_ordering.json", "r") as f:
    channel_names = json.load(f)

z500_idx = channel_names.index("z_500")

mu_list = [stats["mean"][var] if var in stats["mean"] else 0.0 for var in channel_names]
sigma_list = [stats["std"][var] if var in stats["mean"] else 1.0 for var in channel_names]

mu = torch.tensor(mu_list, device=device).view(1, 102, 1, 1)
sigma = torch.tensor(sigma_list, device=device).view(1, 102, 1, 1)

print("Pre-loading Neural Parameter Matrices...")
models = {}
for name, path in [("optimal", "./models/checkpoints/resunet_optimal.pth"), 
                   ("latest", "./models/checkpoints/resunet_latest.pth")]:
    if os.path.exists(path):
        m = ResUNet(in_channels=102, out_channels=102, base_filters=128).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(checkpoint['model_state_dict'])
        m.eval()
        models[name] = m
    else:
        print(f"Warning: {path} not found.")

# Pre-load a single fixed initial condition (Index 35064 represents Jan 1, 2020)
print("Pre-loading Initial Condition Tensor (Jan 1, 2020)...")
eval_dataset = MeteorologicalDataset(tensor_dir="./data/processed/tensors/", rollout_steps=1)
P_initial, _ = eval_dataset[35064]
P_initial = P_initial.unsqueeze(0).to(device)

# 4. Instantiate the Reactive Server
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'monospace', 'backgroundColor': '#111111', 'color': '#ffffff', 'padding': '20px'}, children=[
    html.H1("ResUNet Autoregressive Surrogate: Synoptic Advection Boundary"),
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

# 5. The Inference Execution Graph
@app.callback(
    Output('spatial-forecast-plot', 'figure'),
    [Input('matrix-selector', 'value'),
     Input('lead-time-slider', 'value')]
)
def compute_pushforward_mapping(selected_matrix, lead_time):
    if selected_matrix not in models:
        return go.Figure().update_layout(title="Error: Model checkpoint not found in memory.", template="plotly_dark")
    
    model = models[selected_matrix]
    current_state = P_initial
    
    steps = lead_time // 6
    
    # Execute the strict BPTT loop without TorchDynamo compiler to ensure local compatibility
    with torch.no_grad():
        for _ in range(steps):
            current_state = model(current_state)
            
        # Denormalize strictly to physical ERA5 magnitudes
        P_phys = (current_state.float() * sigma) + mu
        
    # Extract the 2D spatial manifold for Z500
    z500_field = P_phys[0, z500_idx, :, :].cpu().numpy()
    
    # Construct the continuous mathematical projection (Contour Map)
    fig = go.Figure(data=go.Contour(
        z=z500_field,
        colorscale='RdBu_r',
        contours=dict(showlines=False),
        colorbar=dict(title='Geopotential (m²/s²)')
    ))
    
    fig.update_layout(
        title=f"Z500 Projection | Checkpoint: {selected_matrix}.pth | Horizon: +{lead_time}h",
        xaxis_title="Longitude Coordinate Index",
        yaxis_title="Latitude Coordinate Index",
        template="plotly_dark",
        plot_bgcolor='#111111',
        paper_bgcolor='#111111'
    )
    
    return fig

