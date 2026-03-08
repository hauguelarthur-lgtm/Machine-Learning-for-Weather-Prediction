import sys
import os
import json
import torch
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import geopandas as gpd

dash.register_page(__name__, path="/meteofrance", name="Meteo Prediction")

# 1. Path Topology
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'pytorch_pipeline'))
if pipeline_path not in sys.path:
    sys.path.append(pipeline_path)

from architecture import ResUNet
from dataset import MeteorologicalDataset

# 2. Hardware Allocation Matrix
device = torch.device("cpu")

# 3. Global State Initialization
stats_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'global_stats.json')
channels_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'tensors', 'channel_ordering.json')

with open(stats_path, "r") as f:
    stats = json.load(f)
with open(channels_path, "r") as f:
    channel_names = json.load(f)

z500_idx = channel_names.index("z_500")
t2m_idx = channel_names.index("t2m")

mu_list = [stats["mean"].get(var, 0.0) for var in channel_names]
sigma_list = [stats["std"].get(var, 1.0) for var in channel_names]

mu = torch.tensor(mu_list, device=device).view(1, len(channel_names), 1, 1)
sigma = torch.tensor(sigma_list, device=device).view(1, len(channel_names), 1, 1)

# Load Optimal Model Weights
model_path = os.path.join(current_dir, '..', '..', 'data', 'models', 'checkpoints', 'resunet_optimal.pth')
model = ResUNet(in_channels=len(channel_names), out_channels=len(channel_names), base_filters=128).to(device)

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

# Load Initial Condition Matrix
tensor_dir = os.path.join(current_dir, '..', '..', 'data', 'processed', 'tensors')
eval_dataset = MeteorologicalDataset(tensor_dir=tensor_dir, rollout_steps=1)
P_initial, _ = eval_dataset[35064]
P_initial = P_initial.unsqueeze(0).to(device)

# 4. Spatial Grid & Geographic Matrix Computation
print("[meteofrance] Computing Geographic Characteristic Matrix...")
H, W = P_initial.shape[-2:]
# Ensure these linspace bounds strictly match your ERA5 crop coordinates
lat_array = np.linspace(52.0, 41.0, H)
lon_array = np.linspace(-6.0, 10.0, W)

shapefile_path = os.path.join(current_dir, '..', '..', 'data', 'shapefiles', 'departement.shp')
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

# Extract vector boundaries for plotting
shp_lons, shp_lats = [], []
for geom in gdf.geometry:
    if geom is None: continue
    if geom.type == 'Polygon':
        x, y = geom.exterior.coords.xy
        shp_lons.extend(x.tolist() + [None])
        shp_lats.extend(y.tolist() + [None])
    elif geom.type == 'MultiPolygon':
        for poly in geom.geoms:
            x, y = poly.exterior.coords.xy
            shp_lons.extend(x.tolist() + [None])
            shp_lats.extend(y.tolist() + [None])

# Compute spatial mask via unary union PIP evaluation
france_boundary = gdf.geometry.unary_union
Lons, Lats = np.meshgrid(lon_array, lat_array)
points_geoseries = gpd.GeoSeries(gpd.points_from_xy(Lons.flatten(), Lats.flatten()))
mask_1d = points_geoseries.within(france_boundary)
mask_matrix = mask_1d.values.reshape(Lons.shape)

# 5. Declarative DOM Layout
layout = html.Div(style={'fontFamily': 'monospace', 'backgroundColor': '#111111', 'color': '#ffffff', 'padding': '20px'}, children=[
    html.H1("Operational Forecast Execution (Optimal Weights)"),
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

# 6. Inference Execution Graph
@dash.callback(
    Output('forecast-plot', 'figure'),
    [Input('forecast-variable-selector', 'value'),
     Input('forecast-lead-time', 'value')]
)
def compute_forecast(var_idx, lead_time):
    current_state = P_initial
    steps = lead_time // 6
    
    with torch.no_grad():
        for _ in range(steps):
            current_state = model(current_state)
        P_phys = (current_state.float() * sigma) + mu
        
    spatial_field = P_phys[0, var_idx, :, :].cpu().numpy()
    spatial_field_masked = np.where(mask_matrix, spatial_field, np.nan)
    
    colorscale = 'RdBu_r' if var_idx == z500_idx else 'Inferno'
    title = "Z500 Extrapolation" if var_idx == z500_idx else "T2M Extrapolation"
    unit = "m²/s²" if var_idx == z500_idx else "K"
    
    contour_trace = go.Contour(
        z=spatial_field_masked, x=lon_array, y=lat_array,
        colorscale=colorscale, contours=dict(showlines=False),
        colorbar=dict(title=unit)
    )
    
    boundary_trace = go.Scatter(
        x=shp_lons, y=shp_lats, mode='lines',
        line=dict(color='white', width=0.8), hoverinfo='skip', showlegend=False
    )
    
    fig = go.Figure(data=[contour_trace, boundary_trace])
    fig.update_layout(
        title=f"{title} | Horizon: +{lead_time}h",
        xaxis_title="Longitude Coordinate Index", yaxis_title="Latitude Coordinate Index",
        xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(constrain="domain"),
        template="plotly_dark", plot_bgcolor='#111111', paper_bgcolor='#111111'
    )
    return fig