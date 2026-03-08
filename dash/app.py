import dash
from dash import html, dcc

# Initialize Dash with multi-page routing enabled
app = dash.Dash(__name__, use_pages=True)

app.layout = html.Div([
    html.H1("Machine Learning for Weather Predictions in France", style={'fontFamily': 'monospace'}),
    html.Div([
        dcc.Link("Meteorological Forecast", href="/meteofrance"),
        " | ",
        dcc.Link("Models Comparison", href="/modelscomparison"),
    ], className="nav-links", style={'fontFamily': 'monospace', 'marginBottom': '20px'}),
    html.Hr(),
    # dash.page_container acts as the dynamic DOM injection point for page modules
    dash.page_container
], style={'backgroundColor': '#111111', 'color': '#ffffff', 'padding': '20px', 'minHeight': '100vh'})

if __name__ == "__main__":
    app.run(debug=True)