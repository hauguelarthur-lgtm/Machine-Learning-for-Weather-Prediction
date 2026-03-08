import dash
from dash import html, dcc

app = dash.Dash(__name__, use_pages=True)

app.layout = html.Div([
    html.H1("machine Learning for Weather Predictions in France."),
    html.Div([
        dcc.Link("Meteo in France", href="/meteofrance"),
        " | ",
        dcc.Link("Models Comparison", href="/modelscomparison"),
        " | ",
    ], className="nav-links"),
    html.Hr(),
    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=True)