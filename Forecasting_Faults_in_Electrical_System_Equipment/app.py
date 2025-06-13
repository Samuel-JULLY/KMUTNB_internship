import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objs as go
import pandas as pd
import json
import base64
import io
from script import process_csv_data  # Assure-toi que ce script existe et contient une fonction process_csv_data

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

def convert_context_to_json_serializable(context):
    result = {}
    for key, value in context.items():
        if isinstance(value, pd.DataFrame):
            result[key] = value.to_json(date_format='iso', orient='split')
        elif isinstance(value, dict):
            try:
                result[key] = {
                    k: v.to_json(date_format='iso', orient='split') if isinstance(v, pd.DataFrame) else v
                    for k, v in value.items()
                }
            except Exception:
                result[key] = value
        else:
            result[key] = value
    return result

app.layout = html.Div([
    html.H1("Analyse et Prédiction des Défaillances DGA"),

    html.Div([
        html.H2("Charger votre fichier CSV"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Glissez/Déposez ou ', html.A('sélectionnez un fichier')]),
            multiple=False
        ),
    ], className="card upload-section"),

    html.Button("☰", id="hamburger-button", className="hamburger-icon"),

    html.Div(id="sidebar", className="sidebar", children=[
        html.H2("\n"),
        html.H2("Paramètres d'Analyse"),
        html.Div([
            html.Label("Modèle"),
            dcc.Dropdown(id='model-select', options=[], value=None, clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        html.Div([
            html.Label("KV"),
            dcc.Dropdown(id='kv-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        html.Div([
            html.Label("MFG"),
            dcc.Dropdown(id='mfg-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        html.Div([
            html.Label("Intervalle d'années"),
            dcc.RangeSlider(id='year-slider'),
        ], className="control-group"),
        html.Button("Réinitialiser les filtres", id="reset-button", n_clicks=0, className="reset-button")
    ]),

    dcc.Store(id='processed-data', data={}),

    html.Div(id='main-interface', children=html.Div("Veuillez charger un fichier CSV."))
])

@app.callback(
    Output('processed-data', 'data'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def handle_uploaded_file(contents):
    if contents is None:
        return no_update
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')

    try:
        context = process_csv_data(decoded)
        context_serializable = convert_context_to_json_serializable(context)
        return context_serializable
    except Exception as e:
        print(f"Erreur de traitement du fichier : {e}")
        return {}

@app.callback(
    Output('main-interface', 'children'),
    Output('model-select', 'options'),
    Output('model-select', 'value'),
    Output('kv-select', 'options'),
    Output('kv-select', 'value'),
    Output('mfg-select', 'options'),
    Output('mfg-select', 'value'),
    Output('year-slider', 'min'),
    Output('year-slider', 'max'),
    Output('year-slider', 'value'),
    Output('year-slider', 'marks'),
    Input('processed-data', 'data')
)
def display_main_interface(data):
    if not data:
        return html.Div("Aucune donnée traitée. Veuillez uploader un fichier CSV."), [], None, [], "all", [], "all", 0, 0, [0, 0], {}

    df = pd.read_json(data['df'], orient='split')
    fault_labels = {int(k): v for k, v in data['fault_labels'].items()}
    best_model = data['best_temporal_model_name']
    available_models = list(data['all_model_predictions'].keys())
    kvs = sorted(df["KV"].dropna().unique())
    kv_options = [{"label": "Tous", "value": "all"}] + [{"label": k, "value": k} for k in kvs]
    mfgs = sorted(df["MFG"].dropna().unique())
    mfg_options = [{"label": "Tous", "value": "all"}] + [{"label": m, "value": m} for m in mfgs]
    year_min = int(df["Year Test"].min())
    year_max = int(df["Year Test"].max())
    marks = {str(y): str(y) for y in range(year_min, year_max + 1, 2)}

    return (
        html.Div([
            html.H3(id="graph-title"),
            dcc.Graph(id='main-graph'),
            html.Div(id='metrics-output', className="metrics-card")
        ], className="card graph-column"),
        [{"label": m, "value": m} for m in available_models],
        best_model,
        kv_options,
        "all",
        mfg_options,
        "all",
        year_min,
        year_max,
        [year_min, year_max],
        marks
    )

@app.callback(
    Output('main-graph', 'figure'),
    Output('metrics-output', 'children'),
    Output('graph-title', 'children'),
    Input('model-select', 'value'),
    Input('kv-select', 'value'),
    Input('mfg-select', 'value'),
    Input('year-slider', 'value'),
    Input('reset-button', 'n_clicks'),
    State('processed-data', 'data')
)
def update_graph(model_name, kv_selected, mfg_selected, year_range, reset_clicks, data):
    if not data:
        return go.Figure(), html.Div("Aucune donnée disponible."), ""

    df = pd.read_json(data['df'], orient='split')
    all_model_predictions = {k: pd.read_json(v, orient='split') for k, v in data['all_model_predictions'].items()}
    metrics_df_temporal = pd.read_json(data['metrics_df_temporal'], orient='split')
    fault_labels = {int(k): v for k, v in data['fault_labels'].items()}
    best_model = data['best_temporal_model_name']

    ctx_triggered = ctx.triggered_id
    if ctx_triggered == 'reset-button':
        kv_selected = 'all'
        mfg_selected = 'all'
        year_range = [int(df['Year Test'].min()), int(df['Year Test'].max())]
        model_name = best_model

    filtered_df = df[df['Year Test'].between(*year_range)]
    if kv_selected != 'all':
        filtered_df = filtered_df[filtered_df['KV'] == kv_selected]
    if mfg_selected != 'all':
        filtered_df = filtered_df[filtered_df['MFG'] == mfg_selected]

    traces = []
    if not filtered_df.empty:
        temp_real_props = filtered_df.groupby('Year Test')['true_fault_index'].value_counts(normalize=True).unstack(fill_value=0)
        all_faults = [fault_labels[idx] for idx in sorted(fault_labels.keys())]
        temp_real_props_full = pd.DataFrame(0.0, index=temp_real_props.index, columns=all_faults)
        for col in temp_real_props.columns:
            temp_real_props_full[fault_labels[col]] = temp_real_props[col]
        for fault in temp_real_props_full.columns:
            yearly = temp_real_props_full[fault]
            traces.append(go.Scatter(x=yearly.index, y=yearly.values, name=f"Réel {fault}", mode="lines+markers"))

    if model_name in all_model_predictions:
        preds = all_model_predictions[model_name]
        last_train_year = df['Year Test'].max()
        last_year_data = filtered_df[filtered_df['Year Test'] == last_train_year]
        initial_prediction_points = {
            fault_labels[idx]: prop
            for idx, prop in last_year_data['true_fault_index'].value_counts(normalize=True).items()
        }

        for fault in preds.columns:
            plot_preds = preds.copy()
            if last_train_year not in plot_preds.index:
                plot_preds.loc[last_train_year] = 0
                plot_preds = plot_preds.sort_index()
            if fault in initial_prediction_points:
                plot_preds.loc[last_train_year, fault] = initial_prediction_points.get(fault, 0)
            traces.append(go.Scatter(x=plot_preds.index, y=plot_preds[fault],
                                     name=f"{model_name} Prédit {fault}", mode="lines+markers",
                                     line=dict(dash='dash')))

    m = metrics_df_temporal.loc[model_name]
    metrics_text = html.Div([
        html.H4("Résumé des performances"),
        html.Ul([
            html.Li(f"Accuracy: {m.get('Accuracy', 'N/A'):.3f}"),
            html.Li(f"Precision: {m.get('Precision', 'N/A'):.3f}"),
            html.Li(f"Recall: {m.get('Recall', 'N/A'):.3f}"),
            html.Li(f"Specificity: {m.get('Specificity', 'N/A'):.3f}"),
            html.Li(f"F1-Score: {m.get('F1', 'N/A'):.3f}")
        ])
    ])

    title = f"Prédiction des proportions de défaillances par {model_name} (KV: {kv_selected}, MFG: {mfg_selected})"
    figure = {
        'data': traces,
        'layout': go.Layout(
            title=title,
            xaxis=dict(title="Année"),
            yaxis=dict(title="Proportion prédite", range=[0, 1.05]),
            hovermode='x unified',
            legend=dict(orientation="h", x=0.5, xanchor='center', y=-0.2)
        )
    }

    return figure, metrics_text, title

@app.callback(
    Output('sidebar', 'className'),
    Input('hamburger-button', 'n_clicks'),
    State('sidebar', 'className'),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, current_class):
    if 'open' in current_class:
        return 'sidebar'
    else:
        return 'sidebar open'

if __name__ == '__main__':
    app.run(debug=True)
import dash_core_components as dcc
import dash_html_components as html

app.layout = html.Div([
    # ta sidebar, etc.

    dcc.Loading(
        id="loading-graph",
        type="circle",  # ou "dot", "cube", "graph" etc.
        children=html.Div(id="graph-container")
    )
])