import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import joblib

# Charger les objets sauvegardés
# Assurez-vous que le fichier 'dash_context.pkl' est disponible
try:
    context = joblib.load("dash_context.pkl")
except FileNotFoundError:
    print("Erreur: Le fichier 'dash_context.pkl' n'a pas été trouvé.")
    print("Veuillez vous assurer qu'il est généré par votre script de prétraitement.")
    exit()

df = context["df"]
real_proportions_by_year = context["real_proportions_by_year"]
all_model_predictions = context["all_model_predictions"]
metrics_df_temporal = context["metrics_df_temporal"]
fault_labels = context["fault_labels"]
best_temporal_model_name = context["best_temporal_model_name"]

# Options pour les filtres
available_models = list(all_model_predictions.keys())

# Modification ici : Ajouter l'option "Tous" pour KV et MFG
available_kvs = sorted(df["KV"].dropna().unique())
available_kvs_options = [{"label": "Tous", "value": "all"}] + [{"label": k, "value": k} for k in available_kvs]

available_mfgs = sorted(df["MFG"].dropna().unique())
available_mfgs_options = [{"label": "Tous", "value": "all"}] + [{"label": m, "value": m} for m in available_mfgs]

year_range = [int(df["Year Test"].min()), int(df["Year Test"].max())]

# Initialiser l'app Dash
app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Analyse et Prédiction des Défaillances DGA", style={'textAlign': 'center', 'color': '#0A2A5E'}),
    html.H3(id="graph-title", style={'textAlign': 'center', 'color': '#2C3E50'}),
    dcc.Graph(id='main-graph', style={'height': '600px'}),

    html.Div([
        html.Div([
            html.Label("Modèle", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='model-select', options=[{"label": m, "value": m} for m in available_models],
                         value=best_temporal_model_name,
                         clearable=False, # Empêche de vider la sélection
                         style={'marginBottom': '10px'}),

            html.Label("KV", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='kv-select', options=available_kvs_options,
                         value="all", # Valeur par défaut "Tous"
                         clearable=False,
                         style={'marginBottom': '10px'}),

            html.Label("MFG", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='mfg-select', options=available_mfgs_options,
                         value="all", # Valeur par défaut "Tous"
                         clearable=False,
                         style={'marginBottom': '10px'}),

            # Début de la correction pour dcc.RangeSlider style
            html.Div([
                html.Label("Intervalle d'années", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(id='year-slider', min=year_range[0], max=year_range[1],
                                step=1, value=year_range,
                                marks={str(y): {'label': str(y), 'style': {'writing-mode': 'vertical-rl', 'text-orientation': 'sideways'}}
                                       for y in range(year_range[0], year_range[1] + 1, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}),
            ], style={'marginTop': '20px', 'marginBottom': '30px'}),
            # Fin de la correction

            html.Button("Réinitialiser les filtres", id="reset-button", n_clicks=0,
                        style={'backgroundColor': '#E74C3C', 'color': 'white', 'padding': '10px 20px',
                               'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer',
                               'marginTop': '20px', 'width': '100%'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                  'padding': '20px', 'borderRight': '1px solid #ddd', 'boxSizing': 'border-box'}),

        html.Div(id='metrics-output', style={'width': '68%', 'display': 'inline-block',
                                             'verticalAlign': 'top', 'padding': '20px',
                                             'boxSizing': 'border-box'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'flex-start', 'backgroundColor': '#F8F9FA', 'padding': '10px', 'borderRadius': '8px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})
])

@app.callback(
    Output('main-graph', 'figure'),
    Output('metrics-output', 'children'),
    Output('graph-title', 'children'),
    Input('model-select', 'value'),
    Input('kv-select', 'value'),
    Input('mfg-select', 'value'),
    Input('year-slider', 'value'),
    Input('reset-button', 'n_clicks'),
)
def update_graph(model_name, kv_selected, mfg_selected, year_range, reset_clicks):
    # Determine the context of the callback to reset filters if button is clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Reset filters if the reset button is clicked
    # Note: The reset values for dropdowns need to be returned by an Output callback
    # for them to actually change in the UI. For simplicity, we'll just apply them
    # to the filtered_df here and assume the user's manual selection will override
    # if they interact after reset. A more robust solution would involve chained callbacks
    # or storing default values in dcc.Store.
    if button_id == 'reset-button' and reset_clicks > 0:
        kv_selected = "all"
        mfg_selected = "all"
        year_range = [int(df["Year Test"].min()), int(df["Year Test"].max())]
        model_name = best_temporal_model_name # Keep the best model as default on reset


    # Modification ici : Logique de filtrage pour "Tous"
    filtered_df = df[df["Year Test"].between(*year_range)]
    
    if kv_selected != "all":
        filtered_df = filtered_df[filtered_df["KV"] == kv_selected]
    
    if mfg_selected != "all":
        filtered_df = filtered_df[filtered_df["MFG"] == mfg_selected]

    traces = []

    # Proportions réelles
    # Assurez-vous que filtered_df n'est pas vide avant de calculer les proportions
    if not filtered_df.empty:
        # Calculer les proportions réelles basées sur le DataFrame filtré
        # On utilise value_counts puis unstack pour obtenir les proportions par faute et par année
        temp_real_proportions_by_year = filtered_df.groupby("Year Test")["true_fault_index"].value_counts(normalize=True).unstack(fill_value=0)
        # S'assurer que les colonnes sont nommées avec les labels de faute
        # Il faut s'assurer que toutes les fault_labels sont présentes comme colonnes
        # Créez un DataFrame avec toutes les colonnes de faute possibles et joignez-le
        all_fault_cols = [fault_labels[idx] for idx in sorted(fault_labels.keys())]
        temp_real_proportions_by_year_full = pd.DataFrame(0.0, index=temp_real_proportions_by_year.index, columns=all_fault_cols)
        for col in temp_real_proportions_by_year.columns:
            temp_real_proportions_by_year_full[col] = temp_real_proportions_by_year[col]

        for fault in temp_real_proportions_by_year_full.columns:
            yearly = temp_real_proportions_by_year_full[fault]
            if not yearly.empty:
                traces.append(go.Scatter(x=yearly.index, y=yearly.values,
                                         name=f"Réel {fault}", mode="lines+markers",
                                         line=dict(color='black', width=2),
                                         marker=dict(symbol='circle', size=7)))
    else:
        # Si le DataFrame filtré est vide, affichez un message ou une figure vide.
        traces.append(go.Scatter(x=[df["Year Test"].min(), df["Year Test"].max()], y=[0, 0],
                                 mode="lines", name="Aucune donnée réelle filtrée"))


    # Prédictions du modèle (toujours pour toutes les années futures)
    if model_name in all_model_predictions:
        preds = all_model_predictions[model_name]
        
        last_train_year = df["Year Test"].max()
        
        # Récupérer les données réelles pour l'année de fin d'entraînement afin de "joindre" les lignes
        initial_prediction_points = {}
        if not filtered_df.empty:
            last_year_data = filtered_df[filtered_df["Year Test"] == last_train_year]
            if not last_year_data.empty:
                last_year_proportions = (last_year_data["true_fault_index"].value_counts(normalize=True)).to_dict()
                for idx, prop in last_year_proportions.items():
                    initial_prediction_points[fault_labels[idx]] = prop

        for fault in preds.columns:
            # Créer un DataFrame temporaire pour le tracé de la prédiction
            plot_preds = preds.copy()
            
            # Ajouter une entrée pour last_train_year si elle n'existe pas dans les prédictions
            # Cela permet de lier la fin des données réelles au début des prédictions
            if last_train_year not in plot_preds.index:
                 plot_preds.loc[last_train_year] = 0 # Valeur temporaire, sera remplacée
                 plot_preds = plot_preds.sort_index()

            # Mettre à jour la valeur de la dernière année d'entraînement dans les prédictions
            # pour qu'elle corresponde à la valeur réelle de cette année
            # Assurez-vous que le type de faute est bien dans fault_labels et que initial_prediction_points est correctement peuplé
            if fault in initial_prediction_points and last_train_year in plot_preds.index:
                plot_preds.loc[last_train_year, fault] = initial_prediction_points.get(fault, 0)
            
            traces.append(go.Scatter(x=plot_preds.index, y=plot_preds[fault],
                                     name=f"{model_name} Prédit {fault}", mode="lines+markers",
                                     line=dict(dash='dash', width=2),
                                     marker=dict(symbol='x', size=7)))

    # Résumé des performances
    m = metrics_df_temporal.loc[model_name]
    metrics_text = html.Div([
        html.H4("Résumé des performances du modèle sélectionné", style={'color': '#2C3E50', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px'}),
        html.Ul([
            html.Li(f"Accuracy: {m.get('Accuracy', 'N/A'):.3f}", style={'marginBottom': '5px'}),
            html.Li(f"Precision: {m.get('Precision', 'N/A'):.3f}", style={'marginBottom': '5px'}),
            html.Li(f"Recall: {m.get('Recall', 'N/A'):.3f}", style={'marginBottom': '5px'}),
            html.Li(f"Specificity: {m.get('Specificity', 'N/A'):.3f}", style={'marginBottom': '5px'}),
            html.Li(f"F1-Score: {m.get('F1', 'N/A'):.3f}", style={'marginBottom': '5px'})
        ], style={'listStyleType': 'none', 'paddingLeft': '0'})
    ])

    title = f"Prédiction des proportions de défaillances par {model_name} (KV: {kv_selected}, MFG: {mfg_selected})"
    
    # Ajouter une ligne verticale pour marquer la fin des données d'entraînement
    shapes = [
        dict(
            type="line",
            x0=df["Year Test"].max(),
            y0=0,
            x1=df["Year Test"].max(),
            y1=1,
            line=dict(color="red", width=2, dash="dash"),
            name="Fin des données d'entraînement"
        )
    ]

    figure = {
        'data': traces,
        'layout': go.Layout(
            title=title,
            xaxis=dict(title="Année", tickmode='linear', dtick=1),
            yaxis=dict(title="Proportion prédite", range=[0, 1.05]), # Ajuster la plage Y
            hovermode='x unified',
            legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2), # Légende en bas
            shapes=shapes # Ajouter la ligne verticale
        )
    }

    return figure, metrics_text, title

if __name__ == '__main__':
    app.run(debug=True) # CHANGEMENT: app.run_server() remplacé par app.run()