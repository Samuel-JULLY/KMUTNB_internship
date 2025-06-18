import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objs as go
import pandas as pd
import json
import base64
import io
# Assurez-vous que ce script existe et contient une fonction process_csv_data
from script import process_csv_data

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Helper function to convert DataFrame to JSON serializable format
def convert_context_to_json_serialable(context):
    result = {}
    for key, value in context.items():
        if isinstance(value, pd.DataFrame):
            # Convert DataFrame to JSON string
            result[key] = value.to_json(date_format='iso', orient='split')
        elif isinstance(value, dict):
            # Recursively handle dictionaries that might contain DataFrames
            try:
                result[key] = {
                    k: v.to_json(date_format='iso', orient='split') if isinstance(v, pd.DataFrame) else v
                    for k, v in value.items()
                }
            except Exception:
                # Fallback for complex dictionaries not easily serializable
                result[key] = value
        else:
            # Directly assign other types of values
            result[key] = value
    return result

app.layout = html.Div([
    # CSS est maintenant géré par un fichier externe dans le dossier 'assets'
    # Pas besoin de bloc html.Style ici.

    html.H1("Analyse et Prédiction des Défaillances DGA"),

    # Wrapper pour la section de téléchargement - sa visibilité sera contrôlée
    html.Div(id='upload-section-wrapper', children=[
        html.Div([
            html.H2("Charger votre fichier CSV"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Glissez/Déposez ou ', html.A('sélectionnez un fichier')]),
                multiple=False
            ),
        ], className="card upload-section"),
    ]),

    html.Button("☰", id="hamburger-button", className="hamburger-icon"),

    html.Div(id="sidebar", className="sidebar", children=[
        html.H2("\n"), # Nouvelle ligne pour l'espacement sous le bord supérieur
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
        # NOUVEAU : Filtre par Type de Défaillance (Fault)
        html.Div([
            html.Label("Type de Défaillance (Fault)"),
            dcc.Dropdown(id='fault-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        html.Div([
            html.Label("Intervalle d'années"),
            dcc.RangeSlider(
                id='year-slider',
                step=1, # Ajout de step=1 pour s'assurer que seuls les entiers sont sélectionnés et affichés
                # Ajout du tooltip pour afficher la plage sélectionnée
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], className="control-group"),
        html.Button("Réinitialiser les filtres", id="reset-button", n_clicks=0, className="reset-button")
    ]),

    # dcc.Store pour stocker les données traitées
    dcc.Store(id='processed-data', data={}),

    # dcc.Loading enveloppe le contenu principal qui dépend des données traitées
    dcc.Loading(
        id="loading-graph",
        type="circle",   # Vous pouvez choisir 'graph', 'cube', 'dot', etc.
        children=html.Div(id='main-interface', children=html.Div("Veuillez charger un fichier CSV."))
    )
])

# Callback pour gérer le téléchargement de fichier et stocker les données traitées
@app.callback(
    Output('processed-data', 'data'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def handle_uploaded_file(contents):
    if contents is None:
        return no_update # Aucune mise à jour si aucun contenu

    content_type, content_string = contents.split(',')
    # Correction : base64.b64decode attend des octets, pas une chaîne. Encoder content_string en octets d'abord.
    decoded = base64.b64decode(content_string.encode('utf-8')).decode('utf-8')

    try:
        # Traiter les données CSV en utilisant la fonction importée
        context = process_csv_data(decoded)
        # Convertir le contexte (qui peut contenir des DataFrames) dans un format sérialisable en JSON
        context_serialable = convert_context_to_json_serialable(context)
        return context_serialable
    except Exception as e:
        print(f"Erreur de traitement du fichier : {e}")
        # Retourner des données vides et un message d'alerte si le traitement échoue
        return {}

# Callback pour afficher l'interface principale et mettre à jour les options de filtre en fonction des données traitées
@app.callback(
    Output('main-interface', 'children'),
    Output('model-select', 'options'),
    Output('model-select', 'value'),
    Output('kv-select', 'options'),
    Output('kv-select', 'value'),
    Output('mfg-select', 'options'),
    Output('mfg-select', 'value'),
    Output('fault-select', 'options'), # NOUVELLE SORTIE
    Output('fault-select', 'value'),   # NOUVELLE SORTIE
    Output('year-slider', 'min'),
    Output('year-slider', 'max'),
    Output('year-slider', 'value'),
    Output('year-slider', 'marks'),
    Output('upload-section-wrapper', 'style'), # Nouvelle sortie pour la visibilité de la section de téléchargement
    Input('processed-data', 'data')
)
def display_main_interface(data):
    # Valeurs par défaut pour le RangeSlider quand aucune donnée n'est traitée
    default_year_min = 1900
    default_year_max = 2000
    default_year_value = [default_year_min, default_year_max]
    # Générer des marques par défaut pour toutes les 10 années
    default_year_marks = {i: str(i) for i in range(default_year_min, default_year_max + 1, 10)}

    if not data:
        # Si aucune donnée n'est traitée, afficher le message initial et la section de téléchargement
        return (
            html.Div("Aucune donnée traitée. Veuillez uploader un fichier CSV."),
            [], None, [], "all", [], "all",
            [], "all", # Valeurs par défaut pour fault-select
            default_year_min, default_year_max, default_year_value, default_year_marks, # Utiliser les valeurs par défaut sécurisées
            {'display': 'block'} # Afficher la section de téléchargement
        )

    # Reconstruire les DataFrames à partir des chaînes JSON
    df = pd.read_json(data['df'], orient='split')
    # Assurez-vous que 'Year Test' est un entier pour éviter les décimales
    df['Year Test'] = df['Year Test'].astype(int) 
    fault_labels = {int(k): v for k, v in data['fault_labels'].items()}
    best_model = data['best_temporal_model_name']
    available_models = list(data['all_model_predictions'].keys())
    
    # Remplir les options des listes déroulantes pour KV et MFG
    kvs = sorted(df["KV"].dropna().unique())
    kv_options = [{"label": "Tous", "value": "all"}] + [{"label": k, "value": k} for k in kvs]
    mfgs = sorted(df["MFG"].dropna().unique())
    mfg_options = [{"label": "Tous", "value": "all"}] + [{"label": m, "value": m} for m in mfgs]
    
    # NOUVEAU : Remplir les options pour le filtre de défaillance
    fault_options = [{"label": "Tous", "value": "all"}] + \
                    [{"label": label, "value": label} for label in sorted(fault_labels.values())]

    # Configurer la plage et les marques du curseur d'année
    year_min = int(df["Year Test"].min())
    year_max = int(df["Year Test"].max())
    
    # Générer des marques pour le curseur d'année
    # Afficher les marques pour chaque année si la plage est petite (par exemple, <= 10 ans)
    # Sinon, afficher les marques toutes les 5 ou 10 ans pour les plages plus grandes
    if year_max - year_min <= 10:
        marks = {i: str(i) for i in range(year_min, year_max + 1)}
    else:
        # Calculer un pas intelligent pour les marques
        step = (year_max - year_min) // 5
        if step == 0: # Éviter la division par zéro si la plage est très petite
            step = 1
        marks = {i: str(i) for i in range(year_min, year_max + 1, step)}
    # S'assurer que les années min et max sont toujours incluses dans les marques
    marks[year_min] = str(year_min)
    marks[year_max] = str(year_max)

    return (
        # Contenu de l'interface principale (graphique et métriques)
        html.Div([
            html.H3(id="graph-title"),
            dcc.Graph(id='main-graph'),
            html.Div(id='metrics-output', className="metrics-card")
        ], className="card graph-column"),
        # Options des listes déroulantes et valeurs initiales
        [{"label": m, "value": m} for m in available_models],
        best_model,
        kv_options,
        "all",
        mfg_options,
        "all",
        fault_options, # NOUVEAU : Retourne les options de défaillance
        "all",         # NOUVEAU : Définit la valeur initiale "all"
        # Min, max, valeur et marques du curseur d'année
        year_min,
        year_max,
        [year_min, year_max],
        marks,
        {'display': 'none'} # Masquer la section de téléchargement une fois les données traitées
    )

# Callback pour mettre à jour le graphique et les métriques en fonction des sélections de filtre
@app.callback(
    Output('main-graph', 'figure'),
    Output('metrics-output', 'children'),
    Output('graph-title', 'children'),
    Input('model-select', 'value'),
    Input('kv-select', 'value'),
    Input('mfg-select', 'value'),
    Input('fault-select', 'value'), # NOUVEL INPUT
    Input('year-slider', 'value'),
    Input('reset-button', 'n_clicks'),
    State('processed-data', 'data')
)
def update_graph(model_name, kv_selected, mfg_selected, fault_selected, year_range, reset_clicks, data):
    if not data:
        # Retourner un graphique vide et des messages si aucune donnée n'est disponible
        return go.Figure(), html.Div("Aucune donnée disponible."), ""

    # Reconstruire les DataFrames à partir des chaînes JSON
    df = pd.read_json(data['df'], orient='split')
    # Assurez-vous que 'Year Test' est un entier pour éviter les décimales
    df['Year Test'] = df['Year Test'].astype(int)
    all_model_predictions = {k: pd.read_json(v, orient='split') for k, v in data['all_model_predictions'].items()}
    metrics_df_temporal = pd.read_json(data['metrics_df_temporal'], orient='split')
    fault_labels = {int(k): v for k, v in data['fault_labels'].items()}
    best_model = data['best_temporal_model_name']

    ctx_triggered = ctx.triggered_id # Obtenir l'ID du composant qui a déclenché le callback

    # Réinitialiser les filtres si le bouton de réinitialisation a été cliqué
    if ctx_triggered == 'reset-button':
        kv_selected = 'all'
        mfg_selected = 'all'
        fault_selected = 'all' # NOUVEAU : Réinitialiser le filtre de défaillance
        year_range = [int(df['Year Test'].min()), int(df['Year Test'].max())]
        model_name = best_model

    # Appliquer les filtres au DataFrame principal
    filtered_df = df[df['Year Test'].between(*year_range)]
    if kv_selected != 'all':
        filtered_df = filtered_df[filtered_df['KV'] == kv_selected]
    if mfg_selected != 'all':
        filtered_df = filtered_df[filtered_df['MFG'] == mfg_selected]
    
    traces = []

    # NOUVEAU : Mapper les indices de défaillance aux noms de défaillance pour le filtrage
    df_with_fault_names = filtered_df.copy()
    df_with_fault_names['true_fault_name'] = df_with_fault_names['true_fault_index'].map(fault_labels)

    if not df_with_fault_names.empty:
        # Calculer les proportions réelles des défauts pour le tracé
        temp_real_props = df_with_fault_names.groupby('Year Test')['true_fault_name'].value_counts(normalize=True).unstack(fill_value=0)
        
        # NOUVEAU : Déterminer quelles défaillances tracer
        faults_to_plot_real = [fault_selected] if fault_selected != 'all' else sorted(fault_labels.values())

        # Ajouter les traces de données réelles au graphique
        for fault in faults_to_plot_real:
            if fault in temp_real_props.columns:
                yearly = temp_real_props[fault]
                traces.append(go.Scatter(x=yearly.index, y=yearly.values, name=f"Réel {fault}", mode="lines+markers"))

    # Ajouter les traces de données prédites si un modèle est sélectionné
    if model_name in all_model_predictions:
        preds = all_model_predictions[model_name]
        last_train_year = df['Year Test'].max()
        
        # Obtenir les proportions réelles pour la dernière année d'entraînement afin de connecter la prédiction en douceur
        # Nous filtrons d'abord le df pour la dernière année d'entraînement, puis les filtres KV/MFG/Fault
        last_year_filtered_data = df[df['Year Test'] == last_train_year]
        if kv_selected != 'all':
            last_year_filtered_data = last_year_filtered_data[last_year_filtered_data['KV'] == kv_selected]
        if mfg_selected != 'all':
            last_year_filtered_data = last_year_filtered_data[last_year_filtered_data['MFG'] == mfg_selected]

        initial_prediction_points = {}
        if not last_year_filtered_data.empty:
            initial_prediction_points = {
                fault_labels[idx]: prop
                for idx, prop in last_year_filtered_data['true_fault_index'].value_counts(normalize=True).items()
            }

        # NOUVEAU : Filtrer les prédictions en fonction de la défaillance sélectionnée
        preds_to_plot_df = preds.copy()
        if fault_selected != 'all':
            preds_to_plot_df = preds_to_plot_df[[fault_selected]] if fault_selected in preds_to_plot_df.columns else pd.DataFrame(index=preds_to_plot_df.index)

        for fault in preds_to_plot_df.columns:
            plot_preds = preds_to_plot_df[[fault]].copy()
            
            # Ajouter le point de la dernière année d'entraînement si ce n'est pas déjà là
            if last_train_year not in plot_preds.index:
                plot_preds.loc[last_train_year] = initial_prediction_points.get(fault, 0)
                plot_preds = plot_preds.sort_index()
            else:
                   # Définir la prédiction pour la dernière année d'entraînement sur la proportion réelle
                plot_preds.loc[last_train_year, fault] = initial_prediction_points.get(fault, 0)

            # Ajouter les traces de données prédites
            traces.append(go.Scatter(x=plot_preds.index, y=plot_preds[fault],
                                     name=f"{model_name} Prédit {fault}", mode="lines+markers",
                                     line=dict(dash='dash')))

    # Afficher les métriques de performance pour le modèle sélectionné
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

    # NOUVEAU : Construire le titre du graphique de manière plus dynamique
    title_parts = [f"Prédiction des proportions de défaillances par {model_name}"]
    if kv_selected != 'all':
        title_parts.append(f"KV: {kv_selected}")
    if mfg_selected != 'all':
        title_parts.append(f"MFG: {mfg_selected}")
    if fault_selected != 'all':
        title_parts.append(f"Type de Défaillance: {fault_selected}")
    
    title = ", ".join(title_parts)

    figure = {
        'data': traces,
        'layout': go.Layout(
            title=title,
            xaxis=dict(title="Année"),
            yaxis=dict(title="Proportion prédite"), # RANGES SUPPRIMÉS ICI
            hovermode='x unified',
            legend=dict(orientation="h", x=0.5, xanchor='center', y=-0.2) # Légende horizontale en bas
        )
    }

    return figure, metrics_text, title

# Callback pour basculer la visibilité de la barre latérale
@app.callback(
    Output('sidebar', 'className'),
    Input('hamburger-button', 'n_clicks'),
    State('sidebar', 'className'),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, current_class):
    if 'open' in current_class:
        return 'sidebar' # Fermer la barre latérale
    else:
        return 'sidebar open' # Ouvrir la barre latérale

if __name__ == '__main__':
    app.run(debug=True)