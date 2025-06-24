import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objs as go
import pandas as pd
import json
import base64
import io
from script import process_csv_data # Assurez-vous que ce script est bien présent et fonctionnel

# Ajoutez cette ligne pour Font Awesome
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css']

# Modifiez l'initialisation de Dash pour inclure les feuilles de style externes
# Pour changer le titre de la page de "Dash" à "4Cast", ajoutez l'argument `title="4Cast"`.
# Pour l'icône de favori (favicon), placez votre fichier `favicon.ico` dans un dossier nommé `assets`
# à la racine de votre projet Dash.
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets, title="4 Cast")
server = app.server

# Fonction d'aide pour convertir un DataFrame en format sérialisable JSON
def convert_context_to_json_serialable(context):
    result = {}
    # Si le contexte est un dictionnaire d'erreur, le retourner tel quel
    if context.get("error"):
        return context

    for key, value in context.items():
        if isinstance(value, pd.DataFrame):
            # Convertir DataFrame en chaîne JSON
            result[key] = value.to_json(date_format='iso', orient='split')
        elif isinstance(value, dict):
            # Gérer récursivement les dictionnaires qui peuvent contenir des DataFrames
            try:
                result[key] = {
                    k: v.to_json(date_format='iso', orient='split') if isinstance(v, pd.DataFrame) else v
                    for k, v in value.items()
                }
            except Exception:
                # Solution de repli pour les dictionnaires complexes non facilement sérialisables
                result[key] = value
        else:
            # Assigner directement d'autres types de valeurs
            result[key] = value
    return result

app.layout = html.Div([
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
            # C'est ici que le message d'erreur de téléchargement sera affiché
            html.Div(id='upload-error-message', style={'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'})
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
        # Filtre par CODETX
        html.Div([
            html.Label("CODETX"),
            dcc.Dropdown(id='codetx-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        html.Div([
            html.Label("MFG"),
            dcc.Dropdown(id='mfg-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        # Filtre par Type de Défaillance (Fault)
        html.Div([
            html.Label("Type de Défaillance (Fault)"),
            dcc.Dropdown(id='fault-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        html.Div([
            html.Label("Intervalle d'années"),
            dcc.RangeSlider(
                id='year-slider',
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], className="control-group"),
        html.Button("Réinitialiser les filtres", id="reset-button", n_clicks=0, className="reset-button")
    ]),

    # dcc.Store pour stocker les données traitées
    # Initialisé à None pour un état clair avant tout téléchargement
    dcc.Store(id='processed-data', data=None),
    dcc.Store(id='filtered-data-store', data=None), # Store pour les données filtrées à passer au tableau

    # Le conteneur principal de contenu, toujours présent pour que les ID soient valides
    dcc.Loading(
        id="loading-main-content", # ID du composant de chargement
        type="circle",
        children=html.Div(id='main-content-display', children=[ # Ce Div contient tout le contenu dynamique
            # Le message de bienvenue initial (visible par défaut, caché après chargement)
            html.Div(id="initial-welcome-message", children="Veuillez charger un fichier CSV pour commencer l'analyse.", className="card welcome-message"),
            
            # Conteneur pour le graphique, les métriques et les tableaux (caché par défaut, visible après chargement)
            html.Div(id='analysis-content', style={'display': 'none'}, children=[
                html.H3(id="graph-title", children=""),
                dcc.Graph(id='main-graph', figure=go.Figure()), # Graphique initialement vide
                html.Div(id='metrics-output', className="metrics-card"), # Conteneur des métriques
                html.Div(id='prediction-tables-output', className="prediction-tables-container") # Conteneur des tableaux de prédiction
            ])
        ])
    ),

    # --- Bouton Chatbot et Iframe Modale ---
    html.Div(id='chatbot-button', className='chatbot-button', n_clicks=0, children=[
        html.Div(className='chatbot-circle', children=[
            html.I(className="fas fa-robot")
        ])
    ]),
    html.Div(id='chatbot-modal', className='chatbot-modal', children=[
        html.Div(className='chatbot-modal-content', children=[
            html.Button("X", id="chatbot-close-button", className="chatbot-close-button"),
            html.Iframe(
                src="https://648b68375066d2c9e2.gradio.live/",
                width="850",
                height="450",
                style={'border': 'none'}
            )
        ])
    ])
    # --- FIN de la section Chatbot ---

])

# Callback pour gérer le téléchargement de fichier et stocker les données traitées
@app.callback(
    Output('processed-data', 'data'),
    Output('upload-error-message', 'children'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def handle_uploaded_file(contents):
    # Si aucun contenu n'est fourni (ex: l'utilisateur annule le dialogue de fichier)
    if contents is None:
        return no_update, "" # Ne met pas à jour le store et efface le message d'erreur précédent

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string.encode('utf-8')).decode('utf-8')

    context = process_csv_data(decoded)
    context_serialable = convert_context_to_json_serialable(context)

    if context_serialable.get("error"):
        # Si une erreur est présente, retourne un dictionnaire d'erreur dans 'processed-data'
        # et le message d'erreur dans 'upload-error-message'
        return context_serialable, context_serialable.get("message", "Erreur de traitement du fichier.")
    else:
        # Si tout est OK, retourne les données traitées et un message vide
        return context_serialable, ""


# Callback pour mettre à jour les options de filtre et la visibilité des sections de téléchargement/bienvenue
@app.callback(
    Output('model-select', 'options'),
    Output('model-select', 'value'),
    Output('codetx-select', 'options'),
    Output('codetx-select', 'value'),
    Output('mfg-select', 'options'),
    Output('mfg-select', 'value'),
    Output('fault-select', 'options'),
    Output('fault-select', 'value'),
    Output('year-slider', 'min'),
    Output('year-slider', 'max'),
    Output('year-slider', 'value'),
    Output('year-slider', 'marks'),
    Output('upload-section-wrapper', 'style'), # Cacher/afficher la section de téléchargement
    Output('initial-welcome-message', 'style'), # Cacher/afficher le message de bienvenue
    Output('analysis-content', 'style'), # Nouvelle sortie pour contrôler la visibilité du contenu d'analyse
    Input('processed-data', 'data')
)
def update_filter_options_and_visibility(data):
    # Valeurs par défaut pour le RangeSlider quand aucune donnée n'est traitée
    default_year_min = 1900
    default_year_max = 2000
    default_year_value = [default_year_min, default_year_max]
    # Générer des marques par défaut pour toutes les 10 années
    default_year_marks = {i: str(i) for i in range(default_year_min, default_year_max + 1, 10)}

    # Si 'data' est None (état initial avant tout chargement) ou contient une erreur
    if data is None or data.get("error"):
        # Retourne les options vides/par défaut pour les filtres et affiche les sections de téléchargement/bienvenue
        return (
            [], None, # model_select
            [], "all", # codetx_select
            [], "all", # mfg_select
            [], "all", # fault_select
            default_year_min, default_year_max, default_year_value, default_year_marks, # year_slider
            {'display': 'block'}, # upload-section-wrapper style (visible)
            {'display': 'block'},  # initial-welcome-message style (visible)
            {'display': 'none'} # analysis-content style (caché)
        )
    
    # Si les données sont valides : remplir les options des filtres et cacher les sections initiales
    df = pd.read_json(data['df'], orient='split')
    df['Year Test'] = df['Year Test'].astype(int)
    fault_labels = {int(k): v for k, v in data['fault_labels'].items()}
    best_model = data['best_temporal_model_name']
    available_models = list(data['all_model_predictions'].keys())

    # Remplir les options des listes déroulantes pour CODETX et MFG
    codetxs = sorted(df["CODETX"].dropna().unique())
    codetx_options = [{"label": "Tous", "value": "all"}] + [{"label": c, "value": c} for c in codetxs]
    mfgs = sorted(df["MFG"].dropna().unique())
    mfg_options = [{"label": "Tous", "value": "all"}] + [{"label": m, "value": m} for m in mfgs]

    # Remplir les options pour le filtre de défaillance
    fault_options = [{"label": "Tous", "value": "all"}] + \
                    [{"label": label, "value": label} for label in sorted(fault_labels.values())]

    # Configurer la plage et les marques du curseur d'année
    year_min = int(df["Year Test"].min())
    year_max = int(df["Year Test"].max())

    # Générer des marques pour le curseur d'année
    if year_max - year_min <= 10:
        marks = {i: str(i) for i in range(year_min, year_max + 1)}
    else:
        step = (year_max - year_min) // 5
        if step == 0:
            step = 1
        marks = {i: str(i) for i in range(year_min, year_max + 1, step)}
    marks[year_min] = str(year_min)
    marks[year_max] = str(year_max)

    return (
        [{'label': m, 'value': m} for m in available_models], # options pour model_select
        best_model, # valeur par défaut pour model_select
        codetx_options, "all", # options et valeur par défaut pour codetx_select
        mfg_options, "all", # options et valeur par défaut pour mfg_select
        fault_options, "all", # options et valeur par défaut pour fault_select
        year_min, year_max, [year_min, year_max], marks, # min, max, value, marks pour year_slider
        {'display': 'none'}, # upload-section-wrapper style (caché)
        {'display': 'none'},  # initial-welcome-message style (caché)
        {'display': 'block'} # analysis-content style (visible)
    )

# Callback pour mettre à jour le graphique, les métriques et les données filtrées du store
@app.callback(
    Output('main-graph', 'figure'),
    Output('metrics-output', 'children'),
    Output('graph-title', 'children'),
    Output('filtered-data-store', 'data'), # Sortie pour stocker les données filtrées du tableau
    Input('model-select', 'value'),
    Input('codetx-select', 'value'),
    Input('mfg-select', 'value'),
    Input('fault-select', 'value'),
    Input('year-slider', 'value'),
    Input('reset-button', 'n_clicks'),
    State('processed-data', 'data'), # Données traitées (non déclencheur)
    State('model-select', 'options'), # Pour obtenir le meilleur modèle par défaut pour la réinitialisation
    State('year-slider', 'min'), # Pour la réinitialisation
    State('year-slider', 'max')  # Pour la réinitialisation
)
def update_graph(model_name, codetx_selected, mfg_selected, fault_selected, year_range, reset_clicks, data, model_options, year_min_state, year_max_state):
    # Si aucune donnée n'est chargée ou si une erreur est présente, retourner des composants vides
    if data is None or data.get("error"):
        return go.Figure(), html.Div(""), "", None # Figure vide, métriques vides, titre vide, données tableau vides

    # Reconstruire les DataFrames à partir des chaînes JSON
    df = pd.read_json(data['df'], orient='split')
    df['Year Test'] = df['Year Test'].astype(int)
    all_model_predictions = {k: pd.read_json(v, orient='split') for k, v in data['all_model_predictions'].items()}
    metrics_df_temporal = pd.read_json(data['metrics_df_temporal'], orient='split')
    fault_labels = {int(k): v for k, v in data['fault_labels'].items()}
    
    # Obtenir le meilleur modèle par défaut pour la réinitialisation
    best_model_name_from_options = model_options[0]['value'] if model_options else None

    ctx_triggered = ctx.triggered_id # ID du composant qui a déclenché le callback

    # Réinitialiser les filtres si le bouton de réinitialisation a été cliqué
    if ctx_triggered == 'reset-button':
        codetx_selected = 'all'
        mfg_selected = 'all'
        fault_selected = 'all'
        year_range = [year_min_state, year_max_state] # Utiliser les valeurs min/max actuelles du slider
        model_name = best_model_name_from_options # Réinitialiser au modèle par défaut

    # Appliquer les filtres au DataFrame principal pour les données réelles
    filtered_df = df[df['Year Test'].between(*year_range)]
    if codetx_selected != 'all':
        filtered_df = filtered_df[filtered_df['CODETX'] == codetx_selected]
    if mfg_selected != 'all':
        filtered_df = filtered_df[filtered_df['MFG'] == mfg_selected]

    traces = []

    # Mapper les indices de défaillance aux noms de défaillance pour le filtrage
    df_with_fault_names = filtered_df.copy()
    df_with_fault_names['true_fault_name'] = df_with_fault_names['true_fault_index'].map(fault_labels)

    if not df_with_fault_names.empty:
        # Calculer les proportions réelles des défauts pour le tracé
        temp_real_props = df_with_fault_names.groupby('Year Test')['true_fault_name'].value_counts(normalize=True).unstack(fill_value=0)

        # Déterminer quelles défaillances tracer
        faults_to_plot_real = [fault_selected] if fault_selected != 'all' else sorted(fault_labels.values())

        # Ajouter les traces de données réelles au graphique
        for fault in faults_to_plot_real:
            if fault in temp_real_props.columns:
                yearly = temp_real_props[fault]
                traces.append(go.Scatter(x=yearly.index, y=yearly.values, name=f"Réel {fault}", mode="lines+markers"))

    # Préparer les données de prédiction pour le graphique et le tableau
    filtered_predictions_data = None # Initialiser à None

    if model_name in all_model_predictions:
        preds = all_model_predictions[model_name]
        last_real_year = df['Year Test'].max()

        # Pour le graphique : ajouter des points de connexion entre données réelles et prédictions
        preds_for_graph_df = preds.copy()
        if fault_selected != 'all':
            preds_for_graph_df = preds_for_graph_df[[fault_selected]] if fault_selected in preds_for_graph_df.columns else pd.DataFrame(index=preds_for_graph_df.index)

        initial_prediction_points = {}
        last_year_filtered_data = df[df['Year Test'] == last_real_year]
        if codetx_selected != 'all':
            last_year_filtered_data = last_year_filtered_data[last_year_filtered_data['CODETX'] == codetx_selected]
        if mfg_selected != 'all':
            last_year_filtered_data = last_year_filtered_data[last_year_filtered_data['MFG'] == mfg_selected]
        
        if not last_year_filtered_data.empty:
            initial_prediction_points = {
                fault_labels[idx]: prop
                for idx, prop in last_year_filtered_data['true_fault_index'].value_counts(normalize=True).items()
            }

        for fault in preds_for_graph_df.columns:
            plot_preds = preds_for_graph_df[[fault]].copy()
            # Ajouter le point de la dernière année réelle si ce n'est pas déjà là pour connecter les courbes
            if last_real_year not in plot_preds.index:
                plot_preds.loc[last_real_year] = initial_prediction_points.get(fault, 0)
                plot_preds = plot_preds.sort_index()
            else:
                plot_preds.loc[last_real_year, fault] = initial_prediction_points.get(fault, 0)
            
            # Ajouter les traces de données prédites au graphique
            traces.append(go.Scatter(x=plot_preds.index, y=plot_preds[fault],
                                     name=f"{model_name} Prédit {fault}", mode="lines+markers",
                                     line=dict(dash='dash')))
        
        # --- Préparer les données pour le tableau de prédiction ---
        # Filtrer les prédictions pour les années futures (après la dernière année réelle)
        preds_for_table_raw = preds[preds.index > last_real_year].copy()
        
        table_data = []
        # Obtenez les CODETX et MFG uniques du DataFrame original
        unique_transformer_ids = df[['CODETX', 'MFG']].drop_duplicates().to_dict('records')

        if not preds_for_table_raw.empty:
            for year in preds_for_table_raw.index:
                for fault_name in preds_for_table_raw.columns:
                    # Appliquer le filtre de défaillance pour le tableau
                    if fault_selected != 'all' and fault_name != fault_selected:
                        continue
                    
                    # Exclure les types de défaillance "Uncertain"
                    if fault_name == "Uncertain":
                        continue

                    probability = preds_for_table_raw.loc[year, fault_name] * 100

                    # Ajouter l'entrée seulement si la probabilité est significative (pas exactement 0)
                    if probability > 0.001: # Seuil pour éviter les probabilités nulles affichées
                        for item in unique_transformer_ids:
                            table_data.append({
                                'Modèle': model_name,
                                'CODETX+MFG': f"{item['CODETX']}+{item['MFG']}",
                                'Type de Défaillance (Fault)': fault_name,
                                'Année': year,
                                'Probabilité (%)': round(probability, 2)
                            })
        
        filtered_predictions_data = pd.DataFrame(table_data)
        if not filtered_predictions_data.empty:
            # Tri par probabilité décroissante
            filtered_predictions_data = filtered_predictions_data.sort_values(by='Probabilité (%)', ascending=False)
            filtered_predictions_data = filtered_predictions_data.to_json(orient='split', date_format='iso')
        else:
            filtered_predictions_data = None # Aucune prédiction significative à afficher
        # --- Fin de la préparation des données du tableau ---


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

    # Construire le titre du graphique de manière plus dynamique
    title_parts = [f"Prédiction des proportions de défaillances par {model_name}"]
    if codetx_selected != 'all':
        title_parts.append(f"CODETX: {codetx_selected}")
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
            yaxis=dict(title="Proportion prédite"),
            hovermode='x unified',
            legend=dict(orientation="h", x=0.5, xanchor='center', y=-0.2)
        )
    }

    return figure, metrics_text, title, filtered_predictions_data

# Callback pour générer les tableaux de prédiction
@app.callback(
    Output('prediction-tables-output', 'children'),
    Input('filtered-data-store', 'data')
)
def generate_prediction_tables(filtered_predictions_json):
    if filtered_predictions_json is None:
        return html.Div("Aucune prédiction future disponible pour les filtres sélectionnés ou en attente de données.", className="no-predictions-message")

    predictions_df = pd.read_json(filtered_predictions_json, orient='split')

    if predictions_df.empty:
        return html.Div("Aucune prédiction future disponible pour les filtres sélectionnés.", className="no-predictions-message")

    tables = []
    # Assumer que le DataFrame contient les prédictions pour le modèle sélectionné
    model_name_for_table = predictions_df['Modèle'].iloc[0] if not predictions_df.empty else "Modèle sélectionné"

    tables.append(
        html.Div([
            html.H4(f"Prédictions détaillées pour le modèle : {model_name_for_table}"),
            html.Div(
                dash.dash_table.DataTable(
                    id=f'table-{model_name_for_table.replace(" ", "-")}',
                    columns=[{"name": i, "id": i} for i in predictions_df.columns],
                    data=predictions_df.to_dict('records'),
                    sort_action='native', # Permet le tri par l'utilisateur
                    style_header={
                        'backgroundColor': '#0A2A5E',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'left'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_cell={
                        'padding': '8px',
                        'textAlign': 'left',
                        'minWidth': '100px', 'width': '120px', 'maxWidth': '180px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                    },
                    export_format='csv', # Option d'exportation
                    page_size=10, # Nombre de lignes par page
                ),
                className="prediction-table-wrapper"
            )
        ], className="prediction-table-card card")
    )
    return tables


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

# --- Callbacks pour le chatbot ---
@app.callback(
    Output('chatbot-modal', 'className'),
    Input('chatbot-button', 'n_clicks'),
    Input('chatbot-close-button', 'n_clicks'),
    State('chatbot-modal', 'className'),
    prevent_initial_call=True
)
def toggle_chatbot_modal(button_clicks, close_clicks, current_class):
    ctx_triggered = ctx.triggered_id

    # Si le bouton du chatbot est cliqué
    if ctx_triggered == 'chatbot-button':
        # Ajoute la classe 'chatbot-modal-open' pour afficher la modale
        return 'chatbot-modal chatbot-modal-open'
    # Si le bouton de fermeture est cliqué
    elif ctx_triggered == 'chatbot-close-button':
        # Retire la classe 'chatbot-modal-open' pour cacher la modale
        return 'chatbot-modal'
    return current_class # Garde l'état actuel si non déclenché

if __name__ == '__main__':
    app.run(debug=True) 