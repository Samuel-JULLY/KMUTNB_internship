import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objs as go
import pandas as pd
import json
import base64
import io
from script import process_csv_data

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
    dcc.Store(id='processed-data', data={}),

    # dcc.Loading enveloppe le contenu principal qui dépend des données traitées
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=html.Div(id='main-interface', children=html.Div("Veuillez charger un fichier CSV."))
    ),

    # --- Bouton Chatbot et Iframe Modale ---
    html.Div(id='chatbot-button', className='chatbot-button', n_clicks=0, children=[
        html.Div(className='chatbot-circle', children=[ # Ajout de children ici
            html.I(className="fas fa-robot") # C'est ici que l'icône est ajoutée
        ])
    ]),
    html.Div(id='chatbot-modal', className='chatbot-modal', children=[
        html.Div(className='chatbot-modal-content', children=[
            html.Button("X", id="chatbot-close-button", className="chatbot-close-button"),
            html.Iframe(
                src="https://valentin-obert-chatbot.hf.space/",
                width="850",
                height="450",
                style={'border': 'none'} # Correction ici: utilisez 'style' pour la bordure
            )
        ])
    ])
    # --- FIN de la section Chatbot ---

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
    decoded = base64.b64decode(content_string.encode('utf-8')).decode('utf-8')

    # Traiter les données CSV en utilisant la fonction importée
    # process_csv_data renverra un dictionnaire d'erreur si la validation échoue
    context = process_csv_data(decoded)
    # Convertir le contexte (qui peut contenir des DataFrames ou un message d'erreur)
    # dans un format sérialisable en JSON
    context_serialable = convert_context_to_json_serialable(context)
    return context_serialable


# Callback pour afficher l'interface principale et mettre à jour les options de filtre en fonction des données traitées
@app.callback(
    Output('main-interface', 'children'),
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

    # Vérifier si les données contiennent un message d'erreur (format de fichier incorrect, etc.)
    if not data or data.get("error"):
        error_message = data.get("message", "Aucune donnée traitée. Veuillez uploader un fichier CSV.")
        return (
            html.Div([
                html.H3("Erreur de Fichier", style={'color': 'red'}),
                html.P(error_message, style={'color': 'red', 'fontWeight': 'bold'}),
                html.P("Veuillez corriger votre fichier CSV et le recharger.")
            ], className="card error-message"), # Utiliser une classe pour styliser le message d'erreur
            [], None, # Options du modèle vides, valeur nulle
            [], "all", # Options CODETX vides, valeur "all"
            [], "all", # Options MFG vides, valeur "all"
            [], "all", # Options Fault vides, valeur "all"
            default_year_min, default_year_max, default_year_value, default_year_marks, # Valeurs par défaut du slider
            {'display': 'block'} # Afficher la section de téléchargement
        )

    # Reconstruire les DataFrames à partir des chaînes JSON
    df = pd.read_json(data['df'], orient='split')
    # Assurez-vous que 'Year Test' est un entier pour éviter les décimales
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
        # Contenu de l'interface principale (graphique et métriques)
        html.Div([
            html.H3(id="graph-title"),
            dcc.Graph(id='main-graph'),
            html.Div(id='metrics-output', className="metrics-card")
        ], className="card graph-column"),
        # Options des listes déroulantes et valeurs initiales
        [{'label': m, 'value': m} for m in available_models],
        best_model,
        codetx_options,
        "all",
        mfg_options,
        "all",
        fault_options,
        "all",
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
    Input('codetx-select', 'value'),
    Input('mfg-select', 'value'),
    Input('fault-select', 'value'),
    Input('year-slider', 'value'),
    Input('reset-button', 'n_clicks'),
    State('processed-data', 'data')
)
def update_graph(model_name, codetx_selected, mfg_selected, fault_selected, year_range, reset_clicks, data):
    # Si les données sont vides ou contiennent une erreur, retourner un graphique vide et des messages
    if not data or data.get("error"):
        return go.Figure(), html.Div("Aucune donnée disponible pour l'analyse. Veuillez charger un fichier CSV valide."), ""

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
        codetx_selected = 'all'
        mfg_selected = 'all'
        fault_selected = 'all'
        year_range = [int(df['Year Test'].min()), int(df['Year Test'].max())]
        model_name = best_model

    # Appliquer les filtres au DataFrame principal
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

    # Ajouter les traces de données prédites si un modèle est sélectionné
    if model_name in all_model_predictions:
        preds = all_model_predictions[model_name]
        last_train_year = df['Year Test'].max()

        # Obtenir les proportions réelles pour la dernière année d'entraînement afin de connecter la prédiction en douceur
        last_year_filtered_data = df[df['Year Test'] == last_train_year]
        if codetx_selected != 'all':
            last_year_filtered_data = last_year_filtered_data[last_year_filtered_data['CODETX'] == codetx_selected]
        if mfg_selected != 'all':
            last_year_filtered_data = last_year_filtered_data[last_year_filtered_data['MFG'] == mfg_selected]

        initial_prediction_points = {}
        if not last_year_filtered_data.empty:
            initial_prediction_points = {
                fault_labels[idx]: prop
                for idx, prop in last_year_filtered_data['true_fault_index'].value_counts(normalize=True).items()
            }

        # Filtrer les prédictions en fonction de la défaillance sélectionnée
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
