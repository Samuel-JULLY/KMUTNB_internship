import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objs as go
import pandas as pd
import json
import base64
import io
from script import process_csv_data # Ensure this script is present and functional

# Add this line for Font Awesome
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css']

# Modify Dash initialization to include external stylesheets
# To change the page title from "Dash" to "4Cast", add the `title="4Cast"` argument.
# For the favicon, place your `favicon.ico` file in an `assets` folder
# at the root of your Dash project.
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets, title="4Cast")
server = app.server

# Helper function to convert a DataFrame to JSON serializable format
def convert_context_to_json_serialable(context):
    result = {}
    # If the context is an error dictionary, return it as is
    if context.get("error"):
        return context

    for key, value in context.items():
        if isinstance(value, pd.DataFrame):
            # Convert DataFrame to JSON string
            result[key] = value.to_json(date_format='iso', orient='split')
        elif isinstance(value, dict):
            # Recursively handle dictionaries that may contain DataFrames
            try:
                result[key] = {
                    k: v.to_json(date_format='iso', orient='split') if isinstance(v, pd.DataFrame) else v
                    for k, v in value.items()
                }
            except Exception:
                # Fallback for complex non-easily serializable dictionaries
                result[key] = value
        else:
            # Directly assign other value types
            result[key] = value
    return result

# --- Layout for the initial choice screen ---
choice_layout = html.Div([
    html.H1("Welcome to 4Cast", style={'textAlign': 'center', 'marginTop': '100px', 'color': '#0A2A5E'}),
    html.Div([
        html.Button("TGA", id="btn-tga", n_clicks=0, className="choice-button", style={'backgroundColor': '#0A2A5E', 'color': 'white', 'padding': '15px 30px', 'fontSize': '1.2em', 'margin': '10px', 'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer'}),
        html.Button("Oil", id="btn-oil", n_clicks=0, className="choice-button", style={'backgroundColor': '#6c757d', 'color': 'white', 'padding': '15px 30px', 'fontSize': '1.2em', 'margin': '10px', 'border': 'none', 'borderRadius': '8px', 'cursor': 'not-allowed', 'opacity': '0.6'}),
        html.Button("Other", id="btn-other", n_clicks=0, className="choice-button", style={'backgroundColor': '#6c757d', 'color': 'white', 'padding': '15px 30px', 'fontSize': '1.2em', 'margin': '10px', 'border': 'none', 'borderRadius': '8px', 'cursor': 'not-allowed', 'opacity': '0.6'}),
    ], style={'textAlign': 'center', 'marginTop': '50px'}),
    html.Div(id='choice-output', style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '1.1em', 'color': 'red'}) # For messages if needed
])

# --- Layout for the main TGA application ---
tga_app_layout = html.Div([
    html.H1("DGA Fault Analysis and Prediction"),

    # Wrapper for the upload section - its visibility will be controlled
    html.Div(id='upload-section-wrapper', children=[
        html.Div([
            html.H2("Upload your CSV file"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                multiple=False
            ),
            # This is where the upload error message will be displayed
            html.Div(id='upload-error-message', style={'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'})
        ], className="card upload-section"),
    ]),

    html.Button("☰", id="hamburger-button", className="hamburger-icon"),

    html.Div(id="sidebar", className="sidebar", children=[
        html.H2("\n"), # New line for spacing under the top edge
        html.H2("Analysis Settings"),
        html.Div([
            html.Label("Model"),
            dcc.Dropdown(id='model-select', options=[], value=None, clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        # Filter by CODETX
        html.Div([
            html.Label("CODETX"),
            dcc.Dropdown(id='codetx-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        html.Div([
            html.Label("MFG"),
            dcc.Dropdown(id='mfg-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        # Filter by Fault Type
        html.Div([
            html.Label("Fault Type"),
            dcc.Dropdown(id='fault-select', options=[], value="all", clearable=False, className="custom-dropdown"),
        ], className="control-group"),
        html.Div([
            html.Label("Year Range"),
            dcc.RangeSlider(
                id='year-slider',
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], className="control-group"),
        html.Button("Reset Filters", id="reset-button", n_clicks=0, className="reset-button")
    ]),

    # dcc.Store to store processed data
    # Initialized to None for a clear state before any upload
    dcc.Store(id='processed-data', data=None),
    dcc.Store(id='filtered-data-store', data=None), # Store for filtered data to pass to the table

    # The main content container, always present so that IDs are valid
    dcc.Loading(
        id="loading-main-content", # Loading component ID
        type="circle",
        children=html.Div(id='main-content-display', children=[ # This Div contains all dynamic content
            # The initial welcome message (visible by default, hidden after load)
            html.Div(id="initial-welcome-message", children="Please upload a CSV file to start the analysis.", className="card welcome-message"),
            
            # Container for the graph, metrics, and tables (hidden by default, visible after load)
            html.Div(id='analysis-content', style={'display': 'none'}, children=[
                html.H3(id="graph-title", children=""),
                dcc.Graph(id='main-graph', figure=go.Figure()), # Initially empty graph
                html.Div(id='metrics-output', className="metrics-card"), # Metrics container
                html.Div(id='prediction-tables-output', className="prediction-tables-container") # Prediction tables container
            ])
        ])
    ),

    # --- Floating Action Buttons ---
    html.Div(id='floating-buttons-container', children=[
        html.Button(id='home-button', className='floating-button home-button', n_clicks=0, children=[
            html.I(className="fas fa-home")
        ]),
        html.Button(id='info-button', className='floating-button info-button', n_clicks=0, children=[
            html.I(className="fas fa-info")
        ]),
        html.Button(id='chatbot-button', className='floating-button chatbot-button', n_clicks=0, children=[
            html.I(className="fas fa-robot")
        ]),
    ]),
    
    # --- Info Modal ---
    html.Div(id='info-modal', className='info-modal', children=[
        html.Div(className='info-modal-content', children=[
            html.Button("X", id="info-close-button", className="info-close-button"),
            html.H3("Welcome to the 4Cast application!", className="info-title"),
            html.P("This application allows you to analyze DGA fault data and visualize predictions from various temporal models.", className="info-text"),
            html.H4("How to use the application:", className="info-subtitle"),
            html.P("1. Upload your CSV file: Click on the upload area and select your fault data file.", className="info-text"),
            html.P("2. Explore the filters: Once data is loaded, use the dropdown menus and year slider in the sidebar to refine your analysis by model, CODETX, MFG, or fault type.", className="info-text"),
            html.P("3. Interpret the graph: The graph displays actual fault proportions and the selected model's predictions. Solid lines represent actual data, dashed lines represent predictions.", className="info-text"),
            html.P("4. Consult the metrics: The metrics table summarizes the performance of the selected prediction model (Accuracy, Precision, Recall, etc.).", className="info-text"),
            html.P("5. View detailed predictions: Tables at the bottom of the page show future fault probabilities by model and transformer ID (CODETX+MFG).", className="info-text"),
            html.H4("Need help?", className="info-subtitle"),
            html.P("Use the chatbot button (robot icon) at the bottom right to ask the AI your questions.", className="info-text"),
            html.Div(className="info-footer", children=[
                html.P("Developed by [Your Name/Team]", className="info-footer-text"),
                html.P("Version 1.0", className="info-footer-text")
            ])
        ])
    ]),
    # --- END of Info section ---

    # --- Chatbot Modal Iframe ---
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
    # --- END of Chatbot section ---
])

# --- Main App Layout including dcc.Location and page content ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False), # Add dcc.Location to track URL
    html.Div(id='page-content') # This div will contain either choice_layout or tga_app_layout
])

# Callback to handle button clicks and redirect to TGA app or show message
@app.callback(
    Output('url', 'pathname'),
    Output('choice-output', 'children', allow_duplicate=True), # Allow duplicate outputs
    Input('btn-tga', 'n_clicks'),
    Input('btn-oil', 'n_clicks'),
    Input('btn-other', 'n_clicks'),
    prevent_initial_call=True
)
def navigate_to_app(n_tga, n_oil, n_other):
    ctx_triggered = ctx.triggered_id
    if ctx_triggered == 'btn-tga':
        return '/tga-app', "" # Redirect to /tga-app
    elif ctx_triggered == 'btn-oil' or ctx_triggered == 'btn-other':
        return no_update, "Cette section n'est pas encore disponible. Veuillez choisir 'TGA'."
    return no_update, ""


# NEW CALLBACK: To handle the Home button click
@app.callback(
    Output('url', 'pathname', allow_duplicate=True), # Allow duplicate outputs
    Input('home-button', 'n_clicks'),
    prevent_initial_call=True
)
def navigate_to_home(n_clicks):
    if n_clicks and n_clicks > 0:
        return '/' # Redirect to the root path
    return no_update


# Callback to display the correct layout based on the URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/tga-app':
        return tga_app_layout
    else:
        return choice_layout

# Callback to handle file upload and store processed data
@app.callback(
    Output('processed-data', 'data'),
    Output('upload-error-message', 'children'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def handle_uploaded_file(contents):
    # If no content is provided (e.g., user cancels the file dialog)
    if contents is None:
        return no_update, "" # Do not update the store and clear any previous error message

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string.encode('utf-8')).decode('utf-8')

    context = process_csv_data(decoded)
    context_serialable = convert_context_to_json_serialable(context)

    if context_serialable.get("error"):
        # If an error is present, return an error dictionary in 'processed-data'
        # and the error message in 'upload-error-message'
        return context_serialable, context_serialable.get("message", "File processing error.")
    else:
        # If everything is OK, return the processed data and an empty message
        return context_serialable, ""


# Callback to update filter options and visibility of upload/welcome sections
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
    Output('upload-section-wrapper', 'style'), # Hide/show upload section
    Output('initial-welcome-message', 'style'), # Hide/show welcome message
    Output('analysis-content', 'style'), # New output to control analysis content visibility
    Input('processed-data', 'data')
)
def update_filter_options_and_visibility(data):
    # Default values for the RangeSlider when no data is processed
    default_year_min = 1900
    default_year_max = 2000
    default_year_value = [default_year_min, default_year_max]
    # Generate default marks for every 10 years
    default_year_marks = {i: str(i) for i in range(default_year_min, default_year_max + 1, 10)}

    # If 'data' is None (initial state before any upload) or contains an error
    if data is None or data.get("error"):
        # Return empty/default options for filters and display upload/welcome sections
        return (
            [], None, # model_select
            [], "all", # codetx_select
            [], "all", # mfg_select
            [], "all", # fault_select
            default_year_min, default_year_max, default_year_value, default_year_marks, # year_slider
            {'display': 'block'}, # upload-section-wrapper style (visible)
            {'display': 'block'},  # initial-welcome-message style (visible)
            {'display': 'none'} # analysis-content style (hidden)
        )
    
    # If data is valid: populate filter options and hide initial sections
    df = pd.read_json(data['df'], orient='split')
    df['Year Test'] = df['Year Test'].astype(int)
    fault_labels = {int(k): v for k, v in data['fault_labels'].items()}
    best_model = data['best_temporal_model_name']
    available_models = list(data['all_model_predictions'].keys())

    # Populate dropdown options for CODETX and MFG
    codetxs = sorted(df["CODETX"].dropna().unique())
    codetx_options = [{"label": "All", "value": "all"}] + [{"label": c, "value": c} for c in codetxs]
    mfgs = sorted(df["MFG"].dropna().unique())
    mfg_options = [{"label": "All", "value": "all"}] + [{"label": m, "value": m} for m in mfgs]

    # Populate options for fault filter
    fault_options = [{"label": "All", "value": "all"}] + \
                    [{"label": label, "value": label} for label in sorted(fault_labels.values())]

    # Configure year slider range and marks
    year_min = int(df["Year Test"].min())
    year_max = int(df["Year Test"].max())

    # Generate marks for the year slider
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
        [{'label': m, 'value': m} for m in available_models], # options for model_select
        best_model, # default value for model_select
        codetx_options, "all", # options and default value for codetx_select
        mfg_options, "all", # options and default value for mfg_select
        fault_options, "all", # options and default value for fault_select
        year_min, year_max, [year_min, year_max], marks, # min, max, value, marks for year_slider
        {'display': 'none'}, # upload-section-wrapper style (hidden)
        {'display': 'none'},  # initial-welcome-message style (hidden)
        {'display': 'block'} # analysis-content style (visible)
    )

# Callback to update the graph, metrics, and filtered data in the store
@app.callback(
    Output('main-graph', 'figure'),
    Output('metrics-output', 'children'),
    Output('graph-title', 'children'),
    Output('filtered-data-store', 'data'), # Output to store filtered table data
    Input('model-select', 'value'),
    Input('codetx-select', 'value'),
    Input('mfg-select', 'value'),
    Input('fault-select', 'value'),
    Input('year-slider', 'value'),
    Input('reset-button', 'n_clicks'),
    State('processed-data', 'data'), # Processed data (not a trigger)
    State('model-select', 'options'), # To get the default best model for reset
    State('year-slider', 'min'), # For reset
    State('year-slider', 'max')  # For reset
)
def update_graph(model_name, codetx_selected, mfg_selected, fault_selected, year_range, reset_clicks, data, model_options, year_min_state, year_max_state):
    # If no data is loaded or if an error is present, return empty components
    if data is None or data.get("error"):
        return go.Figure(), html.Div(""), "", None # Empty figure, empty metrics, empty title, empty table data

    # Reconstruct DataFrames from JSON strings
    df = pd.read_json(data['df'], orient='split')
    df['Year Test'] = df['Year Test'].astype(int)
    all_model_predictions = {k: pd.read_json(v, orient='split') for k, v in data['all_model_predictions'].items()}
    metrics_df_temporal = pd.read_json(data['metrics_df_temporal'], orient='split')
    fault_labels = {int(k): v for k, v in data['fault_labels'].items()}
    
    # Get the default best model for reset
    best_model_name_from_options = model_options[0]['value'] if model_options else None

    ctx_triggered = ctx.triggered_id # ID of the component that triggered the callback

    # Reset filters if the reset button was clicked
    if ctx_triggered == 'reset-button':
        codetx_selected = 'all'
        mfg_selected = 'all'
        fault_selected = 'all'
        year_range = [year_min_state, year_max_state] # Use current min/max slider values
        model_name = best_model_name_from_options # Reset to default model

    # Apply filters to the main DataFrame for actual data
    filtered_df = df[df['Year Test'].between(*year_range)]
    if codetx_selected != 'all':
        filtered_df = filtered_df[filtered_df['CODETX'] == codetx_selected]
    if mfg_selected != 'all':
        filtered_df = filtered_df[filtered_df['MFG'] == mfg_selected]

    traces = []

    # Map fault indices to fault names for filtering
    df_with_fault_names = filtered_df.copy()
    df_with_fault_names['true_fault_name'] = df_with_fault_names['true_fault_index'].map(fault_labels)

    if not df_with_fault_names.empty:
        # Calculate actual fault proportions for plotting
        temp_real_props = df_with_fault_names.groupby('Year Test')['true_fault_name'].value_counts(normalize=True).unstack(fill_value=0)

        # Determine which faults to plot
        faults_to_plot_real = [fault_selected] if fault_selected != 'all' else sorted(fault_labels.values())

        # Add actual data traces to the graph
        for fault in faults_to_plot_real:
            if fault in temp_real_props.columns:
                yearly = temp_real_props[fault]
                traces.append(go.Scatter(x=yearly.index, y=yearly.values, name=f"Actual {fault}", mode="lines+markers"))

    # Prepare prediction data for graph and table
    filtered_predictions_data = None # Initialize to None

    if model_name in all_model_predictions:
        preds = all_model_predictions[model_name]
        last_real_year = df['Year Test'].max()

        # For the graph: add connection points between actual data and predictions
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
            # Add the last actual year's point if not already there to connect the curves
            if last_real_year not in plot_preds.index:
                plot_preds.loc[last_real_year] = initial_prediction_points.get(fault, 0)
                plot_preds = plot_preds.sort_index()
            else:
                plot_preds.loc[last_real_year, fault] = initial_prediction_points.get(fault, 0)
            
            # Add predicted data traces to the graph
            traces.append(go.Scatter(x=plot_preds.index, y=plot_preds[fault],
                                     name=f"{model_name} Predicted {fault}", mode="lines+markers",
                                     line=dict(dash='dash')))
        
        # --- Prepare data for the prediction table ---
        # Filter predictions for future years (after the last actual year)
        preds_for_table_raw = preds[preds.index > last_real_year].copy()
        
        table_data = []
        # Get unique CODETX and MFG from the original DataFrame
        unique_transformer_ids = df[['LOC', 'NAME', 'CODETX', 'MFG', 'SER', 'KV', 'MVA']].drop_duplicates().to_dict('records')


        if not preds_for_table_raw.empty:
            for year in preds_for_table_raw.index:
                for fault_name in preds_for_table_raw.columns:
                    # Apply fault filter for the table
                    if fault_selected != 'all' and fault_name != fault_selected:
                        continue
                    
                    # Exclude "Uncertain" fault types
                    if fault_name == "Uncertain":
                        continue

                    probability = preds_for_table_raw.loc[year, fault_name] * 100

                    # Add entry only if probability is significant (not exactly 0)
                    if probability > 0.001: # Threshold to avoid displaying zero probabilities
                        for item in unique_transformer_ids:
                            table_data.append({
                                'Model': model_name,
                                'LOC': item['LOC'],
                                'NAME': item['NAME'],
                                'CODETX': item['CODETX'],
                                'MFG': item['MFG'],
                                'SER': item['SER'],
                                'KV': item['KV'],
                                'MVA': item['MVA'],
                                'Fault Type': fault_name,
                                'Year': year,
                                'Probability (%)': round(probability, 2)
                            })
        
        filtered_predictions_data = pd.DataFrame(table_data)
        if not filtered_predictions_data.empty:
            # Sort by probability descending
            filtered_predictions_data = filtered_predictions_data.sort_values(by='Probability (%)', ascending=False)
            filtered_predictions_data = filtered_predictions_data.to_json(orient='split', date_format='iso')
        else:
            filtered_predictions_data = None # No significant predictions to display
        # --- End of table data preparation ---


    # Display performance metrics for the selected model
    m = metrics_df_temporal.loc[model_name]
    metrics_text = html.Div([
        html.H4("Performance Summary"),
        html.Ul([
            html.Li(f"Accuracy: {m.get('Accuracy', 'N/A'):.3f}"),
            html.Li(f"Precision: {m.get('Precision', 'N/A'):.3f}"),
            html.Li(f"Recall: {m.get('Recall', 'N/A'):.3f}"),
            html.Li(f"Specificity: {m.get('Specificity', 'N/A'):.3f}"),
            html.Li(f"F1-Score: {m.get('F1', 'N/A'):.3f}")
        ])
    ])

    # Build the graph title more dynamically
    title_parts = [f"Fault Proportion Prediction by {model_name}"]
    if codetx_selected != 'all':
        title_parts.append(f"CODETX: {codetx_selected}")
    if mfg_selected != 'all':
        title_parts.append(f"MFG: {mfg_selected}")
    if fault_selected != 'all':
        title_parts.append(f"Fault Type: {fault_selected}")
    title = ", ".join(title_parts)

    figure = {
        'data': traces,
        'layout': go.Layout(
            title=title,
            xaxis=dict(title="Year"),
            yaxis=dict(title="Predicted Proportion"),
            hovermode='x unified',
            legend=dict(orientation="h", x=0.5, xanchor='center', y=-0.2)
        )
    }

    return figure, metrics_text, title, filtered_predictions_data

# Callback to generate prediction tables
@app.callback(
    Output('prediction-tables-output', 'children'),
    Input('filtered-data-store', 'data')
)
def generate_prediction_tables(filtered_predictions_json):
    if filtered_predictions_json is None:
        return html.Div("No future predictions available for the selected filters or awaiting data.", className="no-predictions-message")

    predictions_df = pd.read_json(filtered_predictions_json, orient='split')

    if predictions_df.empty:
        return html.Div("No future predictions available for the selected filters.", className="no-predictions-message")

    tables = []
    # Assume the DataFrame contains predictions for the selected model
    model_name_for_table = predictions_df['Model'].iloc[0] if not predictions_df.empty else "Selected Model"

    tables.append(
        html.Div([
            html.H4(f"Detailed predictions for the model: {model_name_for_table}"),
            html.Div(
                dash.dash_table.DataTable(
                    id=f'table-{model_name_for_table.replace(" ", "-")}',
                    columns=[{"name": i, "id": i} for i in predictions_df.columns],
                    data=predictions_df.to_dict('records'),
                    sort_action='native', # Allows user sorting
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
                    export_format='csv', # Export option
                    page_size=10, # Number of rows per page
                ),
                className="prediction-table-wrapper"
            )
        ], className="prediction-table-card card")
    )
    return tables


# Callback to toggle sidebar visibility
@app.callback(
    Output('sidebar', 'className'),
    Input('hamburger-button', 'n_clicks'),
    State('sidebar', 'className'),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, current_class):
    if 'open' in current_class:
        return 'sidebar' # Close sidebar
    else:
        return 'sidebar open' # Open sidebar

# --- Callbacks for the chatbot ---
@app.callback(
    Output('chatbot-modal', 'className'),
    Input('chatbot-button', 'n_clicks'),
    Input('chatbot-close-button', 'n_clicks'),
    State('chatbot-modal', 'className'),
    prevent_initial_call=True
)
def toggle_chatbot_modal(button_clicks, close_clicks, current_class):
    ctx_triggered = ctx.triggered_id

    # If the chatbot button is clicked
    if ctx_triggered == 'chatbot-button':
        # Add 'chatbot-modal-open' class to display the modal
        return 'chatbot-modal chatbot-modal-open'
    # If the close button is clicked
    elif ctx_triggered == 'chatbot-close-button':
        # Remove 'chatbot-modal-open' class to hide the modal
        return 'chatbot-modal'
    return current_class # Keep current state if not triggered

# --- Callbacks for the info modal ---
@app.callback(
    Output('info-modal', 'className'),
    Input('info-button', 'n_clicks'),
    Input('info-close-button', 'n_clicks'),
    State('info-modal', 'className'),
    prevent_initial_call=True
)
def toggle_info_modal(button_clicks, close_clicks, current_class):
    ctx_triggered = ctx.triggered_id

    if ctx_triggered == 'info-button':
        # Add 'info-modal-open' class to display the modal
        return 'info-modal info-modal-open'
    elif ctx_triggered == 'info-close-button':
        # Remove 'info-modal-open' class to hide the modal
        return 'info-modal'
    return current_class # Keep current state if not triggered

if __name__ == '__main__':
    app.run(debug=True)