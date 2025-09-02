from dash import html, dcc, dash_table
from .constants import *



colors = {
    'background': '#e5e5e5',
    'text': '#1f1f1f',
    'accent': '#7a9e7e',         # Muted sage green
    'panel_bg': '#f2f2f2',
    'text_secondary': '#4d4d4d'
}



panel_style = {
    'flex': '0.95',
    'width': '40%',
    'color': colors['text'],
    'backgroundColor': colors['panel_bg'],
    'borderRadius': '8px',
    'boxShadow': '0 0 8px rgba(0,0,0,0.15)',
    'padding': '20px',
    'minHeight': '400px'
}

binding_prediction_layout = html.Div(
    style={'backgroundColor': colors['background'], 'padding': '20px', 'color': colors['text']},
    children=[

        html.Div(style={'display': 'flex', 'gap': '20px'}, children=[

            # Left Side: Input Form (shrunk a bit to pull middle panel left)




html.Div(
    style={'backgroundColor': colors['panel_bg']
, 'padding': '24px 28px', 'width': '300px','borderRadius': '8px',
'color': colors['text'], 'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'},
    children=[

        # Title
        html.H1('BATES-RBP', style={
            "color": colors["accent"],
            "fontWeight": "900",
            "fontSize": "2.8rem",
            "marginBottom": "8px"
        }),

        # Step-by-step instructions
        html.Div([
            html.P("1. Select a Machine Learning / Deep Learning method.", style={'margin': '6px 0', 'fontSize': '1rem'}),
            html.P("2. Choose a model for prediction.", style={'margin': '6px 0', 'fontSize': '1rem'}),
            html.P("3. Input a FASTA sequence or upload a FASTA file.", style={'margin': '6px 0', 'fontSize': '1rem'}),
            html.P("4. Click 'Run Job' to start the prediction.", style={'margin': '6px 0', 'fontSize': '1rem'}),
        ], style={'color': colors['text_secondary'], 'marginBottom': '24px'}),


        # Dropdowns for Method and Model
        html.Div(style={'display': 'flex', 'gap': '12px', 'marginBottom': '18px'}, children=[
            html.Div([
                html.Label('Select Method', style={"color": colors["text"], "fontWeight": "600", "fontSize": "1rem", "marginBottom": "4px"}),
                dcc.Dropdown(
                    options=[{'label': 'DeepCLIP', 'value': 'DeepCLIP'}, {'label': 'RBPNet', 'value': 'RBPNet'}],
                    id='method-dropdown',
                    placeholder='Choose method...',
                    style={'width': '160px', 'color': '#000', 'borderRadius': '6px'}
                )
            ]),

            html.Div([
                html.Label('Select Model', style={"color": colors["text"], "fontWeight": "600", "fontSize": "1rem", "marginBottom": "4px"}),
                dcc.Dropdown(
                    options=[{'label': m, 'value': m} for m in deep_models + rbp_models],
                    id='model-dropdown',
                    placeholder='Choose model...',
                    style={'width': '160px', 'color': '#000', 'borderRadius': '6px'}
                )
            ])
        ]),

        # Sequence input area
        html.Label('Input FASTA Sequence', style={"color": colors["text"], "fontWeight": "600", "fontSize": "1rem", "marginBottom": "6px"}),
        dcc.Textarea(
            id='fasta-input',
            value='>seq1\nUUCUCU\n>seq2\nAUCUCU',
            placeholder="Paste FASTA sequence here...",
            style={
                'width': '100%',
                'height': '110px',
                'padding': '12px',
                'borderRadius': '6px',
                'border': f'1.5px solid {colors["panel_bg"]}',
                'fontFamily': 'monospace',
                'fontSize': '0.9rem',
                'color': '#000',
                'resize': 'vertical'
            }
        ),

        # Buttons for submitting sequence or file upload
        html.Div(style={'display': 'flex', 'gap': '12px', 'marginTop': '14px', 'marginBottom': '24px'}, children=[
            html.Button(
                "Submit Sequence",
                id='submit-fasta',
                n_clicks=0,
                style={
                    'flex': '1',
                    'backgroundColor': colors['accent'],
                    'color': '#fff',
                    'border': 'none',
                    'borderRadius': '8px',
                    'padding': '12px 0',
                    'fontWeight': '700',
                    'fontSize': '1rem',
                    'cursor': 'pointer',
                    'boxShadow': '0 3px 6px rgba(0,0,0,0.3)',
                    'transition': 'background-color 0.3s ease',
                    'minWidth': '150px'

                }
            ),
            dcc.Upload(
                id='upload-fasta',
                children=html.Div(
                    'Select File',
                    style={
                        'flex': '1',
                        'textAlign': 'center',
                        'backgroundColor': colors['accent'],
                        'color': '#fff',
                        'borderRadius': '8px',
                        'padding': '12px 0',
                        'fontWeight': '700',
                        'fontSize': '1rem',
                        'cursor': 'pointer',
                        'userSelect': 'none',
                        'boxShadow': '0 3px 6px rgba(0,0,0,0.3)',
                        'transition': 'background-color 0.3s ease',
                        'minWidth': '150px'

                    }
                ),
                multiple=False,
                style={'flex': '1'}
            ),
        ]),

        # Upload status text
        html.Div(id='upload-status', children="No file uploaded yet.", style={"color": colors["text_secondary"], 'fontSize': '0.85rem', 'minHeight': '22px'}),

        # Status messages (validations etc)
        html.Div(id='fasta-status', style={'marginTop': '8px', 'color': '#7fff7f', 'fontWeight': '600', 'fontSize': '0.9rem'}),
        html.Div(id='fasta-feedback', style={"color": "tomato", "marginTop": "8px", 'fontWeight': '600', 'fontSize': '0.9rem'}),

        # Run Job button
        html.Button(
            "Run Job",
            id='run-job',
            n_clicks=0,
            style={
                'width': '100%',
                'backgroundColor': colors['accent'],
                'color': 'white',
                'padding': '14px 0',
                'fontSize': '1.2rem',
                'border': 'none',
                'borderRadius': '10px',
                'cursor': 'pointer',
                'fontWeight': '800',
                'transition': 'background-color 0.3s ease'
            }
        ),

        html.Br(),

        # Loading spinner & output
        dcc.Loading(
            id="loading-output",
            type="circle",
            children=html.Div(id='output-div', style={"color": colors["text"], "whiteSpace": "pre-wrap", "marginTop": "18px", "minHeight": "120px"})
        ),

        # Hidden stores & debug divs (keep as is)
        dcc.Store(id='selected-method-store', storage_type='session'),
        dcc.Store(id='selected-model-store'),
        dcc.Store(id='counter', data=0, storage_type='session'),
        dcc.Store(id='job-counter', data=0, storage_type='session'),
        dcc.Store(id='sequence-list', storage_type='session'),
        dcc.Store(id="structure-fasta", storage_type='session'),
        html.Div(id='clicked-row-output', style={'display': 'none'}),
        dcc.Store(id='selected-row-store'),
        html.Div(id='debug-selected-row', style={'display': 'none'}),
        dcc.Store(id='viral-genome-sequence-store'),
        dcc.Store(id='search-results-store', data=[]),
        dcc.Store(id='current-seq-id', storage_type='session'),  # or 'memory'
        dcc.Store(id='structure-loaded', data=False),
        dcc.Download(id="download-csv"),
        dcc.Store(id='job-id-store', data=False),
        dcc.Store(id='selected-context-store'),







    ]
),




html.Div(
    style={
        "display": "flex",              # flexbox for side-by-side layout
        "gap": "20px",                  # spacing between the two panels
        "justifyContent": "space-between",
        "alignItems": "flex-start",     # align to top
        "width": "100%",
    },
    children=[
        # Left Panel (Prediction Results)
        html.Div(
            style={
                "backgroundColor": colors["panel_bg"],
                "padding": "24px 14px",
                "width": "350px",
                "borderRadius": "8px",
                "color": colors["text"],
                "fontFamily": "Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
            },
            children=[
                html.H2(
                    "Prediction Results",
                    style={
                        "color": colors["text"],
                        "marginBottom": "0px",
                        "fontSize": "24px",
                    },
                ),
                # html.Div(
                #     id="no-results-message",
                #     children="No jobs yet. Submit a FASTA and click 'Run Job' to see results here.",
                #     style={
                #         "color": colors["text_secondary"],
                #         "marginTop": "0px",
                #         "fontStyle": "italic",
                #         "textAlign": "center",
                #         "paddingTop": "10px",
                #     },
                # ),
                dash_table.DataTable(
                    id="result-table",
                    columns=[
                        {"name": "JobID", "id": "JobID"},
                        {"name": "Protein", "id": "Protein"},
                        {"name": "SequenceID", "id": "SequenceID"},
                        {"name": "Sequence", "id": "Sequence"},
                        {"name": "Score", "id": "Score"},
                    ],
                    data=[],
                    row_selectable="single",
                    selected_rows=[],
                    style_table={
                        "overflowX": "auto",
                        "marginTop": "0px",
                        "margin": "0",
                        "padding": "0",
                        "width": "100%",
                    },
                    style_cell={
                        "textAlign": "left",
                        "padding": "0 4px",
                        "color": colors["text"],
                        "backgroundColor": colors["panel_bg"],
                        "fontFamily": "monospace",
                        "fontSize": "15px",
                        "whiteSpace": "pre-line",
                    },
                    style_header={
                        "backgroundColor": "#687899",
                        "fontWeight": "bold",
                        "color": "#f0f4f8",
                        "fontSize": "16px",
                    },
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#afb6d3"},
                        {
                            "if": {"state": "selected"},
                            "backgroundColor": "#f39c12",
                            "color": "black",
                        },
                    ],
                ),
                html.Div(
                    [
                        dcc.Input(
                            id="results-filename-input",
                            type="text",
                            placeholder="Enter file name...",
                            debounce=True,
                            style={"marginRight": "10px"},
                        ),
                        html.Button(
                            "Download Table",
                            id="download-results",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "backgroundColor": colors["accent"],
                                "color": "white",
                                "padding": "14px 0",
                                "fontSize": "1.2rem",
                                "border": "none",
                                "borderRadius": "10px",
                                "cursor": "pointer",
                                "fontWeight": "800",
                                "transition": "background-color 0.3s ease",
                            },
                        ),
                        dcc.Download(id="download-results-download"),

                    ]
                ),
            ],
        ),
        # Right Panel (Visualisation)
        html.Div(
            style={
                "flex": "0.95",
                "width": "40%",
                "color": colors["text"],
                "backgroundColor": colors["panel_bg"],
                "borderRadius": "8px",
                "boxShadow": "0 0 8px rgba(0,0,0,0.15)",
                "padding": "20px",
                "minHeight": "400px",
            },
            children=[
                html.H2(
                    "Visualisation",
                    style={
                        "color": colors["text"],
                        "marginBottom": "5px",
                        "fontSize": "24px",
                    },
                ),
                dcc.Tabs(
                    id="plot-tabs",
                    value="score-plot",
                    children=[
                        dcc.Tab(
                            label="Score Plot",
                            value="score-plot",
                            children=[
                                html.Div(
                                    id="score-plot-panel",
                                    style={"paddingTop": "10px"},
                                ),
                                html.Button(
                                    "Download Plot",
                                    id="plot_download",
                                    style={
                                        "marginTop": "10px",
                                        "backgroundColor": colors["accent"],
                                        "color": "white",
                                        "border": "none",
                                        "padding": "8px 16px",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                        "fontFamily": "monospace",
                                        "fontSize": "15px",
                                        "fontWeight": "600",
                                        "transition": "background-color 0.3s ease",
                                    },
                                ),
                            ],
                            style={
                                "fontFamily": "monospace",
                                "fontSize": "15px",
                                "padding": "10px 15px",
                                "color": colors["text"],
                                "backgroundColor": colors["panel_bg"],
                            },
                            selected_style={
                                "fontFamily": "monospace",
                                "fontSize": "16px",
                                "fontWeight": "bold",
                                "color": "#f0f4f8",
                                "backgroundColor": colors["accent"],
                                "borderRadius": "6px 6px 0 0",
                                "padding": "12px 15px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
                            },
                        ),
                        dcc.Tab(
                            label="Heatmap",
                            value="heatmap",
                            children=[html.Div(id="heatmap-panel", style={"paddingTop": "10px"})],
                            style={
                                "fontFamily": "monospace",
                                "fontSize": "15px",
                                "padding": "10px 15px",
                                "color": colors["text"],
                                "backgroundColor": colors["panel_bg"],
                            },
                            selected_style={
                                "fontFamily": "monospace",
                                "fontSize": "16px",
                                "fontWeight": "bold",
                                "color": "#f0f4f8",
                                "backgroundColor": colors["accent"],
                                "borderRadius": "6px 6px 0 0",
                                "padding": "12px 15px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
                            },
                        ),
                        dcc.Tab(
                            label="Sequence Logo",
                            value="logo",
                            children=[html.Div(id="logo-panel", style={"paddingTop": "10px"})],
                            style={
                                "fontFamily": "monospace",
                                "fontSize": "15px",
                                "padding": "10px 15px",
                                "color": colors["text"],
                                "backgroundColor": colors["panel_bg"],
                            },
                            selected_style={
                                "fontFamily": "monospace",
                                "fontSize": "16px",
                                "fontWeight": "bold",
                                "color": "#f0f4f8",
                                "backgroundColor": colors["accent"],
                                "borderRadius": "6px 6px 0 0",
                                "padding": "12px 15px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
                            },
                        ),
                    ],
                    style={
                        "marginBottom": "10px",
                        "fontFamily": "monospace",
                        "fontSize": "15px",
                        "color": colors["text"],
                        "backgroundColor": colors["panel_bg"],
                        "border": "1px solid #444",
                        "borderRadius": "8px",
                        "boxShadow": "inset 0 -1px 3px rgba(0,0,0,0.2)",
                    },
                    colors={
                        "border": "#444",
                        "primary": colors["accent"],
                        "background": colors["panel_bg"],
                    },
                ),
            ],
        ),
    ],
)

    



,



    #         dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
    #             dcc.Tab(label='Tab One', value='line-graph'),
    #             dcc.Tab(label='Tab Two', value='heatmap'),
    # ]),
    html.Div(id='tabs-content-example-graph')
        ]),  # end flex container

        dcc.Store(id='completed-jobs', data=[], storage_type='session'),
    ]
)









import dash_bio as dashbio
from dash import Dash, dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate
import dash_bio.utils.ngl_parser as ngl_parser

structure_layout = html.Div([
    html.Div([
        # Left half container that fills height and arranges content vertically
    html.Div([
    html.H3('Sequences and Visualisation Sections', style={
        'color': colors['text'],
        'fontFamily': 'monospace',
        'fontWeight': 'bold',
        'fontSize': '20px',
        'marginBottom': '16px'
    }),
    html.Div([
        html.P("1. Sequences are added to the table automatically after a prediction job or can be entered manually below.", style={'margin': '6px 0', 'fontSize': '1.1rem'}),
        html.P("2. To view a 3D RNA structure, generate the model and open it in the Structure Viewer on the right.", style={'margin': '6px 0', 'fontSize': '1.1rem'}),
        html.P("3. To search an RNA sequence within a viral genome, select a sequence and genome, then run the query below. Results will appear in the Query Results panel on the right.", style={'margin': '6px 0', 'fontSize': '1.1rem'}),
    ], style={'color': colors['text_secondary'], 'marginBottom': '24px'}),

    # Sequence input controls
    html.Div([
        html.Div([




    html.Div([




    ]),


            html.Div(["ID", "Sequence"], style={
                'display': 'flex',
                'gap': '10px',
                'width': '100%',
                'fontFamily': 'monospace',
                'fontWeight': 'bold',
                'fontSize': '15px',
                'color': colors['text'],
                'paddingBottom': '6px'
            }),

            # Inputs and Add Sequence button
            html.Div([
                dcc.Textarea(
                    id='fasta-id',
                    value='Eg.',
                    style={
                        'width': '20%',
                        'height': '28px',
                        'color': colors['text'],
                        'fontFamily': 'monospace',
                        'fontSize': '15px',
                        'padding': '6px',
                        'border': 'none',
                        'border': f'1px solid {colors["accent"]}',
                        'resize': 'vertical'
                    }
                ),
                dcc.Textarea(
                    id='fasta-sequence',
                    value='CGUUCACGA',
                    style={
                        'width': '60%',
                        'height': '28px',
                        'color': colors['text'],
                        'fontFamily': 'monospace',
                        'fontSize': '15px',
                        'padding': '6px',
                        'border': 'none',
                        'resize': 'vertical'
                    }
                ),
                html.Button(
                    "Add sequence",
                    id="add-sequence-button",
                    style={
                        'backgroundColor': colors['accent'],
                        'color': 'white',
                        'border': 'none',
                        'padding': '6px 14px',
                        'borderRadius': '6px',
                        'cursor': 'pointer',
                        'fontFamily': 'monospace',
                        'fontSize': '15px',
                        'fontWeight': '600',
                        'alignSelf': 'center'
                    }
                ),
                dcc.ConfirmDialog(
                    id='duplicate-id-alert',
                    message='This SequenceID already exists in the table!',
                )
            ], style={'display': 'flex', 'gap': '10px', 'width': '100%', 'marginBottom': '20px'}),
            
            
            html.Div(id='fasta-status2', style={'marginTop': '8px', 'color': "#ce0000", 'fontWeight': '600', 'fontSize': '0.9rem'}),

            # Sequence Table
            dash_table.DataTable(
                id="sequence-table",
                editable=False,
                row_deletable=True,
                columns=[
                    {"name": "SequenceID", "id": "SequenceID"},
                    {"name": "Sequence", "id": "Sequence"},
                    {"name": "Structure Generated?", "id": "boltz-status"},
                    {"name": "JobID", "id": "sequence-job-id"}
                ],
                data=[],
                row_selectable='single',
                selected_rows=[],
                style_table={
                    'overflowX': 'auto',
                    'maxHeight': '250px',
                    'width': '100%',
                    'marginBottom': '20px',
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'color': colors['text'],
                    'backgroundColor': colors['panel_bg'],
                    'fontFamily': 'monospace',
                    'fontSize': '15px',
                    'whiteSpace': 'pre-line'
                },
                style_header={
                    'backgroundColor': "#7088B7",
                    'fontWeight': 'bold',
                    'color': '#f0f4f8',
                    'fontSize': '16px'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': "#8790b8",
                    },
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': '#f39c12',
                        'color': 'black'
                    }
                ],
            ),

            # Genrate structure button
            html.Div([
                html.Button(
                    "Generate Structure",
                    id="structure-button",
                    style={
                        'backgroundColor': colors['accent'],
                        'color': 'white',
                        'border': 'none',
                        'padding': '8px 18px',
                        'borderRadius': '6px',
                        'cursor': 'pointer',
                        'fontFamily': 'monospace',
                        'fontSize': '15px',
                        'fontWeight': '600',
                        'marginRight': '12px',
                        'transition': 'background-color 0.3s ease'
                    }
                ),

            ], style={'display': 'flex', 'justifyContent': 'flex-start', 'marginBottom': '20px'}),


            # Structure Viewer placeholder (can hide/show based on callbacks)


            # Loading spinner for structure generation
            dcc.Loading(
                type="default",
                children=html.Div(id='structure-status-div', style={
                    'fontFamily': 'monospace',
                    'fontSize': '14px',
                    'marginTop': '10px',
                    'color': colors['text']
                }),
                color=colors['accent']
            )

            ], style={
                "display": "flex",
                "flexDirection": "column",
                "padding": "10px"
            }),

            html.Div([
                html.Label("Search this RNA sequence in either human or viral genome:", style={
                    'fontWeight': 'bold', 'marginBottom': '6px', 'fontFamily': 'monospace'
                }),
            dcc.RadioItems(
                id='human-or-viral',
                options=[
                    {"label": "Human", "value": "Human"},
                    {"label": "Viral", "value": "Viral"}
                ],
                value='Viral',  # default selection
                labelStyle={'display': 'inline-block', 'margin-right': '20px'}
            ),
                dcc.Dropdown(
                    id='viral-genome-dropdown',
                    placeholder='Choose genome...',
                    style={'width': '100%', 'marginBottom': '10px'}
                ),

            html.Button(
                'Search Selected RNA in Genome',
                id='search-sequence-button',
                n_clicks=0,
                disabled=True,  # initially disabled
                style={
                    'backgroundColor': colors['accent'],
                    'color': 'white',
                    'border': 'none',
                    'padding': '8px 18px',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontFamily': 'monospace',
                    'fontSize': '15px',
                    'fontWeight': '600',
                    'width': '100%'
                }
            ),

            html.Div(id='search-results-div', style={
                'marginTop': '10px',
                'fontFamily': 'monospace',
                'fontSize': '14px',
                'color': colors['text']
    })


            ], style={
                'marginTop': '10px',
                'padding': '12px',
                'borderTop': f'1px solid {colors["accent"]}',
                'marginBottom': '20px'
            }),



                        html.Div([
                    html.Label("Select a number:"),
                    html.Button("Expand Sequence", id="expand-button", n_clicks=0),

                    dcc.Slider(
                        id='my-slider',
                        min=0,
                        max=10,
                        step=1,  # increments of 1
                        value=0,  # initial value
                        marks={i: str(i) for i in range(16)},  # shows numbers on the slider
                        tooltip={"placement": "bottom", "always_visible": True}  # optional
                    ),




            ]),
    ])
], style={
    'backgroundColor': colors['panel_bg'],
    'padding': '20px',
    'color': colors['text'],
    'borderRadius': '8px',
    'boxShadow': '0 0 12px rgba(0,0,0,0.12)',
    'flex': '0 0 400px',
    'maxWidth': '400px',
    'minWidth': '300px',
    'marginRight': '10px',
    'height': '100%',
    'display': 'flex',
    'flexDirection': 'column',


})
        ,

        # Right half container with iframe
    # TABS: Structure Viewer and Query Results
html.Div([
    dcc.Tabs(
        id='structure-query-tabs',
        value='structure',
        children=[
            dcc.Tab(
                label='Structure Viewer',
                value='structure',
                children=[
                    html.Div([
                        # LEFT PANEL – Viewer Options
html.Div([
    html.H4("Viewer Options", style={
        'marginBottom': '16px',
        'fontFamily': 'monospace',
        'color': colors['text']
    }),

    # RNA selection
    html.Label("Select RNA(s) to Visualize", style={
        'fontFamily': 'monospace',
        'marginBottom': '6px',
        'color': colors['text']
    }),
    dcc.Dropdown(
        id='rna-selection',
        options=[],  # to be populated later
        multi=True,
        placeholder="Choose one or more structures...",
        style={'marginBottom': '20px'}
    ),

    html.Hr(style={'borderColor': colors['accent']}),

    # Side-by-side checkboxes
    html.Label("Select Molecular Representation", style={
        'fontFamily': 'monospace',
        'marginBottom': '6px',
        'color': colors['text']
    }),
    dcc.Dropdown(
        id = "molecule-representation",
        options=[
            {"label": "backbone", "value": "backbone"},
            {"label": "ball+stick", "value": "ball+stick"},
            {"label": "cartoon", "value": "cartoon"},
            {"label": "hyperball", "value": "hyperball"},
            {"label": "licorice", "value": "licorice"},
            {"label": "axes+box", "value": "axes+box"},
            {"label": "helixorient", "value": "helixorient"}
        ],
        multi=True,
        value=['cartoon'],
        placeholder="Choose one or more structures...",
        style={'marginBottom': '20px'}
        )
    ,

    html.Hr(style={'borderColor': colors['accent']}),

    # Colouring section
    html.Label("Colouring Options", style={
        'fontFamily': 'monospace',
        'marginBottom': '6px',
        'color': colors['text']
    }),
    dcc.RadioItems(
        id='background-colour',
        options=[{"label": s.capitalize(), "value": s} for s in ["black", "white"]
        ],
        value='white',
        labelStyle={'display': 'block', 'marginBottom': '6px'},
        inputStyle={"marginRight": "6px"},
        style={'marginBottom': '20px', 'fontFamily': 'monospace', 'color': colors['text']}
    ),

    html.Hr(style={'borderColor': colors['accent']}),

    html.Label("Multi-viewer", style={
        'fontFamily': 'monospace',
        'marginBottom': '6px',
        'color': colors['text']
    }),


    dcc.RadioItems(
        id="sidebyside",
        options=[
            {'label': 'sideByside', 'value': "True"},
            {'label': 'Independent', 'value': "False"},
        ],
        value="False",
        labelStyle={'display': 'block', 'marginBottom': '6px'},
        inputStyle={"marginRight": "6px"},
        style={'marginBottom': '20px', 'fontFamily': 'monospace', 'color': colors['text']}
        ),

        
    html.Hr(style={'borderColor': colors['accent']}),

    # Quality setting
    html.Label("Rendering Quality", style={
        'fontFamily': 'monospace',
        'marginBottom': '6px',
        'color': colors['text']
    }),
    dcc.RadioItems(
        id='render-quality',
        options=[
            {'label': 'Low', 'value': 'low'},
            {'label': 'Medium', 'value': 'medium'},
            {'label': 'High', 'value': 'high'}
        ],
        value='medium',
        labelStyle={'display': 'inline-block', 'marginRight': '12px'},
        inputStyle={"marginRight": "6px"},
        style={'marginBottom': '20px', 'fontFamily': 'monospace', 'color': colors['text']}
    ),

    html.Hr(style={'borderColor': colors['accent']}),

    # Download section

    html.Div([
        html.Label("Filename for Download", style={
            'fontFamily': 'monospace',
            'marginBottom': '6px',
            'color': colors['text']
        }),
        dcc.Input(
            id='filename-input',
            type='text',
            placeholder='e.g., structure_1',
            style={
                'width': '100%',
                'marginBottom': '12px',
                'padding': '6px',
                'fontFamily': 'monospace',
                'borderRadius': '4px',
                'border': '1px solid #ccc'
            }
        ),
        html.Button("Download", id='download-image', style={
            'width': '100%',
            'padding': '10px',
            'backgroundColor': colors['accent'],
            'color': 'white',
            'border': 'none',
            'borderRadius': '6px',
            'cursor': 'pointer',
            'fontFamily': 'monospace'
        }),


]),


], style={
    "width": "280px",
    "padding": "20px",
    "marginRight": "10px",
    "flex": "0 0 280px",
    "marginTop": "20px",
    "boxSizing": "border-box",
    "backgroundColor": colors['panel_bg'],
    'minHeight': '900px',
    "borderRadius": "10px",
    "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.1)"
}),


                        # RIGHT – Structure Viewer
                        html.Div(id="structure-viewer-container", style={
                            "flex": "1",
                            "height": "100%",
                            "border": f"1px solid {colors['accent']}",  # Accent color border
                            "borderRadius": "8px",
                            "overflow": "hidden",
                            "boxSizing": "border-box",
                            "backgroundColor": colors['background']
                        }),
                        html.Script("""
                            window.captureNGLImage = function(viewerId, filename) {
                                const stage = window[viewerId + '_viewer']?.stage;
                                if (!stage) {
                                    console.error('NGL stage not found.');
                                    return;
                                }
                                stage.makeImage({
                                    factor: 2,
                                    antialias: true,
                                    trim: true,
                                    transparent: false
                                }).then(function (blob) {
                                    const a = document.createElement('a');
                                    a.href = URL.createObjectURL(blob);
                                    a.download = filename.endsWith('.png') ? filename : filename + '.png';
                                    document.body.appendChild(a);
                                    a.click();
                                    document.body.removeChild(a);
                                });
                            }
                            """)

                    ], style={
                        "display": "flex",
                        "flexDirection": "row",
                        "height": "100%",
                        "width": "100%",
                        "backgroundColor": colors['background'],
                        "paddingTop": "20px"  # <-- Add this

                    })
                ],
                style={
                    'fontFamily': 'monospace',
                    'fontSize': '15px',
                    'padding': '10px 15px',
                    'color': colors['text']
                },
                selected_style={
                    'fontFamily': 'monospace',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'color': '#f0f4f8',
                    'backgroundColor': colors['accent'],  # Accent for selected tab
                    'borderRadius': '6px 6px 0 0',
                    'padding': '12px 15px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'
                }
            ),
            dcc.Tab(
                label='Query Results',
                value='query',
                children=[
                    html.Div(id="query-results-container", style={
                        "width": "100%",
                        "backgroundColor": colors['background'],
                        "color": colors['text']
                    })
                ],
                style={
                    'fontFamily': 'monospace',
                    'fontSize': '15px',
                    'padding': '10px 15px',
                    'color': colors['text']
                },
                selected_style={
                    'fontFamily': 'monospace',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'color': '#f0f4f8',
                    'backgroundColor': colors['accent'],  # Accent for selected tab
                    'borderRadius': '6px 6px 0 0',
                    'padding': '12px 15px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'
                }
            )
        ],
        style={
            'width': '100%',
            'fontFamily': 'monospace',
            'backgroundColor': colors['background'],
            'color': colors['text']
        }
    )
], style={
    "flex": "1",
    "display": "flex",
    "flexDirection": "column",
   # "overflow": "hidden",
    "backgroundColor": colors['background'],
    "color": colors['text'],
    'height': '100%',
    
})






    ], style={
        "display": "flex",
        "height": "100%"   # important for the container of left+right
    })

]
, style={"height": "100vh"}

)





rbp_layout = html.Div(
    children=[
        dash_table.DataTable(
            id="rbp-table",

            columns= [
                {"name": "Accession", "id": "Accession"},
                {"name": "Symbol", "id": "Symbol"},
                {"name": "Protein Name", "id": "Protein Name"},
                {"name": "Alt Names", "id": "Alt Names"},
                {"name": "NCBI ID", "id": "NCBI ID"},
                {"name": "Ensembl ID", "id": "Ensembl ID"},
                {"name": "Keywords", "id": "Keywords"},
                {"name": "Motifs", "id": "Motifs"},
                {"name": "Available Algorithm", "id": "Available Algorithm"},
                {"name": "GO Terms", "id": "GO Terms"},
            ]
,
            filter_action="native",  # <-- this enables column filtering

            data=table_data,
            page_size=8,
            style_table={
                "overflowX": "auto",
                "backgroundColor": colors['panel_bg'],
                "border": "none",
            },
            style_cell={
                "textAlign": "left",
                "padding": "8px 10px",
                "maxWidth": 180,
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
                "color": colors['text'],
                "backgroundColor": colors['panel_bg'],
                "borderBottom": f"1px solid {colors['background']}",
                "fontSize": "14px",
            },
            style_header={
                "backgroundColor": colors['background'],
                "color": colors['accent'],
                "fontWeight": "bold",
                "borderBottom": f"2px solid {colors['accent']}",
                "fontSize": "15px",
            },
            tooltip_data=[
                {
                    col: {"value": str(row[col]), "type": "markdown"}
                    for col in row.keys()
                } for row in table_data
            ],
            tooltip_duration=None,
        )
    ],
    style={
        "backgroundColor": colors['panel_bg'],
        "padding": "15px",
        "borderRadius": "10px",
    }
)





binding_prediction_div = html.Div(
    id='binding-tab',
    children=[binding_prediction_layout],
    style={'display': 'block'}  # visible by default
)


structure_div = html.Div(
    id='structure-tab',
    children=[structure_layout],
    style={'display': 'none'}  # hidden by default
)


rbp_div = html.Div(
    id='rbp-tab',
    children=[rbp_layout],
    style={'display': 'none'}  # hidden by default
)



# 2) The Tabs in your top‐level layout:
layout = html.Div([
    dcc.Tabs(id='tabs', value='binding-prediction', children=[
        dcc.Tab(label='Binding prediction', value='binding-prediction'),
        dcc.Tab(label='Structure prediction', value='structure-prediction'),
        dcc.Tab(label='RBP Catalogue', value='RBP-catalogue')

       # dcc.Tab(label='Sequence search', value='sequence-search')  # <--- Add this line
    ]),
    html.Div([
        binding_prediction_div,
        structure_div,
        rbp_div  # <--- Add this line
    ])
])

