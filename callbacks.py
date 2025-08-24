import dash
from dash import Input, Output, State, MATCH, ALL, ctx
from dash.exceptions import PreventUpdate
#from .helpers import *       # your helper functions
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from pathlib import Path
import uuid
import glob
    # your helper functions
import shutil
import os
import shutil
import subprocess
from pathlib import Path
import glob
import json
from Bio import SeqIO

import sys

# Standard libraries
import base64
import glob
import io
import json
import os
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from pprint import pprint

# 3rd Party
# Scientific / Data
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

# Plotly / Dash
import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, callback
from dash import dash_table
from dash.dependencies import MATCH, ALL, ALLSMALLER
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

# Bioinformatics
from Bio import SeqIO
from Bio.Seq import Seq
import gffutils
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.obo_parser import GODag

# Dash Bio
import dash_bio as dashbio
import dash_bio.utils.ngl_parser as ngl_parser
from dash_bio import NglMoleculeViewer

# Web
import flask
from flask import send_from_directory

from setuptools import setup





from .constants import *
from .helpers import *   

def register_callbacks(app):
    """Register all Dash callbacks"""



    @app.callback(
        Output('viral-genome-dropdown', 'options'),
        Output('viral-genome-dropdown', 'value'),  # reset value
        Input('human-or-viral', 'value')
    )
    def update_genome_dropdown(selection):
        if selection == 'Human':
            return human_chromosomes, None
        elif selection == 'Viral':
            return viral_genomes, None
        return [], None  # default empty

    @app.callback(
        Output('binding-tab', 'style'),
        Output('structure-tab', 'style'),
        Output('rbp-tab', 'style'),  # <--- Add this
        Input('tabs', 'value')
    )
    def display_tab(tab):
        return (
            {'display': 'block'} if tab == 'binding-prediction' else {'display': 'none'},
            {'display': 'block'} if tab == 'structure-prediction' else {'display': 'none'},
            {'display': 'block'} if tab == 'RBP-catalogue' else {'display': 'none'}  # <--- Add this
        )












    BASE_DIR = Path(__file__).parent  # folder of the current file
    FASTA_UPLOAD_DIR = BASE_DIR / "user_uploads"  # folder for user-uploaded FASTA




    ##use this for 2 


    @app.callback([
        Output('sequence-table', 'data', allow_duplicate=True),
        Output('duplicate-id-alert', 'displayed'),
        Output('fasta-status2', 'children')  # new output
    ],
    Input('add-sequence-button', 'n_clicks'),
    State('fasta-id', 'value'),
    State('fasta-sequence', 'value'),
    State('sequence-table', 'data'),
    State('sequence-table', 'columns'),
    prevent_initial_call=True
    )
    def add_sequence(n_clicks, fasta_id, fasta_sequence, existing_data, columns):
        if not fasta_id or not fasta_sequence:
            return dash.no_update, False, ""

        # Normalize and sanitize inputs
        fasta_id = fasta_id.strip().lstrip('>').replace(' ', '_')
        fasta_sequence = fasta_sequence.strip()

        for character in fasta_sequence:
            if character.upper() not in ["A", "U", "C", "G"]:
                return dash.no_update, False, "❌ Invalid sequence: only A, U, C, and G are allowed."

        if existing_data:
            for row in existing_data:
                if row.get('SequenceID') == fasta_id:
                    return dash.no_update, True, ""  # duplicate, but no error message in fasta-status2

        # Save FASTA
        FASTA_BOLTZ_DIR.mkdir(exist_ok=True, parents=True)


        new_path = FASTA_BOLTZ_DIR / f"{fasta_id}_b.fasta"


        with open(new_path, 'w') as f:
            f.write(f">{fasta_id}|rna\n{fasta_sequence}\n")


        # Add new row
        new_row = {col['id']: '' for col in columns}
        new_row['SequenceID'] = fasta_id
        new_row['Sequence'] = fasta_sequence
        new_row["boltz-status"] = "❌"

        updated_data = existing_data + [new_row] if existing_data else [new_row]

        return updated_data, False, ""  # success, clear any message










    @app.callback(
        Output('selected-method-store', 'data', allow_duplicate=True),
        Output('fasta-status', 'children', allow_duplicate=True),
        Output('output-div', 'children', allow_duplicate=True),
        Input('method-dropdown', 'value'),
        Input('selected-model-store', 'data'),
        prevent_initial_call=True
    )
    def handle_method_selection(selected_method, selected_model):
        # Clear uploaded FASTA files
        for file in FASTA_UPLOAD_DIR.iterdir():
            if file.is_file():
                file.unlink()  # safer Path method instead of os.remove

        return selected_method, "", ""  # Clear the message

















    @app.callback(
        Output('upload-status', 'children'),
        Input('upload-fasta', 'contents'),
        State('upload-fasta', 'filename'),
    )
    def handle_fasta_upload(list_of_contents, list_of_names):
        if list_of_contents is None:
            return "No files uploaded."

        paths = []
        for contents, name in zip(list_of_contents, list_of_names):
            if not name.lower().endswith('.fasta'):
                return "Error: {} is not a .fasta file.".format(name)
            path = save_uploaded_file(contents, name)
            paths.append(path)

        return html.Ul([
            html.Li("Saved: {}".format(p)) for p in paths
        ])






    # Base directory for the app
    BASE_DIR = Path(__file__).parent

    # Folder for user-uploaded FASTA files
    FASTA_UPLOAD_DIR = BASE_DIR / "user_uploads"
    FASTA_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # ensure it exists

    # Messages
    s1 = "FASTA saved:-)"
    e1 = "Error: FASTA format requires the first line to start with '>'"
    e2 = "Error: Sequence contains invalid characters. Only A, U, G, C are allowed."
    e3 = "Please select method first"
    fasta_outcomes = [e1, e2, e3]


    @app.callback(
        Output('fasta-status', 'children', allow_duplicate=True),
        Input('submit-fasta', 'n_clicks'),
        State('fasta-input', 'value'),
        State('selected-method-store', 'data'),
        prevent_initial_call=True
    )
    def save_user_fasta(n_clicks, fasta_text, selected_method):
        if n_clicks == 0:
            return ""
        
        if selected_method is None:
            return e3

        lines = fasta_text.strip().split('\n')
        if not lines[0].startswith('>'):
            return e1

        valid_chars = set("AUGC")
        processed_lines = []

        for line in lines:
            if line.startswith('>'):
                processed_lines.append(line.strip())  # header
                continue
            seq_line = line.strip().upper().replace("T", "U")  # T → U
            if any(char not in valid_chars for char in seq_line):
                return e2
            processed_lines.append(seq_line)

        # Rebuild normalized FASTA
        fasta_text_normalized = "\n".join(processed_lines)

        # Save file using Path
        filename = FASTA_UPLOAD_DIR / f"user_input_{uuid.uuid4().hex}.fasta"
        filename.write_text(fasta_text_normalized)

        return s1


    @app.callback(
        Output('fasta-status', 'children', allow_duplicate=True),
        Input('fasta-input', 'value'),
        Input('run-job', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_fasta_status(fasta_value, run_job_clicks):
        return ""







    @app.callback(
        [Output('model-dropdown', 'options'),
        Output('model-dropdown', 'value')],
        Input('selected-method-store', 'data'),
        State('model-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_model_dropdown_options(selected_method, current_value):
        if selected_method == "DeepCLIP":
            options = [{'label': m, 'value': m} for m in deep_models]
        elif selected_method == "RBPNet":
            options = [{'label': m, 'value': m} for m in rbp_models]
        else:
            options = []

        #Only reset value if it's not in the updated options
        model_values = [opt['value'] for opt in options]
        if current_value not in model_values:
            return options, None
        else:
            return options, current_value





    @app.callback(
        Output('selected-model-store', 'data'),
        Input('model-dropdown', 'value'),
        State('selected-method-store', 'data'),  # avoid circular triggering
        prevent_initial_call=True
    )
    def store_selected_model_path(selected_model, selected_method):
        if selected_model and selected_method:
            if selected_method == "DeepCLIP":
                model_path = DEEPCLIP_MODEL_DIR / f"{selected_model}.pkl"
                return str(model_path)  # convert Path to string for Dash storage
            elif selected_method == "RBPNet":
                model_path = RBPNet_MODEL_DIR / f"{selected_model}.h5"
                return str(model_path)

        return "Please select model(s)"








    def run_method(method, model_path, selected_model, output_path, job_id):
        BASE_DIR = Path(__file__).parent
        UPLOAD_DIR = BASE_DIR / "user_uploads"
        FASTA_DIR = BASE_DIR / "fasta_files"
        RESULTS_DIR = BASE_DIR / "results"

        # Ensure directories exist
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        FASTA_DIR.mkdir(exist_ok=True, parents=True)
        FASTA_BOLTZ_DIR.mkdir(exist_ok=True, parents=True)

        # Get latest uploaded FASTA
        fasta_files = sorted(UPLOAD_DIR.glob("*.fasta"), key=os.path.getmtime, reverse=True)
        if not fasta_files:
            return "No FASTA files found to run prediction.", []

        fasta_path = fasta_files[0]

        # Copy FASTA for this job
        fasta_copy_path = FASTA_DIR / f"job_{job_id}.fasta"
        shutil.copy(fasta_path, fasta_copy_path)

        # Write per-sequence FASTAs
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id = record.id
            seq = str(record.seq)
            new_path = FASTA_BOLTZ_DIR / f"{seq_id}_b.fasta"
            with open(new_path, "w") as f:
                f.write(f">{seq_id}|rna\n{seq}\n")

        if method.lower() == "deepclip":
            cmd = [
                "conda", "run", "-n", "deepclip_env",
                "python",
                "DeepCLIP.py",
                "--runmode", "predict",
                "-P", str(model_path),
                "--sequences", str(fasta_path.resolve()),
                "--predict_output_file", str(output_path)
            ]
            cwd = BASE_DIR.parent / "deepclip"  # DeepCLIP repo location

        elif method.lower() == "rbpnet":
            cmd = [
                    "conda", "run", "-n", "rbpnet_env",
                    "python", "-m", "rbpnet", "predict",
                    "-m", str(model_path),
                    str(fasta_path.resolve()),
                    "-o", str(output_path)
                ]
            cwd = None
        else:
            return f"Unknown method: {method}", []

        try:
            # Copy environment and remove LD_PRELOAD to avoid preloading errors
            env = os.environ.copy()
            env.pop("LD_PRELOAD", None)

            # Run the subprocess
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                env=env
            )

            # Parse sequences from the input FASTA
            parsed_sequences = [{"id": r.id, "seq": str(r.seq)} for r in SeqIO.parse(fasta_path, "fasta")]

            if result.returncode == 0:
                return f"{method.capitalize()} job complete.", parsed_sequences
            else:
                return f"{method.capitalize()} failed:\n{result.stderr}", []

        except Exception as e:
            return f"Error running {method.capitalize()}: {str(e)}", []
        BASE_DIR = Path(__file__).parent
        RESULTS_DIR = BASE_DIR / "results"
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)


    @app.callback(
        Output('output-div', 'children'),
        Output('job-counter', 'data'),
        Output('completed-jobs', 'data'),
        Output('sequence-list', 'data'),
        Input('run-job', 'n_clicks'),
        State('selected-method-store', 'data'),
        State('selected-model-store', 'data'),
        State('model-dropdown', 'value'),
        State('job-counter', 'data'),
        State('completed-jobs', 'data'),
        prevent_initial_call=True
    )
    def run_job(n_clicks, selected_method, selected_model_path, selected_model, job_counter, job_list):
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)

        matching_models = glob.glob(selected_model_path)
        if not matching_models:
            return f"Error: No model file found for pattern: {selected_model_path}", job_counter, job_list, []

        actual_model_path = matching_models[0]

        if selected_method.lower() in ["deepclip", "rbpnet"]:
            job_id = job_counter + 1
            ext = ".json" if selected_method.lower() == "deepclip" else ".txt"
            output_path = RESULTS_DIR / f"job_{job_id}_{selected_model}{ext}"

            result_message, parsed_sequences = run_method(
                selected_method.lower(),
                actual_model_path,
                selected_model,
                str(output_path),
                job_id
            )

            sequences = []

            if "complete" in result_message.lower():
                if selected_method.lower() == "deepclip":
                    with open(output_path, 'r') as f:
                        data = json.load(f)
                    for idx, pred in enumerate(data["predictions"], start=1):
                        seq_id = pred.get("id", f"Sequence_{idx}")
                        seq = pred.get("sequence", "")
                        sequences.append({'id': seq_id, 'seq': seq})

                elif selected_method.lower() == "rbpnet":
                    current_id = None
                    current_seq = []
                    with open(output_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith(">"):
                                if current_id is not None:
                                    sequences.append({'id': current_id, 'seq': ''.join(current_seq)})
                                current_id = line[1:].split()[0]
                                current_seq = []
                            else:
                                current_seq.append(line)
                        if current_id is not None:
                            sequences.append({'id': current_id, 'seq': ''.join(current_seq)})

                job_list.append({
                    'job_id': job_id,
                    'output_path': str(output_path),
                    'sequences': parsed_sequences,
                    'method': selected_method,
                    'model': selected_model
                })

                return f"Job {job_id}: {result_message}", job_id, job_list, sequences

            else:
                return result_message, job_counter, job_list, dash.no_update

        else:
            return f"Job {job_counter + 1}: {result_message}", job_counter, job_list, []



    @app.callback(
        Output('result-table', 'data'),
        Output('sequence-table', 'data', allow_duplicate=True),
        Input('completed-jobs', 'data'),
        State('sequence-table', 'data'),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def update_job_results(job_list, existing_sequences):


        table_data = []
        if existing_sequences is None:
            existing_sequences = []

        sequence_data = existing_sequences.copy()

        for job in job_list:
            job_id = job['job_id']
            sequences = job.get("sequences", [])
            model = job.get('model', 'Unknown model')
            method = job.get('method', 'Unknown').strip().lower()
            output_file = job.get('output_path')

            # Load DeepCLIP scores if available
            score_lookup = {}
            if method == "deepclip" and output_file and os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        data = json.load(f)
                        for p in data.get("predictions", []):
                            seq_id = p.get("id")
                            score = p.get("score")
                            if seq_id is not None and score is not None:
                                score_lookup[seq_id] = score
                except Exception as e:
                    print(f"Error loading DeepCLIP output for job {job_id}: {e}")

            for seq in sequences:
                sequence_id = seq.get('id', 'Unknown ID')
                sequence = seq.get('seq', 'N/A')
                score = score_lookup.get(sequence_id, "N/A") if method == "deepclip" else "N/A"

                table_data.append({
                    'JobID': job_id,
                    'Protein': model,
                    'SequenceID': sequence_id,
                    "Sequence": sequence,
                    "Score": score  # <-- New score column
                })

                if not any(d['SequenceID'] == sequence_id and d['Sequence'] == sequence for d in sequence_data):
                    sequence_data.append({
                        'SequenceID': sequence_id,
                        'Sequence': sequence,
                        "boltz-status": "❌",
                        "sequence-job-id": job_id
                    })

        return table_data, sequence_data






    @app.callback(
        Output("download-results-download", "data"),
        Input("download-results", "n_clicks"),
        State("result-table", "data"),
        State("results-filename-input", "value"),
        prevent_initial_call=True
    )
    def download_result_table(n_clicks, result_data, filename_input):
        if not result_data:
            return dcc.send_string("No results to download.", filename="empty.txt")

        df = pd.DataFrame(result_data)

        # Clean and set filename
        filename = filename_input.strip() if filename_input else "results"
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        return dcc.send_data_frame(df.to_csv, filename=filename, index=False)




















    @app.callback(
        Output('clicked-row-output', 'children'),  # You can replace this with anything
        Input('result-table', 'active_cell'),
        State('result-table', 'data'),
    )
    def on_row_click(active_cell, table_data):
        if active_cell:
            row_index = active_cell['row']
            clicked_row = table_data[row_index]
            return html.Div([
                html.P(f"Job ID: {clicked_row['JobID']}"),
                html.P(f"Protein: {clicked_row['Protein']}"),
                html.P(f"Sequence ID: {clicked_row['SequenceID']}"),
                html.P(f"Sequence: {clicked_row['Sequence'][:50]}...")  # Truncate for display
            ])
        return "Click on a row to see details."





    @app.callback(
        Output({'type': 'zip-download', 'job_id': MATCH}, 'data'),
        Input({'type': 'zip-button', 'job_id': MATCH}, 'n_clicks'),
        State('completed-jobs', 'data'),
        prevent_initial_call=True
    )
    def download_zip(n_clicks, job_list):
        job_id = ctx.triggered_id['job_id']
        print(f"Download button clicked for job {job_id}")

        job = next((j for j in job_list if j['job_id'] == job_id), None)
        if not job:
            raise PreventUpdate

        # Create zip file path
        plots_dir = os.path.join("plots", f"job_{job_id}")
        zip_path = os.path.join(plots_dir, "plots.zip")

        # Recreate the zip file every time (optional)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in os.listdir(plots_dir):
                if file.endswith(".png"):
                    file_path = os.path.join(plots_dir, file)
                    zipf.write(file_path, arcname=file)

        # Return file for download
        return dcc.send_file(zip_path)







    @app.callback(
        Output({'type': 'download-component', 'job_id': MATCH}, 'data'),
        Input({'type': 'results-button', 'job_id': MATCH}, 'n_clicks'),
        State('completed-jobs', 'data'),
        prevent_initial_call=True
    )
    def download_result(n_clicks, job_list):
        trigger = dash.callback_context.triggered[0]['prop_id']
        #job_id = eval(trigger.split('.')[0])['index']  # safe since it's controlled
        job_id = eval(trigger.split('.')[0])['job_id']  # corrected

        # Find matching job
        job = next((j for j in job_list if j['job_id'] == job_id), None)
        if job is None:
            raise dash.exceptions.PreventUpdate

        with open(job['output_path'], 'r') as f:
            content = f.read()

        filename = os.path.basename(job['output_path'])
        return dict(content=content, filename=filename)


    @app.callback(
        Output('heatmap-panel', 'children'),
        Input('plot-tabs', 'value'),
        Input('selected-row-store', 'data'),
        State('completed-jobs', 'data'),
        prevent_initial_call=True
    )
    def update_heatmap(tab_value, selected_row, job_list):

        if tab_value != 'heatmap' or not selected_row:
            raise dash.exceptions.PreventUpdate

        job_id = selected_row['job_id']
        job = next((j for j in job_list if j['job_id'] == job_id), None)
        if not job:
            return html.Div("Job not found.")

        method = job.get('method', 'Unknown')
        model = job.get('model', 'Unknown')
        output_file = job.get('output_path')
        seq_info = {s['id']: s['seq'] for s in job.get("sequences", [])}

        # === Data Parsing ===
        seq_ids = []
        weights_matrix = []

        if method.lower() == "deepclip":
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                predictions = data.get("predictions", [])
            except Exception as e:
                return html.Div(f"Error loading DeepCLIP file: {e}")

            if len(predictions) < 2:
                return html.Div("Need at least two sequences for a heatmap.")

            max_len = max(len(p["weights"]) for p in predictions)
            for p in predictions:
                seq_id = p["id"]
                padded_weights = p["weights"] + [np.nan] * (max_len - len(p["weights"]))
                seq_ids.append(seq_id)
                weights_matrix.append(padded_weights)

        elif method.lower() == "rbpnet":
            try:
                with open(output_file, "r") as f:
                    lines = f.readlines()
            except Exception as e:
                return html.Div(f"Error loading RBPNet file: {e}")

            entries = []
            i = 0
            while i < len(lines):
                if lines[i].startswith(">"):
                    title = lines[i][1:].split()[0]
                    sequence = lines[i + 1].strip()
                    ig_target = list(map(float, lines[i + 4].split()))
                    entries.append((title, sequence, ig_target))
                    i += 5
                else:
                    i += 1

            if len(entries) < 2:
                return html.Div("Need at least two sequences for a heatmap.")

            max_len = max(len(e[2]) for e in entries)
            for title, seq, ig_target in entries:
                padded = ig_target + [np.nan] * (max_len - len(ig_target))
                seq_ids.append(title)
                weights_matrix.append(padded)

        else:
            return html.Div(f"Heatmap not supported for method: {method}")

        # === Build Hover Text ===
        hover_text = []
        for i, seq_id in enumerate(seq_ids):
            sequence = seq_info.get(seq_id, "")
            row_scores = weights_matrix[i]
            row_hover = []
            for j, score in enumerate(row_scores):
                if j < len(sequence) and score is not None:
                    base = sequence[j]
                    row_hover.append(
                        f"Sequence: {seq_id}<br>Position: {j+1}<br>Residue: {base}<br>Score: {score:.3f}"
                    )
                else:
                    row_hover.append("No data")
            hover_text.append(row_hover)

        # === Plot with Plotly ===
        weights_array = np.array(weights_matrix, dtype=float)
        fig = go.Figure(data=go.Heatmap(
            z=weights_array,
            x=list(range(1, weights_array.shape[1] + 1)),
            y=seq_ids,
            colorscale="YlGnBu",
            colorbar=dict(title="Binding<br>Score"),
            hoverongaps=False,
            text=hover_text,
            hoverinfo="text"
        ))

        fig.update_layout(
            title=f"{method} Binding Heatmap — {model}",
            xaxis_title="Position",
            yaxis_title="Sequence ID",
            margin=dict(l=60, r=40, t=60, b=40),
            height=max(300, len(seq_ids) * 30),
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
        )

        # === Return clean panel ===
        return html.Div([
            html.H5("Binding Score Heatmap", style={"marginBottom": "5px"}),
            html.Div(f"{method} model: {model}", style={"color": "#666", "marginBottom": "10px"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False})
        ], style={
            "padding": "15px",
            "backgroundColor": "#f9f9f9",
            "borderRadius": "8px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
        })




    @app.callback(
        Output('logo-panel', 'children'),
        Input('plot-tabs', 'value'),
        Input('selected-row-store', 'data'),
        State('completed-jobs', 'data'),
        prevent_initial_call=True
    )
    def update_logo(tab_value, selected_row, job_list):

        if tab_value != 'logo' or not selected_row:
            raise dash.exceptions.PreventUpdate

        job_id = selected_row['job_id']
        job = next((j for j in job_list if j['job_id'] == job_id), None)
        if not job:
            return html.Div("Job not found.")

        method = job.get('method', 'Unknown')
        output_file = job.get('output_path')
        seq_info = {s['id']: s['seq'] for s in job.get("sequences", [])}

        sequences = []
        scores = []

        if method.lower() == "deepclip":
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                predictions = data.get("predictions", [])
            except Exception as e:
                return html.Div(f"Error loading DeepCLIP file: {e}")

            max_len = max(len(p["weights"]) for p in predictions)
            for p in predictions:
                seq_id = p["id"]
                seq = seq_info.get(seq_id)
                if not seq:
                    continue
                padded_weights = p["weights"] + [0.0] * (max_len - len(p["weights"]))
                padded_seq = seq + "-" * (max_len - len(seq))
                sequences.append(padded_seq.upper())
                scores.append(padded_weights)

        elif method.lower() == "rbpnet":
            try:
                with open(output_file, "r") as f:
                    lines = f.readlines()
            except Exception as e:
                return html.Div(f"Error loading RBPNet file: {e}")

            entries = []
            i = 0
            while i < len(lines):
                if lines[i].startswith(">"):
                    title = lines[i][1:].split()[0]
                    sequence = lines[i + 1].strip()
                    ig_target = list(map(float, lines[i + 4].split()))
                    entries.append((title, sequence, ig_target))
                    i += 5
                else:
                    i += 1

            max_len = max(len(e[2]) for e in entries)
            for title, seq, ig_target in entries:
                padded_weights = ig_target + [0.0] * (max_len - len(ig_target))
                padded_seq = seq + "-" * (max_len - len(seq))
                sequences.append(padded_seq.upper())
                scores.append(padded_weights)

        else:
            return html.Div(f"Sequence logo not supported for method: {method}")

        if len(sequences) < 2:
            return html.Div("Need at least two sequences for a logo.")

        # === Build weighted frequency matrix ===
        L = len(scores[0])
        bases = ['A', 'C', 'G', 'U']
        matrix = {b: [0.0] * L for b in bases}

        for seq, score in zip(sequences, scores):
            for i, (base, weight) in enumerate(zip(seq, score)):
                base = base.replace("T", "U")
                if base in bases:
                    matrix[base][i] += weight

        # Normalize
        df = pd.DataFrame(matrix)
        df.fillna(0.0, inplace=True)
        df_sum = df.sum(axis=1)
        df = df.div(df_sum, axis=0).fillna(0.0)

        # === Create sequence logo using logomaker ===
        fig_width = min(20, max(6, L * 0.3))
        fig, ax = plt.subplots(figsize=(fig_width, 2.5))
        logomaker.Logo(df, ax=ax)
        ax.set_ylabel("Relative Importance", fontsize=13.5, fontweight="bold")
        ax.set_xlabel("Position", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # No tight_layout() – instead use bbox_inches='tight'
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        return html.Img(src=img_src, style={'width': '100%', 'maxWidth': '1000px'})







    @app.callback(
        Output({'type': 'output-div', 'job_id': MATCH}, 'children'),
        Input({'type': 'view-plot-button', 'job_id': MATCH}, 'n_clicks'),
        State({'type': 'sequence-checklist', 'job_id': MATCH}, 'value'),
        State('completed-jobs', 'data'),
        prevent_initial_call=True
    )
    def plot_selected_sequences(n_clicks, selected_seqs, job_list):

        triggered = ctx.triggered_id
        job_id = triggered['job_id']

        if not selected_seqs:
            return html.Div("Please select at least one sequence before plotting.")

        job = next((j for j in job_list if j['job_id'] == job_id), None)
        if not job:
            return html.Div("Job data not found.")

        method = job.get('method', 'Unknown').strip().lower()
        model = job.get('model', 'Unknown')
        output_file = job['output_path']
        plots_dir = os.path.join("plots", f"job_{job_id}")
        os.makedirs(plots_dir, exist_ok=True)

        header = html.Div([
            html.P(f"Method: {method}"),
            html.P(f"Model: {model}")
        ], style={"fontWeight": "bold", "marginBottom": "10px"})

        output_elements = []

        if method == "deepclip":
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                return html.Div(f"Error loading DeepCLIP output: {e}")

            for selected_seq in selected_seqs:
                pred = next((p for p in data["predictions"] if p.get("id") == selected_seq), None)
                if not pred:
                    output_elements.append(html.Div(f"No match found for sequence: {selected_seq}"))
                    continue

                sequence = pred["sequence"]
                weights = pred["weights"]
                score = pred["score"]
                filename = f"{selected_seq}.png"
                filepath = os.path.join(plots_dir, filename)

                plt.figure(figsize=(6, 3))
                plt.plot(range(len(weights)), weights, marker='o', linestyle='-', color='mediumseagreen')
                plt.xticks(range(len(sequence)), list(sequence))
                plt.title(f"{selected_seq} (Score: {score:.3f}) - Method: {method}")
                plt.xlabel("Position")
                plt.ylabel("Weight")
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()
                image_url = f"/plots/job_{job_id}/{filename}"

                plot_div_id = {'type': 'plot-container', 'job_id': job_id, 'seq_id': selected_seq}

                output_elements.append(html.Div([
                    html.Div("×", id={'type': 'close-button', 'job_id': job_id, 'seq_id': selected_seq},
                            style={
                                'position': 'absolute',
                                'top': '5px',
                                'right': '10px',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'fontSize': '16px',
                                'color': '#900',
                            }),
                    html.P(f"Plot for {selected_seq}"),
                    html.Img(src=image_url, style={"width": "100%", "maxWidth": "700px"})
                ], id=plot_div_id, style={'position': 'relative', 'border': '1px solid #ccc', 'padding': '10px', 'marginBottom': '10px'}))

            return html.Div([header] + output_elements)  # ← Add this return




        # --- RBPNet ---
        if method.lower() == "rbpnet":
            print(job)
            print(output_file)
            if not output_file.endswith(".txt"):
                return html.Div("Invalid RBPNet output: expected a .txt file.")

            try:
                with open(output_file, "r") as f:
                    lines = f.readlines()
            except Exception as e:
                return html.Div(f"Error loading RBPNet output: {e}")

            # Parse all entries
            entries = []
            i = 0
            while i < len(lines):
                if lines[i].startswith('>'):
                    title = lines[i][1:].split()[0]
                    sequence = lines[i + 1].strip()
                    ig_total, ig_control, ig_target = [], [], []
                    ig_total = list(map(float, lines[i + 2].split()))
                    ig_control = list(map(float, lines[i + 3].split()))
                    ig_target = list(map(float, lines[i + 4].split()))

                    entries.append((title, sequence, ig_total, ig_control, ig_target))

                    i += 5 
                else:
                    i += 1

            if not all(len(arr) == len(sequence) for arr in [ig_total, ig_control, ig_target]):
                return html.Div(f"Mismatch between sequence and attribution lengths for {title}")


            # Plot only selected sequences
            for selected_seq in selected_seqs:
                entry = next((e for e in entries if e[0] == selected_seq), None)
                if not entry:
                    output_elements.append(html.Div(f"No match found for sequence: {selected_seq}"))
                    continue

                title, sequence, ig_total, ig_control, ig_target = entry
                filename = f"{selected_seq}.png"
                filepath = os.path.join(plots_dir, filename)

                positions = list(range(1, len(sequence) + 1))
                plt.figure(figsize=(8, 3))
                plt.plot(positions, ig_total, label='Total IG', color='blue', linewidth=2)
                plt.plot(positions, ig_control, label='Control IG', color='orange', linestyle='--')
                plt.plot(positions, ig_target, label='Target IG', color='green', linestyle='--')
                plt.xticks(positions, list(sequence), fontsize=9)
                plt.xlabel('Position (nt)')
                plt.ylabel('IG Attribution Score')
                plt.title(f'RBPNet IG Attribution: {title}')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()
                image_url = f"/plots/job_{job_id}/{filename}"

                image_url = f"/plots/job_{job_id}/{filename}"
                plot_div_id = {'type': 'plot-container', 'job_id': job_id, 'seq_id': selected_seq}

                output_elements.append(html.Div([
                    html.Div("×", id={'type': 'close-button', 'job_id': job_id, 'seq_id': selected_seq},
                            style={
                                'position': 'absolute',
                                'top': '5px',
                                'right': '10px',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'fontSize': '16px',
                                'color': '#900',
                            }),
                    html.P(f"Plot for {selected_seq}"),
                    html.Img(src=image_url, style={"width": "100%", "maxWidth": "700px"})
                ], id=plot_div_id, style={'position': 'relative', 'border': '1px solid #ccc', 'padding': '10px', 'marginBottom': '10px'}))

        else:
            return html.Div(f"Plotting not supported for method: {method}")

        return html.Div([header] + output_elements)









    @app.callback(
        Output({'type': 'plot-container', 'job_id': MATCH, 'seq_id': ALL}, 'style'),
        Input({'type': 'close-button', 'job_id': MATCH, 'seq_id': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def hide_plot(close_clicks):
        styles = []
        for n in close_clicks:
            if n:
                # Hide this div
                styles.append({'display': 'none'})
            else:
                # Keep showing
                styles.append({'display': 'block'})
        return styles





    @app.callback(
        Output('debug-selected-row', 'children'),
        Input('selected-row-store', 'data')
    )
    def display_selected_row(data):
        if not data:
            return "No row selected yet."
        return f"Selected Job: {data.get('job_id')}, Sequence: {data.get('seq_id')}"



    @app.callback(
        Output('selected-row-store', 'data'),
        Input('result-table', 'selected_rows'),
        State('result-table', 'data'),
        prevent_initial_call=True
    )
    def update_selected_row(selected_rows, table_data):
        if not selected_rows:
            return dash.no_update

        selected_index = selected_rows[0]
        selected_row = table_data[selected_index]

        return {
            'job_id': selected_row['JobID'],
            'seq_id': selected_row['SequenceID']
        }










    @app.callback(
        Output('score-plot-panel', 'children'),
        Input('selected-row-store', 'data'),
        State('completed-jobs', 'data'),
        prevent_initial_call=True
    )
    def update_score_plot_panel(selected_row, job_list):
        if not selected_row:
            return html.Div("No row selected.")

        job_id = selected_row['job_id']
        seq_id = selected_row['seq_id']

        job = next((j for j in job_list if j['job_id'] == job_id), None)
        if not job:
            return html.Div("Job not found.")

        method = job.get('method', 'unknown').lower()
        model = job.get('model', 'unknown')
        output_file = job.get('output_path')
        plots_dir = os.path.join("plots", f"job_{job_id}")
        os.makedirs(plots_dir, exist_ok=True)

        filename = f"{seq_id}.png"
        filepath = os.path.join(plots_dir, filename)
        image_url = f"/plots/job_{job_id}/{filename}"

        # ------------------
        # DeepCLIP
        # ------------------
        if method == "deepclip":
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                return html.Div(f"Error loading DeepCLIP output: {e}")

            pred = next((p for p in data["predictions"] if p.get("id") == seq_id), None)
            if not pred:
                return html.Div(f"No prediction found for sequence ID '{seq_id}'.")

            sequence = pred["sequence"]
            weights = pred["weights"]
            score = pred["score"]

            # Generate DeepCLIP plot


    # New Plotly graph block
            x = list(range(len(sequence)))
            y = weights
            tick_labels = list(sequence)

            trace = go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                line=dict(color='mediumseagreen'),
                marker=dict(size=6),
                hovertext=tick_labels,
                name=seq_id
            )

            layout = go.Layout(
                title=f"{seq_id} (Score: {score:.3f}) - Method: {method}",
                xaxis=dict(
                    title='Position',
                    tickmode='array',
                    tickvals=x,
                    ticktext=tick_labels,
                    tickangle=0
                ),
                yaxis=dict(title='Weight'),
                height=300,
                margin=dict(l=50, r=30, t=50, b=50)
            )

            fig = go.Figure(data=[trace], layout=layout)

            return html.Div([
                html.H4(f"{seq_id} - Score Plot"),
                html.P(f"Method: {method} | Model: {model} | Score: {score:.3f}"),
                dcc.Download(id={'type': 'score-download', 'index': seq_id}),
                dcc.Graph(figure=fig)
            ])

    

        # ------------------
        # RBPNet
        # ------------------
        elif method == "rbpnet":
            try:
                with open(output_file, "r") as f:
                    lines = f.readlines()
            except Exception as e:
                return html.Div(f"Error loading RBPNet output: {e}")

            entries = []
            i = 0
            while i < len(lines):
                if lines[i].startswith(">"):
                    title = lines[i][1:].split()[0]
                    sequence = lines[i + 1].strip()
                    ig_total = list(map(float, lines[i + 2].split()))
                    ig_control = list(map(float, lines[i + 3].split()))
                    ig_target = list(map(float, lines[i + 4].split()))
                    entries.append((title, sequence, ig_total, ig_control, ig_target))
                    i += 5
                else:
                    i += 1

            entry = next((e for e in entries if e[0] == seq_id), None)
            if not entry:
                return html.Div(f"No RBPNet prediction found for sequence ID '{seq_id}'.")

            title, sequence, ig_total, ig_control, ig_target = entry

            if not all(len(arr) == len(sequence) for arr in [ig_total, ig_control, ig_target]):
                return html.Div(f"Mismatch between sequence and attribution lengths for {seq_id}.")

            # Generate RBPNet plot
            positions = list(range(1, len(sequence) + 1))
            nucleotides = list(sequence)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=positions,
                y=ig_total,
                mode='lines',
                name='Total IG',
                line=dict(color='blue', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=positions,
                y=ig_control,
                mode='lines',
                name='Control IG',
                line=dict(color='orange', dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=positions,
                y=ig_target,
                mode='lines',
                name='Target IG',
                line=dict(color='green', dash='dash')
            ))

            fig.update_layout(
                title=f'RBPNet IG Attribution: {seq_id}',
                xaxis=dict(
                    title='Position (nt)',
                    tickmode='array',
                    tickvals=positions,
                    ticktext=nucleotides,
                    tickangle=0
                ),
                yaxis=dict(title='IG Attribution Score'),
                height=350,
                margin=dict(l=50, r=30, t=50, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            return html.Div([
                html.H4(f"{seq_id} - IG Attribution Plot"),
                html.P(f"Method: {method} | Model: {model}"),
                html.Button("Download Scores", id={'type': 'download-score-btn', 'index': seq_id}, n_clicks=0),
                dcc.Download(id={'type': 'score-download', 'index': seq_id}),
                dcc.Graph(figure=fig)
            ])
        # ------------------
        # Unsupported
        # ------------------
        else:
            return html.Div(f"Plotting not yet supported for method '{method}'.")






    @app.callback(
        Output({'type': 'score-download', 'index': MATCH}, 'data'),
        Input({'type': 'download-score-btn', 'index': MATCH}, 'n_clicks'),
        State('selected-row-store', 'data'),
        State('completed-jobs', 'data'),
        prevent_initial_call=True
    )
    def download_score_data(n_clicks, selected_row, job_list):
        if not selected_row:
            return

        job_id = selected_row['job_id']
        seq_id = selected_row['seq_id']

        job = next((j for j in job_list if j['job_id'] == job_id), None)
        if not job:
            return

        method = job.get('method', 'unknown').lower()
        output_file = job.get('output_path')

        # ---------------- DeepCLIP ----------------
        if method == "deepclip":
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                pred = next((p for p in data["predictions"] if p.get("id") == seq_id), None)
            except Exception:
                return

            if not pred:
                return

            df = pd.DataFrame({
                'Position': list(range(len(pred['sequence']))),
                'Nucleotide': list(pred['sequence']),
                'Score': pred['weights']
            })

        # ---------------- RBPNet ----------------
        elif method == "rbpnet":
            try:
                with open(output_file, "r") as f:
                    lines = f.readlines()
            except Exception:
                return

            entries = []
            i = 0
            while i < len(lines):
                if lines[i].startswith(">"):
                    title = lines[i][1:].split()[0]
                    sequence = lines[i + 1].strip()
                    ig_total = list(map(float, lines[i + 2].split()))
                    ig_control = list(map(float, lines[i + 3].split()))
                    ig_target = list(map(float, lines[i + 4].split()))
                    entries.append((title, sequence, ig_total, ig_control, ig_target))
                    i += 5
                else:
                    i += 1

            entry = next((e for e in entries if e[0] == seq_id), None)
            if not entry:
                return

            _, sequence, ig_total, ig_control, ig_target = entry
            df = pd.DataFrame({
                'Position': list(range(1, len(sequence)+1)),
                'Nucleotide': list(sequence),
                'IG_Total': ig_total,
                'IG_Control': ig_control,
                'IG_Target': ig_target
            })

        # -------------- Convert and Return --------------
        csv_string = df.to_csv(index=False)
        return dict(content=csv_string, filename=f"{method}_{seq_id}_scores.csv")

























    def run_boltz_structure(selected_seq_ids):
        """
        Run Boltz structure prediction for a list of sequence IDs.
        
        Each sequence should have a corresponding FASTA file in FASTA_DIR:
            fasta_files/{seq_id}_b.fasta

        Predicted structures will be stored in STRUCTURE_OUTPUT_DIR/{seq_id}/
        """
        env = os.environ.copy()
        completed = []
        failed = []

        for seq_id in selected_seq_ids:
            fasta_path = FASTA_BOLTZ_DIR / f"{seq_id}_b.fasta"
            if not fasta_path.exists():
                print(f"FASTA file not found for {seq_id}: {fasta_path}")
                failed.append(seq_id)
                continue

            output_dir = STRUCTURE_OUTPUT_DIR / seq_id
            output_dir.mkdir(exist_ok=True, parents=True)

            cmd = [
                "conda", "run", "-n", "boltz", "boltz",
                "predict",
                str(fasta_path),
                "--model", "boltz2",
                "--out_dir", str(output_dir),
                "--accelerator", "cpu"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(FASTA_DIR),
                    capture_output=True,
                    text=True,
                    env=env
                )

                if result.returncode == 0:
                    completed.append(seq_id)
                    print(f"Boltz completed for {seq_id}")
                else:
                    print(f"Boltz failed for {seq_id}:\n{result.stderr.strip()}")
                    failed.append(seq_id)

            except Exception as e:
                print(f"Exception running Boltz for {seq_id}: {str(e)}")
                failed.append(seq_id)

        summary = []
        if completed:
            summary.append(f"Completed: {', '.join(completed)}")
        if failed:
            summary.append(f"Failed: {', '.join(failed)}")

        return " | ".join(summary) if summary else "No valid inputs."







    @app.callback(
        Output('structure-status-div', 'children'),
        Output('sequence-table', 'data', allow_duplicate=True),  # <-- Add this line
        Input('structure-button', 'n_clicks'),
        State('sequence-table', 'selected_rows'),
        State('sequence-table', 'data'),
        prevent_initial_call=True
    )
    def generate_structure(n_clicks, sel_first, data_first):
        if not sel_first:
            return "No row selected.", data_first

        selected_index = sel_first[0]
        selected_row = data_first[selected_index]
        seq_id = selected_row['SequenceID']

        try:
            summary = run_boltz_structure([seq_id])
            
            # Set status emoji
            if "Completed" in summary:
                status_icon = "✅"
            else:
                status_icon = "❌"

            # Update the table data
            updated_data = data_first.copy()
            updated_data[selected_index]['boltz-status'] = status_icon

            return f"Structure generation complete for {seq_id}: {summary}", updated_data

        except Exception as e:
            return f"Error generating structure for {seq_id}: {str(e)}", data_first


    @app.callback(
        Output('structure-viewer-container', 'children'),
        Input('rna-selection', 'value'),
        Input('molecule-representation', 'value'),
        Input('background-colour', 'value'),
        Input('sidebyside', 'value'),
        Input('render-quality', 'value'),
        prevent_initial_call=True
    )
    def show_structure(rna_selection, selected_reps, colour, sidebyside, quality):
        if not rna_selection:
            return html.Div("No RNA selected.", style={'color': colors['text_secondary']})

        if not isinstance(rna_selection, list):
            rna_selection = [rna_selection]


        if not selected_reps:
            selected_reps = ['cartoon']

        if not quality:
            quality = 'medium'

        data_list = []

        for seq_id in rna_selection:
            cif_path = STRUCTURE_OUTPUT_DIR / seq_id / f"boltz_results_{seq_id}_b/predictions/{seq_id}_b/{seq_id}_b_model_0.cif"

            if not os.path.exists(cif_path):
                return html.Div(f"Structure file not found for sequence {seq_id}.", style={'color': 'red'})

            with open(cif_path, "r") as f:
                cif_content = f.read()
            
            sidebyside_bool = sidebyside == "True"

            data_list.append({
                'filename': f'{seq_id}_b_model_0.cif',
                'ext': 'cif',
                'selectedValue': f'{seq_id}_b_model_0.cif',
                'chain': 'ALL',
                'aaRange': 'ALL',
                'chosen': {'chosenAtoms': '', 'chosenResidues': ''},
                'color': 'red',
                'config': {'input': cif_content, 'type': 'text/plain'},
                'uploaded': False,
                'resetView': True
            })

        return html.Div([
            html.Div("Structure Viewer", style={'fontFamily': 'monospace', 'marginBottom': '6px'}),
            dashbio.NglMoleculeViewer(
                id='ngl-viewer-multi',
                data=data_list,
                molStyles={'representations': selected_reps,
                #"chosenAtomsRadius": 1,
                "molSpacingXaxis": 20,
                "sideByside": sidebyside_bool},
                stageParameters={
                    'backgroundColor': colour,
                    'quality': quality
                },
                height="600px"
            )
        ])









    #this one works with jobIDs
    @app.callback(
        Output('search-results-store', 'data'),
        Output('query-results-container', 'children'),
        Output("structure-query-tabs", "value"),
        Output('job-id-store', 'data'),
        Input('search-sequence-button', 'n_clicks'),
        State('sequence-table', 'derived_virtual_selected_rows'),
        State('sequence-table', 'data'),
        State('viral-genome-dropdown', 'value'),  # used for both viral and human
        State('search-results-store', 'data'),
        State('job-id-store', 'data'),
        State('human-or-viral', 'value'),
        prevent_initial_call=True
    )
    def append_search_result(n_clicks, selected_rows, table_data, genome_key, existing_results, job_id, genome_type):

        if not selected_rows or not genome_key:
            return existing_results, dash.no_update, dash.no_update, job_id

        query = table_data[selected_rows[0]]['Sequence'].upper()

        # -------------------
        # Human genome setup
        # -------------------
        human_paths = {f'chr{i}': FASTA_DIR / f"human/Homo_sapiens.GRCh38.dna.chromosome.{i}.fa" for i in range(1, 23)}
        human_paths.update({
            'chrX': FASTA_DIR / "human/Homo_sapiens.GRCh38.dna.chromosome.X.fa",
            'chrY': FASTA_DIR / "human/Homo_sapiens.GRCh38.dna.chromosome.Y.fa"
        })
        human_db_path = DB_DIR / "human/human_please.db"

        # -------------------
        # Viral genome setup
        # -------------------
        viral_paths = {
            'SARS_CoV_2': FASTA_DIR / "SARS_CoV_2.fna",
            'IAV': FASTA_DIR / "IAV.fna",
            'zika': FASTA_DIR / "zika.fna",
            'CHIKV': FASTA_DIR / "CHIKV.fna",
            'COV-OC43': FASTA_DIR / "COV-OC43.fna",
            'dengue': FASTA_DIR / "dengue.fna",
            'HIV': FASTA_DIR / "HIV.fna",
            'RV': FASTA_DIR / "RV.fna",
            'SINV': FASTA_DIR / "SINV.fna",
            'VEEV': FASTA_DIR / "VEEV.fna"
        }
        viral_labels = {
            'SARS_CoV_2': 'SARS-CoV-2',
            'IAV': 'Influenza A Virus (H1N1)',
            'zika': 'Zika Virus',
            'CHIKV': 'Chikungunya Virus',
            'COV-OC43': 'Coronavirus OC43',
            'dengue': 'Dengue Virus',
            'HIV': 'Human Immunodeficiency Virus',
            'RV': 'Rhinovirus',
            'SINV': 'Sindbis Virus',
            'VEEV': 'Venezuelan Equine Encephalitis Virus'
        }

        # -------------------
        # Select paths and DB based on genome type
        # -------------------
        if genome_type == "Human":
            genome_paths = human_paths
            db_path = human_db_path
            genome_labels = {k: k for k in human_paths.keys()}
        else:
            genome_paths = viral_paths
            db_path = DB_DIR / f"{genome_key}.db"
            genome_labels = viral_labels

        # Safety check
        if genome_key not in genome_paths:
            return existing_results, dash.no_update, dash.no_update, job_id

        # Load genome sequence
        with open(genome_paths[genome_key], 'r') as handle:
            records = list(SeqIO.parse(handle, "fasta"))
        full_seq = str(records[0].seq).upper().replace('T', 'U')

        # Load GFF database
        db = gffutils.FeatureDB(db_path)

        try:
            with open(genome_paths[genome_key], 'r') as handle:
                records = list(SeqIO.parse(handle, "fasta"))
            full_seq = str(records[0].seq).upper().replace('T', 'U')

            # Forward strand matches
            fwd_positions = [m.start() for m in re.finditer(f'(?={query})', full_seq)]
            # Reverse strand matches
            rev_query = str(Seq(query.replace('U', 'T')).reverse_complement()).replace('T', 'U')
            rev_positions = [m.start() for m in re.finditer(f'(?={rev_query})', full_seq)]

            if not fwd_positions and not rev_positions:
                new_result = {
                    "label": f"Sequence '{query}' in {genome_labels[genome_key]}",
                    "matches": [],
                    "no_match": True
                }
            else:
                context_len = 10
                matches = []
                bed_rows = []

                for pos, strand in [(p, '+') for p in fwd_positions] + [(p, '-') for p in rev_positions]:
                    start = max(0, pos - context_len)
                    end = pos + len(query) + context_len
                    context = full_seq[start:end]



    # Decide which seqid to use
                    seqid = records[0].id
                    if genome_type == "Human" and not seqid.startswith("chr"):
                        seqid = "chr" + seqid

                    # Now query features
                    features = list(
                        db.region(
                            seqid=seqid,
                            start=pos + 1,
                            end=pos + len(query),
                            featuretype=None
                        )
                    )


                    if features:
                        all_genes = set()
                        for f in features:
                            if 'gene' in f.attributes:
                                all_genes.add(f.attributes['gene'][0])
                            elif 'gene_id' in f.attributes:
                                all_genes.add(f.attributes['gene_id'][0])
                            elif 'gene_name' in f.attributes:
                                all_genes.add(f.attributes['gene_name'][0])
                            else:
                                try:
                                    for parent in db.parents(f, level=None):
                                        if 'gene' in parent.attributes:
                                            all_genes.add(parent.attributes['gene'][0])
                                            break
                                        elif 'gene_id' in parent.attributes:
                                            all_genes.add(parent.attributes['gene_id'][0])
                                            break
                                        elif 'gene_name' in parent.attributes:
                                            all_genes.add(parent.attributes['gene_name'][0])
                                            break
                                except Exception:
                                    continue

                        # Instead of filtering by priority, just keep all overlapping features
                        feature_types = ','.join(set(f.featuretype for f in features if f.featuretype))
                        feature_ids = ','.join(set(f.id for f in features if f.id))
                        genes = ','.join(all_genes)











                    matches.append({
                        "Match #": len(matches) + 1,
                        "Start": start,
                        "End": end,
                        "Context": context,
                        "Strand": "+",   # always forward strand
                        "Feature Type": feature_types,
                        "Feature ID": feature_ids,
                        "Gene": genes,
                    })


                    #BED row (0-based start, 1-based end) 
                    bed_rows.append([
                        records[0].id, 
                        pos, 
                        pos + len(query), 
                        genes if genes else query,
                        ".",
                        "+" ])
                    bed_dir = RESULTS_DIR / "bed_files"
                    bed_dir.mkdir(exist_ok=True, parents=True)  # ensure directory exists

                    bed_path = bed_dir / f"{genome_key}_{query}.bed"

                    os.makedirs(os.path.dirname(bed_path), exist_ok=True) 
                    with open(bed_path, "w") as bedfile: 
                        for row in bed_rows: 
                            bedfile.write("\t".join(map(str, row)) + "\n")

                    # Count feature types for this search result
                    feature_counter = Counter()
                    for m in matches:
                        for ft in m["Feature Type"].split(","):
                            if ft:  # ignore empty strings
                                feature_counter[ft] += 1

                    if feature_counter:
                        feature_df = pd.DataFrame({
                            "Feature Type": list(feature_counter.keys()),
                            "Count": list(feature_counter.values())
                        })

                        feature_bar = px.bar(
                            feature_df,
                            x="Feature Type",
                            y="Count",
                            text="Count",
                            labels={"Count": "Number of Matches", "Feature Type": "Feature Type"},
                            title="Feature Type Counts"
                        )

                        feature_bar.update_traces(textposition='outside', marker_color='#ff6361')
                        feature_graph = dcc.Graph(figure=feature_bar)
                    else:
                        feature_graph = html.Div("No feature types found for this query.", style={'color': '#ff6b6b'})



                new_result = {
                    "label": f"Sequence '{query}' in {genome_labels[genome_key]}",
                    "matches": matches,
                    "bed_path": bed_path,
                    "no_match": False
                }

            # ✅ Increment job_id only for successful search
            job_id = job_id + 1 if job_id is not None else 1
            new_result["job_id"] = job_id
            updated_results = existing_results + [new_result]

            children = []
            for res in updated_results:
                if isinstance(res.get("bed_path"), Path):
                    res["bed_path"] = str(res["bed_path"])
                query_seq = res["label"].split(" in ")[0].replace("Sequence '", "").replace("'", "")
                virus_label = res["label"].split(" in ")[1]
                job_label = res.get("job_id", "?")

                label_header = html.Div(
                    [
                        html.Span(f"Job {job_label} Query: {query} | Virus: {virus_label}",
                                style={'flex': '1'}),

                        html.Button(
                            "⬇ Download CSV",
                            id={'type': 'download-btn', 'index': job_label},
                            n_clicks=0,
                            style={
                                'backgroundColor': colors['accent'],
                                'color': '#1e1e2f',
                                'border': 'none',
                                'borderRadius': '8px',
                                'padding': '8px 16px',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'fontSize': '14px',
                                'marginLeft': '15px'
                            }
                        ),
                        dcc.Download(id={'type': 'download-data', 'index': job_label}),
                        html.Button(
                            "⬇ Download BED",
                            id={'type': 'download-bed-btn', 'index': job_label},
                            n_clicks=0,
                            style={
                                'backgroundColor': colors['accent'],
                                'color': '#1e1e2f',
                                'border': 'none',
                                'borderRadius': '8px',
                                'padding': '8px 16px',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'fontSize': '14px',
                                'marginLeft': '10px'
                            }
                        ),
                        dcc.Download(id={'type': 'download-bed', 'index': job_label}),
                                        
                    
                    ],
                    style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'space-between'
                    }
                )

                if res["no_match"]:
                    body = html.Div("No matches found for this sequence.", style={
                        'color': '#ff6b6b',
                        'backgroundColor': '#3b2c2c',
                        'padding': '10px',
                        'borderRadius': '6px',
                        'fontFamily': 'monospace',
                        'textAlign': 'center',
                        'fontWeight': 'bold'
                    })
                else:
                    df = pd.DataFrame(res["matches"])

                    def highlight_context(context):
                        return context.replace(query_seq,
                            f"<mark style='background-color:{colors['accent']}; "
                            f"color:{colors['background']}; padding:2px 4px; border-radius:3px;'>{query_seq}</mark>"
                        )

                    df["Context"] = df["Context"].apply(highlight_context)

                    download_id = f"download-csv-{job_label}"
                    body = html.Div([
                        dash_table.DataTable(
                            id={'type': 'result-table', 'index': job_label},
                            data=df.to_dict("records"),
                            columns=[{"name": c, "id": c, 'presentation': 'markdown'} if c == "Context" else {"name": c, "id": c} for c in df.columns],
                            row_selectable="single",
                            filter_action='native',
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_data={'color': colors['text_secondary'], 'backgroundColor': colors['panel_bg'],
                                        'fontFamily': 'monospace', 'fontSize': '14px', 'padding': '6px'},
                            style_header={'backgroundColor': colors['accent'], 'color': '#1e1e2f', 'fontWeight': 'bold'},
                            style_cell={'textAlign': 'left'},
                            markdown_options={'html': True}
                        ),

                        # CSV download button + dcc.Download
                        html.Div([
                            html.Button("Download CSV", id={'type': 'download-btn', 'index': job_label}, n_clicks=0,
                                        style={'marginTop': '10px'}),
                            dcc.Download(id={'type': 'download-data', 'index': job_label})
                        ]),

                        html.Div(feature_graph, style={'marginTop': '20px'}),
                    ])

                    collapsible = html.Details([
                        html.Summary(label_header, style={'cursor': 'pointer', 'padding': '10px 0'}),
                        html.Div(body, style={'marginTop': '10px'})
                    ], style={
                        'border': f'1px solid {colors["accent"]}',
                        'borderRadius': '10px',
                        'padding': '20px',
                        'marginBottom': '20px',
                        'backgroundColor': colors['panel_bg'],
                        'boxShadow': '0 0 10px rgba(0,0,0,0.2)',
                        'color': colors['text']
                    })

                    children.append(collapsible)

            print(f"Completed Job {job_id} for query '{query_seq}' in {virus_label}")

            return updated_results, children, "query", job_id

        except Exception as e:
            return existing_results, html.Div(f"Error: {str(e)}, {genome_key}", style={'color': 'red'}), dash.no_update, job_id






    @app.callback(
        Output({'type': 'download-data', 'index': MATCH}, 'data'),
        Input({'type': 'download-btn', 'index': MATCH}, 'n_clicks'),
        State({'type': 'result-table', 'index': MATCH}, 'data'),
        prevent_initial_call=True
    )
    def download_table(n_clicks, table_data):
        if n_clicks:
            df = pd.DataFrame(table_data)
            return dcc.send_data_frame(df.to_csv, "results.csv", index=False)







    @app.callback(
        Output({'type': 'download-bed', 'index': MATCH}, 'data'),
        Input({'type': 'download-bed-btn', 'index': MATCH}, 'n_clicks'),
        State('search-results-store', 'data'),  # assuming you store results in a dcc.Store
        prevent_initial_call=True
    )
    def download_bed(n_clicks, results):
        ctx = dash.callback_context
        if not ctx.triggered or not results:
            return dash.no_update

        job_index = ctx.triggered[0]['prop_id'].split('.')[0]
        job_id = eval(job_index)['index']

        # find correct result
        for res in results:
            if res.get("job_id") == job_id:
                bed_path = res.get("bed_path")
                if bed_path and os.path.exists(bed_path):
                    return dcc.send_file(bed_path)

        return dash.no_update






















    @app.callback(
        Output("download-csv", "data"),
        Input({'type': 'download-button', 'index': ALL}, "n_clicks"),
        State('search-results-store', 'data'),
        prevent_initial_call=True
    )
    def download_csv(n_clicks_list, search_results):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update

        # Identify the index of the clicked download button
        for i, n in enumerate(n_clicks_list):
            if n and n > 0:
                triggered_index = i
                break
        else:
            return dash.no_update  # No valid clicks

        if not search_results or triggered_index >= len(search_results):
            return dash.no_update

        result = search_results[triggered_index]
        if result.get('no_match', True):
            return dash.no_update

        df = pd.DataFrame(result['matches'])
        return dcc.send_data_frame(df.to_csv, filename=f"search_result_{triggered_index+1}.csv", index=False)




    @app.callback(
        Output('search-sequence-button', 'disabled'),
        Input('sequence-table', 'selected_rows'),
        Input('viral-genome-dropdown', 'value'),
        State('sequence-table', 'data')
    )
    def toggle_search_button(selected_rows, selected_genome, table_data):
        if not selected_rows or selected_genome is None:
            return True  # disable button

        # Optionally ensure the selected row has a sequence (non-empty)
        selected_row = table_data[selected_rows[0]]
        return not bool(selected_row.get('Sequence'))  # True disables button



    @app.callback(
        Output('rna-selection', 'options'),
        Output('rna-selection', 'value'),
        Input('sequence-table', 'data'),
        State('rna-selection', 'value'),
        prevent_initial_call=True
    )
    def update_rna_dropdown(table_data, current_selection):
        if not table_data:
            return [], []

        available_ids = []

        for row in table_data:
            seq_id = row.get("SequenceID")
            if seq_id:
                output_dir = STRUCTURE_OUTPUT_DIR / seq_id
                if output_dir.exists():
                    available_ids.append(seq_id)

        options = [{"label": seq_id, "value": seq_id} for seq_id in available_ids]

        # Preserve selections if they're still in the list of available IDs
        if current_selection:
            new_value = [v for v in current_selection if v in available_ids]
        else:
            new_value = []

        return options, new_value




    @app.callback(
        Output("sequence-table", "data"),
        Input("expand-button", "n_clicks"),
        State({'type': 'result-table', 'index': ALL}, 'derived_virtual_selected_rows'),
        State({'type': 'result-table', 'index': ALL}, 'data'),
        State("sequence-table", "selected_rows"),
        State("sequence-table", "data"),
        State("my-slider", "value"),
    )
    def add_expanded_sequence(n_clicks, selected_rows_list, data_list, sequence_selected_rows, sequence_data, expand_length):
        if n_clicks is None or n_clicks == 0:
            raise dash.exceptions.PreventUpdate

        if not sequence_selected_rows:
            raise dash.exceptions.PreventUpdate

        # Get selected sequence
        sequence_index = sequence_selected_rows[-1]
        original_row = sequence_data[sequence_index]
        selected_sequence = original_row["Sequence"]
        original_id = original_row["SequenceID"]

        # Get selected row from first result table
        table_index = 0
        selected_rows = selected_rows_list[table_index]
        if not selected_rows:
            raise dash.exceptions.PreventUpdate

        row_index = selected_rows[-1]
        row_data = data_list[table_index][row_index]
        plain_context = re.sub(r'<.*?>', '', row_data['Context'])

        # Find occurrences of the sequence
        matches = [m.start() for m in re.finditer(re.escape(selected_sequence), plain_context)]
        if not matches:
            start_index = row_data['Start']
        else:
            mid_point = len(plain_context) // 2
            start_index = min(matches, key=lambda x: abs(x - mid_point))

        end_index = start_index + len(selected_sequence)

        # Expand sequence
        expanded_start = max(0, start_index - expand_length)
        expanded_end = min(len(plain_context), end_index + expand_length)
        expanded_sequence = plain_context[expanded_start:expanded_end]


        new_path = FASTA_BOLTZ_DIR / f"{original_id}_b.fasta"


        with open(new_path, 'w') as f:
            f.write(f">{original_id}_e|rna\n{expanded_sequence}\n")

        # Build new row
        new_row = {
            "SequenceID": f"{original_id}_e",
            "Sequence": expanded_sequence,
            "boltz-status": "❌",  # red X
            "sequence-job-id": "NA"
        }
        print(new_row)

        # Append new row to table
        updated_data = sequence_data + [new_row]

        return updated_data