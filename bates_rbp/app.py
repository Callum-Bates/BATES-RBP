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
from dash import Dash
from flask import send_from_directory

from .layout import layout
from . import callbacks


# 3rd Party
# Scientific / Data
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
#import logomaker

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




# app.py
from dash import Dash
import dash_bootstrap_components as dbc
from flask import send_from_directory

app = Dash(__name__)
server = app.server  # Needed for deployment

# Layout
from .layout import layout
app.layout = layout

# Register callbacks
from . import callbacks
callbacks.register_callbacks(app)  # <-- Callbacks are now properly connected

# Serve plots folder
@app.server.route('/plots/<path:path>')
def serve_plot(path):
    return send_from_directory('plots', path)

def main():
    """Run the app"""
    app.run(debug=True)

if __name__ == "__main__":
    main()


    

    








        