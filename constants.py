from pathlib import Path

BASE_DIR = Path(__file__).parent

# Existing dirs
UPLOAD_DIR   = BASE_DIR / "user_uploads"
RESULTS_DIR  = BASE_DIR / "results"
FASTA_DIR    = BASE_DIR / "fasta_files"
STRUCTURE_DIR = BASE_DIR / "structure_output"

# New: genome FASTA/GFF dirs
GENOME_DIR   = BASE_DIR / "fasta_gff"
HUMAN_GENOME_DIR = GENOME_DIR / "human"

# Ensure directories exist
for d in [UPLOAD_DIR, RESULTS_DIR, FASTA_DIR, STRUCTURE_DIR, GENOME_DIR, HUMAN_GENOME_DIR]:
    d.mkdir(parents=True, exist_ok=True)




import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HUMAN_FASTAS = {f'chr{i}': os.path.join(BASE_DIR, "gff_files", "human", f"Homo_sapiens.GRCh38.dna.chromosome.{i}.fa") for i in range(1, 23)}
HUMAN_FASTAS.update({
    'chrX': os.path.join(BASE_DIR, "gff_files", "human", "Homo_sapiens.GRCh38.dna.chromosome.X.fa"),
    'chrY': os.path.join(BASE_DIR, "gff_files", "human", "Homo_sapiens.GRCh38.dna.chromosome.Y.fa")
})
HUMAN_DB = os.path.join(BASE_DIR, "gff_files", "human", "human_please.db")

VIRAL_FASTAS = {
    'SARS_CoV_2': os.path.join(BASE_DIR, "gff_files", "viral", "SARS_CoV_2.fna"),
    'IAV': os.path.join(BASE_DIR, "gff_files", "viral", "IAV.fna"),
    # etc.
}
VIRAL_DB_PATHS = {
    'SARS_CoV_2': os.path.join(BASE_DIR, "gff_files", "viral", "SARS_CoV_2.db"),
    'IAV': os.path.join(BASE_DIR, "gff_files", "viral", "IAV.db"),
    # etc.
}

VIRAL_LABELS = {
    'SARS_CoV_2': 'SARS-CoV-2',
    'IAV': 'Influenza A Virus (H1N1)',
    # etc.
}

import os

# Base directory for FASTA and GFF files
BASE_DIR = "/home4/2185048b/fasta_gff"

# ------------------------
# Human genome FASTAs
# ------------------------
HUMAN_CHROMOSOME_FASTAS = {
    f'chr{i}': os.path.join(BASE_DIR, "human", f"Homo_sapiens.GRCh38.dna.chromosome.{i}.fa")
    for i in range(1, 23)
}
HUMAN_CHROMOSOME_FASTAS.update({
    'chrX': os.path.join(BASE_DIR, "human", "Homo_sapiens.GRCh38.dna.chromosome.X.fa"),
    'chrY': os.path.join(BASE_DIR, "human", "Homo_sapiens.GRCh38.dna.chromosome.Y.fa")
})

# Human GFF database path
HUMAN_DB_PATH = os.path.join(BASE_DIR, "human", "human_please.db")

# ------------------------
# Viral genome FASTAs
# ------------------------
VIRAL_FASTAS = {
    'SARS_CoV_2': os.path.join(BASE_DIR, "SARS_CoV_2.fna"),
    'IAV': os.path.join(BASE_DIR, "IAV.fna"),
    'zika': os.path.join(BASE_DIR, "zika.fna"),
    'CHIKV': os.path.join(BASE_DIR, "CHIKV.fna"),
    'COV-OC43': os.path.join(BASE_DIR, "COV-OC43.fna"),
    'dengue': os.path.join(BASE_DIR, "dengue.fna"),
    'HIV': os.path.join(BASE_DIR, "HIV.fna"),
    'RV': os.path.join(BASE_DIR, "RV.fna"),
    'SINV': os.path.join(BASE_DIR, "SINV.fna"),
    'VEEV': os.path.join(BASE_DIR, "VEEV.fna")
}

# ------------------------
# Viral labels
# ------------------------
VIRAL_LABELS = {
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


# Base models directory
MODELS_DIR = Path(__file__).parent / "models"

# RBPNet models
RBPNet_MODEL_DIR = MODELS_DIR / "RBPNet"

# DeepCLIP models
DEEPCLIP_MODEL_DIR = MODELS_DIR / "DeepCLIP"



BASE_DIR = Path(__file__).parent
FASTA_DIR = BASE_DIR / "fasta_files"

FASTA_BOLTZ_DIR = BASE_DIR / "fasta_boltz_files"

STRUCTURE_OUTPUT_DIR = BASE_DIR / "structure_output"
BOLTZ_EXECUTABLE = BASE_DIR / "environments" / "boltz" / "bin" / "boltz"  # adj



STRUCTURE_OUTPUT_DIR = BASE_DIR / "structure_output"


FASTA_DIR = BASE_DIR / "fasta_files"
DB_DIR = BASE_DIR / "dbs"

# Ensure dirs exist
for d in [UPLOAD_DIR, RESULTS_DIR, FASTA_DIR, STRUCTURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)





# FASTA_DIR = BASE_DIR / "fasta_files"
# BOLTZ_EXECUTABLE = BASE_DIR / "bin" / "boltz"  # adjust if needed
# STRUCTURE_OUTPUT_DIR = BASE_DIR / "structure_output"
# STRUCTURE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)




SARS_CoV_2_genes = ['UPF1', 'PTBP1', 'MATR3', 'AQR', 'BCLAF1', 'U2AF2', 'SUGP2', 'KHSRP', 'XRN2', 'HNRNPM', 'PABPN1', 'CSTF2', 'HNRNPUL1', 'SFPQ', 'HNRNPA1', 'SF3B4', 'RPS3', 'TIAL1', 'HNRNPU', 'HNRNPK', 'PRPF8', 'XRCC6', 'SND1', 'EIF3D', 'DKC1', 'RBM15', 'NOLC1', 'LARP7', 'CSTF2T', 'SF3A3', 'DDX52', 'FKBP4', 'STAU2', 'FAM120A', 'YBX3', 'RBM22', 'FUS', 'HNRNPC', 'RBFOX2', 'HNRNPL', 'FUBP3', 'BCCIP', 'EFTUD2', 'DDX6', 'SRSF9', 'QKI', 'SUB1', 'SRSF7', 'TIA1', 'FXR2', 'ILF3', 'IGF2BP3', 'SRSF1', 'SLTM', 'SSB', 'G3BP1', 'EIF3H', 'IGF2BP1', 'U2AF1', 'SAFB', 'LARP4', 'TRA2A', 'CDC40', 'PCBP1', 'LIN28B', 'PCBP2', 'DDX3X', 'TAF15', 'DHX30', 'GRSF1']
dengue_genes = ['UPF1', 'PTBP1', 'MATR3', 'AQR', 'BCLAF1', 'STAU2', 'FAM120A', 'ZC3H11A', 'YBX3', 'U2AF2', 'SUGP2', 'RBM22', 'KHSRP', 'XRN2', 'FUS', 'HNRNPC', 'HNRNPM', 'RBFOX2', 'EIF3D', 'PABPN1', 'HNRNPL', 'HNRNPUL1', 'FUBP3', 'EFTUD2', 'DDX6', 'SRSF9', 'QKI', 'SUB1', 'SRSF7', 'TIA1', 'SFPQ', 'XPO5', 'FXR2', 'ILF3', 'HNRNPA1', 'IGF2BP3', 'SRSF1', 'SLTM', 'SSB', 'G3BP1', 'RPS3', 'TIAL1', 'HNRNPU', 'IGF2BP1', 'SAFB', 'LARP4', 'RBM15', 'TRA2A', 'HNRNPK', 'PCBP1', 'PRPF8', 'SF3A3', 'LIN28B', 'XRCC6', 'PCBP2', 'SND1', 'DDX3X', 'TAF15']
zika_genes =  ['UPF1', 'PTBP1', 'MATR3', 'STAU2', 'FAM120A', 'YBX3', 'U2AF2', 'SUGP2', 'XRN2', 'FUS', 'HNRNPC', 'HNRNPM', 'RBFOX2', 'EIF3D', 'PABPN1', 'HNRNPL', 'HNRNPUL1', 'FUBP3', 'DDX6', 'TIA1', 'SFPQ', 'FXR2', 'ILF3', 'HNRNPA1', 'IGF2BP3', 'SRSF1', 'G3BP1', 'RPS3', 'TIAL1', 'HNRNPU', 'IGF2BP1', 'SAFB', 'LARP4', 'TRA2A', 'HNRNPK', 'PCBP1', 'PRPF8', 'LIN28B', 'XRCC6', 'PCBP2', 'SND1', 'DDX3X']
CHIKV_genes =  ['UPF1', 'PTBP1', 'BCLAF1', 'XRN2', 'HNRNPM', 'HNRNPL', 'SFPQ', 'ILF3', 'HNRNPA1', 'SLTM', 'SSB', 'HNRNPU', 'RBM15', 'HNRNPK', 'PCBP1', 'PRPF8', 'XRCC6', 'PCBP2', 'SND1', 'DDX3X']
HIV_genes =  ['UPF1', 'PTBP1', 'MATR3', 'BCLAF1', 'FAM120A', 'YBX3', 'U2AF2', 'XRN2', 'HNRNPC', 'HNRNPM', 'HNRNPL', 'FUBP3', 'SFPQ', 'ILF3', 'HNRNPA1', 'IGF2BP3', 'SLTM', 'G3BP1', 'RPS3', 'HNRNPU', 'IGF2BP1', 'HNRNPK', 'PCBP1', 'PRPF8', 'XRCC6', 'PCBP2', 'SND1', 'DDX3X']
IAV_genes =  ['UPF1', 'MATR3', 'SFPQ', 'ILF3', 'HNRNPU', 'RBM15', 'HNRNPK', 'PCBP1', 'PRPF8', 'XRCC6', 'SND1', 'DDX3X']
RV_genes =  ['UPF1', 'PTBP1', 'MATR3', 'BCLAF1', 'FAM120A', 'U2AF2', 'XRN2', 'HNRNPC', 'HNRNPM', 'EIF3D', 'HNRNPL', 'FUBP3', 'TIA1', 'SFPQ', 'ILF3', 'HNRNPA1', 'IGF2BP3', 'SLTM', 'SSB', 'G3BP1', 'RPS3', 'TIAL1', 'HNRNPU', 'IGF2BP1', 'LARP4', 'RBM15', 'HNRNPK', 'PCBP1', 'PRPF8', 'XRCC6', 'SND1', 'DDX3X']
SINV_genes =  ['UPF1', 'PTBP1', 'MATR3', 'FAM120A', 'YBX3', 'U2AF2', 'HNRNPC', 'HNRNPM', 'EIF3D', 'FUBP3', 'TIA1', 'SFPQ', 'FXR2', 'HNRNPA1', 'IGF2BP3', 'SSB', 'G3BP1', 'RPS3', 'TIAL1', 'HNRNPU', 'IGF2BP1', 'LARP4', 'HNRNPK', 'PCBP1', 'PRPF8', 'XRCC6', 'PCBP2', 'SND1', 'DDX3X']
VEEV_genes =  ['SFPQ', 'HNRNPA1', 'HNRNPK', 'PCBP2']
flavi_genes =  ['UPF1', 'PTBP1', 'MATR3', 'AQR', 'BCLAF1', 'FAM120A', 'ZC3H11A', 'YBX3', 'U2AF2', 'SUGP2', 'KHSRP', 'XRN2', 'FUS', 'HNRNPC', 'HNRNPM', 'EIF3D', 'PABPN1', 'HNRNPL', 'HNRNPUL1', 'FUBP3', 'EFTUD2', 'DDX6', 'SRSF9', 'QKI', 'SRSF7', 'TIA1', 'SFPQ', 'FXR2', 'ILF3', 'HNRNPA1', 'IGF2BP3', 'SRSF1', 'SLTM', 'SSB', 'G3BP1', 'RPS3', 'TIAL1', 'HNRNPU', 'IGF2BP1', 'SAFB', 'LARP4', 'RBM15', 'TRA2A', 'HNRNPK', 'PRPF8', 'SF3A3', 'LIN28B', 'XRCC6', 'PCBP2', 'SND1', 'DDX3X', 'TAF15']
COV_OC43_genes =  ['UPF1', 'PTBP1', 'MATR3', 'FAM120A', 'YBX3', 'U2AF2', 'KHSRP', 'XRN2', 'FUS', 'HNRNPC', 'HNRNPM', 'EIF3D', 'PABPN1', 'HNRNPL', 'HNRNPUL1', 'FUBP3', 'SRSF7', 'TIA1', 'SFPQ', 'FXR2', 'ILF3', 'DKC1', 'HNRNPA1', 'IGF2BP3', 'SRSF1', 'SLTM', 'SSB', 'G3BP1', 'RPS3', 'TIAL1', 'HNRNPU', 'SAFB', 'LARP4', 'RBM15', 'HNRNPK', 'PCBP1', 'PRPF8', 'LARP7', 'XRCC6', 'PCBP2', 'SND1', 'DDX3X', 'TAF15']


#hep deepclip models
deep_models = ["FUBP3_HepG2", "HNRNPA1_HepG2", "HNRNPM_HepG2", "IGF2BP3_HepG2", "PTBP1_HepG2", "RBM22_HepG2", "SRSF1_HepG2", "TRA2A_HepG2",
"FUS_HepG2", "HNRNPC_HepG2", "HNRNPU_HepG2", "KHSRP_HepG2", "QKI_HepG2", "RBM5_HepG2", "SRSF7_HepG2", "U2AF1_HepG2",
"G3BP1_HepG2", "HNRNPK_HepG2", "HNRNPUL1_HepG2", "LARP7_HepG2", "RBFOX2_HepG2", "SF3A3_HepG2", "SRSF9_HepG2", "U2AF2_HepG2",
"GRSF1_HepG2", "HNRNPL_HepG2", "IGF2BP1_HepG2", "MATR3_HepG2", "RBM15_HepG2", "SFPQ_HepG2", "TIAL1_HepG2", "FUS_K562", "HNRNPL_K562", "IGF2BP1_K562", "KHSRP_K562", "NONO_K562", "QKI_K562", "SF3B1_K562", "TARDBP_K562", "U2AF2_K562",
"HNRNPA1_K562", "HNRNPM_K562", "IGF2BP2_K562", "LARP4_K562", "PTBP1_K562", "RBFOX2_K562", "SF3B4_K562", "TIA1_K562",
"HNRNPC_K562", "HNRNPU_K562", "ILF3_K562", "LARP7_K562", "PUM1_K562", "RBM15_K562", "SRSF1_K562", "TRA2A_K562",
"HNRNPK_K562", "HNRNPUL1_K562", "KHDRBS1_K562", "MATR3_K562", "PUM2_K562", "RBM22_K562", "SRSF7_K562", "U2AF1_K562", "PTBP1_RCMPT", "FUS_RNCMPT"]


rbp_models = [
    "AGGF1", "DDX6", "FUBP3", "HNRNPU", "NKRF", "RBM15", "SRSF7", "U2AF1",
    "AKAP1", "DGCR8", "FUS", "HNRNPUL1", "NOL12", "RBM22", "SRSF9", "U2AF2",
    "AQR", "DHX30", "FXR2", "IGF2BP1", "NOLC1", "RBM5", "SSB", "UCHL5",
    "BCCIP", "DKC1", "G3BP1", "IGF2BP3", "PABPN1", "RPS3", "STAU2", "UPF1",
    "BCLAF1", "DROSHA", "GRSF1", "ILF3", "PCBP1", "SAFB", "SUB1", "UTP18",
    "BUD13", "EFTUD2", "GRWD1", "KHSRP", "PCBP2", "SDAD1", "SUGP2", "WDR43",
    "CDC40", "EIF3D", "GTF2F1", "LARP4", "POLR2G", "SF3A3", "SUPV3L1", "XPO5",
    "CSTF2", "EIF3H", "HLTF", "LARP7", "PPIG", "SF3B4", "TAF15", "XRCC6",
    "CSTF2T", "EXOSC5", "HNRNPA1", "LIN28B", "PRPF4", "SFPQ", "TBRG4", "XRN2",
    "DDX3X", "FAM120A", "HNRNPC", "LSM11", "PRPF8", "SLTM", "TIA1", "YBX3",
    "DDX52", "FASTKD2", "HNRNPK", "MATR3", "PTBP1", "SMNDC1", "TIAL1", "ZC3H11A",
    "DDX55", "FKBP4", "HNRNPL", "NCBP2", "QKI", "SND1", "TRA2A", "ZNF800",
    "DDX59", "FTO", "HNRNPM", "NIP7", "RBFOX2", "SRSF1", "TROVE2"
]


human_chromosomes = [
    {'label': f'Chromosome {i}', 'value': f'chr{i}'} for i in range(1, 23)
] + [{'label': 'Chromosome X', 'value': 'chrX'},
     {'label': 'Chromosome Y', 'value': 'chrY'}]

viral_genomes = [
    {'label': 'SARS-CoV-2', 'value': 'SARS_CoV_2'},
    {'label': 'Influenza A (H1N1)', 'value': 'IAV'},
    {'label': 'Dengue', 'value': 'dengue'},
    {'label': 'Chikungunya', 'value': 'CHIKV'},
    {'label': 'COV-OC43', 'value': 'COV-OC43'},
    {'label': 'HIV', 'value': 'HIV'},
    {'label': 'Rhinovirus', 'value': 'RV'},
    {'label': 'Sindbis', 'value': 'SINV'},
    {'label': 'Venezuelan equine encephalitis', 'value': 'VEEV'},
    {'label': 'Zika', 'value': 'zika'},
]



import json
import dash
from dash import Dash, dcc, html
from dash import dash_table

# Load your JSON


base_dir = os.path.dirname(__file__)  # points to BATES_RBP/
json_path = os.path.join(base_dir, "gene_infos.json")

with open(json_path, "r") as f:
    gene_infos = json.load(f)


# Flatten JSON -> list of rows for DataTable
table_data = []
for gene in gene_infos:
    row = {
        "Accession": gene.get("accession", ""),
        "Symbol": gene.get("symbol", ""),
        "Protein Name": gene.get("protein_name", ""),
        "Alt Names": ", ".join(gene.get("alt_names", [])),
        "NCBI ID": ", ".join([gid.get("ncbi", "") for gid in gene.get("gene_ids", [])]),
        "Ensembl ID": ", ".join([gid.get("ensembl", "") for gid in gene.get("gene_ids", [])]),
        "Keywords": ", ".join(gene.get("keywords", [])),
        "Motifs": ", ".join(gene.get("motifs", [])),
        "Available Algorithm": gene.get("available_algorithm", ""),
        "GO Terms": ", ".join(gene.get("go_terms", [])),
    }
    table_data.append(row)

columns = [{"name": col, "id": col} for col in table_data[0].keys()]




colors = {
    'background': '#e5e5e5',
    'text': '#1f1f1f',
    'accent': '#7a9e7e',         # Muted sage green
    'panel_bg': '#f2f2f2',
    'text_secondary': '#4d4d4d'
}


