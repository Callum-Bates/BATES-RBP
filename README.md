# BATES-RBP

## About

BATES-RBP is an RNA-RBP binding prediction and analysis suite.  BATES-RBP allows use of Machine Learning methods to predict RNA-Protein interactions at a per-nucleotide resolution.  It allows visualisation options for results, 3d predictions of RNA structure, and RNA mapping to viral genomes.  BATES-RBP is currently only available on Linux operating systems.


## Prerequisites

- Conda 
- Git
- Python 3.8+
- Linux system
- CUDA (optional, for GPU acceleration for Boltz-2)


## Installation

**1. Clone the repository:**
```
git clone https://github.com/Callum-Bates/BATES-RBP.git
cd BATES-RBP
```

**2. Run the installation script:**
```
chmod +x scripts/install.sh
scripts/install.sh
```

**The installation script will:**

Create conda environments for each tool <br>
Install all required dependencies <br>
Download necessary models <br>
Set up the web interface <br>



## Quick start

**1. Activate the main environment**
```
conda activate bates_rbp
```

**2. Launch the app (opens browser)**

```
bates-rbp
```
note: must be in BATES-RBP environment as per installation step 1.



## Using BATES-RBP

**1. ML Predictions**

The default tab is the ML prediction tab.  Here users can select a ML method (currently DeepCLIP and RBPNet are available) and an associated protein model (A).
From here, users can either enter RNA sequences in FASTA format, or upload a FASTA file (B).  Once ML prediction model and FASTA sequences are stored, clicking "Run Job" button runs the prediction.


**2.1 Motif Mapping**
In the RNA characterisation tab, users can run RNA structure prediction using Boltz-2.  RNA sequences can be added to the sequence table (A) and are added from any sequences used in ML predictions.  
The user then selects a viral genome for mapping and clicks "Search Selected RNA in Genome" button (B).  Results are displayed in a panel to the right (C).

**2.2 RNA Extension**
Users can extend RNA motifs based on genomic context taken from RNA motif mapping.  The user has to select an RNA sequence from the sequence table (A), a RNA match (B) and a nucleotide length (C).  The RNA sequence will be expanded on either side by the chosen number and added to the sequence table with a "_e" prefix (A).


**3. 3D Structure Generation**



