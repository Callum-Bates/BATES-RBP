# BATES-RBP

## About

BATES-RBP is an RNA-RBP binding prediction and analysis suite. 


## Prerequisites

- Conda 
- Git
- Python 3.8+
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
conda activate bates_rbp_TEST
```

**2. Launch the app (opens browser)**

```
bates-rbp
```
note: must be in BATES_RBP environment as per installation step 1.






