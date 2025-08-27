### Note to MSc assessors
In terms of the tool itself, the code is the same as I submitted for assessment.  I have changed things / added scripts to help with the installation process. 
I intend to add more features to the program, but when I do this will be clearly marked.  I am still optimising the installation process, but I hope a version of BATES-RBP will be available to you should you wish to install it.


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

Create conda environments for each tool
Install all required dependencies
Download necessary models
Set up the web interface



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






