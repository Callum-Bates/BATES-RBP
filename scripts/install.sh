#!/bin/bash

set -e  # Exit on any error

echo "Installing BATES-RBP and dependencies..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed. Please install Git first."
    exit 1
fi

# Create main environment
echo "Creating main conda environment..."
conda env create -f environments/bates_rbp.yml  

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bates_rbp_TEST

# Install package in development mode
echo "Installing BATES-RBP package..."
pip install -e .

# Create tools directory
echo "Creating tools directory..."
mkdir -p tools

# Install DeepCLIP (Python 2.7 environment)
echo "Installing DeepCLIP (Python 2.7)..."
echo "NOTE: DeepCLIP requires Python 2.7 (legacy support)"

# Create DeepCLIP environment (conda packages only)
echo "Creating DeepCLIP environment with conda packages..."
if conda env create -f environments/deepclip_env.yml; then
    echo "SUCCESS: DeepCLIP conda environment created"
    
    # Activate the environment
    echo "Installing pip and Python packages..."
    conda activate deepclip_env
    
    # Install pip for Python 2.7
    curl -s https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
    python get-pip.py --quiet
    rm get-pip.py
    
    # Install the working versions
    pip install theano==0.8.2
    pip install lasagne==0.1
    
    # Test the installation
    echo "Testing DeepCLIP environment..."
    python -c "import theano; print('Theano:', theano.__version__)" || echo "WARNING: Theano import failed"
    python -c "import lasagne; print('Lasagne imported successfully')" || echo "WARNING: Lasagne import failed"
    
    # Clone DeepCLIP repository if needed
    if [ ! -d "deepclip" ]; then
        echo "Cloning DeepCLIP repository..."
        git clone https://github.com/deepclip/deepclip.git
        echo "SUCCESS: DeepCLIP cloned successfully"
    else
        echo "INFO: DeepCLIP already exists, skipping clone"
    fi
    
    conda deactivate
    echo "SUCCESS: DeepCLIP environment setup complete"
    
else
    echo "ERROR: DeepCLIP environment creation failed"
fi



# Install RBPNet environment and package
echo "Setting up RBPNet environment..."
conda activate bates_rbp_TEST
conda env create -f environments/rbpnet_env.yml  
conda activate rbpnet_env_TEST
pip install git+https://github.com/mhorlacher/rbpnet.git
echo "SUCCESS: RBPNet installed successfully"

# Install Boltz environment and package
echo "Setting up Boltz environment..."
conda activate bates_rbp_TEST
conda env create -f environments/boltz_env.yml  
conda activate boltz_env_TEST

# Check if CUDA is available for Boltz
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing Boltz with CUDA support..."
    pip install boltz[cuda] -U
else
    echo "No CUDA detected, installing Boltz CPU version..."
    pip install boltz -U
fi
echo "SUCCESS: Boltz installed successfully"

# Return to main environment
conda activate bates_rbp_TEST

# Download models
echo "Downloading prediction models..."
bates-rbp-download-models

# Test installations
echo "Testing installations..."
python scripts/test_installations.py

echo "SUCCESS: Installation complete!"
echo ""
echo "IMPORTANT NOTES:"
echo "  - DeepCLIP uses Python 2.7 (legacy support)"
echo "  - If you encounter SSL issues with Python 2.7, see troubleshooting docs"
echo ""
echo "To launch BATES-RBP:"
echo "  conda activate bates_rbp"
echo "  bates-rbp"
echo ""
echo "The web interface will be available at http://localhost:8050"
