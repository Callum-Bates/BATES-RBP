#!/usr/bin/env python3
"""
Download and organize models for BATES-RBP
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil
import re

def download_file(url, output_path):
    """Download a file from URL"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {output_path}")
import os
def download_rbpnet_models():
    print(os.getcwd())

    print("Downloading RBPNet models from Zenodo...")
    
    models_dir = Path("bates_rbp/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    rbpnet_dir = models_dir / "rbpnet"
    rbpnet_dir.mkdir(exist_ok=True)
    
    zenodo_url = "https://zenodo.org/records/10185223/files/RBPNet_models.zip"
    zip_path = rbpnet_dir / "RBPNet_models.zip"
    
    try:
        download_file(zenodo_url, zip_path)
        print("Extracting RBPNet models...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(rbpnet_dir)
        
        zip_path.unlink()
        
        print("Renaming model files...")
        model_files = list(rbpnet_dir.rglob("*.model.h5"))
        print(f"Found {len(model_files)} model files to rename")
        
        for model_file in model_files:
            old_name = model_file.name
            print(f"Processing: {old_name}")
            
            if old_name.endswith('.model.h5'):
                base_name = old_name[:-9]
                if '_' in base_name:
                    protein_name = base_name.split('_')[0]
                else:
                    protein_name = base_name
                
                new_name = f"{protein_name}.h5"
                destination = rbpnet_dir / new_name
                
                if destination.exists():
                    print(f"  Duplicate found, removing {old_name}")
                    model_file.unlink()  # delete the new duplicate
                else:
                    shutil.move(str(model_file), str(destination))
                    print(f"  Renamed to: {new_name}")
        
        for item in rbpnet_dir.iterdir():
            if item.is_dir() and not any(item.iterdir()):
                item.rmdir()
                print(f"Removed empty directory: {item.name}")
        
        print("SUCCESS: RBPNet models downloaded and extracted successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to download RBPNet models: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


rbpnet_dir = Path("bates_rbp/models/rbpnet")  # change if your files are in another folder

for item in rbpnet_dir.iterdir():
    if item.is_file() and item.suffix == ".h5":
        match = re.match(r"(.+?)_[A-Za-z0-9]+\.model\.h5", item.name)
        if match:
            protein_name = match.group(1)
            new_name = f"{protein_name}.h5"
            new_path = rbpnet_dir / new_name
            item.rename(new_path)
            print(f"Renamed {item.name} -> {new_name}")


def download_deepclip_models():
    """Download DeepCLIP models from GitHub and rename them"""
    print("Downloading DeepCLIP models from GitHub...")
    
    # Changed: models now go in bates_rbp/models/
    models_dir = Path("bates_rbp/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    deepclip_dir = models_dir / "deepclip"
    deepclip_dir.mkdir(exist_ok=True)
    
    # GitHub raw URLs for model files
    base_urls = {
        "HepG2": "https://raw.githubusercontent.com/deepclip/models/master/ENCODE/ENCODE-HepG2/",
        "K562": "https://raw.githubusercontent.com/deepclip/models/master/ENCODE/ENCODE-K562/"
    }
    
    # Get list of model files from GitHub API
    def get_github_files(cell_line):
        api_url = f"https://api.github.com/repos/deepclip/models/contents/ENCODE/ENCODE-{cell_line}"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            files = response.json()
            return [f['name'] for f in files if f['name'].endswith('.pkl')]
        except Exception as e:
            print(f"ERROR: Failed to get file list for {cell_line}: {e}")
            return []
    
    success = True
    
    for cell_line in ["HepG2", "K562"]:
        print(f"Downloading {cell_line} models...")
        
        # Get list of .pkl files
        pkl_files = get_github_files(cell_line)
        
        if not pkl_files:
            print(f"WARNING: No .pkl files found for {cell_line}")
            continue
        
        for filename in pkl_files:
            try:
                # Original filename: FUS_ENCODE-HepG2.pkl or FUS_ENCODE-K562.pkl
                # New filename: FUS_HepG2.pkl or FUS_K562.pkl
                
                # Extract protein name (everything before _ENCODE-)
                match = re.match(r'(.+)_ENCODE-(.+)\.pkl', filename)
                if match:
                    protein_name = match.group(1)
                    original_cell_line = match.group(2)
                    new_filename = f"{protein_name}_{cell_line}.pkl"
                else:
                    # Fallback if pattern doesn't match
                    new_filename = filename.replace(f"_ENCODE-{cell_line}", f"_{cell_line}")
                
                # Download the file
                file_url = base_urls[cell_line] + filename
                output_path = deepclip_dir / new_filename
                
                download_file(file_url, output_path)
                
            except Exception as e:
                print(f"ERROR: Failed to download {filename}: {e}")
                success = False
    
    if success:
        print("SUCCESS: DeepCLIP models downloaded and renamed successfully")
    else:
        print("WARNING: Some DeepCLIP models failed to download")
    
    return success

def main():
    """Main function to download all models"""
    print("BATES-RBP Model Downloader")
    print("=" * 40)
    
    # Changed: Create models directory inside bates_rbp/
    models_dir = Path("bates_rbp/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Download RBPNet models
    if not download_rbpnet_models():
        success = False
    
    print()  # Empty line for separation
    
    # Download DeepCLIP models
    if not download_deepclip_models():
        success = False
    
    print()
    print("=" * 40)
    
    if success:
        print("SUCCESS: All models downloaded successfully!")
        print(f"Models are located in: {models_dir.resolve()}")
        
        # Show directory structure
        print("\nDirectory structure:")
        for root, dirs, files in os.walk(models_dir):
            level = root.replace(str(models_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent},{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent},{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
    else:
        print("WARNING: Some models failed to download. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
