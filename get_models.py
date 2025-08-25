import os
import gdown

# Google Drive links to the individual zip files for each model
ZENODO_MODELS = {
    "RBPNet": "https://drive.google.com/uc?id=<RBPNet_file_id>&export=download",
    "DeepCLIP": "https://drive.google.com/uc?id=<DeepCLIP_file_id>&export=download",
}

MODEL_DIR = "models"

def download_and_extract(url, folder_name):
    """Download a zip from Google Drive and extract to models/folder_name."""
    target_zip = f"{folder_name}.zip"
    target_path = os.path.join(MODEL_DIR, folder_name)

    if os.path.exists(target_path):
        print(f"{folder_name} already exists. Skipping download.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Downloading {folder_name} models...")
    gdown.download(url, output=target_zip, quiet=False)

    print(f"Extracting {folder_name} models...")
    import zipfile
    with zipfile.ZipFile(target_zip, "r") as zip_ref:
        zip_ref.extractall(os.path.join(MODEL_DIR, folder_name))

    os.remove(target_zip)
    print(f"{folder_name} models saved in {target_path}")

def main():
    for name, url in ZENODO_MODELS.items():
        download_and_extract(url, name)

if __name__ == "__main__":
    main()
