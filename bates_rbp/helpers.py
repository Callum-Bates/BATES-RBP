from pathlib import Path
import os
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "user_uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
from .constants import *
def save_uploaded_file(contents, filename, output_dir=UPLOAD_DIR):
    os.makedirs(output_dir, exist_ok=True)
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(decoded)
    return filepath





times_file = "times.tsv"
if not os.path.exists(times_file):
    with open(times_file, "w") as file:
        file.write("method\tmodel\tRNA length\ttime\n")



def run_method(method, model_name, job_id):
    # Find latest uploaded FASTA
    fasta_files = sorted(
        glob.glob(str(UPLOAD_DIR / "*.fasta")),
        key=os.path.getmtime,
        reverse=True
    )
    if not fasta_files:
        return "No FASTA files found to run prediction.", []

    fasta_path = Path(fasta_files[0])

    # Copy FASTA file for reproducibility
    fasta_copy_path = FASTA_DIR / f"job_{job_id}.fasta"
    shutil.copy(fasta_path, fasta_copy_path)

    # Write per-sequence FASTAs
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        seq = str(record.seq)
        with open(FASTA_DIR / f"{seq_id}_b.fasta", "w") as f:
            f.write(f">{seq_id}|rna\n{seq}\n")

    # Method-specific setup
    env_map = {
        "deepclip": {
            "env": "deepclip_env",  # <-- from environments/deepclip_env.yml
            "model_dir": DEEPCLIP_MODEL_DIR,
            "cmd": lambda model, fasta, out: [
                "python", "DeepCLIP.py",
                "--runmode", "predict",
                "-P", str(model),
                "--sequences", str(fasta),
                "--predict_output_file", str(out)
            ],
            "cwd": str(Path(__file__).resolve().parent.parent / "deepclip")  # repo checkout
        },
        "rbpnet": {
            "env": "rbpnet_env",
            "model_dir": RBPNet_MODEL_DIR,
            "cmd": lambda model, fasta, out: [
                "rbpnet", "predict",
                "-m", str(model),
                str(fasta),
                "-o", str(out)
            ],
            "cwd": None
        }
    }

    if method not in env_map:
        return f"Unknown method: {method}", []

    env = env_map[method]

    # Build model path
    model_path = env["model_dir"] / model_name
    if not model_path.exists():
        return f"No model file found: {model_path}", []

    output_path = RESULTS_DIR / f"{method}_job_{job_id}.txt"

    # Build command with conda run
    cmd = ["conda", "run", "-n", env["env"]] + env["cmd"](model_path, fasta_path, output_path)

    try:
        result = subprocess.run(
            cmd,
            cwd=env["cwd"],
            capture_output=True,
            text=True
        )

        parsed_sequences = [
            {"id": record.id, "seq": str(record.seq)}
            for record in SeqIO.parse(fasta_path, "fasta")
        ]

        if result.returncode == 0:
            return f"{method.capitalize()} job complete.", parsed_sequences
        else:
            return f"{method.capitalize()} failed:\n{result.stderr}", []

    except Exception as e:
        return f"Error running {method}: {str(e)}", []

def run_boltz_structure(selected_seq_ids):
    fasta_dir = "/home4/2185048b/fasta_files"
    boltz_executable = "/home4/2185048b/anaconda3/envs/boltz/bin/boltz"
    out_base = "/home4/2185048b/structure_output"
    os.makedirs(out_base, exist_ok=True)
    out_base = "/home4/2185048b/structure_output"

    env = os.environ.copy()
    # Only needed if you get libstdc++ errors, otherwise comment out
    # env["LD_PRELOAD"] = "/home4/2185048b/anaconda3/envs/boltz/lib/libstdc++.so.6"

    completed = []
    failed = []

    for seq_id in selected_seq_ids:
        fasta_filename = f"{seq_id}_b.fasta"
        fasta_path = os.path.join(fasta_dir, fasta_filename)
        print(f"Looking for FASTA file: {fasta_path}")

        if not os.path.exists(fasta_path):
            failed.append(seq_id)
            continue

        output_dir = os.path.join(out_base, f"{seq_id}")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            boltz_executable, "predict", fasta_path,
            "--model", "boltz2",
            "--out_dir", output_dir,
            "--accelerator", "cpu"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=fasta_dir,
                capture_output=True,
                text=True,
                env=env
            )
            if result.returncode == 0:
                completed.append(seq_id)
            else:
                print(f"Boltz error for {seq_id}:\n{result.stderr.strip()}")
                failed.append(seq_id)

        except Exception as e:
            print(f"Exception for {seq_id}: {str(e)}")
            failed.append(seq_id)

    summary = []
    if completed:
        summary.append(f"Completed: {', '.join(completed)}")
    if failed:
        summary.append(f"Failed: {', '.join(failed)}")
    return " | ".join(summary) if summary else "No valid inputs."




