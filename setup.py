# setup.py
#!/usr/bin/env python
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"Command failed: {command}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Setup NeuroNest project")
    parser.add_argument("--download", action="store_true", help="Download checkpoint")
    args = parser.parse_args()
    
    # Create directory structure
    dirs = ["app", "utils", "utils/contrast", "configs", "data"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    if args.download:
        # Download the OneFormer checkpoint
        checkpoint_dir = "data/oneformer_ade20k_swin_large"
        if not os.path.exists(checkpoint_dir):
            print(f"Creating checkpoint directory: {checkpoint_dir}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        if not os.path.exists(f"{checkpoint_dir}/model_final.pth"):
            print("Downloading OneFormer ADE20K checkpoint...")
            # Use git-lfs if available, otherwise use curl/wget
            if subprocess.call("which git-lfs", shell=True, stdout=subprocess.DEVNULL) == 0:
                run_command("git lfs install")
                run_command(f"git clone https://huggingface.co/shi-labs/oneformer_ade20k_swin_large {checkpoint_dir}")
            else:
                # Alternative download method using huggingface_hub
                run_command("pip install huggingface_hub")
                run_command(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('shi-labs/oneformer_ade20k_swin_large', local_dir='{checkpoint_dir}')\"")
    
    print("✅ Setup complete!")

if __name__ == "__main__":
    main()
