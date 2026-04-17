#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def run_cmd(cmd, shell=True):
    print(f"🚀 Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Command failed but continuing acquisition: {e}")

def main():
    target_dir = Path("dataset_audit/incoming")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧹 Preparing download environment...")
    
    # 1. TrashBox (GitHub Mirror) ~17k images
    print("\n📦 Acquiring TrashBox (Medical & E-Waste)...")
    trashbox_dir = target_dir / "TrashBox"
    if not trashbox_dir.exists():
        run_cmd(f"git clone https://github.com/nikhilvenkatkumsetty/TrashBox {trashbox_dir}")
    
    # 2. ModelScope Household (150k images)
    print("\n📦 Acquiring ModelScope Household Waste (Massive Diversity)...")
    modelscope_dir = target_dir / "ModelScope_Household"
    modelscope_dir.mkdir(exist_ok=True)
    # Using the ModelScope CLI for efficient multi-threaded download
    run_cmd(f"modelscope download --dataset damo/household_garbage_classification --local_dir {modelscope_dir}")

    # 3. Waste_pictures (Specialized Textures) ~24k images
    print("\n📦 Acquiring Waste_Pictures (24k targets)...")
    waste_pics_dir = target_dir / "Waste_Pictures"
    if not waste_pics_dir.exists():
        run_cmd(f"git clone https://github.com/AgaMiko/waste-datasets-review {waste_pics_dir}")

    # 4. ZeroWaste (Alternative Source)
    print("\n📦 Acquiring ZeroWaste (Industrial Sorting Line)...")
    # Using a secondary research mirror if main is down
    zerowaste_dir = target_dir / "ZeroWaste"
    zerowaste_dir.mkdir(exist_ok=True)
    # Note: If wget fails, consider using kaggle-cli for https://www.kaggle.com/datasets/saimun777/zerowaste
    run_cmd(f"wget -c https://files.webis.de/datasets/zerowaste/zerowaste-f.tar.gz -P {zerowaste_dir}")

    print("\n✅ All acquisition processes initiated.")
    print(f"Check {target_dir} for progress.")

if __name__ == "__main__":
    main()
