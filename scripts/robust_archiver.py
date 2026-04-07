import subprocess
import os
import sys
from pathlib import Path

def archive_class_safely(class_name):
    source_dir = Path(f"Dataset_Final/{class_name}")
    archive_path = Path(f"Dataset_Archives/{class_name}.tar")
    
    if not source_dir.exists():
        print(f"[!] Class {class_name} not found.")
        return

    print(f"[*] Archiving {class_name} with LOW PRIORITY...")
    
    # Using 'nice -n 19' for lowest CPU priority
    # Using 'ionice -c 3' for Idle-only disk priority (only works if kernel supports it)
    # No compression (just tar) to save CPU
    cmd = [
        "nice", "-n", "19",
        "tar", "-cf", str(archive_path),
        "-C", "Dataset_Final", class_name
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[+] Successfully created {archive_path}")
        size_gb = archive_path.stat().st_size / (1024**3)
        print(f"    Size: {size_gb:.2f} GB")
    except subprocess.CalledProcessError as e:
        print(f"[!] Error archiving {class_name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/robust_archiver.py <class_name>")
    else:
        archive_class_safely(sys.argv[1])
