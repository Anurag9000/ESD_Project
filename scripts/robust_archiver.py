import os
import zipfile
import json
from pathlib import Path

# THE INDUSTRIAL-GRADE ARCHIVER (1.06M PIXELS)
ZIP_PATH = 'Dataset_Final.zip'
ROOT_DIR = 'Dataset_Final'
LOG_PATH = 'archival_status.log'

def archive_sanctuary():
    print(f"--- STARTING ROBUST ARCHIVAL ---")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        
    count = 0
    with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(ROOT_DIR):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, start=os.path.join(ROOT_DIR, '..'))
                z.write(abs_path, arcname=rel_path)
                count += 1
                if count % 10000 == 0:
                    print(f"Archived {count} / 1,045,691 images...")
                    with open(LOG_PATH, "a") as log:
                        log.write(f"Archived {count} images...\n")
    print(f"--- ARCHIVAL COMPLETE: {count} images in master zip ---")

if __name__ == "__main__":
    archive_sanctuary()
