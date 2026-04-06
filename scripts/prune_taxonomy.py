import json
import random
from pathlib import Path
import sys

# Append scripts folder so we can import integrate_external_dataset
sys.path.append(str(Path("scripts").absolute()))
from integrate_external_dataset import regenerate_auto_split_manifest, KAGGLE_GENERIC_CLASS_MAP

def prune():
    dataset_root = Path("Dataset_Final")
    metadata_path = dataset_root / "dataset_metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    if isinstance(metadata, dict):
        sources = metadata.get("sources", [])
    else:
        sources = metadata
        
    print(f"Original entries: {len(sources)}")
    
    # Filter out trash and wood
    new_sources = []
    dropped_count = 0
    for entry in sources:
        if entry.get("label") in ["trash", "wood"]:
            dropped_count += 1
        else:
            new_sources.append(entry)
            
    print(f"Dropped {dropped_count} entries. Remaining: {len(new_sources)}")
    
    # Rewrite metadata
    final_metadata = {"sources": new_sources} # Enforce dict structure
    with open(metadata_path, 'w') as f:
        json.dump(final_metadata, f, indent=4)
        
    print("Metadata updated. Regenerating Split Manifest...")
    regenerate_auto_split_manifest(dataset_root, final_metadata)
    print("Split manifest regenerated successfully.")

if __name__ == "__main__":
    prune()
