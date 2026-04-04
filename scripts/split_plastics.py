#!/usr/bin/env python3

import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from integrate_external_dataset import regenerate_auto_split_manifest

def main():
    root = Path("Dataset_Final")
    meta_path = root / "dataset_metadata.json"
    manifest_path = root / "auto_split_manifest.json"
    
    if not meta_path.exists():
        print(f"Error: {meta_path} not found.")
        return

    with open(meta_path) as f:
        metadata = json.load(f)
        
    soft_labels = {
        "soft plastic", "soft_plastic", "Plastic film", "Other plastic wrapper", 
        "Crisp packet", "Single-use carrier bag", "Garbage bag", "Six pack rings", 
        "Polypropylene bag", "Plastic glooves"
    }
    
    rigid_labels = {
        "rigid_plastic", "water_bottles", "hard plastic", "plastic_bottle", 
        "Clear plastic bottle", "Other plastic bottle", "Plastic bottle cap", 
        "Plastic straw", "Styrofoam piece", "Disposable plastic cup", "Plastic lid", 
        "Disposable food container", "Plastic utensils", "Foam food container", 
        "Foam cup", "Spread tub", "Squeezable tube", "Other plastic container", 
        "Tupperware", "Other plastic cup"
    }
    
    soft_dir = root / "soft_plastic"
    rigid_dir = root / "rigid_plastic"
    soft_dir.mkdir(exist_ok=True)
    rigid_dir.mkdir(exist_ok=True)
    
    soft_count = 0
    rigid_count = 0
    missing_count = 0
    
    for item in metadata:
        if item["label"] == "plastic":
            src_label = str(item.get("source_label", "")).strip()
            dest_label = None
            
            if src_label in soft_labels:
                dest_label = "soft_plastic"
            elif src_label in rigid_labels:
                dest_label = "rigid_plastic"
            
            if dest_label:
                old_path = Path(item["file_path"])
                new_path = root / dest_label / old_path.name
                
                if old_path.exists():
                    shutil.move(old_path, new_path)
                    item["file_path"] = f"Dataset_Final/{dest_label}/{old_path.name}"
                    item["label"] = dest_label
                    item["mapping_rule"] = f"heuristic_remap_from_plastic:{src_label}->{dest_label}"
                    
                    if dest_label == "soft_plastic":
                        soft_count += 1
                    else:
                        rigid_count += 1
                else:
                    missing_count += 1
                
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
        
    print(f"Moved {soft_count} to soft_plastic, {rigid_count} to rigid_plastic.")
    if missing_count > 0:
        print(f"Warning: {missing_count} files were listed in JSON but missing on disk.")
        
    print("Regenerating auto-split manifest...")
    manifest = regenerate_auto_split_manifest(root, 42, "0.7,0.2,0.1")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
        
    print("Manifest regenerated. Done.")

if __name__ == "__main__":
    main()
