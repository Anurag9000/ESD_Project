"""
Plastic Segregation Engine
Migrates 'plastic' items into hard_plastic or soft_plastic based on source_label semantics.
"""
import json
import os
import shutil
from pathlib import Path

METADATA_PATH = "Dataset_Final/dataset_metadata.json"
DATASET_ROOT = "Dataset_Final"

# HARD plastic = rigid, structural, thick-walled containers (PET bottles, trays, equipment)
HARD_PLASTIC_LABELS = {
    "plastic, bottle", "soda bottle", "water bottle", "pet",
    "bottle-transp", "bottle-transp-full", "bottle-blue", "bottle-blue5l",
    "bottle-blue-full", "bottle-dark", "bottle-green", "bottle-green-full",
    "bottle-dark-full", "bottle-milk", "bottle-oil", "bottle-yogurt",
    "bottle_150cl", "bottle_50cl", "bottle_100cl", "bottle_200cl",
    "bottle_33cl", "bottle_25cl", "bottle_200cl",
    "alucan", "hdpem",
    "plastic, tub", "plastic, utensil", "7-plastic",
    "plastic_equipment_packaging", "other plastic",
    "plastic bottles", "water_bottles",
    "plastic_bottle", "plastic_bottle_takeaway_cup",
    "clear plastic bottle", "other plastic bottle",
    "disposable plastic cup", "plastic bottle cap",
    "plastic lid", "pop tab",
}

# SOFT plastic = flexible, film-based, bags, wrappers
SOFT_PLASTIC_LABELS = {
    "plastic, bag", "plastic, packaging", "plastic, film",
    "plastic film", "plastic bag images",
    "plastics", "other plastic wrapper",
    "plastic utensils", "plastic glooves",
    "single-use carrier bag", "plastic straw",
    "crisp packet", "polypropylene bag",
    "retort_pouch", "six pack rings",
    "waste, styrofoam", "styrofoam piece",
    "foam food container", "foam cup",
    "spread tub", "squeezable tube",
    "other plastic container", "other plastic cup",
    "plastic utensils",
}

def run_plastic_segregation():
    print("=== INITIATING PLASTIC SEGREGATION ENGINE ===")
    
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    sources = meta.get("sources", meta)
    
    hard_count = 0
    soft_count = 0
    clean_items = []
    
    for item in sources:
        if item.get("label") == "plastic":
            source_lbl = item.get("source_label", "").lower().strip()
            
            if source_lbl in HARD_PLASTIC_LABELS:
                target_label = "hard_plastic"
            elif source_lbl in SOFT_PLASTIC_LABELS:
                target_label = "soft_plastic"
            else:
                clean_items.append(item)
                continue
            
            original_path = item.get("file_path")
            target_dir = os.path.join(DATASET_ROOT, target_label)
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, os.path.basename(original_path))
            
            if os.path.exists(original_path):
                try:
                    shutil.move(original_path, new_path)
                    item["label"] = target_label
                    item["file_path"] = new_path
                    if target_label == "hard_plastic":
                        hard_count += 1
                    else:
                        soft_count += 1
                except:
                    pass
        
        clean_items.append(item)
    
    print(f"[SUCCESS] Re-classified {hard_count} -> hard_plastic")
    print(f"[SUCCESS] Re-classified {soft_count} -> soft_plastic")
    
    with open(METADATA_PATH, "w") as f:
        json.dump({"sources": clean_items}, f, indent=2)
    
    from integrate_external_dataset import regenerate_auto_split_manifest
    try:
        regenerate_auto_split_manifest(Path(DATASET_ROOT), 42, "0.8,0.1,0.1")
        print("[*] Manifest regenerated.")
    except Exception as e:
        print(f"Manifest error: {e}")
    
    print("=== PLASTIC SEGREGATION COMPLETE ===")

if __name__ == "__main__":
    run_plastic_segregation()
