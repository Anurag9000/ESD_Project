import os
import json
from pathlib import Path

DATASET_ROOT = Path("Dataset_Final")
METADATA_FILE = DATASET_ROOT / "dataset_metadata.json"

# Rules for purging: dictionary mapping class_name to list of forbidden signature substrings in filename
# If class_name is "*", applies to all classes.
PURGE_RULES = {
    "battery": ["spent-lithium-ion", "waste_classification__hazardous"],
    "*": ["taco__annotations"]
}

def matches_purge_rule(class_name, filename):
    fname_lower = filename.lower()
    
    # Check class-specific rules
    if class_name in PURGE_RULES:
        for rule in PURGE_RULES[class_name]:
            if rule.lower() in fname_lower:
                return True
                
    # Check global rules
    if "*" in PURGE_RULES:
        for rule in PURGE_RULES["*"]:
            if rule.lower() in fname_lower:
                return True
                
    return False

def main():
    print("🧹 Starting targeted source purge...")
    
    deleted_count = 0
    valid_files_set = set()

    for class_dir in DATASET_ROOT.iterdir():
        if not class_dir.is_dir() or class_dir.name in ["hard_plastic", "soft_plastic"] or class_dir.name.startswith("."):
            continue
            
        class_name = class_dir.name
        
        for file_path in class_dir.iterdir():
            if not file_path.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                continue
                
            if matches_purge_rule(class_name, file_path.name):
                file_path.unlink()
                deleted_count += 1
            else:
                # Add to set of valid paths exactly as they appear in metadata
                rel_path = f"Dataset_Final/{class_name}/{file_path.name}"
                valid_files_set.add(rel_path)
                
    print(f"✅ Deleted {deleted_count} files matching purge rules.")
    
    # Sync metadata
    if METADATA_FILE.exists():
        print("🔄 Syncing dataset_metadata.json...")
        with open(METADATA_FILE, 'r') as f:
            meta = json.load(f)
            
        initial_meta_len = len(meta)
        new_meta = [entry for entry in meta if entry.get('file_path') in valid_files_set]
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(new_meta, f, indent=2)
            
        print(f"✅ Metadata sync complete. Removed {initial_meta_len - len(new_meta)} entries.")
        print(f"📊 Final metadata track count: {len(new_meta)}")

        # Also remove auto_split_manifest as it is now invalid and needs to be rebuilt by the pipeline
        manifest = DATASET_ROOT / "auto_split_manifest.json"
        if manifest.exists():
            manifest.unlink()
            print("🗑️ Deleted stale auto_split_manifest.json.")
            
if __name__ == "__main__":
    main()
