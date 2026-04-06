import json
import os
import shutil
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# USER-DEFINED THRESHOLD FOR CHERRY-PICKING
def is_smart_match(item_name, search_term):
    if not search_term: return True # Stage 1 fallback
    target = search_term.lower().replace("_", " ").strip()
    item = item_name.lower().replace("_", " ").strip()
    if target in item: return True
    if target.rstrip('s') in item or target + 's' in item: return True
    return False

def get_file_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(81920), b''): # 80KB chunks for speed
                h.update(chunk)
        return h.hexdigest()
    except:
        return None

def worker_task(file_info):
    file_path, dest_label, source_name, search_term = file_info
    filename = os.path.basename(file_path)
    parent_folder = os.path.basename(os.path.dirname(file_path))
    
    # SMART MATCHING
    is_match = is_smart_match(filename, search_term) or is_smart_match(parent_folder, search_term)
    
    file_hash = get_file_hash(file_path)
    if not file_hash:
        return None
        
    return {
        "file_path": file_path,
        "hash": file_hash,
        "dest_label": dest_label if is_match else "trash",
        "source_name": source_name,
        "parent_folder": parent_folder,
        "is_match": is_match
    }

def run_turbo_jit_integration():
    DATASET_ROOT = Path("Dataset_Final")
    EXT_ROOT = Path("external_datasets/mega_scourge_phase_10")
    METADATA_PATH = DATASET_ROOT / "dataset_metadata.json"

    print("\n=== STARTING PHASE 14.1 JIT-PURGE INTEGRATOR ===")
    
    if not METADATA_PATH.exists():
        print("Error: metadata.json not found!")
        return

    with open(METADATA_PATH, 'r') as f:
        master_meta = json.load(f)

    master_sources = master_meta.get("sources", master_meta)
    seen_hashes = {item['hash'] for item in master_sources if 'hash' in item}
    
    if not EXT_ROOT.exists():
        print(f"Nothing to integrate in {EXT_ROOT}")
        return

    # GATHER ALL DATASET FOLDERS
    dataset_queue = []
    for process_bin in EXT_ROOT.iterdir():
        if not process_bin.is_dir(): continue
        for dataset_folder in process_bin.iterdir():
            if not dataset_folder.is_dir(): continue
            dataset_queue.append((dataset_folder, process_bin.name))

    print(f"[*] Found {len(dataset_queue)} datasets to process sequentially.")
    
    total_reclaimed = 0
    total_accepted = 0
    
    # SHARED POOL (Re-use for efficiency)
    with ProcessPoolExecutor(max_workers=16) as executor:
        for idx, (dataset_folder, dest_label) in enumerate(dataset_queue):
            source_name = dataset_folder.name
            print(f"\n[{idx+1}/{len(dataset_queue)}] Locking onto: {source_name} ({dest_label})")
            
            # Load search term metadata
            search_query = ""
            meta_path = dataset_folder / "search_query.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        q_data = json.load(f)
                        search_query = q_data.get("query", "")
                except Exception as e:
                    print(f"    [!] Corrupted search_query.json detected. Treating as empty.")
                    search_query = ""
            
            # Gather tasks for THIS dataset
            dataset_tasks = []
            for path in dataset_folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    dataset_tasks.append((str(path), dest_label, source_name, search_query))
            
            if not dataset_tasks:
                print(f"    [!] No images found. Purging folder...")
                shutil.rmtree(dataset_folder)
                continue

            print(f"    [*] Dispatching {len(dataset_tasks)} images...")
            
            futures = {executor.submit(worker_task, task): task for task in dataset_tasks}
            
            dataset_accepted = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    f_hash = result['hash']
                    if f_hash not in seen_hashes:
                        # Physically move/copy file
                        orig_path = Path(result['file_path'])
                        final_label = result['dest_label']
                        
                        target_dir = DATASET_ROOT / final_label
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        dest_file = target_dir / f"{result['source_name']}_{f_hash[:8]}{orig_path.suffix}"
                        shutil.copy2(orig_path, dest_file)
                        
                        master_sources.append({
                            "file_path": str(dest_file),
                            "label": final_label,
                            "source_name": f"mega_scourge_p10/{result['source_name']}",
                            "source_label": result['parent_folder'],
                            "hash": f_hash
                        })
                        seen_hashes.add(f_hash)
                        dataset_accepted += 1
            
            # RECLAIM SPACE
            folder_size = sum(f.stat().st_size for f in dataset_folder.rglob('*') if f.is_file())
            shutil.rmtree(dataset_folder)
            total_reclaimed += folder_size
            total_accepted += dataset_accepted
            
            print(f"    [SUCCESS] Ingested {dataset_accepted} images. RECLAIMED: {folder_size / (1024**3):.2f} GB.")

    # SAVE MASTER METADATA
    with open(METADATA_PATH, 'w') as f:
        json.dump({"sources": master_sources}, f, indent=2)

    print(f"\n=== JIT-PURGE COMPLETE! ===")
    print(f"Total Accepted: {total_accepted}")
    print(f"Total Disk Space Reclaimed: {total_reclaimed / (1024**3):.2f} GB")

    print("\nRegenerating Auto Split Manifest...")
    sys.path.append('scripts')
    from integrate_external_dataset import regenerate_auto_split_manifest
    try:
        regenerate_auto_split_manifest(DATASET_ROOT, 42, "0.8,0.1,0.1")
        print("ALL PROCESSES FINISHED FLAWLESSLY!")
    except Exception as e:
        print(f"Manifest error: {e}")

if __name__ == "__main__":
    run_turbo_jit_integration()
