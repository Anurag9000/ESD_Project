import os
import logging
from tqdm import tqdm
from huggingface_hub import HfApi

# Configure Formal Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='archival_upload.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

def upload_wss_dataset():
    api = HfApi()
    
    username = "Anurag1011" 
    repo_id = f"{username}/WSS-1.04M"
    file_path = "Dataset_Final.zip"
    path_in_repo = "WSS-1.04M_Master_Baseline.zip"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    file_size = os.path.getsize(file_path)
    
    print(f"--- WSS-1.04M RESEARCH UPLINK ---")
    print(f"Target Repository: {repo_id}")
    print(f"File Size: {file_size / (1024**3):.2f} GB")
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        logger.info(f"Repository {repo_id} verified.")
    except Exception as e:
        logger.error(f"Error creating repo: {e}")

    print(f"\nStarting 58GB Uplink. Streaming via binary buffer...")

    # Manual progress bar that updates as the API consumes the file object
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Uploading")

    def read_callback(chunk_size):
        # This is a simple wrapper to update the bar
        pbar.update(chunk_size)

    try:
        # We pass the file path directly. 
        # In older HF Hub versions, if 'callback' isn't supported, 
        # the best way is to let HF Hub handle the file and just trust the OS buffer.
        # However, to ENSURE a bar, we use the path_or_fileobj as the actual file.
        
        with open(file_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
        
        pbar.n = file_size # Force bar to 100% on success
        pbar.refresh()
        pbar.close()
        print("\n--- MISSION ACCOMPLISHED: WSS-1.04M IS LIVE ---")
        logger.info("Upload completed successfully.")
        
    except Exception as e:
        pbar.close()
        print(f"\nUpload failed: {e}")
        logger.error(f"Upload failed: {e}")

if __name__ == "__main__":
    upload_wss_dataset()