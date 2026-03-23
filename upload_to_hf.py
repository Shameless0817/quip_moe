#!/usr/bin/env python
"""
Upload mixtral_8x7b_quip_full folder to Hugging Face Hub
"""
import os
from huggingface_hub import HfApi, create_repo
import argparse

def upload_model(repo_name, folder_path, private=False):
    """
    Upload a folder to Hugging Face Hub
    
    Args:
        repo_name: Name of the repository (e.g., "mixtral-8x7b-quip-2bit")
        folder_path: Path to the folder to upload
        private: Whether the repository should be private
    """
    api = HfApi()
    
    # Get username
    user_info = api.whoami()
    username = user_info['name']
    full_repo_id = f"{username}/{repo_name}"
    
    print(f"Preparing to upload to: {full_repo_id}")
    print(f"Folder: {folder_path}")
    print(f"Private: {private}")
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: {full_repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Upload folder
    print(f"\nUploading files from {folder_path}...")
    print("This may take a while for large files (12GB)...")
    
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=full_repo_id,
            repo_type="model",
            commit_message="Upload quantized Mixtral-8x7B model (QuIP# 2-bit)"
        )
        print(f"\n✓ Upload complete!")
        print(f"View your model at: https://huggingface.co/{full_repo_id}")
    except Exception as e:
        print(f"Error during upload: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--repo-name",
        type=str,
        default="mixtral-8x7b-quip-2bit",
        help="Name of the repository on Hugging Face (default: mixtral-8x7b-quip-2bit)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="/fact_home/zeyuli/quip_sharp/mixtral_8x7b_quip_full",
        help="Path to the folder to upload"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    args = parser.parse_args()
    
    # Verify folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder {args.folder} does not exist!")
        exit(1)
    
    print(f"Found folder: {args.folder}")
    folder_size = os.popen(f'du -sh "{args.folder}"').read().split()[0]
    print(f"Size: {folder_size}")
    
    # Confirm before upload
    response = input(f"\nUpload to {args.repo_name}? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Upload cancelled.")
        exit(0)
    
    upload_model(args.repo_name, args.folder, args.private)
