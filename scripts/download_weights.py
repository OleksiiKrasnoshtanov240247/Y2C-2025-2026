import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_models():
    print("==================================================")
    print(" Downloading Official HuggingFace Checkpoints ")
    print("==================================================")

    # 1. Wav2Lip GAN
    print("\n--- Pulling Wav2Lip GAN ---")
    hf_hub_download(
        repo_id="Nekochu/Wav2Lip", 
        filename="wav2lip_gan.pth", 
        local_dir="Baseline_video/Wav2Lip/checkpoints",
        local_dir_use_symlinks=False
    )
    print("[OK] Wav2Lip secured.")

    # 2. Deep-Live-Cam Inswapper 128 (bypassing GitHub LFS HTML pointer)
    print("\n--- Pulling Deep-Live-Cam face swapper ---")
    hf_hub_download(
        repo_id="hacksider/deep-live-cam", 
        filename="inswapper_128.onnx", 
        local_dir="realtime_video/Deep-Live-Cam/models",
        local_dir_use_symlinks=False
    )
    print("[OK] Deep-Live-Cam secured.")

    # 3. LivePortrait Array
    print("\n--- Pulling LivePortrait Checkpoints Array ---")
    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir="Baseline_video/LivePortrait/pretrained_weights"
    )
    print("[OK] LivePortrait array secured.")

    print("\n==================================================")
    print(" All weights securely downloaded via HuggingFace ")
    print("==================================================")

if __name__ == "__main__":
    download_models()
