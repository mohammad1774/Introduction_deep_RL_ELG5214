import os 
import platform 
import sys 
from datetime import datetime 

import torch 

def metadata_info(output_path: str = "environment_info.txt") -> None: 
    """
    Writes environment details to a text file: 
    - OS 
    - Python Version
    - GPU Availability
    - CUDA Version
    """

    os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
    python_version = sys.version.replace("\n", " ")

    cuda_avlbl = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_avlbl else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if cuda_avlbl else "N/A"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"ENV INFORMATION",
        f"Generated at: {timestamp}",
        f"OS: {os_info}",
        f"Python Version: {python_version}",
        f"PyTorch Version: {torch.__version__}",
        f"CUDA Available: {cuda_avlbl}",
        f"CUDA Version: {cuda_version}",
        f"GPU Name: {gpu_name}"
    ]

    with open(output_path,"w", encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(lines)
    print(f"Environment Information written into {output_path}")

if __name__ == "__main__":
    metadata_info()