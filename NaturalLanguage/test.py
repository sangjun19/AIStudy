import torch
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
print(f"현재 CUDA 버전: {torch.version.cuda}")
print(f"사용 GPU: {torch.cuda.get_device_name(0)}")