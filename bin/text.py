import torch

if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
else:
    print("❌ CUDA is NOT available. Using CPU.")
