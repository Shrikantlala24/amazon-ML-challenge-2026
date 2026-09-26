import torch

# Check if CUDA is available
has_cuda = torch.cuda.is_available()
print("CUDA Available:", has_cuda)
if has_cuda:
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM Available (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
