import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")
    x = torch.ones(10).cuda()
    print("Tensor on GPU:", x.device)
    print("OK - RTX 5060 ready for training")
else:
    print("CUDA not available")
