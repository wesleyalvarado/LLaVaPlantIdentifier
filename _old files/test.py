import torch

def test_m2_torch():
    # Test basic tensor creation
    print("Creating test tensor...")
    x = torch.rand(3, 3)
    print(x)
    
    # Check M2 (MPS) availability
    print("\nChecking M2 support...")
    print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
    print(f"MPS (Metal) built: {torch.backends.mps.is_built()}")
    
    # Try creating a tensor on MPS if available
    if torch.backends.mps.is_available():
        print("\nCreating tensor on M2 GPU...")
        device = torch.device("mps")
        x_gpu = x.to(device)
        print(f"Tensor device: {x_gpu.device}")
        
    print("\nPyTorch version:", torch.__version__)

if __name__ == "__main__":
    test_m2_torch()