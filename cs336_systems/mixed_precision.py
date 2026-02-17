import torch
import torch.nn as nn


def accumulation_mixed_precision():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(f"FP32 total: {s.item():.4f}") 
    
    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(f"FP16 total: {s.item():.4f}") 
    
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(f"Mixed total: {s.item():.4f}") 
    
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(f"Manual Upcast total: {s.item():.4f}") 

from torch.amp import autocast

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        print(f"FC1 output: {x_fc1.dtype}")
        
        x = self.relu(x_fc1)
        
        x_ln = self.ln(x)
        print(f"LN output:  {x_ln.dtype}")
        
        logits = self.fc2(x_ln)
        print(f"Logits:     {logits.dtype}")
        return logits

def main():
    device = "cuda"
    model = ToyModel(5, 2).to(device)
    x = torch.randn(8, 5).to(device)
    target = torch.randint(0, 2, (8,)).to(device)
    criterion = nn.CrossEntropyLoss()
    print(f"Params:     {model.fc1.weight.dtype}")
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(x)
        loss = criterion(output, target)
        print(f"Loss:       {loss.dtype}")

    loss.backward()
    print(f"Gradients:  {model.fc1.weight.grad.dtype}")

if __name__ == "__main__":
    main()

