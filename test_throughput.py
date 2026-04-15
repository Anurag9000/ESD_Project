import time
import torch
import gc
from torch.cuda.amp import autocast
from torchvision.models import efficientnet_b0
import random

# Hardware setup
device = torch.device('cuda')
model = efficientnet_b0().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda')
criterion = torch.nn.CrossEntropyLoss()

# Synthetic data
batch_size = 64
dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
dummy_labels = torch.randint(0, 8, (batch_size,), device=device)

# Warmup
for _ in range(5):
    optimizer.zero_grad()
    with autocast():
        outs = model(dummy_images)
        loss = criterion(outs, dummy_labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
iters = 50
for _ in range(iters):
    optimizer.zero_grad()
    with autocast():
        outs = model(dummy_images)
        loss = criterion(outs, dummy_labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

torch.cuda.synchronize()
end = time.perf_counter()

total_images = batch_size * iters
total_time = end - start
fps = total_images / total_time
print(f"GPU Throughput: {fps:.1f} images/sec")
