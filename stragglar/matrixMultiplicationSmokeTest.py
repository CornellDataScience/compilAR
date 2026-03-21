import torch
import time

SIZE = 4096
WARMUP = 5
RUNS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}")

a = torch.randn(SIZE, SIZE, device=device)
b = torch.randn(SIZE, SIZE, device=device)

# warmup
for _ in range(WARMUP):
    _ = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()

# timed runs
times = []
for _ in range(RUNS):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    times.append(end - start)

avg = sum(times) / len(times)
print(f"Matrix size: {SIZE}x{SIZE}")
print(f"Avg time over {RUNS} runs: {avg*1000:.2f} ms")