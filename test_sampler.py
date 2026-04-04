import torch
import time
def test():
    t0 = time.time()
    def gen():
        batch_size = 100000
        for _ in range(100):
            yield from torch.randint(0, 1000000, (batch_size,)).tolist()
    
    count = 0
    for x in gen():
        count += 1
    print(f"Time for {count} samples: {time.time() - t0:.2f}s")

test()
