from tqdm import tqdm
import time

for i in tqdm(range(100_000), mininterval=1):
    time.sleep(0.005)