from trainer.filequeue import *
from multiprocess.dummy import Pool
import time


def fn(arg):
    with FileQueue(n=4, verbose=False):
        time.sleep(1)
        print(f'run: {arg}')

start = time.time()
with Pool(10) as pool:
    pool.map(fn, range(10))
print(time.time() - start)