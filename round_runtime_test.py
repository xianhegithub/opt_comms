import time
import numpy as np

TOLERANCE = 3
some_float = 12.2439924563634564564564
nb_loops = 1000

start_time = time.time()
for _ in range(nb_loops):
    np.int_(np.random.random(2**18)+0.5)
print('np.int_', time.time()-start_time)

start_time = time.time()
for _ in range(nb_loops):
    (np.random.random(2**18)+0.5).astype(int)
print('astype', time.time()-start_time)

start_time = time.time()
for _ in range(nb_loops):
    np.array(np.random.random(2**18)+0.5, dtype=int)
print('dtype', time.time()-start_time)


