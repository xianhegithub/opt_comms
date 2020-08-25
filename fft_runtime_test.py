import mkl_fft
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

sig = np.random.random(2**18) + np.random.random(2**18) * 1j



start = timer()
for i in range(100):
    mkl_fft.fft(sig)
end = timer()
print("mkl_fft.fft, 1 slice", (end - start)/100)

start = timer()
for i in range(100):
    np.fft.fft(sig)
end = timer()
print("np.fft.fft, 1 slice", (end - start)/100)


