import mkl_fft
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pyfftw

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

start = timer()
a = pyfftw.empty_aligned(2**18, dtype='complex64')
b = pyfftw.empty_aligned(2**18, dtype='complex64')
fft_object = pyfftw.FFTW(a, b)
end = timer()
print("pyfftw overhead", (end - start))
a[:] = sig

start = timer()
for i in range(100):
    fft_object()
end = timer()
print("pyfftw, 1 slice", (end - start)/100)

