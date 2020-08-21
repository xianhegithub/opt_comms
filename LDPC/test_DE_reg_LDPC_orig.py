"""
This script runs the original density evolution for regular LDPC code ensemble in AWGN channel.
Ref. [1] Channel Codes classical and modern - - William E.Ryan and Shu Lin Algorithm 9.1 and Example 9.2
"""
import matplotlib.pyplot as plt
from DE_orig_LDPC_AWGN import *


iter_max = 10000
Pe_th = 1e-6

dv = 3
dc = 6
ll = 0
sigma = 0.6
m_sup = [-30, 30, 6001]
z_sup = [-10, -2e-4, 50000]

pc_0, __ = ch_msg(m_sup, sigma)
pe_res = de_reg_ldpc_awgn_orig(pc_0, iter_max, m_sup, z_sup, Pe_th, dv, dc)

plt.subplot(111)
plt.semilogy(pe_res[1:])
plt.title('Density Evolution')
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Error Probability")
plt.show()

