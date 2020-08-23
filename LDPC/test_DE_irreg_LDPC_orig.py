"""
This script runs the original density evolution for irregular LDPC code ensemble in AWGN channel.
Ref. [1] Channel Codes classical and modern - - William E.Ryan and Shu Lin Algorithm 9.1 and Example 9.2
"""
import matplotlib.pyplot as plt
from DE_orig_LDPC_AWGN import *


iter_max = 10000
Pe_th = 1e-6
on_remote = 1

lmbda = np.zeros(20)
rho = np.zeros(9)
lmbda[1] = 0.23403
lmbda[2] = 0.21242
lmbda[5] = 0.1469
lmbda[6] = 0.10284
lmbda[19] = 0.30381
rho[7] = 0.71875
rho[8] = 0.28125
ll = 0
ebn0_db = 0.627
ebn0 = 10 ** (ebn0_db/10)
sigma = 0.96#np.sqrt(1/ebn0)
m_sup = [-30, 30, 6001]
z_sup = [-10, -2e-4, 50000]

pc_0, __ = ch_msg(m_sup, sigma)
pe_res = de_irreg_ldpc_awgn_orig(pc_0, iter_max, m_sup, z_sup, Pe_th, lmbda, rho)

if not on_remote:
    plt.subplot(111)
    plt.semilogy(pe_res[1:])
    plt.title('Density Evolution')
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Error Probability")
    plt.show()

