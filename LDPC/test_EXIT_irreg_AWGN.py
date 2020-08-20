"""
This script calculate the EXIT chart for irregular LDPC ensemble
"""
import matplotlib.pyplot as plt
from EXIT_AWGN import *


# bit node degree polynomial
lmbda = [0, 0.267, 0.176, 0.127, 0, 0, 0, 0, 0, 0.43]
# check node degree polynomial
rho = [0, 0, 0, 0, 0.113, 0, 0, 0.887]
code_rate = 1 - np.divide(rho, list(range(1, len(rho)+1))).sum() / np.divide(lmbda, list(range(1, len(lmbda)+1))).sum()
print(code_rate)

Ps = 1
EbN0_dB = 0.55
EbN0 = 10 ** (EbN0_dB/10)
sigma = np.sqrt(Ps/EbN0)
sigma_ch = np.sqrt(8 * code_rate * EbN0)

I_ev, I_ec, I_a = exit_irreg_awgn(lmbda, rho, EbN0_dB)


fig = plt.figure()
EXIT_chart = fig.add_subplot(1, 1, 1)
EXIT_chart.plot(I_a, I_ev)
EXIT_chart.plot(I_ec, I_a)
EXIT_chart.grid()
EXIT_chart.legend(['variable node', 'check node'])
EXIT_chart.set_xlabel("I_ev(I_ac)")
EXIT_chart.set_ylabel("I_av(I_ec)")
fig.show()