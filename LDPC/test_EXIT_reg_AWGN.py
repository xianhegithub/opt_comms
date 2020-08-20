"""
This script calculate the EXIT chart for regular LDPC ensemble
"""
import matplotlib.pyplot as plt
from EXIT_AWGN import *


dv = 3
dc = 6
code_rate = dv/dc
Ps = 1
EbN0_dB = 1.1

I_ev, I_ec, I_a = exit_reg_awgn(dv, dc, EbN0_dB)


fig = plt.figure()
EXIT_chart = fig.add_subplot(1, 1, 1)
EXIT_chart.plot(I_a, I_ev)
EXIT_chart.plot(I_ec, I_a)
EXIT_chart.grid()
EXIT_chart.legend(['variable node', 'check node'])
EXIT_chart.set_xlabel("I_ev(I_ac)")
EXIT_chart.set_ylabel("I_av(I_ec)")
fig.show()

