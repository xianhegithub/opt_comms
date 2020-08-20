import numpy as np
"""
This is a set of functions that calculate the EXIT chart for regular and irregular LDPC code 
in AWGN channel.
"""


def exit_reg_awgn(dv, dc, ebn0_db):
    """
    This function calculates the EXIT chart for regular LDPC ensemble
    Ref. [1] Channel Codes classical and modern -- William E. Ryan and Shu Lin (9.42) (9.43)
    """

    code_rate = dv / dc
    ebn0 = 10 ** (ebn0_db / 10)
    sigma_ch = np.sqrt(8 * code_rate * ebn0)
    i_a = np.linspace(0, 1, num=21)

    i_ev = iev_iav(i_a, sigma_ch, dv)

    i_ec = iec_iac(i_a, dc)

    return i_ev, i_ec, i_a


def exit_irreg_awgn(lmbda, rho, ebn0_db):
    """
    This function calculates the EXIT chart for irregular LDPC ensemble
    Ref. [1] Channel Codes classical and modern -- William E. Ryan and Shu Lin (9.44) (9.45)
    """

    code_rate = 1 - np.divide(rho, list(range(1, len(rho) + 1))).sum() / np.divide(lmbda, list(range(1, len(lmbda) + 1))).sum()
    ebn0 = 10 ** (ebn0_db / 10)
    sigma_ch = np.sqrt(8 * code_rate * ebn0)

    i_a = np.linspace(0, 1, num=21)

    i_ev = np.zeros(i_a.size)

    for ii in range(len(lmbda)):
        i_ev = lmbda[ii] * iev_iav(i_a, sigma_ch, ii + 1) + i_ev

    i_ec = np.zeros(i_a.size)

    for ii in range(len(rho)):
        i_ec = rho[ii] * iec_iac(i_a, ii + 1) + i_ec

    return i_ev, i_ec, i_a


def iev_iav(i_a, sigma_ch, dv):
    """
    this function calculate the EXIT curve for variable nodes with degree dv
    Ref. [1] Channel Codes classical and modern -- William E. Ryan and Shu Lin (9.42)
    :param i_a: a priori mutual information that goes into the variable node
    :param sigma_ch: 8 * code_rate * EbN0
    :param dv: variable node degree
    :return: extrinsic mutual information that goes out of the variable node
    """
    i_ev = np.zeros(i_a.size)

    for ii in range(i_a.size):

        tmp = j_sigma_inv(i_a[ii])
        j_arg = ((dv - 1) * tmp ** 2 + sigma_ch ** 2) ** 0.5
        i_ev[ii] = j_sigma(j_arg)

    return i_ev


def iec_iac(i_a,  dc):
    """
    this function calculate the EXIT curve for check nodes with degree dc
    Ref. [1] Channel Codes classical and modern -- William E. Ryan and Shu Lin (9.43)
    :param i_a: a priori mutual information that goes into the check node
    :param dc: check node degree
    :return: extrinsic mutual information that goes out of the check node
    """
    i_ec = np.zeros(i_a.size)

    for ii in range(i_a.size):

        tmp = j_sigma_inv(1 - i_a[ii])
        j_arg = ((dc - 1) * tmp ** 2) ** 0.5
        i_ec[ii] = 1 - j_sigma(j_arg)

    return i_ec


def j_sigma(sigma):
    """
    this function is one of the steps in the EXIT calculation
    Ref. [1] Design of Low-Density Parity-Check Codes for Modulation and
    Detection -- Stephan ten Brink et al. Appendix
    """

    sigma_star = 1.6363
    aj1 = -0.0421061
    bj1 = 0.209252
    cj1 = -0.00640081
    aj2 = 0.00181491
    bj2 = -0.142675
    cj2 = -0.0822054
    dj2 = 0.0549608

    if 0 <= sigma <= sigma_star:

        out = np.multiply([aj1, bj1, cj1], np.power(sigma, [3, 2, 1])).sum()

    elif sigma > sigma_star:

        out = 1 - np.exp(np.multiply([aj2, bj2, cj2, dj2], np.power(sigma, [3, 2, 1, 0])).sum())

    else:

        out = 1

    return out


def j_sigma_inv(ei):
    """
    this function is one of the steps in the EXIT calculation
    Ref.[1] Design of Low - Density Parity - Check Codes for Modulation and
    Detection - - Stephan ten Brink et al. Appendix
    """

    ei_star = 0.3646
    as1 = 1.09542
    bs1 = 0.214217
    cs1 = 2.33727
    as2 = 0.706692
    bs2 = 0.386013
    cs2 = -1.75017

    if 0 <= ei <= ei_star:

        out = np.multiply([as1, bs1, cs1], np.power(ei, [2, 1, 0.5])).sum()

    elif ei_star < ei < 1:

        out = - as2 * np.log(bs2 * (1 - ei)) - cs2 * ei

    else:

        out = 10
        print('Numerical error in the inverse J_sigma function\n')

    return out
