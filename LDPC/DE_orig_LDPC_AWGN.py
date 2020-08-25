import numpy as np
from scipy import stats
from scipy.special import comb
import mkl_fft
"""
This is a set of functions that calculate the original density evolution for regular and 
irregular LDPC code in AWGN channel.
"""


def de_reg_ldpc_awgn_orig(pc_0, itermax, m_sup, z_sup, pe_th, dv, dc):
    """
    This function runs the original density evolution for regular LDPC code ensemble in AWGN channel.
    Ref. [1] Channel Codes classical and modern - - William E.Ryan and Shu Lin Algorithm 9.1 and Example 9.2
    :param pc_0: pdf of the message from the channel
    :param itermax: maximum number of iterations for the density evolution
    :param m_sup: vector that generate the grid of m value, m stands for the message passed between nodes.
    m_sup is of form [m_min, m_max, num_ele].
    :param z_sup: vector that generate the grid of z value, z is the phi transform of m.
    z_sup is of form [z_min, z_max, num_ele].
    :param pe_th: the threshold of error probability
    :param dv: variable node degree
    :param dc: check node degree
    :return pe_res: error probability at each iteration above the threshold
    """
    pv = pc_0
    ll = 0
    pe_curr = 0.5
    m_inc = (m_sup[1] - m_sup[0]) / (m_sup[2] - 1)
    pe_res = np.zeros(itermax)

    while ll < itermax and pe_curr > pe_th:

        pe_res[ll] = pe_curr
        ll = ll + 1
        pc = cn_update(m_sup, pv, dc - 1, z_sup)
        pv = vn_update(pc_0, pc, dv - 1, m_sup, m_sup)
        pe_curr = pv[:int((m_sup[2] - 1) / 2 + 1 + 0.5)].sum() * m_inc
        print(pe_curr)

    pe_res[ll] = pe_curr

    return pe_res[:ll+1]


def de_irreg_ldpc_awgn_orig(pc_0, itermax, m_sup, z_sup, pe_th, lmbda, rho):

    pv_aver = pc_0
    ll = 0
    pe_curr = 0.5
    m_inc = (m_sup[1] - m_sup[0])/(m_sup[2] - 1)
    pe_res = np.zeros(itermax)
    max_dv = len(lmbda)
    max_dc = len(rho)

    while ll < itermax and pe_curr > pe_th:

        pe_res[ll] = pe_curr
        ll = ll + 1

        pc_aver = np.zeros(len(pc_0))
        for idx in range(1, max_dc):
            if rho[idx] != 0:
                pc = cn_update(m_sup, pv_aver, idx, z_sup)
                pc_aver = pc_aver + rho[idx] * pc

        pv_aver = np.zeros(len(pc_0))
        for idx in range(1, max_dv):
            if lmbda[idx] != 0:
                pv = vn_update(pc_0, pc_aver, idx, m_sup, m_sup)
                pv_aver = pv_aver + lmbda[idx] * pv

        pe_curr = pv_aver[:int((m_sup[2] - 1) / 2 + 1 + 0.5)].sum() * m_inc
        print(pe_curr)

    return pe_res


def ch_msg(m_sup, sigma):
    """
    channel output signal is of 'distribution is N(2/sigma^2,4/sigma^2)
    Ref. [1] Channel Codes classical and modern -- William E. Ryan and Shu Lin
    page 394, example 9.2
    """
    mu_ch = 2/(sigma ** 2)
    var_ch_sqrt = 2/sigma
    m_inc = (m_sup[1] - m_sup[0])/(m_sup[2] - 1)
    m_grid = np.linspace(m_sup[0], m_sup[1], m_sup[2])
    pc0 = stats.norm.pdf(m_grid, mu_ch, var_ch_sqrt)

    return pc0, m_grid


def cn_update(m_sup, pv, dcm1, z_sup):
    """
    This function updates the check nodes pdf in density evolution.
    :param m_sup: vector that generate the grid of m value, m stands for the message passed between nodes.
    m_sup is of form [m_min, m_max, num_ele].
    :param pv: pdf of variable check nodes
    :param dcm1: check node degree minus 1, i.e., dc - 1
    :param z_sup: vector that generate the grid of z value, z is the phi transform of m.
    z_sup is of form [z_min, z_max, num_ele].
    :return pm_update: p^{(c)} in Algorithm 9.1 step 3 [1]
    Ref. [1] Channel Codes classical and modern - - William E.Ryan and Shu Lin Algorithm 9.1 step 3
    """
    m_inc = (m_sup[1] - m_sup[0]) / (m_sup[2] - 1)
    z_inc = (z_sup[1] - z_sup[0]) / (z_sup[2] - 1)

    p0_zi, p1_zi, excess = phi_trans(m_sup, pv, z_sup)

    # the probability density of pv(0) is not handled in the above transform,
    # but separately in the each step followed.
    p_zero = pv[int((m_sup[2] - 1) / 2 + 0.5)] * m_inc
    # z_uni_extn contains the new bins for the convolved function, because
    # convolution will expand the support of the pdf, unavoidably
    z_extn_sup = [z_sup[0]*dcm1, z_sup[0]*dcm1 + z_inc*z_sup[2]*dcm1, z_sup[2]*dcm1 + 1]

    pomg_pos, pomg_neg, p_res_zero = cn_fft_convolve(p0_zi, p1_zi, dcm1, z_extn_sup, excess, p_zero)

    pm_update, ofl_pos, ofl_neg = phi_trans_inv(pomg_pos, pomg_neg, p_res_zero, z_extn_sup, m_sup)

    pm_update = cn_overflow(pm_update, ofl_pos, ofl_neg, m_sup, z_sup)

    # normalisation of the pdf to sum 100
    pm_update = pm_update / (pm_update.sum() * m_inc)

    return pm_update


def phi_trans(m_sup, pv, z_sup):
    """
    converts a probability from LLR to log(tanh(L/2)) form
    preserving sign information
    note, the zero is dropped! ... but not the underflow
    :param m_sup: vector that generate the grid of m value, m stands for the
    message passed between nodes. m_sup is of form [m_min, m_max, num_ele]
    :param pv: pdf of variable nodes
    :param z_sup: vector that generate the grid of z value, z is the phi transform of m.
    z_sup is of form [z_min, z_max, num_ele].
    :return p0_zi, p1_zi:
    Ref. [1] Channel Codes classical and modern - - William E.Ryan and Shu Lin Algorithm 9.1 step 3, sub-step 1
    """

    z_uni_grid = np.linspace(z_sup[0], z_sup[1], z_sup[2])

    m_inc = (m_sup[1] - m_sup[0])/(m_sup[2] - 1)
    lw_up_grid = np.zeros([2, int((m_sup[2]-1)/2)])
    tmp_min = m_inc/2
    tmp_max = m_inc/2 + m_inc * (m_sup[2]-3)/2
    lw_up_grid[0, :] = np.linspace(tmp_min, tmp_max, int((m_sup[2]-1)/2))
    tmp_min = m_inc/2 + m_inc
    tmp_max = m_inc/2 + m_inc * (m_sup[2]-3)/2 + m_inc
    lw_up_grid[1, :] = np.linspace(tmp_min, tmp_max, int((m_sup[2]-1)/2))
    lw_up_grid[0, 0] = m_inc
    lw_up_grid[1, -1] = m_inc * (m_sup[2]-3)/2 + m_inc

    z_non_grid = np.log(np.tanh(lw_up_grid/2))
    coeff = 2 * np.exp(z_uni_grid)/(1 - np.exp(2 * z_uni_grid))

    excess = np.zeros(4)
    pv_pos = pv[int((m_sup[2]-1)/2)+1:]
    p0_zi, ofl, ufl = pm2pz2pm(m_inc, pv_pos, z_non_grid, z_sup, coeff)
    excess[0] = ofl
    excess[1] = ufl

    pv_neg = pv[:int((m_sup[2] - 1)/2)]
    pv_neg_rev = pv_neg[::-1]
    p1_zi, ofl, ufl = pm2pz2pm(m_inc, pv_neg_rev, z_non_grid, z_sup, coeff)
    excess[2] = ofl
    excess[3] = ufl

    return p0_zi, p1_zi, excess


def phi_trans_inv(pomg_pos, pomg_neg, p_res_zero, z_extn_sup, m_sup):
    """
    converts a log(tanh(L/2)) form random variable to LLR
    recall that sign information is preserved
    :param pomg_pos: p(\omega) z>0
    :param pomg_neg: p(\omega) z<0
    :param p_res_zero: probability density of z = 0
    :param z_extn_sup: vector that generate the EXTENDED grid of z value due to convolution,
    zzextn_sup is of form [z_min, z_max, num_ele].
    :param m_sup: vector that generate the grid of m value, m stands for the
    message passed between nodes. m_sup is of form [m_min, m_max, num_ele]
    :return:
    Ref. [1] Channel Codes classical and modern - - William E.Ryan and Shu Lin Algorithm 9.1 step 3, sub-step 3
    """

    m_inc = (m_sup[1] - m_sup[0]) / (m_sup[2] - 1)
    m_pos_min = m_inc
    m_pos_num = int((m_sup[2] - 1)/2)
    m_pos_max = m_pos_min + m_inc * (m_pos_num - 1)
    m_pos_sup = [m_pos_min, m_pos_max, m_pos_num]
    m_pos_grid = np.linspace(m_pos_min, m_pos_max, m_pos_num)

    z_extn_inc = (z_extn_sup[1] - z_extn_sup[0]) / (z_extn_sup[2] - 1)
    lw_up_grid = np.zeros([2, z_extn_sup[2]])
    tmp_min = z_extn_sup[0] - z_extn_inc / 2
    tmp_max = z_extn_sup[0] - z_extn_inc / 2 + z_extn_inc * (z_extn_sup[2] - 1)
    lw_up_grid[0, :] = np.linspace(tmp_min, tmp_max, z_extn_sup[2])
    tmp_min = z_extn_sup[0] + z_extn_inc / 2
    tmp_max = z_extn_sup[0] + z_extn_inc / 2 + z_extn_inc * (z_extn_sup[2] - 1)
    lw_up_grid[1, :] = np.linspace(tmp_min, tmp_max, z_extn_sup[2])
    lw_up_grid[0, 0] = z_extn_sup[0]
    tmp = z_extn_sup[0] + z_extn_inc * (z_extn_sup[2] - 1)
    if tmp == 0:
        tmp = -1.e-6
    lw_up_grid[1, -1] = tmp
    # this is just 2 times atanh(lw_up_grid)
    m_non_grid = np.log((1+np.exp(lw_up_grid)) / (1-np.exp(lw_up_grid)))

    tmp_vc = np.tanh(m_pos_grid / 2)
    coeff = 0.5 / tmp_vc * (1 - np.power(tmp_vc, 2))

    pm_pos, ofl_pos, ufl_pos = pm2pz2pm(z_extn_inc, pomg_pos, m_non_grid, m_pos_sup, coeff)
    pm_neg, ofl_neg, ufl_neg = pm2pz2pm(z_extn_inc, pomg_neg, m_non_grid, m_pos_sup, coeff)

    pm_update = np.zeros(m_sup[2])
    tmp_vc = pm_neg[:int((m_sup[2] - 1) / 2)]
    pm_update[:int((m_sup[2] - 1) / 2)] = tmp_vc[::-1]
    pm_update[int((m_sup[2] - 1) / 2 + 1):] = pm_pos
    pm_update[int((m_sup[2] - 1) / 2)] = (p_res_zero + ufl_pos + ufl_neg) / m_inc

    return pm_update, ofl_pos, ofl_neg


def pm2pz2pm(m_inc, pv_half, z_non_grid, z_sup, coeff):

    itmax = len(z_non_grid[0])
    pzi = np.zeros(z_sup[2])
    ofl = 0.
    ufl = 0.
    z_inc = (z_sup[1] - z_sup[0])/(z_sup[2] - 1)
    min_res_bin = z_sup[0] - 0.5 * z_inc
    max_res_bin = z_sup[1] + 0.5 * z_inc
    for cc in range(itmax):
        z_in_z_uni = np.array((z_non_grid[:, cc] - z_sup[0]) / z_inc + 0.5, dtype=int)
        flag = 0
        partflag = 0
        # higher range exceeded by both, this part of pv is added into ofl
        if z_in_z_uni[0] > z_sup[2]-1 and z_in_z_uni[1] > z_sup[2]-1:
            ofl = ofl + pv_half[cc] * m_inc
            flag = 1
        # lower range exceeded by both, this part of pv is added into ufl
        if z_in_z_uni[0] < 0 and z_in_z_uni[1] < 0:
            ufl = ufl + pv_half[cc] * m_inc
            flag = 1
        # lower range exceeded only in one part
        if flag == 0 and z_in_z_uni[0] < 0:
            z_in_z_uni[0] = 0
            z_non_grid[0, cc] = min_res_bin
            partflag = 1
        # higher range exceeded by a single index. Special care needs to be taken to deal with the value inf in Python
        if flag == 0 and (z_in_z_uni[1] >= z_sup[2] or z_in_z_uni[1] == np.iinfo(np.int64).min):
            z_in_z_uni[1] = z_sup[2] - 1
            z_non_grid[1, cc] = max_res_bin
            partflag = 1
        if flag == 0:
            if z_in_z_uni[0] == z_in_z_uni[1]:
                tmp = pv_half[cc] * m_inc/z_inc
                pzi[z_in_z_uni[0]] = pzi[z_in_z_uni[0]] + tmp
            elif z_in_z_uni[1]-z_in_z_uni[0] == 1:
                # find the fractional probabilities associated with each bin
                # the bin boundary is (obviously) halfway between round(2) and round(1)
                bdy = (z_in_z_uni[0] + 0.5) * z_inc + z_sup[0]
                lowfrac = abs(z_non_grid[0, cc] - bdy) / z_inc
                highfrac = abs(z_non_grid[1, cc] - bdy) / z_inc
                tmp = np.multiply([lowfrac, highfrac], pv_half[cc])
                tmp = np.multiply(tmp, [coeff[z_in_z_uni[0]], coeff[z_in_z_uni[1]]])
                pzi[z_in_z_uni[0]: z_in_z_uni[1]+1] = pzi[z_in_z_uni[0]: z_in_z_uni[1]+1] + tmp
            else:
                # find the fractional probabilities associated with the end bins
                # then the probabilities associated with the intervening bins
                lowbdy = (z_in_z_uni[0] + 0.5) * z_inc + z_sup[0]
                highbdy = (z_in_z_uni[1] - 0.5) * z_inc + z_sup[0]
                lowfrac = abs(z_non_grid[0, cc] - lowbdy) / z_inc
                highfrac = abs(z_non_grid[1, cc] - highbdy) / z_inc
                tmp = np.zeros(z_in_z_uni[1] - z_in_z_uni[0] + 1)
                tmp[0] = lowfrac * pv_half[cc]
                tmp[-1] = highfrac * pv_half[cc]
                tmp[1: -1] = pv_half[cc]
                tmp = tmp * coeff[z_in_z_uni[0]:z_in_z_uni[1]+1]
                pzi[z_in_z_uni[0]: z_in_z_uni[1]+1] = pzi[z_in_z_uni[0]: z_in_z_uni[1]+1] + tmp

        if partflag == 1:
            # part of the probability lies outside of the range
            # calculate how much is accounted for; rest is overflow
            pprob = tmp.sum() * z_inc
            if pprob < pv_half[cc] * m_inc:
                if z_in_z_uni[0] < 0:  # underflow
                    ufl = ufl + (pv_half[cc] * m_inc - pprob)
                else:
                    ofl = ofl + (pv_half[cc] * m_inc - pprob)

    return pzi, ofl, ufl


def cn_fft_convolve(p0_zi, p1_zi, dcm1, z_extn_sup, excess, p_zero):
    """
    perform FFTs over R x GF(2) to obtain probabilities with sign information
    I will *assume* that the range of p0_zi and p1_zi are over the range
    z_extn_sup

    the input ofl_pos is the probability of the input "p(z)" message
    being zero ... can directly include this in the sum

    furthermore, by the symmetry property, "p(z)" can only be zero
    if the sign is positive

    find the probabilities of being positive or negative ... used in calculating the
    conditional probability
    these form the extended (by 1) messages, which give room for an extra zero message
    :param p0_zi:
    :param p1_zi:
    :param dcm1: means dc - 1
    :param z_extn_sup:
    :param excess:
    :param p_zero:
    :return:
    Ref. [1] Channel Codes classical and modern - - William E.Ryan and Shu Lin Algorithm 9.1 step 3, sub-step 2
    """

    z_extn_inc = (z_extn_sup[1] - z_extn_sup[0])/(z_extn_sup[2] - 1)
    q_pos = np.pad(p0_zi, (0, 1))
    q_neg = np.pad(p1_zi, (0, 1))

    # sf = size of the extended message
    sf = len(q_pos)
    # find length to nearest higher power of 2 to help with fft
    fft_size = 2 ** (int(np.ceil(np.log2(sf * dcm1 - dcm1 + 1))))
    q_pos_fft_size = np.zeros(fft_size)
    q_neg_fft_size = np.zeros(fft_size)

    # preserve *direct* probabilities under convolution
    # also note that overflow is included in q_pos_fft_size
    q_pos_fft_size[:sf] = q_pos * z_extn_inc
    q_neg_fft_size[:sf] = q_neg * z_extn_inc
    q_pos_fft_size[sf-1] = excess[0]
    q_neg_fft_size[sf-1] = excess[2]

    # find the probabilities of being positive or negative and finite ...
    # as well as marginal and conditional probabilities
    p_pos_fin = q_pos_fft_size.sum()
    p_pos = q_pos_fft_size.sum() + excess[1]
    p_neg_fin = q_neg_fft_size.sum()
    p_neg = q_neg_fft_size.sum() + excess[3]

    # here we take the FFT of the *conditional* density
    f_pos_fin = mkl_fft.fft(q_pos_fft_size / p_pos_fin)
    f_neg_fin = mkl_fft.fft(q_neg_fft_size / p_neg_fin)

    res_pos = np.zeros(fft_size)
    res_neg = np.zeros(fft_size)
    p_res_zero = 0

    for cc in range(dcm1+1):
        # this is roughly a binomial extension
        # no underflow
        # any sign = zero -- attached to p_res_zero

        # the following two lines are dealing with p_zero
        tmp_sv = (1 - (p_pos / (p_pos + p_zero)) ** (dcm1 - cc)) * comb(dcm1, cc) \
                 * p_neg ** cc * (p_pos + p_zero) ** (dcm1 - cc)
        p_res_zero = p_res_zero + tmp_sv
        # the following lines are the most important steps in the convolution loop
        # tmp1 = (1-excess(4))^c * (1-excess(2))^(dcm1-c);
        tmp2 = comb(dcm1, cc) * p_neg ** cc * p_pos ** (dcm1 - cc)
        tmp_vec = np.power(f_pos_fin, dcm1 - cc) * np.power(f_neg_fin, cc)
        # tmp_vec = tmp_vec * tmp1 * tmp2;
        tmp_vec = tmp_vec * tmp2

        if np.mod(cc, 2) == 0:
            res_pos = res_pos + tmp_vec
        else:
            res_neg = res_neg + tmp_vec

        for dd in range(dcm1 - cc + 1):

            w = (1-excess[1]) ** cc * (1-excess[3]) ** dd
            tmp_sv = (1 - w) * np.math.factorial(dcm1) / (np.math.factorial(cc)
                            * np.math.factorial(dd) * np.math.factorial(dcm1 - cc - dd))
            tmp_sv = tmp_sv * p_neg * cc * p_pos ** dd * p_zero ** (dcm1 - cc - dd)
            p_res_zero = p_res_zero + tmp_sv

    res_pos = mkl_fft.ifft(res_pos)
    res_neg = mkl_fft.ifft(res_neg)
    res_pos = abs(res_pos) / z_extn_inc
    res_neg = abs(res_neg) / z_extn_inc

    # reduce the size needed for the convenience of fft
    res_pos = res_pos[:(sf * dcm1 - dcm1 + 1)]
    res_neg = res_neg[:(sf * dcm1 - dcm1 + 1)]

    return res_pos, res_neg, p_res_zero


def cn_overflow(pm_update, ofl_pos, ofl_neg, m_sup, z_sup):
    """
    delicate handling of overflow of p0_zi and p1_zi
    :return:
    """

    m_inc = (m_sup[1] - m_sup[0]) / (m_sup[2] - 1)
    z_inc = (z_sup[1] - z_sup[0]) / (z_sup[2] - 1)
    tmp = np.log((1 + np.exp(-z_inc/2)) / (1 - np.exp(-z_inc/2)))
    m_pos_index = int((tmp - m_sup[0]) / m_inc + 0.5)
    m_neg_index = int((-tmp - m_sup[0]) / m_inc + 0.5)

    if m_pos_index < m_sup[2] - 1:
        pm_update[m_pos_index] = pm_update[m_pos_index] + ofl_pos/m_inc
    else:
        pm_update[-1] = pm_update[-1] + ofl_pos / m_inc

    if m_neg_index > 0:
        pm_update[m_neg_index] = pm_update[m_neg_index] + ofl_neg/m_inc
    else:
        pm_update[0] = pm_update[0] + ofl_neg/m_inc

    return pm_update


def vn_update(pc0, pc, dvm1, pc0_sup, m_sup):
    """
    This function updates the variable nodes pdf in density evolution.
    :param pc0: pdf of the message from the channel
    :param pc: pdf of check nodes
    :param dvm1: variable node degree minus one, i.e., dv - 1
    :param pc0_sup:
    :param m_sup:
    :return:
    Ref. [1] Channel Codes classical and modern - - William E.Ryan and Shu Lin Algorithm 9.1 step 4
    """

    sc = len(pc0)
    sf = len(pc)
    pc0_inc = (pc0_sup[1] - pc0_sup[0]) / (pc0_sup[2] - 1)
    m_inc = (m_sup[1] - m_sup[0]) / (m_sup[2] - 1)

    pc_cvl = np.zeros(sc + sf * dvm1 - dvm1)
    pc0_cvl = np.zeros(sc + sf * dvm1 - dvm1)
    pc0_cvl[:sc] = pc0 * pc0_inc
    pc_cvl[:sf] = pc * m_inc

    f_pc_cvl = mkl_fft.fft(pc_cvl)
    tmp = np.power(f_pc_cvl, dvm1) * mkl_fft.fft(pc0_cvl)
    pv_cvl = mkl_fft.ifft(tmp)

    minx = pc0_sup[0] + dvm1 * m_sup[0]
    ext_min_idx = int((m_sup[0] - minx) / m_inc + 0.5 + 1)
    ext_max_idx = ext_min_idx + m_sup[2] - 1
    ufl = abs(np.sum(pv_cvl[:(ext_min_idx - 1)]))
    ofl = abs(np.sum(pv_cvl[ext_max_idx:]))

    pv_cvl[ext_min_idx-1] = pv_cvl[ext_min_idx-1] + ufl
    pv_cvl[ext_max_idx-1] = pv_cvl[ext_max_idx-1] + ofl
    res = abs(pv_cvl[ext_min_idx-1:ext_max_idx]) / m_inc

    return res
