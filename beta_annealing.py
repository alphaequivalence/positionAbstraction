import math


def beta(t, T=1000, R=0.5, M=4):
    """
    Cyclical Annealing Schedule [1] Hao Fu et al. 2019 (NAACL)
    t: the iteration number
    T:  total number of training iterations
    R: proportion used to increase within a cycle (default R=0.5)
    M: number of cycles (default M=4);

    [1] Fu, Hao, et al. "Cyclical Annealing Schedule: A Simple Approach to
    Mitigating KL Vanishing." Proceedings of the 2019 Conference of the North
    American Chapter of the Association for Computational Linguistics: Human
    Language Technologies, Volume 1 (Long and Short Papers). 2019.
    """
    def f(tau):
        return math.pow(tau, 2)

    tau = ((t-1) % math.ceil(T/M))/(T/M)
    print(t)
    print(tau)
    if tau<=R:
        beta_ = f(tau)
    else:
        beta_ = 1
    return beta_
