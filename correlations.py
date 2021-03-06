import numpy as np
from scipy.stats import kendalltau

# spearman coefficient
def spearman(R, Q):
    N = len(R)
    denominator = N*(N**2-1)
    numerator = 6*sum((R-Q)**2)
    rS = 1-(numerator/denominator)
    return rS


# weighted spearman coefficient rw
def weighted_spearman(R, Q):
    N = len(R)
    denominator = N**4 + N**3 - N**2 - N
    numerator = 6 * sum((R - Q)**2 * ((N - R + 1) + (N - Q + 1)))
    rW = 1 - (numerator / denominator)
    return rW


# rank similarity coefficient WS
def coeff_WS(R, Q):
    N = len(R)
    numerator = 2**(-np.float64(R)) * np.abs(R - Q)
    denominator = np.max((np.abs(R - 1), np.abs(R - N)), axis = 0)
    ws = 1 - np.sum(numerator / denominator)
    return 


# pearson coefficient
def pearson_coeff(R, Q):
    numerator = np.sum((R - np.mean(R)) * (Q - np.mean(Q)))
    denominator = np.sqrt(np.sum((R - np.mean(R))**2) * np.sum((Q - np.mean(Q))**2))
    corr = numerator / denominator
    return corr


# kendall rank correlation coefficient
def kendall(R, Q):
    N = len(R)
    Ns, Nd = 0, 0
    for i in range(1, N):
        for j in range(i):
            if ((R[i] > R[j]) and (Q[i] > Q[j])) or ((R[i] < R[j]) and (Q[i] < Q[j])):
                Ns += 1
            elif ((R[i] > R[j]) and (Q[i] < Q[j])) or ((R[i] < R[j]) and (Q[i] > Q[j])):
                Nd += 1

    tau = (Ns - Nd) / ((N * (N - 1))/2)
    return tau


# goodman kruskal correlation coefficient
def goodman_kruskal(R, Q):
    N = len(R)
    Ns, Nd = 0, 0
    for i in range(1, N):
        for j in range(i):
            if ((R[i] > R[j]) and (Q[i] > Q[j])) or ((R[i] < R[j]) and (Q[i] < Q[j])):
                Ns += 1
            elif ((R[i] > R[j]) and (Q[i] < Q[j])) or ((R[i] < R[j]) and (Q[i] > Q[j])):
                Nd += 1

    coeff = (Ns - Nd) / (Ns + Nd)
    return coeff


def kendall_tau(R, Q):
    corr, _ = kendalltau(R, Q)
    return corr