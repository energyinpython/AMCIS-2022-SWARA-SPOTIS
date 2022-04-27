import copy
import numpy as np
import pandas as pd
from correlations import pearson_coeff
from normalizations import sum_normalization, minmax_normalization


# equal weighting
def equal_weighting(X):
    N = np.shape(X)[1]
    return np.ones(N) / N


# entropy weighting
def entropy_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)
    m, n = np.shape(pij)

    H = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            if pij[i, j] != 0:
                H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))
    d = 1 - h
    w = d / (np.sum(d))

    return w


# standard deviation weighting
def std_weighting(X):
    stdv = np.std(X, axis = 0)
    return stdv / np.sum(stdv)


# CRITIC weighting
def critic_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    x_norm = minmax_normalization(X, criteria_type)
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    correlations = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            correlations[i, j] = pearson_coeff(x_norm[:, i], x_norm[:, j])

    difference = 1 - correlations
    suma = np.sum(difference, axis = 0)
    C = std * suma
    w = C / (np.sum(C, axis = 0))
    return w


# gini weighting
def gini_weighting(X):
        m, n = np.shape(X)
        G = np.zeros(n)
        # iteration over criteria j = 1, 2, ..., n
        for j in range(0, n):
            # iteration over alternatives i = 1, 2, ..., m
            Yi = np.zeros(m)
            if np.mean(X[:, j]) != 0:
                for i in range(0, m):
                    for k in range(0, m):
                        Yi[i] += np.abs(X[i, j] - X[k, j]) / (2 * m**2 * (np.sum(X[:, j]) / m))
            else:
                for i in range(0, m):
                    for k in range(0, m):
                        Yi[i] += np.abs(X[i, j] - X[k, j]) / (m**2 - m)
            G[j] = np.sum(Yi)
        return G / np.sum(G)


# MEREC weighting
def merec(matrix, types):
    X = copy.deepcopy(matrix)
    m, n = X.shape
    X = np.abs(X)
    norm_matrix = np.zeros(X.shape)
    norm_matrix[:, types == 1] = np.min(X[:, types == 1], axis = 0) / X[:, types == 1]
    norm_matrix[:, types == -1] = X[:, types == -1] / np.max(X[:, types == -1], axis = 0)
    
    S = np.log(1 + ((1 / n) * np.sum(np.abs(np.log(norm_matrix)), axis = 1)))
    Sp = np.zeros(X.shape)

    for j in range(n):
        norm_mat = np.delete(norm_matrix, j, axis = 1)
        Sp[:, j] = np.log(1 + ((1 / n) * np.sum(np.abs(np.log(norm_mat)), axis = 1)))

    E = np.sum(np.abs(Sp - S.reshape(-1, 1)), axis = 0)
    w = E / np.sum(E)
    return w


# statistical variance weighting
def stat_variance_weighting(X):
    criteria_type = np.ones(np.shape(X)[1])
    xn = minmax_normalization(X, criteria_type)
    v = np.mean(np.square(xn - np.mean(xn, axis = 0)), axis = 0)
    w = v / np.sum(v)
    return w


# SWARA weighting
def swara_weighting(s):
    list_of_years = [str(y) for y in range(2015, 2021)]
    df_swara = pd.DataFrame(index = list_of_years[::-1])
    df_swara['cp'] = s
    k = np.ones(len(s))
    q = np.ones(len(s))
    for j in range(1, len(s)):
        k[j] = s[j] + 1
        q[j] = q[j - 1] / k[j]

    df_swara['kp'] = k
    df_swara['vp'] = q
    df_swara['wp'] = q / np.sum(q)
    df_swara.to_csv('results/swara_results.csv')

    return q / np.sum(q)