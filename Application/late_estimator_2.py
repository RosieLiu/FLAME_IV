# -*- coding: utf-8 -*-
# @Author: marco
# @Date:   2019-02-23 17:33:53
# @Last Modified by:   marco
# @Last Modified time: 2019-02-26 16:25:34

'''
Average measure across groups
'''

import numpy as np


def ITTj(Y, Z):
    return np.mean(Y[Z == 1]) - np.mean(Y[Z == 0])


def ITT(Y, Z, G):
    return (1.0 / len(Y)) * np.sum([np.sum(G == j) * ITTj(Y[G == j], Z[G == j])
                                    for j in set(G)])


def lambda_hat(Y, T, Z, G):
    return ITT(Y, Z, G) / ITT(T, Z, G)


def S_squared(Y, mean):
    n = len(Y)
    return (1.0 / (n)) * np.sum((Y - mean) ** 2)


def Var_ITTy(Y, Z, G):

    mean_c = np.mean( Y[(Z == 0)]) # mean across all control units
    mean_t = np.mean( Y[(Z == 1)]) # mean across all treatment units
        
    return (1.0 / len(Y) ** 2) * \
        np.sum([(S_squared(Y[(Z == 1) & (G == j)], mean_t) / np.sum((Z == 1) & (G == j))
                + S_squared(Y[(Z == 0) & (G == j)], mean_c) / np.sum((Z == 0) & (G == j)))
                * np.sum(G == j) ** 2 
                for j in set(G)])


def Var_ITTt(T, Z, G):
    mean_t = np.mean( T[(Z == 1)]) # mean across all treatment units
    
    return (1.0 / len(T) ** 2) * \
        np.sum([S_squared(T[(Z == 1) & (G == j)] ,mean_t) / np.sum((Z == 1) & (G == j))
                * np.sum(G == j) ** 2 
                for j in set(G)])


def Cov_ITTj(Y, T):
    return (1.0 / len(Y) ** 2) * \
        np.sum((Y - np.mean(Y)) * (T - np.mean(T)))


def Cov_ITT(Y, T, Z, G):
    return (1.0 / len(Y) ** 2) * \
        np.sum([Cov_ITTj(Y[(Z == 1) & (G == j)], T[(Z == 1) & (G == j)]) * np.sum(G == j) ** 2
                for j in set(G)])


def Var_lambda_hat(Y, T, Z, G):
    itty = ITT(Y, Z, G)
    ittt = ITT(T, Z, G)
    Vy = Var_ITTy(Y, Z, G)
    Vt = Var_ITTt(T, Z, G)
    Cyt = Cov_ITT(Y, T, Z, G)
    return (1 / ittt ** 2) * Vy + ((itty ** 2) / (ittt ** 4)) * Vt - \
        2 * (itty / (ittt ** 3)) * Cyt
