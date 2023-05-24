#STEADY STATE PROCESS ERROR COVARIANCE MATRIX ESTIMATION ALGORITHM USED TO INITIALIZE P

import numpy as np
from scipy import linalg

def calc_P_est(input_vec, alpha, var_p, var_m, max_iters, dim):

    for_toe = input_vec[0:dim]

    Toeplitz_input = linalg.toeplitz(for_toe)

    gamma = np.sqrt(1 - alpha)

    thres = 0.1 * var_p

    #SVD of ip Toeplitz to get Q, V'
    Q, eigendiag, QT = np.linalg.svd(Toeplitz_input)

    a = 1 - (1/(gamma * gamma))
    b = (var_p) / (gamma * gamma)

    V = np.identity(dim)
    v_vec = np.diag(V)

    for i in range (0, max_iters):
        sum = 0
        for j in range(0, len(eigendiag) - 1):
            sum += v_vec[j] * eigendiag[j]
            break
        f_V = sum + var_m

        temp = 0
        for l in range(0, len(eigendiag) - 1):
            temp += ((v_vec[l] * v_vec[l] * eigendiag[l]) - (f_V * (a * v_vec[l] + b))) ** 2

        epsilon = (1/len(eigendiag)) * temp

        if epsilon < thres:
            break

        v_new = np.zeros(v_vec.shape)
        for k in range(0, len(v_vec) - 1):
            v_new[k] = (f_V/(2 * eigendiag[k])) * (a + np.sqrt((a*a) + ((4*eigendiag[k]*b)/f_V)))

        v_vec = v_new


    V = np.diag(v_vec)

    P = Q @ V @ Q.T

    return P