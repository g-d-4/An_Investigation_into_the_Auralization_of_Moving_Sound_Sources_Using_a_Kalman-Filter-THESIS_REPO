#FUNTIONS TO IMPLEMENT KALMAN FILTER: Single iteration + run filter across length of measurement

#Region Imports
import numpy as np
import csv
from scipy import linalg
#endregion

#region Single Kalman Filter Iteration Function
def kalman_filter(w_est, P_est, x, y, A, Q, R):
    dim = len(w_est) #Needed to parametrize implementation
    #Predict
    w_pred = np.dot(A, w_est) #A matrix
    # w_pred = A * w_est #A scalar

    P_pred = (A @ P_est @ A.T) + Q #A matrix
    # P_pred = (A * A * P_est) + Q #A scalar

    #Update
    K = np.dot(P_pred, x).dot(1 / (np.dot(np.dot(x.T, P_pred), x) + R))
    w_est_post = w_pred + np.dot(K, y - np.dot(x.T, w_pred))
    # P_est_post = np.dot(np.identity(dim) - np.dot(K, x.T), P_pred) # original implementation
    P_est_post = P_pred - np.dot(K, np.dot(x.T, P_pred)) #Faster implementation
    return w_est_post, P_est_post
#endregion

#region Kalman Over Entire Signal
def run_kalman(rir, P_est, measurement, input, state_transition, process_noise, measure_noise):
    dim = len(rir)
    rir_buffer = np.zeros(rir.shape)
    P_buffer = np.zeros(P_est.shape)
    out_rirs = []
    # out_P_set = [] #not needed to be saved: causes drastic inc of memory requirement

    if len(P_est) != dim or len(process_noise) != dim or len(state_transition) != dim:
       print("Incorrect dimensions for error covariance estimate, state transition matrix "
             "and/or process noise covariance matrix")

    else:
        i = 0
        for i in range(0, len(measurement) - 1):
            try:
                y = measurement[i]
                x = input[i : i + dim]

                if i == 0:
                    w_est = rir
                    P = P_est

                else:
                    w_est = rir_buffer
                    P = P_buffer

                rir_buffer, P_buffer = kalman_filter(w_est, P, x, y, state_transition, process_noise, measure_noise)
                out_rirs.append(rir_buffer)

            except IndexError:
                print("Kalman filter ended because input signal passed through")
                break

        #save output state estimates to file
        with open('out_sine_9_5_R.npy', 'wb') as f:
            np.save(f, out_rirs)

        return rir_buffer, P_buffer
#endregion