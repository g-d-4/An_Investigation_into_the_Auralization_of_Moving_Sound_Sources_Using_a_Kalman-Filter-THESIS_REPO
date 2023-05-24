import numpy as np
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt
from scipy import signal
from scipy import optimize
from scipy import linalg
from sklearn import preprocessing
from kalman_filter import kalman_filter
from kalman_filter import run_kalman
import time
import sys

#REad in measurements as np array
str_dir = './Signals_Binary_npyz/wh_faster_data.npz'
data_dir = pjoin(dirname(str_dir), 'wh_faster_data.npz')
mob_dir = pjoin(dirname(str_dir), 'mob_audio_wh_faster.npy')

wh_faster_data = np.load(data_dir)

wh_faster_dhl = wh_faster_data['wh_faster_dhl']
wh_faster_dhr = wh_faster_data['wh_faster_dhr']
wh_faster_input = wh_faster_data['wh_faster_input']

mob_signal = np.load(mob_dir)
mob_fs = 48000
measure_fs = 44100

# Find start of audio signal in mobile recording
start_mob = np.argmax(mob_signal[:, 0] > 0.25)
t_mob = start_mob / mob_fs

# Find start of audio signal in input signal
start_input = np.argmax(wh_faster_input > 0.1)
t_input = start_input / measure_fs

# Calculate time delay and sample delay
t_delay = t_mob - t_input
sample_delay = int(t_delay*measure_fs)

t_stop = 7.2

rir_dir_l = pjoin(dirname(str_dir), 'rirs_L_measured.npz')
rir_l_data = np.load(rir_dir_l)

rir_l6_l = rir_l_data['rir_l6_l']

rir_dir_r = pjoin(dirname(str_dir), 'rirs_R_measured.npz')
rir_r_data = np.load(rir_dir_r)
#
rir_l6_r = rir_r_data['rir_l6_r']

len_wh_faster_dhl = len(wh_faster_dhl)/measure_fs
len_wh_faster_dhr = len(wh_faster_dhr)/measure_fs
len_input = len(wh_faster_input)/measure_fs
len_rir = len(rir_l6_l)/measure_fs
len_rir_r = len(rir_l6_r)/measure_fs

#Declare new freq
new_freq = 8000

#Resampling to 8kHz
wh_faster_dhl_down = signal.resample(wh_faster_dhl, int(new_freq*len_wh_faster_dhl))
wh_faster_dhr_down = signal.resample(wh_faster_dhr, int(new_freq*len_wh_faster_dhr))
wh_faster_input_down = signal.resample(wh_faster_input, int(new_freq*len_input))
rir_l6_l_down = signal.resample(rir_l6_l, int(new_freq*len_rir))
rir_l6_r_down = signal.resample(rir_l6_r, int(new_freq*len_rir_r))
rir_l6_r_down = rir_l6_r_down/np.max(rir_l6_r_down)

rir_l6_l_down_reqd = rir_l6_l_down[0:int(0.1*new_freq)]
x_t_2 = np.linspace(0, len(rir_l6_l_down_reqd)/new_freq, len(rir_l6_l_down_reqd))

remaining_brir = rir_l6_l_down[int(0.1*new_freq)::]
x_t_3 = np.linspace(0.15, len(remaining_brir)/new_freq, len(remaining_brir))

rir_l6_r_down_reqd = rir_l6_r_down[0:int(0.1*new_freq)]
remaining_brir_r = rir_l6_r_down[int(0.1*new_freq) : :]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.plot(x_t_2, rir_l6_r_down_reqd)
ax2.plot(x_t_3, remaining_brir_r)

#Plotting Initial State Estimate and RIR
ax1.set_title('INITIAL STATE ESTIMATE')
ax2.set_title('REMAINING PORTION OF RIR')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

plt.plot(x_t_2, rir_l6_r_down_reqd)
plt.title("INITIAL STATE ESTIMATE")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

plt.plot(np.linspace(0, len(rir_l6_l_down[0:10*new_freq])/new_freq, len(rir_l6_l_down[0:10*new_freq])), rir_l6_l_down[0:10*new_freq]/np.max(rir_l6_l_down))
plt.title("RIR MEASURED AT POSITION L6 THROUGH DECONVOLUTION")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

q = len(rir_l6_r_down_reqd)

#Initial estimate
w_init = rir_l6_r_down_reqd

start_sample = int((5 - t_delay)*new_freq)
y_measure = wh_faster_dhr_down[start_sample]

print(start_sample)

#Final a posteriori estimate
w_fin = np.zeros(np.shape(w_init))

x_measure = wh_faster_input_down[start_sample : start_sample + q]

end_sample = int((10 - t_delay)*new_freq)
measurement = wh_faster_dhr_down[start_sample : end_sample]
input_snippet = wh_faster_input_down[start_sample : end_sample]

#Estimating P, Q, R

alpha = 0.98

#Measurement noise read as npy file (saved earlier)
R_vector_dir = pjoin(dirname(str_dir), 'R_vector.npy')
R_vector = np.load(R_vector_dir)

R_error = R_vector[0]
R_matrix = np.diag(R_vector[0:q])

var_m = np.var(measurement)

#region P estimation
#Steady state process noise covar. matrix algo. not used as was first tried on white noise here!`
var_p = np.var(remaining_brir[len(remaining_brir_r) - q : :])
for_toe = input_snippet[0:q]
Toeplitz_input = linalg.toeplitz(for_toe)

print("Toeplitz IP = ", Toeplitz_input)

gamma = np.sqrt(1 - alpha)

thres = 0.1 * var_p

#SVD of ip Toeplitz to get Q, REV. V
Q, eigendiag, QT = np.linalg.svd(Toeplitz_input)

a = 1 - (1/(gamma * gamma))
b = (var_p) / (gamma * gamma)

max_iters = 50

V = np.identity(q)
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

# Plotting P estimate obtained
plt.matshow(P)
plt.colorbar(orientation='horizontal')
plt.title("Steady-State Process Error Covariance Matrix Estimate")
plt.show()
#endregion

#Initializing Parameters for Kalman filter based on P, input and measurement
Q_pn = var_p * np.identity(q)

A = np.sqrt(alpha) * P

input = wh_faster_input_down[start_sample : :]

#Estimating memory requirements
# print("Size of final OP = ", int(len(measurement)*sys.getsizeof(w_out)))
# print("Mem. over tot iters = ", int(len(measurement)*size_1_iter))

#Running Kalman filter across measurement signal length
t1 = time.time()
# out_rir_wh, out_P_wh = run_kalman(w_init, P, measurement, input, A, Q_pn, var_m)
t2 = time.time()
#
t_kalman_full = t2 - t1
print(t_kalman_full)

# plt.matshow(out_P_wh)
# plt.colorbar()
# plt.title("Last P obtd.")
# plt.show()
# #
# plt.plot(out_rir_wh)
# plt.title("last RIR Obtd.")
# plt.show()