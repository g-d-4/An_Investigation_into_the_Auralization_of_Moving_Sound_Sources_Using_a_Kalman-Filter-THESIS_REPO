from scipy.io import wavfile
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy import signal
from P_ss_est_svd import calc_P_est
import time
from kalman_filter import run_kalman

#Storing measurements as np arrays
str_dir = './Signals_Binary_npyz/sine_500_data.npz'
data_dir = pjoin(dirname(str_dir), 'sine_500_data.npz')

sine_500_data = np.load(data_dir)
fs_og = 44100
sine500_dhl = sine_500_data['sine500_DHL']
sine500_dhr = sine_500_data['sine500_DHR']
sine500_input = sine_500_data['sine500_input']

plt.plot(np.linspace(0,len(sine500_input)/fs_og, len(sine500_input)), sine500_input)
plt.title("IP sig.")
plt.show()

rir_dir_l = pjoin(dirname(str_dir), 'rirs_L_measured.npz')
rir_l_data = np.load(rir_dir_l)

rir_l6_l = rir_l_data['rir_l6_l']


rir_dir_r = pjoin(dirname(str_dir), 'rirs_R_measured.npz')
rir_r_data = np.load(rir_dir_r)

rir_l6_r = rir_r_data['rir_l6_r']

mob_dir = pjoin(dirname(str_dir), 'sine500_mob.wav')
mob_data, mob_fs = sf.read(mob_dir)

strt = np.argmax(sine500_input > 0.1)

start_t = strt/fs_og

t_end_mob = 16
t_strt_mob = np.argmax(mob_data[:,0] > 0.25)/mob_fs

end_t = start_t + (t_end_mob - t_strt_mob)

#Downsampling to 8kHz
len_dhl = len(sine500_dhl)/fs_og
len_dhr = len(sine500_dhr)/fs_og
len_input = len(sine500_input)/fs_og
len_rir_l = len(rir_l6_l)/fs_og
len_rir_r = len(rir_l6_r)/fs_og

new_freq = 8000

sine_dhl_down = signal.resample(sine500_dhl, int(len_dhl * new_freq))
sine_dhr_down = signal.resample(sine500_dhr, int(len_dhr * new_freq))
sine_input_down = signal.resample(sine500_input, int(len_input * new_freq))

rir_l6_l_down = signal.resample(rir_l6_l, int(len_rir_l * new_freq))
rir_l6_r_down = signal.resample(rir_l6_r, int(len_rir_r * new_freq))

rir_l6_l_down_reqd = rir_l6_l_down[0 : int(0.1 * new_freq)]
remaining_rir_l = rir_l6_l_down[int(0.1 * new_freq) : :]

#Splitting RIR measurement into state estimate + remaining portion
rir_l6_r_down_reqd = rir_l6_r_down[0 : int(0.1 * new_freq)]
remaining_rir_r = rir_l6_r_down[int(0.1 * new_freq) : :]

q = len(rir_l6_r_down_reqd) #same as len(rir_l6_l_down_reqd)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(np.linspace(0, 0.1, q), rir_l6_l_down_reqd)
ax2.plot(np.linspace(0, len(remaining_rir_l)/new_freq, len(remaining_rir_l)), remaining_rir_l)

#Plotting state estimate
ax1.set_title('REQUIRED PORTION OF RIR: L')
ax2.set_title('REMAINING PORTION OF RIR: L')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

print("start t = ", start_t)
print("end t = ", end_t)
start_sample = int(start_t * new_freq)
end_sample = int(end_t * new_freq)

#Initializing w_init, x, y, alpha
w_init_l = rir_l6_l_down_reqd
w_init_r = rir_l6_r_down_reqd

input_sig = sine_input_down[start_sample : :]
measurement_l = sine_dhl_down[start_sample : end_sample]
measurement_r = sine_dhr_down[start_sample : end_sample]

var_p_l = np.var(remaining_rir_l[len(remaining_rir_l) - q : :])
var_p_r = np.var(remaining_rir_r[len(remaining_rir_r) - q : :])

var_m_l = np.var(measurement_l)
var_m_r = np.var(measurement_r)

alpha = 0.98

#Calculating steady state process error covariance matrix as P estimate
P_l = calc_P_est(input_sig, alpha, var_p_l, var_m_l, 50, q)
P_r = calc_P_est(input_sig, alpha, var_p_r, var_m_r, 50, q)

plt.matshow(P_l)
plt.colorbar()
plt.title("P estimate L")
plt.show()

plt.matshow(P_r)
plt.colorbar()
plt.title("P estimate R")
plt.show()

#Initializing A, Q, r
A_l = np.sqrt(alpha) * P_l
Q_l = var_p_l * np.identity(q)
R_l = var_m_l

A_r = np.sqrt(alpha) * P_r
Q_r = var_p_r * np.identity(q)
R_r = var_m_r

#Running Kalman Filter
t1 = time.time() #Time variable(s) to understand computational intensity
# last_rir_l, last_P_l = run_kalman(w_init_l, P_l, measurement_l, input_sig, A_l, Q_l, R_l)
# last_rir_r, last_P_r = run_kalman(w_init_r, P_r, measurement_r, input_sig, A_r, Q_r, R_r)
t2 = time.time()
tkal = t2 - t1
#print(tkal)

# plt.plot(np.linspace(0, 0.1, q), last_rir_r)
# plt.title("Last RIR R")
# plt.show()
#
# plt.matshow(last_P_r)
# plt.colorbar()
# plt.title("Last P R")
# plt.show()