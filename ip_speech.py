from scipy.io import wavfile
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy import signal
from P_ss_est_svd import calc_P_est
from sklearn import preprocessing
from kalman_filter import run_kalman
import time

#region using sf.read
#sf. read was initially used, later required signals were stored in npy/npz arrys for easier access
# str_dir = './Recordings_ESAT_DHMics_LSMic/speech'
# data_dir = pjoin(dirname(str_dir), 'speech')
#
# #File directory vars
# file_DHL = pjoin(data_dir, 'speech_DHL.wav')
# file_DHR = pjoin(data_dir, 'speech_DHR.wav')
# file_LSMic = pjoin(data_dir, 'speech_LSmic.wav')
#
# #Data vars
# data_L, fs_L = sf.read(file_DHL)
# data_R, fs_R = sf.read(file_DHR)
# data_LS, fs_LS = sf.read(file_LSMic)
#
# with open('speech_data.npz', 'wb') as f:
#     np.savez(f, speech_DHL=data_L, speech_DHR=data_R, speech_input=data_LS)
#
# plt.plot(data_R)
# plt.show()
#endregion

#Reading in input, measurement, rir signals as np arrays
str_dir = './Signals_Binary_npyz/speech_data.npz'
data_dir = pjoin(dirname(str_dir), 'speech_data.npz')

speech_data = np.load(data_dir)

speech_dhl = speech_data['speech_DHL']
speech_dhr = speech_data['speech_DHR']
speech_input = speech_data['speech_input']
signals_fs = 44100
x_sigs = np.linspace(0, len(speech_input)/signals_fs, len(speech_input))

mob_dir = pjoin(dirname(str_dir), 'speech_mob_video.wav')

mob_data, mob_fs = sf.read(mob_dir)
x_mob = np.linspace(0, len(mob_data)/mob_fs, len(mob_data))

mob_delay = np.argmax(mob_data[:,0] > 0.2)
# print("t at start mob = ", mob_delay/mob_fs)

t_start_mob = mob_delay/mob_fs
# print("t start mob = ", t_start_mob)

sig_start = np.argmax(speech_input > 0.05)

# print("t at start DHL/IP", sig_start/signals_fs)
#DHL/R and input signal starts from sample where ip > 0.05

end_t = 20

t_start_sigs = sig_start/signals_fs
end_t_sig = t_start_sigs + (end_t - t_start_mob)

rir_dir_l = pjoin(dirname(str_dir), 'rirs_L_measured.npz')
rir_l_data = np.load(rir_dir_l)

rir_l6_l = rir_l_data['rir_l6_l']

rir_dir_r = pjoin(dirname(str_dir), 'rirs_R_measured.npz')
rir_r_data = np.load(rir_dir_r)

rir_l6_r = rir_r_data['rir_l6_r']

len_dhl = len(speech_dhl)/signals_fs
len_dhr = len(speech_dhr)/signals_fs
len_input = len(speech_input)/signals_fs

len_rir_l6 = len(rir_l6_l)/signals_fs
len_rir_l6_r = len(rir_l6_r)/signals_fs

new_freq = 8000

#Downsampling to 8kHz
speech_dhl_down = signal.resample(speech_dhl, int(len_dhl * new_freq))
speech_dhr_down = signal.resample(speech_dhr, int(len_dhr * new_freq))
speech_input_down = signal.resample(speech_input, int(len_input * new_freq))

rir_l6_l_down = signal.resample(rir_l6_l, int(len_rir_l6 * new_freq))
rir_l6_r_down = signal.resample(rir_l6_r, int(len_rir_l6_r * new_freq))

rir_l6_l_down_reqd = rir_l6_l_down[0:int(0.1*new_freq)]
x_t_2 = np.linspace(0, len(rir_l6_l_down_reqd)/new_freq, len(rir_l6_l_down_reqd))

remaining_brir_l = rir_l6_l_down[int(0.1*new_freq)::]
x_t_3 = np.linspace(0.1, len(remaining_brir_l)/new_freq, len(remaining_brir_l))

# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# ax1.plot(x_t_2, rir_l6_l_down_reqd)
# ax2.plot(x_t_3, remaining_brir)
#
# ax1.set_title('REQUIRED PORTION OF RIR')
# ax2.set_title('REMAINING PORTION OF RIR')
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

rir_l6_r_down_reqd = rir_l6_r_down[0:int(0.1*new_freq)]
x_t_2 = np.linspace(0, len(rir_l6_r_down_reqd)/new_freq, len(rir_l6_r_down_reqd))

remaining_brir_r = rir_l6_r_down[int(0.1*new_freq)::]
x_t_3 = np.linspace(0.1, len(remaining_brir_r)/new_freq, len(remaining_brir_r))

# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# ax1.plot(x_t_2, rir_l6_r_down_reqd)
# ax2.plot(x_t_3, remaining_brir_r)

# ax1.set_title('REQUIRED PORTION OF RIR: R')
# ax2.set_title('REMAINING PORTION OF RIR')
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

# print("t start = ", t_start_sigs)
# print("end t = ", end_t_sig)

start_sample = int(t_start_sigs * new_freq)
end_sample = int(end_t_sig * new_freq)

q = len(rir_l6_l_down_reqd)

#Declare initial estimate of RIR + Input + measurement
w_init = rir_l6_r_down_reqd
measurement_r = speech_dhr_down[start_sample : end_sample]
measurement_l = speech_dhl_down[start_sample : end_sample]
input_sig = speech_input_down[start_sample : :]

#Declare var_p, var_m, alpha

var_p_l = np.var(remaining_brir_l[len(remaining_brir_l) - q : :])
var_m_l = np.var(measurement_l)

var_m_r = np.var(measurement_r)
var_p_r = np.var(remaining_brir_r[len(remaining_brir_r) - q : :])

alpha = 0.98

#Estimating P using steady state process error covar. matrix estimation algo
P = calc_P_est(input_sig, alpha, var_p_l, var_m_l, 50, q)
P_r = calc_P_est(input_sig, alpha, var_p_r, var_m_r, 50, q)

print("P est: ", P)

plt.matshow(P)
plt.colorbar()
plt.title("P estimate")
plt.show()

#Initialize A, Q, r
A = np.sqrt(alpha) * P_r

R = var_m_r

Q = var_p_r * np.identity(q)

#Run Kalman filter
t1 = time.time()
# last_rir, last_P = run_kalman(w_init, P_r, measurement_r, input_sig, A, Q, R)
t2 = time.time()

t_full = t2 - t1
# print("t for Kalman: ", t_full, "s")

# plt.plot(x_t_2, last_rir)
# plt.title("Last RIR speech")
# plt.show()
#
# plt.matshow(last_P)
# plt.colorbar()
# plt.title('Last P')
# plt.show()