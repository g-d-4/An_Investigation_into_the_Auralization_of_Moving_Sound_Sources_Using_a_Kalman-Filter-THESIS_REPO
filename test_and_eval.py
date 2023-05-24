#File for generating auralized signals and plotting quantitative assesment figures for testing and evaluation

#region Imports
import numpy as np
from scipy import signal
import tikzplotlib as tikz
from scipy import linalg
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
import soundfile as sf
import time
import sys
#endregion

#region Test OP Wh noise faster
str_dir = './Outputs_npyz/wh_faster/output_rirs_wh_faster.npy'
str_dir = './out_whFaster_3_5_wSVD.npy'
op_dir_wh_faster = pjoin(dirname(str_dir), 'out_whFaster_3_5_wSVD.npy')
#
op_rirs_wh_faster = np.load('./out_whFaster_3_5_wSVD.npy')
print("Len of RIR set WH L = ", len(op_rirs_wh_faster))

str_dir = './out_whFaster_3_5_wSVD_R.npy'
op_dir_wh_faster = pjoin(dirname(str_dir), 'out_whFaster_3_5_wSVD_R.npy')

op_rirs_whFaster_r = np.load('./out_whFaster_3_5_wSVD_R.npy')

freq = 8000

str_dir = './Signals_Binary_npyz/wh_faster_data.npz'
data_dir = pjoin(dirname(str_dir), 'wh_faster_data.npz')
wh_faster_data = np.load(data_dir)
wh_faster_input = wh_faster_data['wh_faster_input']
len_input = len(wh_faster_input)/44100
wh_faster_input_down = signal.resample(wh_faster_input, int(freq*len_input))

start_sample = 20664 #Taken from previous computation in wh noise py file

input_eval = wh_faster_input_down[start_sample : :]
print(len(input_eval))

q = len(op_rirs_wh_faster[0])

#Convoluting signals to obtain auralized stereo audio signals
op_wh_l = []

for i in range(0, len(op_rirs_wh_faster) - 1):
    y = np.dot(op_rirs_wh_faster[i].T, input_eval[i : i + q])
    op_wh_l.append(y)

op_wh_l = np.array(op_wh_l)

op_wh_r = []

for i in range(0, len(op_rirs_whFaster_r) - 1):
    y = np.dot(op_rirs_whFaster_r[i].T, input_eval[i : i + q])
    op_wh_r.append(y)

op_wh_r = np.array(op_wh_r)

x_t_2 = np.linspace(0, len(op_wh_l)/freq, len(op_wh_l))
# plt.plot(x_t_2, op_wh_l)
# plt.title("Output signal obtained from generated RIRs : L")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

x_t_2 = np.linspace(0, len(op_wh_r)/freq, len(op_wh_r))
# plt.plot(x_t_2, op_wh_r)
# plt.title("Output signal obtained from generated RIRs : R")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()
#
x_t_1 = np.linspace(0, len(op_rirs_wh_faster[0])/freq, len(op_rirs_wh_faster[0]))
# fig, ax = plt.subplots()
# ax.plot(x_t_1, op_rirs_wh_faster[0], label='Out RIR from 1st measurement')
# ax.plot(x_t_1, op_rirs_wh_faster[105], label='Out RIR 106th measurement')
# ax.legend()
# ax.set_title("Comparison of RIR evolution acorss Kalman : L")
# plt.show()

x_t_1 = np.linspace(0, len(op_rirs_whFaster_r[0])/freq, len(op_rirs_whFaster_r[0]))
# fig, ax = plt.subplots()
# ax.plot(x_t_1, op_rirs_whFaster_r[0], label='Out RIR from 1st measurement')
# ax.plot(x_t_1, op_rirs_whFaster_r[105], label='Out RIR 106th measurement')
# ax.legend()
# ax.set_title("Comparison of RIR evolution acorss Kalman : R")
# plt.show()

#Create stereo signal to be heard
stereo_sig = np.column_stack((op_wh_l, op_wh_r))

# sf.write('test_op_whFaster_stereo.wav', stereo_sig, freq)
#endregion

#region Test OP speech

out_rirs_speech_l = np.load(pjoin(dirname('./Outputs_npyz/speech/out_speech_8_5_L_02.npy'), 'out_speech_8_5_L_02.npy'))
dim = len(out_rirs_speech_l[0])

out_rirs_speech_r = np.load(pjoin(dirname('./Outputs_npyz/speech/out_speech_8_5_R_02.npy'), 'out_speech_8_5_R_02.npy'))

freq = 8000

str_dir = './Signals_Binary_npyz/speech_data.npz'
data_dir = pjoin(dirname(str_dir), 'speech_data.npz')

speech_data = np.load(data_dir)
speech_input = speech_data['speech_input']

fs = 44100

sig_start = np.argmax(speech_input > 0.05)
t_start = sig_start/fs

freq = 8000

start_sample = int(t_start * freq)
print("start speech = ", start_sample)

ip = signal.resample(speech_input, int((len(speech_input)/fs) * freq))
reqd_ip = ip[start_sample : :]

x_t_1 = np.linspace(0, dim/freq, dim)

# fig, ax = plt.subplots()
# ax.plot(x_t_1, out_rirs_speech_l[1], label='Out RIR L from 1st measurement')
# ax.plot(x_t_1, out_rirs_speech_r[1], label='Out RIR R from 1st measurement')
# ax.legend()
# ax.set_title("Comparison of RIR_ L/R")
# plt.show()

#Create auralized signals
op_speech_l = []
for i in range(0, len(out_rirs_speech_l) - 1):
    y = np.dot(out_rirs_speech_l[i].T, reqd_ip[i : i + dim])
    op_speech_l.append(y)

op_speech_l = np.array(op_speech_l)

op_speech_r = []
for i in range(0, len(out_rirs_speech_r) - 1):
    y = np.dot(out_rirs_speech_r[i].T, reqd_ip[i : i + dim])
    op_speech_r.append(y)

op_speech_r = np.array(op_speech_r)
x_op = np.linspace(0, len(op_speech_l)/freq, len(op_speech_l))

# plt.plot(x_op, op_speech_l)
# plt.title("Output obtd. for Speech : L")
# plt.show()

#Create stereo audio signal
stereo_speech = np.column_stack((op_speech_l, op_speech_r))

# sf.write('stereo_speech_8_5_02.wav', stereo_speech, freq)

#endregion

#region Test Sine Wave

sine_rirs_l = np.load('out_sine_9_5_L.npy')
sine_rirs_r = np.load('out_sine_9_5_R.npy')
dim = len(sine_rirs_l[0])

ip_dir = './Signals_Binary_npyz/sine_500_data.npz'
dir_sine = pjoin(dirname(ip_dir), 'sine_500_data.npz')

sine_data = np.load(dir_sine)

sine_input = sine_data['sine500_input']

fs_og = 44100
new_freq = 8000

len_ip = len(sine_input)/fs_og

t_start = 1.0893197278911564
reqd_input = signal.resample(sine_input, int(len_ip * new_freq))
start_sample = int(t_start * new_freq)
reqd_input = reqd_input[start_sample : :]

#Create auralized signals
op_sine_l = []

for i in range (0, len(sine_rirs_l) - 1):
    y = np.dot(sine_rirs_l[i].T, reqd_input[i : i + dim])
    op_sine_l.append(y)

op_sine_l = np.array(op_sine_l)

op_sine_r = []

for i in range(0, len(sine_rirs_r) - 1):
    y = np.dot(sine_rirs_r[i].T, reqd_input[i : i + dim])
    op_sine_r.append(y)

op_sine_r = np.array(op_sine_r)

x_op = np.linspace(0, len(op_sine_l)/new_freq, len(op_sine_l))

# plt.plot(x_op, op_sine_r)
# plt.title("OP signal R")
# plt.show()
#
# plt.plot(x_op, op_sine_l)
# plt.title("OP signal L")
# plt.show()

#Create stereo signal
stereo_sine = np.column_stack((op_sine_l, op_sine_r))

# sf.write('sine_stereo_out_10_5.wav', stereo_sine, new_freq)

#endregion


#TESTING AND PLOTTING FOR EVALUATION BEGINS HERE
# * TESTS POSSIBLE:
#   Analysis of cross-correlation b/w L/R signals of DH recs and synthesized L/R auralized signals
#   Spectrogram visualization: b/w DH L/R signals and auralized stereo signal
#   Spectrogram/graprintphical representation comparing RIR estimate at particular time instance with recorded RIR at that position
# *

#Reading measured signals
wh_faster_data = np.load('Signals_Binary_npyz/wh_faster_data.npz')

wh_rec_l = wh_faster_data['wh_faster_dhl']
wh_rec_r = wh_faster_data['wh_faster_dhr']

speech_data = np.load('Signals_Binary_npyz/speech_data.npz')

speech_rec_l = speech_data['speech_DHL']
speech_rec_r = speech_data['speech_DHR']

sine_data = np.load('Signals_Binary_npyz/sine_500_data.npz')

sine_rec_l = sine_data['sine500_DHL']
sine_rec_r = sine_data['sine500_DHR']

#region Correlation Analysis

freq = 8000

#whNoise Covariances
len_rec_l = len(wh_rec_l)/fs
len_rec_r = len(wh_rec_r)/fs
wh_rec_l_down = signal.resample(wh_rec_l, int(len_rec_l * freq))
wh_rec_r_down = signal.resample(wh_rec_r, int(len_rec_r * freq))

t_delay = 2.4169767573696146
start_sample = int((5 - t_delay)*freq)
end_sample = int((10 - t_delay)*freq)

wh_rec_l_reqd = wh_rec_l_down[start_sample : end_sample]
wh_rec_r_reqd = wh_rec_r_down[start_sample : end_sample]

wh_corr_rec = signal.correlate(wh_rec_l_reqd, wh_rec_r_reqd, method='fft')

wh_corr_op = signal.correlate(op_wh_l, op_wh_r, method='fft')

wh_corr_err = np.abs(wh_corr_rec[0 : len(wh_corr_op)] - wh_corr_op) ** 2
wh_corr_err = np.log10(wh_corr_err/(np.abs(wh_corr_rec[0 : len(wh_corr_op)]) ** 2))

x_t_rec = np.linspace(-1, 1, len(wh_corr_rec))
x_t_op = np.linspace(-1, 1, len(wh_corr_op))

# Plotting relative error signal and correlation singals obtained
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(x_t_rec, wh_corr_rec, label='ρ(Measured L, Measured R)')
# axs[0].plot(x_t_op, wh_corr_op, label='ρ(Auralized L, Auralized R)')
# axs[0].set_title('Cross-correlation comparison between Measured and Auralized Signals')
# axs[0].set_xlabel("Overlap Ratio between Signals")
# axs[0].set_ylabel("Degree of +/- Correlation")
# axs[0].legend()
#
# axs[1].plot(x_t_op, wh_corr_err, color='green', label='Absolute error signal between cross correlations')
# axs[1].set_xlabel("Overlap Ratio Between Signals")
# axs[1].set_title("Squared Error Relative to Measurement Correlation")
# axs[1].set_ylabel("Error [dB]")
#
# fig.suptitle("WHITE NOISE")
# plt.tight_layout()
# plt.show()
# tikz.clean_figure()
# tikz.save("wh_op_corr.tex")

#speech covariance
t_start_sigs = 1.6708616780045351
start_sample = int(t_start_sigs * freq)
end_t = 14.141861678004535 # Got from ip_speech.py file
end_sample = int(end_t * freq)

len_rec_l_sp = len(speech_rec_l)/fs
len_rec_r_sp = len(speech_rec_r)/fs

speech_rec_l_down = signal.resample(speech_rec_l, int(len_rec_l_sp * freq))
speech_rec_r_down = signal.resample(speech_rec_r, int(len_rec_r_sp * freq))

speech_rec_l_reqd = speech_rec_l_down[start_sample : end_sample]
speech_rec_r_reqd = speech_rec_r_down[start_sample : end_sample]


speech_corr_recs = signal.correlate(speech_rec_l_reqd, speech_rec_r_reqd, method='fft')
speech_corr_op = signal.correlate(op_speech_l, op_speech_r, method='fft')

#Calc. error signal
speech_corr_err = np.abs(speech_corr_recs[0 : len(speech_corr_op)] - speech_corr_op) ** 2
speech_corr_err = np.log10(speech_corr_err/(np.abs(speech_corr_recs[0 : len(speech_corr_op)]) ** 2))

x_t_rec = np.linspace(-1, 1, len(speech_corr_recs))
x_t_op = np.linspace(-1, 1, len(speech_corr_op))

# Plotting relative error signal and correlation singals obtained
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(x_t_rec, speech_corr_recs, label='ρ(Measured L, Measured R)')
# axs[0].plot(x_t_op, speech_corr_op, label='ρ(Auralized L, Auralized R)')
# axs[0].set_title('Cross-correlation Comparison between Measured and Auralized Signals')
# # axs[0].set_xlabel("Overlap Ratio between Signals")
# axs[0].set_ylabel("Degree of +/- Correlation")
# axs[0].legend()
#
# axs[1].plot(x_t_op, speech_corr_err, color='green', label='Absolute error signal between cross correlations')
# axs[1].set_xlabel("Overlap Ratio Between Signals")
# axs[1].set_title("Squared Error Relative to Measurement Correlation")
# axs[1].set_ylabel("Error [dB]")
#
# fig.suptitle("SPEECH SIGNAL")
# plt.tight_layout()
# plt.show()
# # tikz.clean_figure()
# tikz.save("speech_op_corr.tex")

#SINE CORRELATIONS
t_start = 1.0893197278911564
t_end = 12.360173894557823 #Got from ip_sine.py
start_sample = int(t_start * freq)
end_sample = int(t_end * freq)

len_rec_l_si = len(sine_rec_l)/fs
len_rec_r_si = len(sine_rec_r)/fs

sine_rec_l_down = signal.resample(sine_rec_l, int(len_rec_l_si * freq))
sine_rec_r_down = signal.resample(sine_rec_r, int(len_rec_r_si * freq))

sine_rec_l_reqd = sine_rec_l_down[start_sample : end_sample]
sine_rec_r_reqd = sine_rec_r_down[start_sample : end_sample]

sine_corr_recs = signal.correlate(sine_rec_l_reqd, sine_rec_r_reqd, method='fft')
sine_corr_op = signal.correlate(op_sine_l, op_sine_r, method='fft')

sine_corr_err = np.abs(sine_corr_recs[0 : len(sine_corr_op)] - sine_corr_op) ** 2
sine_corr_err = np.log10(sine_corr_err/(np.abs(sine_corr_recs[0 : len(sine_corr_op)]) ** 2))

x_t_rec = np.linspace(-1, 1, len(sine_corr_recs))
x_t_op = np.linspace(-1, 1, len(sine_corr_op))

# Plotting relative error signal and correlation singals obtained
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(x_t_rec, sine_corr_recs, label='ρ(Measured L, Measured R)')
# axs[0].plot(x_t_op, sine_corr_op, label='ρ(Auralized L, Auralized R)')
# axs[0].set_title('Cross-correlation Comparison between Measured and Auralized Signals')
# # axs[0].set_xlabel("Overlap Ratio between Signals")
# axs[0].set_ylabel("Degree of +/- Correlation")
# axs[0].legend()
#
# axs[1].plot(x_t_op, sine_corr_err, color='green', label='Absolute error signal between cross correlations')
# axs[1].set_xlabel("Overlap Ratio Between Signals")
# axs[1].set_title("Squared Error Relative to Measurement Correlation")
# axs[1].set_ylabel("Error [dB]")
#
# fig.suptitle("SINE SIGNAL")
# plt.tight_layout()
# plt.show()
# # tikz.clean_figure()
# tikz.save("sine_op_corr.tex")
#endregion

#region Spectrogram/plot analysis of RIR estimates vs measured

#Measured RIRs
str_dir = './Signals_Binary_npyz/rirs_L_measured.npz'
rir_dir_l = pjoin(dirname(str_dir), 'rirs_L_measured.npz')
rir_l_data = np.load('./Signals_Binary_npyz/rirs_L_measured.npz')

rir_l1_l = rir_l_data['rir_l1_l']
rir_l2_l = rir_l_data['rir_l2_l']
rir_l3_l = rir_l_data['rir_l3_l']
rir_l4_l = rir_l_data['rir_l4_l']
rir_l5_l = rir_l_data['rir_l5_l']
rir_l6_l = rir_l_data['rir_l6_l']

len_rir_l5_l = len(rir_l5_l)/fs

rir_l5_l = signal.resample(rir_l5_l, int(len_rir_l5_l * freq))
rir_l4_l = signal.resample(rir_l4_l, int((len(rir_l4_l)/fs) * freq))
rir_l3_l = signal.resample(rir_l3_l, int((len(rir_l3_l)/fs) * freq))
rir_l2_l = signal.resample(rir_l2_l, int((len(rir_l2_l)/fs) * freq))
rir_l1_l = signal.resample(rir_l1_l, int((len(rir_l1_l)/fs) * freq))

rir_l1_l_measure = rir_l1_l[0:dim]
rir_l2_l_measure = rir_l2_l[0:dim]
rir_l3_l_measure = rir_l3_l[0:dim]
rir_l4_l_measure = rir_l4_l[0:dim]
rir_l5_l_measure = rir_l5_l[0:dim]

# rir_dir_r = pjoin(dirname(str_dir), 'rirs_R_measured.npz')
rir_r_data = np.load('./Signals_Binary_npyz/rirs_R_measured.npz')
#
rir_l1_r = rir_r_data['rir_l1_r']
rir_l2_r = rir_r_data['rir_l2_r']
rir_l3_r = rir_r_data['rir_l3_r']
rir_l4_r = rir_r_data['rir_l4_r']
rir_l5_r = rir_r_data['rir_l5_r']
rir_l6_r = rir_r_data['rir_l6_r']

len_rir_l5_r = len(rir_l5_r)/fs

rir_l5_r = signal.resample(rir_l5_r, int(len_rir_l5_r * freq))
rir_l4_r = signal.resample(rir_l4_r, int((len(rir_l4_r)/fs) * freq))
rir_l3_r = signal.resample(rir_l3_r, int((len(rir_l3_r)/fs) * freq))
rir_l2_r = signal.resample(rir_l2_r, int((len(rir_l2_r)/fs) * freq))
rir_l1_r = signal.resample(rir_l1_r, int((len(rir_l1_r)/fs) * freq))

rir_l1_r_measure = rir_l1_r[0:dim]
rir_l2_r_measure = rir_l2_r[0:dim]
rir_l3_r_measure = rir_l3_r[0:dim]
rir_l4_r_measure = rir_l4_r[0:dim]
rir_l5_r_measure = rir_l5_r[0:dim]

#White noise
t_l5_wh = 6.65

#Speech timing at which fixed positions(P1-P6) were passed
t_l5 = 10.35
t_l4 = 12.55
t_l3 = 15.75
t_l2 = 18.1
t_l1 = 21

sig_start = np.argmax(speech_input > 0.05)
t_start = sig_start/fs
start_sample = int(t_start * freq)

t_start_mob = 7.529

l5_sample = (t_l5 - t_start_mob) * freq
l4_sample = (t_l4 - t_start_mob) * freq
l3_sample = (t_l3 - t_start_mob) * freq
l2_sample = (t_l2 - t_start_mob) * freq
l1_sample = (t_l1 - t_start_mob) * freq

#White Noise: RIR estimate at position P5
rir_l5_l_est_wh = op_rirs_wh_faster[int(t_l5_wh * freq) - 20664]

rir_l5_r_est_wh = op_rirs_whFaster_r[int(t_l5_wh * freq) - 20664]

nperseg = 8

l5_l_est_f, l5_l_est_t, l5_l_est_Sxx = signal.spectrogram(rir_l5_l_est_wh, freq, nperseg=nperseg, window='hann')#, noverlap=0.75*nperseg)
l5_r_est_f, l5_r_est_t, l5_r_est_Sxx = signal.spectrogram(rir_l5_r_est_wh, freq, nperseg=nperseg, window='hann')#, noverlap=0.75*nperseg)
l5_l_est_f = l5_l_est_f/1000
l5_r_est_f = l5_r_est_f/1000

l5_l_measure_f, l5_l_measure_t, l5_l_measure_Sxx = signal.spectrogram(rir_l5_l_measure, freq, nperseg=nperseg, window='hann')#, noverlap=0.75*nperseg)
l5_r_measure_f, l5_r_measure_t, l5_r_measure_Sxx = signal.spectrogram(rir_l5_r_measure, freq, nperseg=nperseg, window='hann')#, noverlap=0.75*nperseg)
l5_l_measure_f = l5_l_measure_f/1000
l5_r_measure_f = l5_r_measure_f/1000

# plt.pcolormesh(l5_l_est_t, l5_l_est_f, 10*np.log10(l5_l_est_Sxx**2), shading='gouraud')
# plt.clim(-190, -230)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P5 (L-channel) Auralized from White Noise")
# plt.show()
#
# plt.pcolormesh(l5_l_measure_t, l5_l_measure_f, 10*np.log10(l5_l_measure_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P5 (L-channel) Measured")
# plt.show()
#
# plt.pcolormesh(l5_r_est_t, l5_r_est_f, 10*np.log10(l5_r_est_Sxx**2), shading='gouraud')
# plt.clim(-240, -300)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P5 (R-channel) Auralized from White Noise")
# plt.show()
#
# plt.pcolormesh(l5_r_measure_t, l5_r_measure_f, 10*np.log10(l5_r_measure_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P5 (R-channel) Measured")
# plt.show()

#SPEECH SIGNAL
#RIRs estimated from positions L1-L5
rir_l1_l_est_speech = out_rirs_speech_l[int(l1_sample) - 13366]
rir_l1_r_est_speech = out_rirs_speech_r[int(l1_sample) - 13366]

rir_l5_l_est_speech = out_rirs_speech_l[int(l5_sample) - 13366]
rir_l5_r_est_speech = out_rirs_speech_r[int(l5_sample) - 13366]

l1_est_l_f, l1_est_l_t, l1_est_l_Sxx = signal.spectrogram(rir_l1_l_est_speech, freq, nperseg=nperseg, window='hann')
l1_est_r_f, l1_est_r_t, l1_est_r_Sxx = signal.spectrogram(rir_l1_r_est_speech, freq, nperseg=nperseg, window='hann')
l1_est_l_f = l1_est_l_f/1000
l1_est_r_f = l1_est_r_f/1000

# l5_est_l_f, l5_est_l_t, l5_est_l_Sxx = signal.spectrogram(rir_l5_l_est_speech, freq, nperseg=nperseg, window='hann')
# l5_est_r_f, l5_est_r_t, l5_est_r_Sxx = signal.spectrogram(rir_l5_r_est_speech, freq, nperseg=nperseg, window='hann')

l1_measure_l_f, l1_measure_l_t, l1_measure_l_Sxx = signal.spectrogram(rir_l1_l_measure, freq, nperseg=nperseg, window='hann')
l1_measure_r_f, l1_measure_r_t, l1_measure_r_Sxx = signal.spectrogram(rir_l1_r_measure, freq, nperseg=nperseg, window='hann')
l1_measure_l_f = l1_measure_l_f/1000
l1_measure_r_f = l1_measure_r_f/1000

# plt.pcolormesh(l1_est_l_t, l1_est_l_f, 10*np.log10(l1_est_l_Sxx**2), shading='gouraud')
# plt.clim(-230, -260)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P1 (L-channel) Auralized from Speech Signal")
# plt.show()
#
# plt.pcolormesh(l1_measure_l_t, l1_measure_l_f, 10*np.log10(l1_measure_l_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P1 (L-channel) Measured")
# plt.show()
#
# plt.pcolormesh(l1_est_r_t, l1_est_r_f, 10*np.log10(l1_est_r_Sxx**2), shading='gouraud')
# plt.clim(-240, -270)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P1 (R-channel) Auralized from Speech Signal")
# plt.show()
#
# plt.pcolormesh(l1_measure_r_t, l1_measure_r_f, 10*np.log10(l1_measure_r_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P1 (R-channel) Measured")
# plt.show()


rir_l2_l_est_speech = out_rirs_speech_l[int(l2_sample) - 13366]
rir_l2_r_est_speech = out_rirs_speech_r[int(l2_sample) - 13366]

l2_est_l_f, l2_est_l_t, l2_est_l_Sxx = signal.spectrogram(rir_l2_l_est_speech, freq, nperseg=nperseg, window='hann')
l2_est_r_f, l2_est_r_t, l2_est_r_Sxx = signal.spectrogram(rir_l2_r_est_speech, freq, nperseg=nperseg, window='hann')
l2_est_l_f = l2_est_l_f/1000
l2_est_r_f = l2_est_r_f/1000

# l5_est_l_f, l5_est_l_t, l5_est_l_Sxx = signal.spectrogram(rir_l5_l_est_speech, freq, nperseg=nperseg, window='hann')
# l5_est_r_f, l5_est_r_t, l5_est_r_Sxx = signal.spectrogram(rir_l5_r_est_speech, freq, nperseg=nperseg, window='hann')

l2_measure_l_f, l2_measure_l_t, l2_measure_l_Sxx = signal.spectrogram(rir_l2_l_measure, freq, nperseg=nperseg, window='hann')
l2_measure_r_f, l2_measure_r_t, l2_measure_r_Sxx = signal.spectrogram(rir_l2_r_measure, freq, nperseg=nperseg, window='hann')
l2_measure_l_f = l2_measure_l_f/1000
l2_measure_r_f = l2_measure_r_f/1000

# plt.pcolormesh(l2_est_l_t, l2_est_l_f, 10*np.log10(l2_est_l_Sxx**2), shading='gouraud')
# plt.clim(-180, -220)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P2 (L-channel) Auralized from Speech Signal")
# plt.show()

# plt.pcolormesh(l2_measure_l_t, l2_measure_l_f, 10*np.log10(l2_measure_l_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P2 (L-channel) Measured")
# plt.show()

# plt.pcolormesh(l2_est_r_t, l2_est_r_f, 10*np.log10(l2_est_r_Sxx**2), shading='gouraud')
# plt.clim(-180, -220)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P2 (R-channel) Auralized from Speech Signal")
# plt.show()

# plt.pcolormesh(l2_measure_r_t, l2_measure_r_f, 10*np.log10(l2_measure_r_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P2 (R-channel) Measured")
# plt.show()

rir_l3_l_est_speech = out_rirs_speech_l[int(l3_sample) - 13366]
rir_l3_r_est_speech = out_rirs_speech_r[int(l3_sample) - 13366]

l3_est_l_f, l3_est_l_t, l3_est_l_Sxx = signal.spectrogram(rir_l3_l_est_speech, freq, nperseg=nperseg, window='hann')
l3_est_r_f, l3_est_r_t, l3_est_r_Sxx = signal.spectrogram(rir_l3_r_est_speech, freq, nperseg=nperseg, window='hann')
l3_est_l_f = l3_est_l_f/1000
l3_est_r_f = l3_est_r_f/1000

# l5_est_l_f, l5_est_l_t, l5_est_l_Sxx = signal.spectrogram(rir_l5_l_est_speech, freq, nperseg=nperseg, window='hann')
# l5_est_r_f, l5_est_r_t, l5_est_r_Sxx = signal.spectrogram(rir_l5_r_est_speech, freq, nperseg=nperseg, window='hann')

l3_measure_l_f, l3_measure_l_t, l3_measure_l_Sxx = signal.spectrogram(rir_l3_l_measure, freq, nperseg=nperseg, window='hann')
l3_measure_r_f, l3_measure_r_t, l3_measure_r_Sxx = signal.spectrogram(rir_l3_r_measure, freq, nperseg=nperseg, window='hann')
l3_measure_l_f = l3_measure_l_f/1000
l3_measure_r_f = l3_measure_r_f/1000

# plt.pcolormesh(l3_est_l_t, l3_est_l_f, 10*np.log10(l3_est_l_Sxx**2), shading='gouraud')
# plt.clim(-190, -230)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P3 (L-channel) Auralized from Speech Signal")
# plt.show()

# plt.pcolormesh(l3_measure_l_t, l3_measure_l_f, 10*np.log10(l3_measure_l_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P3 (L-channel) Measured")
# plt.show()

# plt.pcolormesh(l3_est_r_t, l3_est_r_f, 10*np.log10(l3_est_r_Sxx**2), shading='gouraud')
# plt.clim(-210, -260)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P3 (R-channel) Auralized from Speech Signal")
# plt.show()

# plt.pcolormesh(l3_measure_r_t, l3_measure_r_f, 10*np.log10(l3_measure_r_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P3 (R-channel) Measured")
# plt.show()


rir_l4_l_est_speech = out_rirs_speech_l[int(l4_sample) - 13366]
rir_l4_r_est_speech = out_rirs_speech_r[int(l4_sample) - 13366]

l4_est_l_f, l4_est_l_t, l4_est_l_Sxx = signal.spectrogram(rir_l4_l_est_speech, freq, nperseg=nperseg, window='hann')
l4_est_r_f, l4_est_r_t, l4_est_r_Sxx = signal.spectrogram(rir_l4_r_est_speech, freq, nperseg=nperseg, window='hann')
l4_est_l_f = l4_est_l_f/1000
l4_est_r_f = l4_est_r_f/1000

# l5_est_l_f, l5_est_l_t, l5_est_l_Sxx = signal.spectrogram(rir_l5_l_est_speech, freq, nperseg=nperseg, window='hann')
# l5_est_r_f, l5_est_r_t, l5_est_r_Sxx = signal.spectrogram(rir_l5_r_est_speech, freq, nperseg=nperseg, window='hann')

l4_measure_l_f, l4_measure_l_t, l4_measure_l_Sxx = signal.spectrogram(rir_l4_l_measure, freq, nperseg=nperseg, window='hann')
l4_measure_r_f, l4_measure_r_t, l4_measure_r_Sxx = signal.spectrogram(rir_l4_r_measure, freq, nperseg=nperseg, window='hann')
l4_measure_l_f = l4_measure_l_f/1000
l4_measure_r_f = l4_measure_r_f/1000

# plt.pcolormesh(l4_est_l_t, l4_est_l_f, 10*np.log10(l4_est_l_Sxx**2), shading='gouraud')
# plt.clim(-180, -220)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P4 (L-channel) Auralized from Speech Signal")
# plt.show()

# plt.pcolormesh(l4_measure_l_t, l4_measure_l_f, 10*np.log10(l4_measure_l_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P4 (L-channel) Measured")
# plt.show()

# plt.pcolormesh(l4_est_r_t, l4_est_r_f, 10*np.log10(l4_est_r_Sxx**2), shading='gouraud')
# plt.clim(-190, -230)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P4 (R-channel) Auralized from Speech Signal")
# plt.show()

# plt.pcolormesh(l4_measure_r_t, l4_measure_r_f, 10*np.log10(l4_measure_r_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P4 (R-channel) Measured")
# plt.show()


rir_l5_l_est_speech = out_rirs_speech_l[int(l5_sample) - 13366]
rir_l5_r_est_speech = out_rirs_speech_r[int(l5_sample) - 13366]

l5_est_l_f, l5_est_l_t, l5_est_l_Sxx = signal.spectrogram(rir_l5_l_est_speech, freq, nperseg=nperseg, window='hann')
l5_est_r_f, l5_est_r_t, l5_est_r_Sxx = signal.spectrogram(rir_l5_r_est_speech, freq, nperseg=nperseg, window='hann')
l5_est_l_f = l5_est_l_f/1000
l5_est_r_f = l5_est_r_f/1000

# l5_est_l_f, l5_est_l_t, l5_est_l_Sxx = signal.spectrogram(rir_l5_l_est_speech, freq, nperseg=nperseg, window='hann')
# l5_est_r_f, l5_est_r_t, l5_est_r_Sxx = signal.spectrogram(rir_l5_r_est_speech, freq, nperseg=nperseg, window='hann')

l5_measure_l_f, l5_measure_l_t, l5_measure_l_Sxx = signal.spectrogram(rir_l5_l_measure, freq, nperseg=nperseg, window='hann')
l5_measure_r_f, l5_measure_r_t, l5_measure_r_Sxx = signal.spectrogram(rir_l5_r_measure, freq, nperseg=nperseg, window='hann')
l5_measure_l_f = l5_measure_l_f/1000
l5_measure_r_f = l5_measure_r_f/1000

# plt.pcolormesh(l5_est_l_t, l5_est_l_f, 10*np.log10(l5_est_l_Sxx**2), shading='gouraud')
# plt.clim(-200, -240)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P5 (L-channel) Auralized from Speech Signal")
# plt.show()

# plt.pcolormesh(l5_measure_l_t, l5_measure_l_f, 10*np.log10(l5_measure_l_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 5], [1, 2, 3, 5])
# plt.ylim(0,4)
# plt.title("RIR at P5 (L-channel) Measured")
# plt.show()

# plt.pcolormesh(l5_est_r_t, l5_est_r_f, 10*np.log10(l5_est_r_Sxx**2), shading='gouraud')
# plt.clim(-210, -250)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 5], [1, 2, 3, 5])
# plt.ylim(0,4)
# plt.title("RIR at P5 (R-channel) Auralized from Speech Signal")
# plt.show()
#
# plt.pcolormesh(l5_measure_r_t, l5_measure_r_f, 10*np.log10(l5_measure_r_Sxx**2), shading='gouraud')
# plt.clim(0, -40)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.ylabel('Frequency [kHz]')
# plt.xlabel('Time [s]')
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0,4)
# plt.title("RIR at P5 (R-channel) Measured")
# plt.show()

#endregion

#region Spectrogram of I/O Signals

#White noise
op_wh_l_f, op_wh_l_t, op_wh_l_Sxx = signal.spectrogram(op_wh_l, freq, window='hann')
op_wh_r_f, op_wh_r_t, op_wh_r_Sxx = signal.spectrogram(op_wh_r, freq, window='hann')
op_wh_r_f = op_wh_r_f/1000
op_wh_l_f = op_wh_l_f/1000

measure_wh_l_f, measure_wh_l_t, measure_wh_l_Sxx = signal.spectrogram(wh_rec_l_reqd, freq, window='hann')
measure_wh_r_f, measure_wh_r_t, measure_wh_r_Sxx = signal.spectrogram(wh_rec_r_reqd, freq, window='hann')
measure_wh_r_f = measure_wh_r_f/1000
measure_wh_l_f = measure_wh_l_f/1000

#Plotting
# plt.pcolormesh(op_wh_l_t, op_wh_l_f, 10*np.log10(op_wh_l_Sxx**2), shading='auto')
# plt.clim(-100, -130)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Auralized white noise : L-channel")
# plt.show()
#
# plt.pcolormesh(op_wh_r_t, op_wh_r_f, 10*np.log10(op_wh_r_Sxx**2), shading='auto')
# plt.clim(-100, -130)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Auralized white noise : R-channel")
# plt.show()
#
# plt.pcolormesh(measure_wh_l_t, measure_wh_l_f, 10*np.log10(measure_wh_l_Sxx**2), shading='auto')
# plt.clim(-100, -130)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Measured white noise : L-channel")
# plt.show()
#
# plt.pcolormesh(measure_wh_r_t, measure_wh_r_f, 10*np.log10(measure_wh_r_Sxx**2), shading='auto')
# plt.clim(-100, -130)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Measured white noise : R-channel")
# plt.show()

#SPEECH
op_speech_l_f, op_speech_l_t, op_speech_l_Sxx = signal.spectrogram(op_speech_l, freq, window='hann')
op_speech_r_f, op_speech_r_t, op_speech_r_Sxx = signal.spectrogram(op_speech_r, freq, window='hann')
op_speech_l_f = op_speech_l_f/1000
op_speech_r_f = op_speech_r_f/1000

measure_speech_l_f, measure_speech_l_t, measure_speech_l_Sxx = signal.spectrogram(speech_rec_l_reqd, freq, window='hann')
measure_speech_r_f, measure_speech_r_t, measure_speech_r_Sxx = signal.spectrogram(speech_rec_r_reqd, freq, window='hann')
measure_speech_r_f = measure_speech_r_f/1000
measure_speech_l_f = measure_speech_l_f/1000

#Plotting
# plt.pcolormesh(op_speech_l_t, op_speech_l_f, 10*np.log10(op_speech_l_Sxx**2), shading='auto')
# plt.clim(-120, -170)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Auralized speech signal : L-channel")
# plt.show()
#
# plt.pcolormesh(op_speech_r_t, op_speech_r_f, 10*np.log10(op_speech_r_Sxx**2), shading='auto')
# plt.clim(-120, -170)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Auralized speech signal : R-channel")
# plt.show()
#
# plt.pcolormesh(measure_speech_l_t, measure_speech_l_f, 10*np.log10(measure_speech_l_Sxx**2), shading='auto')
# plt.clim(-120, -170)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Measured speech signal : L-channel")
# plt.show()
#
# plt.pcolormesh(measure_speech_r_t, measure_speech_r_f, 10*np.log10(measure_speech_r_Sxx**2), shading='auto')
# plt.clim(-120, -170)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Measured speech signal : R-channel")
# plt.show()

#SINE
op_sine_l_f, op_sine_l_t, op_sine_l_Sxx = signal.spectrogram(op_sine_l[0:10*freq], freq, window='hann')
op_sine_r_f, op_sine_r_t, op_sine_r_Sxx = signal.spectrogram(op_sine_r[0:10*freq], freq, window='hann')
op_sine_l_f = op_sine_l_f/1000
op_sine_r_f = op_sine_r_f/1000

measure_sine_l_f, measure_sine_l_t, measure_sine_l_Sxx = signal.spectrogram(sine_rec_l_reqd[0:10*freq], freq, window='hann')
measure_sine_r_f, measure_sine_r_t, measure_sine_r_Sxx = signal.spectrogram(sine_rec_r_reqd[0:10*freq], freq, window='hann')
measure_sine_r_f = measure_sine_r_f/1000
measure_sine_l_f = measure_sine_l_f/1000

#Plotting
# plt.pcolormesh(op_sine_l_t, op_sine_l_f, 10*np.log10(op_sine_l_Sxx**2), shading='auto')
# plt.clim(-170, -190)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Auralized sine signal : L-channel")
# plt.show()
#
# plt.pcolormesh(op_sine_r_t, op_sine_r_f, 10*np.log10(op_sine_r_Sxx**2), shading='auto')
# plt.clim(-170, -190)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Auralized sine signal : R-channel")
# plt.show()
#
# plt.pcolormesh(measure_sine_l_t, measure_sine_l_f, 10*np.log10(measure_sine_l_Sxx**2), shading='auto')
# plt.clim(-170, -190)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Measured sine signal : L-channel")
# plt.show()
#
# plt.pcolormesh(measure_sine_r_t, measure_sine_r_f, 10*np.log10(measure_sine_r_Sxx**2), shading='auto')
# plt.clim(-170, -190)
# plt.colorbar(label='Power Spectral Density (dB/Hz)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (kHz)")
# plt.yscale('log')
# plt.yticks([1, 2, 3, 4], [1, 2, 3, 4])
# plt.ylim(0, 4)
# plt.title("Measured sine signal : R-channel")
# plt.show()

#endregion