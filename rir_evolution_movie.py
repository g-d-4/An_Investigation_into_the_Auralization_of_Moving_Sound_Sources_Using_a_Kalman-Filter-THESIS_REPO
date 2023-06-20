import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from gen_2_1_subplots import gen_plots
from make_evo_movie import make_vid

#Read in RIRS
str_dir = './Outputs_npyz/wh_faster/output_rirs_wh_faster.npy'
str_dir = './out_whFaster_3_5_wSVD.npy'
op_dir_wh_faster = pjoin(dirname(str_dir), 'out_whFaster_3_5_wSVD.npy')
#
wh_rirs_l = np.load('./out_whFaster_3_5_wSVD.npy')

str_dir = './out_whFaster_3_5_wSVD_R.npy'
op_dir_wh_faster = pjoin(dirname(str_dir), 'out_whFaster_3_5_wSVD_R.npy')

wh_rirs_r = np.load('./out_whFaster_3_5_wSVD_R.npy')

speech_rirs_l = np.load(pjoin(dirname('./Outputs_npyz/speech/out_speech_8_5_L_02.npy'), 'out_speech_8_5_L_02.npy'))

speech_rirs_r = np.load(pjoin(dirname('./Outputs_npyz/speech/out_speech_8_5_R_02.npy'), 'out_speech_8_5_R_02.npy'))

sine_rirs_l = np.load('out_sine_9_5_L.npy')
sine_rirs_r = np.load('out_sine_9_5_R.npy')

q = len(speech_rirs_l[0])

#Measured RIRs
fs = 44100
freq = 8000

rir_l_data = np.load('./Signals_Binary_npyz/rirs_L_measured.npz')

rir_l1_l = rir_l_data['rir_l1_l']
rir_l2_l = rir_l_data['rir_l2_l']
rir_l3_l = rir_l_data['rir_l3_l']
rir_l4_l = rir_l_data['rir_l4_l']
rir_l5_l = rir_l_data['rir_l5_l']
rir_l6_l = rir_l_data['rir_l6_l']

len_rir_l5_l = len(rir_l5_l)/fs

rir_l6_l = signal.resample(rir_l6_l, int((len(rir_l6_l)/fs) * freq))
rir_l5_l = signal.resample(rir_l5_l, int(len_rir_l5_l * freq))
rir_l4_l = signal.resample(rir_l4_l, int((len(rir_l4_l)/fs) * freq))
rir_l3_l = signal.resample(rir_l3_l, int((len(rir_l3_l)/fs) * freq))
rir_l2_l = signal.resample(rir_l2_l, int((len(rir_l2_l)/fs) * freq))
rir_l1_l = signal.resample(rir_l1_l, int((len(rir_l1_l)/fs) * freq))

rir_l1_l_measure = rir_l1_l[0:q]
rir_l2_l_measure = rir_l2_l[0:q]
rir_l3_l_measure = rir_l3_l[0:q]
rir_l4_l_measure = rir_l4_l[0:q]
rir_l5_l_measure = rir_l5_l[0:q]
rir_l6_l_measure = rir_l6_l[0:q]

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

rir_l6_r = signal.resample(rir_l6_r, int((len(rir_l6_r)/fs) * freq))
rir_l5_r = signal.resample(rir_l5_r, int(len_rir_l5_r * freq))
rir_l4_r = signal.resample(rir_l4_r, int((len(rir_l4_r)/fs) * freq))
rir_l3_r = signal.resample(rir_l3_r, int((len(rir_l3_r)/fs) * freq))
rir_l2_r = signal.resample(rir_l2_r, int((len(rir_l2_r)/fs) * freq))
rir_l1_r = signal.resample(rir_l1_r, int((len(rir_l1_r)/fs) * freq))

rir_l1_r_measure = rir_l1_r[0:q]
rir_l2_r_measure = rir_l2_r[0:q]
rir_l3_r_measure = rir_l3_r[0:q]
rir_l4_r_measure = rir_l4_r[0:q]
rir_l5_r_measure = rir_l5_r[0:q]
rir_l6_r_measure = rir_l6_r[0:q]

measure_rirs_l = np.vstack((rir_l6_l_measure, rir_l5_l_measure, rir_l4_l_measure, rir_l3_l_measure, rir_l2_l_measure, rir_l1_l_measure))
measure_rirs_r = np.vstack((rir_l6_r_measure, rir_l5_r_measure, rir_l4_r_measure, rir_l3_r_measure, rir_l2_r_measure, rir_l1_r_measure))

#2x1 subplot: measured on L, estimated on R
#switch case: case pos. P6, P5, ... , P1 (sample corres. to position => plot)
#Between positions, keep last measured RIR on L subplot
#Default: some generic plot

#Save plots as image: make movie with openCV

#Timestamps corresponding to passing of positions

t_start_speech = 1.6708616780045351

t_l5_speech = 8.85
t_l4_speech = 11.95
t_l3_speech = 14.7
t_l2_speech = 16.95
t_l1_speech = 19.55

t_start_wh = 5 - 2.6864399092970523

t_l5_wh = 6.15
t_l4_wh = 7.2
t_l3_wh = 8.05
t_l2_wh = 9
t_l1_wh = 9.95

t_start_sine = 1.0893197278911564

t_l5_sine = 6.55
t_l4_sine = 8.45
t_l3_sine = 10.05
t_l2_sine = 12.05
t_l1_sine = 14.4

#Normailze RIRs
for i in range(0, len(measure_rirs_l) - 1):
    measure_rirs_l[i] = measure_rirs_l[i] / np.max(measure_rirs_l[i])
    measure_rirs_r[i] = measure_rirs_r[i] / np.max(measure_rirs_r[i])

for i in range(0, len(wh_rirs_l) - 1):
    if np.max(wh_rirs_l[i]) > 1:
        wh_rirs_l[i] = wh_rirs_l[i] / np.max(wh_rirs_l[i])

    if np.max(wh_rirs_r[i]) > 1:
        wh_rirs_r[i] = wh_rirs_r[i] / np.max(wh_rirs_r[i])

for i in range(0, len(speech_rirs_l) - 1):
    if np.max(speech_rirs_l[i]) > 1:
        speech_rirs_l[i] = speech_rirs_l[i] / np.max(speech_rirs_l[i])

    if np.max(speech_rirs_r[i]) > 1:
        speech_rirs_r[i] = speech_rirs_r[i] / np.max(speech_rirs_r[i])

for i in range(0, len(sine_rirs_l) - 1):
    if np.max(sine_rirs_l[i]) > 1:
        sine_rirs_l[i] = sine_rirs_l[i] / np.max(sine_rirs_l[i])

    if np.max(sine_rirs_r[i]) > 1:
        sine_rirs_r[i] = sine_rirs_r[i] / np.max(sine_rirs_r[i])

# print(len(speech_rirs_l)) # 99767
# print(len(sine_rirs_l)) # 90166
# print(len(wh_rirs_l)) # 39999

# gen_plots(measure_rirs_l, wh_rirs_l, 400, t_l1_wh, t_l2_wh, t_l3_wh, t_l4_wh, t_l5_wh, t_start_wh)
# gen_plots(measure_rirs_r, wh_rirs_r, 400, t_l1_wh, t_l2_wh, t_l3_wh, t_l4_wh, t_l5_wh, t_start_wh)
# gen_plots(measure_rirs_l, speech_rirs_l, 980, t_l1_speech, t_l2_speech, t_l3_speech, t_l4_speech, t_l5_speech, t_start_speech)
# gen_plots(measure_rirs_r, speech_rirs_r, 980, t_l1_speech, t_l2_speech, t_l3_speech, t_l4_speech, t_l5_speech, t_start_speech)
# gen_plots(measure_rirs_l, sine_rirs_l, 920, t_l1_sine, t_l2_sine, t_l3_sine, t_l4_sine, t_l5_sine, t_start_sine)
# gen_plots(measure_rirs_r, sine_rirs_r, 920, t_l1_sine, t_l2_sine, t_l3_sine, t_l4_sine, t_l5_sine, t_start_sine)

path = './rir_evolutions/speech/L_channel/'
op = './rir_evolutions/speech/speech_l.mp4'
make_vid(path, 15, op)