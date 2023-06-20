# An_Investigation_into_the_Auralization_of_Moving_Sound_Sources_Using_a_Kalman-Filter-THESIS_REPO
Hello! Welcome to the Git repository of our Master thesis. Here you may find the necessary code implementations for the procedures and results found in our thesis. Happy reading and testing! - Borys &amp; Gautam

Thesis Abstract - 
Auralization, a procedure designed to model and simulate the experience of an acoustic phenomenon, plays a vital role in simulating acoustic environments and understanding acoustic systems. So far, a majority of the research in this field is dedicated to scenarios where the receiver and the sound source are static (i.e., Linear Time Invariant systems). The primary motivation behind this thesis was the lack of an effective algorithmic framework for the auralization of moving sound sources. The objective of this Master's thesis is to investigate the challenge of auralization of moving sound sources using a Linear Time-Varying system formulation, which was done by employing a Kalman filter. To achieve this objective, a state-space model was developed to estimate the state of a time-varying system with largely unknown system dynamics, where the state is an RIR that varies over time. An experimental procedure that involved recording static and non-static sound signals with a dummy head was defined to verify our model to collect all the necessary audio files. The proposed Kalman filter implementation proved its ability to provide a state estimation framework for the auralization of moving sound sources, showing stable steady-state behavior and generating an estimate of a time-varying state capable of delivering the acoustic perception of a moving sound source. Potential future work could include exploring the shortcomings of the existing framework, such as working with narrowband signals and studying a movement trajectory that is not a straight line.

The code can be split into 2 parts: 2 .py files containing functions that are called in other py files. The role of each file can be found below:

kalman_filter.py -> Contains 2 functions, one to implement a single iteration of the Kalman filter implementation described in our thesis and one to run the Kalman filter across the length of the measurement signal

P_ss_est_svd.py -> Contains a function to compute the steady-state process error covariance matrix to be used as an initial estimate of the process noise covariance matrix used in the Kalman filter implementation

wh_noise_kalman_run.py -> The Kalman filter implementation is applied to the recorded moving white noise signal in order to auralize the same

ip_speech.py -> The Kalman filter implementation is applied to the recorded moving speech signal in order to auralize the same

ip_sine.py -> The Kalman filter implementation applied to the recorded moving sine signal in order to auralize the same

test_and_eval.py -> Contains code to evaluate our Kalman filter implementation. The correlation analysis and plotting of spectrograms are detailed here, along with the code necessary to generate stereo audio files using sf.write()
All the code necessary to plot the desired spectrograms/correlation analysis plots are commented out to avoid generating numerous plots simultaneously.

The files rir_evolution_movie.py, make_evo_movie.py, and gen_2_1_subplots.py are utilized to create a video demonstrating the evolution of the state in a 2x1 subplot with the nearest estimated state on the left subplot and the current state estimate on the right subplot which was used in the defense presentation. 

PS: All the audio files are read in as arrays using np.load() as the signals were previously stored as np arrays using sf.read() and then written to binary files using np.save()/ np.savez() for increased portability

Link to the Google Drive to hear the auralized audio files: https://drive.google.com/drive/folders/1jSTFwkyU0omvLX7l1e6CRNismbH63cZk?usp=sharing
