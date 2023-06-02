# An_Investigation_into_the_Auralization_of_Moving_Sound_Sources_Using_a_Kalman-Filter-THESIS_REPO
Hello! Welcome to the Git repository of our Master thesis. Here you may find the necessary code implementations for the procedures and results found in our thesis. Happy reading and testing! - Borys &amp; Gautam

The code can be split into 2 parts: 2 .py files containing functions that are called in other py files. The funtion of each file can be found below:

kalman_filter.py -> Contains 2 funtions, one to implement a single iteration of the Kalman filter implementation described in our thesis, and one to run the Kalman filter across the length of the measurement signal

P_ss_est_svd.py -> Contains a funtion to compute the steady state process error covariance matrix to be used as an initial estimate of the process noise covariance matrix used in the Kalman filter implementation

wh_noise_kalman_run.py -> The Kalman filter implementation applied to the recorded moving white noise signal in order to auralize the same

ip_speech.py -> The Kalman filter implementation applied to the recorded moving speech signal in order to auralize the same

ip_sine.py -> The Kalman filter implementation applied to the recorded moving sine signal in order to auralize the same

test_and_eval.py -> Contains code to evaluate our Kalman filter implementation. The correlation analysis and plotting of spectrograms are detialed here, along with code necessary to generate stereo audio files using sf.write()
All the code necessary to plot the desired spectrograms / correlation analysis plots are commented out to avoid generating numerous plots simultaneously

PS: All the audio files are read in as arrays using np.load() as the signals were previously stored as np arrays using sf.read() and then written to binary files using np.save()/ np.savez() for increased portability

Link to the Google Drive to hear the auralized audio files: https://drive.google.com/drive/folders/1jSTFwkyU0omvLX7l1e6CRNismbH63cZk?usp=sharing
