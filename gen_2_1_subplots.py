import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gen_plots(measure_rirs, est_rirs, gap, t_l1, t_l2, t_l3, t_l4, t_l5, t_start):
    freq = 8000
    dim = len(est_rirs[0])
    x_t = np.linspace(0, dim / freq, dim)

    l1_sample = int((freq * t_l1) - (freq * t_start))
    l2_sample = int((freq * t_l2) - (freq * t_start))
    l3_sample = int((freq * t_l3) - (freq * t_start))
    l4_sample = int((freq * t_l4) - (freq * t_start))
    l5_sample = int((freq * t_l5) - (freq * t_start))

    i = 0
    j = 0
    while i < len(est_rirs):
        try:
            # if-else : 0 < i < l5, l4 < i < l3, ..., l2 < i < l1, else (l1 case)...
            # make sure comparison at specific location is done BRODER CONDTS!!
            str_name = "./rir_evolutions/sine/R_channel/" + str(i) + ".png"
            if i >= 0 and i < l5_sample:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.plot(x_t, measure_rirs[0])  # 0, 1, ..., 4, 5 #Normailized to [0,1]??
                ax2.plot(x_t, est_rirs[i])

                # Plotting Initial State Estimate and RIR
                ax1.set_title('NEAREST MEASURED STATE ESTIMATE')
                ax2.set_title('ESTIMATED STATE ESTIMATE')
                ax1.set_ylabel("Amplitude")
                ax1.set_xlabel("Time [s]")
                ax2.set_xlabel("Time [s]")
                plt.tight_layout()
                plt.savefig(str_name)
                plt.close()
                # plt.show()

            elif i >= l5_sample and i < l4_sample:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.plot(x_t, measure_rirs[1])  # 0, 1, ..., 4, 5 #Normailized to [0,1]??
                ax2.plot(x_t, est_rirs[i])

                # Plotting Initial State Estimate and RIR
                ax1.set_title('NEAREST MEASURED STATE ESTIMATE')
                ax2.set_title('ESTIMATED STATE ESTIMATE')
                ax1.set_ylabel("Amplitude")
                ax1.set_xlabel("Time [s]")
                ax2.set_xlabel("Time [s]")
                plt.tight_layout()
                plt.savefig(str_name)
                plt.close()
                # plt.show()

            elif i >= l4_sample and i < l3_sample:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.plot(x_t, measure_rirs[2])  # 0, 1, ..., 4, 5 #Normailized to [0,1]??
                ax2.plot(x_t, est_rirs[i])

                # Plotting Initial State Estimate and RIR
                ax1.set_title('NEAREST MEASURED STATE ESTIMATE')
                ax2.set_title('ESTIMATED STATE ESTIMATE')
                ax1.set_ylabel("Amplitude")
                ax1.set_xlabel("Time [s]")
                ax2.set_xlabel("Time [s]")
                plt.tight_layout()
                plt.savefig(str_name)
                plt.close()
                # plt.show()

            elif i >= l3_sample and i < l2_sample:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.plot(x_t, measure_rirs[3])  # 0, 1, ..., 4, 5 #Normailized to [0,1]??
                ax2.plot(x_t, est_rirs[i])

                # Plotting Initial State Estimate and RIR
                ax1.set_title('NEAREST MEASURED STATE ESTIMATE')
                ax2.set_title('ESTIMATED STATE ESTIMATE')
                ax1.set_ylabel("Amplitude")
                ax1.set_xlabel("Time [s]")
                ax2.set_xlabel("Time [s]")
                plt.tight_layout()
                plt.savefig(str_name)
                plt.close()
                # plt.show()

            elif i >= l2_sample and i < l1_sample:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.plot(x_t, measure_rirs[4])  # 0, 1, ..., 4, 5 #Normailized to [0,1]??
                ax2.plot(x_t, est_rirs[i])

                # Plotting Initial State Estimate and RIR
                ax1.set_title('NEAREST MEASURED STATE ESTIMATE')
                ax2.set_title('ESTIMATED STATE ESTIMATE')
                ax1.set_ylabel("Amplitude")
                ax1.set_xlabel("Time [s]")
                ax2.set_xlabel("Time [s]")
                plt.tight_layout()
                plt.savefig(str_name)
                plt.close()
                # plt.show()

            elif i>= l1_sample:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.plot(x_t, measure_rirs[5])  # 0, 1, ..., 4, 5 #Normailized to [0,1]??
                ax2.plot(x_t, est_rirs[i])

                # Plotting Initial State Estimate and RIR
                ax1.set_title('NEAREST MEASURED STATE ESTIMATE')
                ax2.set_title('ESTIMATED STATE ESTIMATE')
                ax1.set_ylabel("Amplitude")
                ax1.set_xlabel("Time [s]")
                ax2.set_xlabel("Time [s]")
                plt.tight_layout()
                plt.savefig(str_name)
                plt.close()
                # plt.show()

            else: break

            if j % 4 != 0 or j == 0:
                i += 1
                j += 1
            else:
                i += gap
                j = 0

        except IndexError:
            print("Estimated RIRs passed through")
            break

