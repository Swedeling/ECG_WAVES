import math
import matplotlib.pyplot as plt
import numpy as np


class PanTompkins:
    def __init__(self, raw_signal):
        self.raw_signal = raw_signal
        self.frequency = 360
        self.window_size = 15
        self.upper_threshold = 15
        self.lower_threshold = 5

        self.r_peaks, self.qrs_onset, self.qrs_end = self.run_pan_tompkins()

        self.plot_signal()

    def lowpass_filter(self):
        upper_frequency = self.upper_threshold / (self.frequency/2)
        lowpass_window = []

        for coef in [i for i in range(-self.window_size, self.window_size+1)]:
            if coef == 0:
                lowpass_window.append(2*upper_frequency)
            else:
                lowpass_window.append(math.sin(2 * math.pi*upper_frequency * coef)/(math.pi * coef))

        return lowpass_window

    def highpass_filter(self):
        lower_frequency = self.lower_threshold / (self.frequency/2)
        highpass_window = []

        for coef in [i for i in range(-self.window_size, self.window_size+1)]:
            if coef == 0:
                highpass_window.append(1 - 2*lower_frequency)
            else:
                highpass_window.append(-1*(math.sin(2 * math.pi*lower_frequency * coef)/(math.pi * coef)))

        return highpass_window

    def get_filtered_signal(self, signal):
        hamming = np.hamming(2*self.window_size+1)

        lowpass_filter = hamming * self.lowpass_filter()
        highpass_filter = hamming * self.highpass_filter()

        filtered_signal = np.convolve(signal, lowpass_filter, 'same')
        filtered_signal = np.convolve(filtered_signal, highpass_filter, 'same')

        return filtered_signal

    @staticmethod
    def differentiate_signal(signal):
        diff_signal = np.diff(signal)
        return diff_signal / max(abs(diff_signal))

    @staticmethod
    def amplifying_signal(signal):
        ampl_signal = signal * signal
        return ampl_signal / max(abs(ampl_signal))

    def integrate_signal(self, signal):
        h = np.ones(2 * self.window_size + 1) / (2 * self.window_size + 1)
        i_signal = np.convolve(signal, h, 'same')
        return i_signal / max(abs(i_signal))

    @staticmethod
    def threshold_signal(signal):
        threshold1, threshold2 = None, None
        diff_signal = np.diff(signal)
        max_values = [1 if diff_signal[k - 1] > 0 > diff_signal[k] else 0 for k in range(1, len(diff_signal))]

        npki = np.mean(signal[:2000])
        spki = max(signal[:2000])

        for loc in [n for n in max_values if n == 1]:
            threshold1 = npki + 0.25 * (spki - npki)
            threshold2 = 0.5 * threshold1

            if threshold2 < signal[loc] < threshold1:
                npki = 0.125 * signal[loc] + 0.875 * npki
            if signal[loc] > threshold1:
                spki = 0.875 * signal[loc] + 0.125 * npki

        qrs_locs = [idx for idx, n in enumerate(signal) if n > threshold1]
        thr_signal = signal > threshold1

        return qrs_locs, thr_signal

    @staticmethod
    def get_r_peaks(filtered_signal, thr_signal):
        qs_points = [idx for idx, n in enumerate(np.diff(thr_signal)) if n == 1]

        r_locs = []
        for i in range(1, len(qs_points), 2):
            r_loc = np.argmax(filtered_signal[qs_points[i-1]:qs_points[i]])
            r_locs.append(r_loc + qs_points[i-1])

        r_values = [filtered_signal[loc] for loc in r_locs]

        return r_locs, r_values

    @staticmethod
    def get_qrs_onset(filtered_signal, thr_signal):

        left = [idx for idx, n in enumerate(np.diff(np.append(thr_signal, 0))) if n == 1]
        right = [idx for idx, n in enumerate(np.diff(np.append(0, thr_signal))) if n == 1]

        qrs_onset_locs = [np.argmax(filtered_signal[left[i]:right[i]]) + left[i] for i in range(min(len(left),
                                                                                                    len(right)))]
        qrs_onset_values = [filtered_signal[loc] for loc in qrs_onset_locs]

        return qrs_onset_locs, qrs_onset_values

    @staticmethod
    def get_qrs_end(filtered_signal, thr_signal):
        left = [idx for idx, n in enumerate(np.diff(np.append(thr_signal, 0))) if n == -1]
        right = [idx for idx, n in enumerate(np.diff(np.append(0, thr_signal))) if n == -1]

        qrs_end_locs = [np.argmax(filtered_signal[left[i]:right[i]]) + left[i] for i in range(min(len(left),
                                                                                                  len(right)))]
        qrs_end_values = [filtered_signal[loc] for loc in qrs_end_locs]

        return qrs_end_locs, qrs_end_values

    def run_pan_tompkins(self):
        filtered_signal = self.get_filtered_signal(self.raw_signal)
        diff_signal = self.differentiate_signal(filtered_signal)
        ampl_signal = self.amplifying_signal(diff_signal)
        integrated_signal = self.integrate_signal(ampl_signal)
        qrs_locs, thr_signal = self.threshold_signal(integrated_signal)

        r_peaks = self.get_r_peaks(filtered_signal, thr_signal)
        qrs_onset = self.get_qrs_onset(filtered_signal, thr_signal)
        qrs_end = self.get_qrs_end(filtered_signal, thr_signal)

        return r_peaks, qrs_onset, qrs_end

    def plot_signal(self):

        plt.plot(self.get_filtered_signal(self.raw_signal), color='black')

        plt.plot(self.r_peaks[0], self.r_peaks[1], 'o', color='blue', label='R peak')
        plt.plot(self.qrs_onset[0], self.qrs_onset[1], 'o', color='green', label='QRS-onset')
        plt.plot(self.qrs_end[0], self.qrs_end[1], 'o', color='red', label='QRS-end')

        plt.title('Pan-Tompkins algorithm result')
        plt.legend(loc="upper right")
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [mV]')

        plt.xlim([0, 2000])
        plt.ylim([-1, 1])

        plt.show()
