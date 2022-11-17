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

    def get_filtered_signal(self):
        hamming = np.hamming(2*self.window_size+1)

        lowpass_filter = hamming * self.lowpass_filter()
        highpass_filter = hamming * self.highpass_filter()

        filtered_signal = np.convolve(self.raw_signal, lowpass_filter)
        filtered_signal = np.convolve(filtered_signal, highpass_filter)

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
        signal_intg = np.convolve(signal, h)
        signal_intg = signal_intg / max(abs(signal_intg))
        return signal_intg

    def threshold_signal(self):
        pass

    def run_pan_tompkins(self):
        filtered_signal = self.get_filtered_signal()
        diff_signal = self.differentiate_signal(filtered_signal)
        ampl_signal = self.amplifying_signal(diff_signal)
        intg_signal = self.integrate_signal(ampl_signal)

        return intg_signal

    def plot_signal(self):
        output_signal = self.run_pan_tompkins()
        plt.plot(output_signal)
        plt.title('Filtered signal')
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [mV]')
        plt.xlim([0, 2000])
        plt.ylim([-1, 1])
        plt.show()
