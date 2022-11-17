import matplotlib.pyplot as plt
import numpy as np


class DataLoader:
    def __init__(self):
        self.data = None
        self.data_path = 'data/100_MLII.dat'

        self.load_data()
        self.plot_data()

    def load_data(self):
        self.data = np.loadtxt(self.data_path)

    def get_signal(self):
        return self.data

    def plot_data(self):
        plt.plot(self.data)
        plt.title('Raw signal')
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [mV]')
        plt.xlim([0, 1000])
        plt.show()
