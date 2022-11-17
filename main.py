from utils.data_loader import DataLoader
from modules.R_detection import PanTompkins


if __name__ == '__main__':
    data_loader = DataLoader()
    raw_signal = data_loader.get_signal()
    r_peaks_detector = PanTompkins(raw_signal)




