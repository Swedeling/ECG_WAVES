from utils.data_loader import DataLoader
from modules.R_detection import PanTompkins


if __name__ == '__main__':
    data_loader = DataLoader()
    raw_signal = data_loader.get_signal()

    pan_tompkins = PanTompkins(raw_signal)
