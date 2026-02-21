import pandas as pd
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#from matplotlib import rcParams
#rcParams['text.usetex'] = True

def analyze_data(file_path):
    df = pd.read_csv(file_path, skiprows=5, usecols=[0, 1])
    t = df['Time'].values
    v = df['Voltage'].values
    
    mask = (t >= 0) & (t <= 0.1)

    t_trim = t[mask]
    v_trim = v[mask]

    dt = t[1] - t[0]
    n = len(t_trim)

    F = fft.ifft(v_trim)
    w = fft.fftfreq(n, d=dt)

    F = fft.fftshift(F)
    w = fft.fftshift(w)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t_trim, v_trim)
    ax2.plot(w, np.abs(F)**2)

    ax2.set_xlim([-500, 500])
    ax2.set_xticks(np.arange(-500, 501, 100))


if __name__ == "__main__":
    analyze_data("./Short/ShortBalloonOscilloscopeData.csv")
    analyze_data("./Medium/MediumBalloonOscilloscopeData.csv")
    analyze_data("./Long/LongBalloonOscilloscopeData.csv")

    plt.show()