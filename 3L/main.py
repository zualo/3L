import pandas as pd
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'cm'
rcParams['ytick.labelsize'] = 10.0
rcParams['xtick.labelsize'] = 10.0
rcParams['legend.handletextpad'] = 0.3
rcParams['legend.handlelength'] = 0.85
rcParams['legend.borderaxespad'] = 0.4
rcParams['legend.labelspacing'] = 0.10
rcParams['legend.framealpha'] = 1.00
rcParams['legend.borderpad'] = 0.5
rcParams['legend.handleheight'] = 1.0
rcParams['legend.edgecolor'] = 'k'
rcParams['axes.linewidth'] = 0.5
rcParams['lines.linewidth'] = 1.0

def omega_model(l, A, B):
    return (A / l) * (0.5 + B / l)**(-0.5)

def zeta_model(l, C, B):
    return C * l * (0.5 + B / l)**0.5

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

    F = np.abs(F)**2

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t_trim, v_trim)
    ax2.plot(w, F)

    ax1.set_xlabel(r"Time ($s$)", fontsize=15)
    ax1.set_ylabel(r"Voltage ($V$)", fontsize=15)

    ax2.set_xlim([-500, 500])
    ax2.set_xticks(np.arange(-500, 501, 100))

    w_vis = (w >= -500) & (w <= 500)
    F_vis = F[w_vis]

    peaks, _ = find_peaks(F_vis, height=np.max(F_vis)*0.05, distance=5)

    peak_freqs = w[w_vis][peaks]
    peak_powers = F_vis[peaks]

    ax2.plot(peak_freqs[1], peak_powers[1], "x", color='red', label=f'Sensor Peak ({peak_freqs[1]:.0f} hz)', markersize=8, markeredgewidth=2)
    ax2.plot(peak_freqs[2], peak_powers[2], "x", color='green', label=f'Hose Peak ({peak_freqs[2]:.0f} hz)', markersize=8, markeredgewidth=2)
    ax2.set_xlabel(r"Frequency ($hz$)", fontsize=15)
    ax2.set_ylabel(r"$| \widehat{V}(\omega) |^2$", fontsize=15)
    ax2.legend()

    title_part = file_path.split('/')[1]
    fig.suptitle(f"{title_part} Tube Balloon Response", fontsize=18)
    fig.tight_layout()

def plot_omega_zeta(lengths, w_balloon, w_valve, z_balloon, z_valve):
    l_all = np.concatenate((lengths, lengths))
    w_all = np.concatenate((w_balloon, w_valve))
    z_all = np.concatenate((z_balloon, z_valve))
    
    mask_w = ~np.isnan(w_all)
    mask_z = ~np.isnan(z_all)
    
    l_valid_w = l_all[mask_w]
    w_valid = w_all[mask_w]
    
    l_valid_z = l_all[mask_z]
    z_valid = z_all[mask_z]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Deliverable 4: Dynamic Characteristics vs. Tube Length", fontweight='bold')
    
    ax1.plot(lengths, w_balloon, 'bo', label='Balloon Data')
    ax1.plot(lengths, w_valve, 'ro', label='Valve Data')
    
    ax1.set_xlabel("Tube Length, $l$ (m)")
    ax1.set_ylabel(r"Natural Frequency, $\omega$ (rad/s)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    ax2.plot(lengths, z_balloon, 'bo', label='Balloon Data')
    ax2.plot(lengths, z_valve, 'ro', label='Valve Data')
    
    ax2.set_xlabel("Tube Length, $l$ (m)")
    ax2.set_ylabel(r"Damping Ratio, $\zeta$")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_data("./Short/ShortBalloonOscilloscopeData.csv")
    analyze_data("./Medium/MediumBalloonOscilloscopeData.csv")
    analyze_data("./Long/LongBalloonOscilloscopeData.csv")

    lengths_measured = np.array([0.1, 0.5, 1.0])
    w_ball = np.array([1811.4, 679.3, 452.8])
    w_valv = np.array([np.nan, 708.31, 463.11])
    z_ball = np.array([0.0450, 0.0500, 0.0650])
    z_valv = np.array([np.nan, 0.051, 0.081])
    plot_omega_zeta(lengths_measured, w_ball, w_valv, z_ball, z_valv)

    plt.show()
