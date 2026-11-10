from typing import Dict, Tuple, List

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from load_ecg import load_ecg_signals

def plot_signals(
    signals: List[ArrayLike], 
    sampling_rate: float = 300, 
    title: str = "ECG Signal",
) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    for signal in signals:
        signal = np.array(signal)
        total_time = len(signal) / sampling_rate
        time = np.linspace(0, total_time, len(signal))
        ax.plot(time, signal)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(title)

def fast_fourier_transform(
    signal: ArrayLike,
    sampling_rate: float = 300,
) -> Tuple[ArrayLike, ArrayLike]:
    signal = np.array(signal)
    n = len(signal)
    freq = np.fft.fftfreq(n, 1 / sampling_rate)
    fft_values = np.fft.fft(signal)
    freq = np.fft.fftshift(freq)
    fft_values = np.fft.fftshift(fft_values)
    return freq, fft_values

def plot_fft(
    freq: ArrayLike, 
    fft_values: ArrayLike, 
    title: str = "FFT of ECG Signal",
) -> None:
    fft_values = np.array(fft_values)
    fig = plt.figure(figsize=(10, 5))

    ax_mag = fig.add_subplot(211)
    ax_mag.plot(freq, np.abs(fft_values))
    ax_mag.set_ylabel("Amplitude")

    ax_phase = fig.add_subplot(212)
    ax_phase.plot(freq, np.angle(fft_values))
    ax_phase.set_ylabel("Phase (radians)")
    ax_phase.set_xlabel("Frequency (Hz)")
    fig.suptitle(title)

def main() -> None:
    ecg_noisy, ecg_clean = load_ecg_signals()

    # Change the following ecg_clean to ecg_noisy to plot the noisy signal
    signal = ecg_clean
    
    plot_signals([signal])
    freq, fft_values = fast_fourier_transform(signal)
    plot_fft(freq, fft_values)
    plt.show()


if __name__ == "__main__":
    main()

