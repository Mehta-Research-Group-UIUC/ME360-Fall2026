import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal
import matplotlib.pyplot as plt
from load_ecg import load_ecg_signals
from problem1 import plot_signals, fast_fourier_transform, plot_fft
from problem2 import print_transfer_function

### Copy and paste the code from your designed filter function here in this file. ###


def main() -> None:
    ecg_noisy, ecg_clean = load_ecg_signals()
    sos = design_filter()
    b, a = signal.sos2tf(sos)
    print_transfer_function(b, a)

    noisy_signal = ecg_noisy

    filtered_signal: ArrayLike = signal.sosfilt(sos, noisy_signal)

    plot_signals([filtered_signal, ecg_clean])
    freq, fft_values = fast_fourier_transform(filtered_signal)
    plot_fft(freq, fft_values)

    filtered_signal: ArrayLike = signal.sosfiltfilt(sos, noisy_signal)

    plot_signals([filtered_signal, ecg_clean])
    freq, fft_values = fast_fourier_transform(filtered_signal)
    plot_fft(freq, fft_values)

    plt.show()

if __name__ == "__main__":
    main()