from numpy.typing import ArrayLike
from scipy import signal
import matplotlib.pyplot as plt
from load_ecg import load_ecg_signals
from problem1 import plot_signals, fast_fourier_transform, plot_fft
from problem2 import design_filter


def main() -> None:
    ecg_noisy, ecg_clean = load_ecg_signals()
    b, a = design_filter()
    noisy_signal = ecg_noisy

    # change the following lfilter to filtfilt to have zero-phase filtering
    filtered_signal: ArrayLike = signal.lfilter(b, a, noisy_signal)

    plot_signals([filtered_signal, ecg_clean])
    freq, fft_values = fast_fourier_transform(filtered_signal)
    plot_fft(freq, fft_values)
    plt.show()

if __name__ == "__main__":
    main()