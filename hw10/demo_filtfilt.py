import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def demonstrate_filtfilt():
    # Create a sample signal with noise
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    noisy_signal = clean_signal + np.random.normal(0, 0.5, clean_signal.shape)
    
    # Design Butterworth filter
    b, a = signal.butter(4, 0.1, btype='low')  # 4th order, cutoff = 0.1
    
    # Apply different filtering methods
    # 1. Forward-only filtering (lfilter)
    forward_filtered = signal.lfilter(b, a, noisy_signal)
    
    # 2. Zero-phase filtering (filtfilt)
    zero_phase_filtered = signal.filtfilt(b, a, noisy_signal)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(t, noisy_signal, label='Noisy Signal')
    plt.plot(t, clean_signal, 'r--', label='Original Clean')
    plt.legend()
    plt.title('Original vs Noisy Signal')
    
    plt.subplot(312)
    plt.plot(t, forward_filtered, label='Forward-only Filter')
    plt.plot(t, clean_signal, 'r--', label='Original Clean')
    plt.legend()
    plt.title('Forward-only Filtering (lfilter)')
    
    plt.subplot(313)
    plt.plot(t, zero_phase_filtered, label='Zero-phase Filter')
    plt.plot(t, clean_signal, 'r--', label='Original Clean')
    plt.legend()
    plt.title('Zero-phase Filtering (filtfilt)')
    
    plt.tight_layout()
    return t, noisy_signal, forward_filtered, zero_phase_filtered

# Run demonstration
t, noisy, forward, zero_phase = demonstrate_filtfilt()
plt.show()
