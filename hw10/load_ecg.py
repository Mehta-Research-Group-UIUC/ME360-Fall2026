from typing import Dict, Tuple

from numpy.typing import NDArray
import scipy.io

def load_ecg_signals(
    filename: str = "ecg_signals.mat",
) -> Tuple[NDArray, NDArray]:
    ecg_signals: Dict = scipy.io.loadmat(filename)
    ecg_noisy: NDArray = ecg_signals["ecg_noisy"][0]
    ecg_clean: NDArray = ecg_signals["ecg_clean"][0]
    return ecg_noisy, ecg_clean
