import numpy as np
from scipy.signal import stft, istft

def transposer_audio_brut(signal: np.ndarray, ratio: float, fs: int = 44100) -> np.ndarray:
    """
    Décale la fréquence d'un signal audio en conservant sa durée (Pitch Shifting par STFT).
    """
    if abs(ratio - 1.0) < 1e-4:
        return signal

    nperseg = 1024
    noverlap = nperseg * 3 // 4

    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    Zxx_transpose = np.zeros_like(Zxx)
    n_freqs = Zxx.shape[0]
    
    for i in range(n_freqs):
        cible = int(i / ratio)
        if 0 <= cible < n_freqs:
            Zxx_transpose[cible] += Zxx[i]

    _, signal_transpose = istft(Zxx_transpose, fs=fs, nperseg=nperseg, noverlap=noverlap)

    if len(signal_transpose) < len(signal):
        signal_transpose = np.pad(signal_transpose, (0, len(signal) - len(signal_transpose)))
    
    max_val = np.max(np.abs(signal_transpose))
    if max_val > 0:
        signal_transpose = signal_transpose / max_val
        
    return signal_transpose[:len(signal)]
