import numpy as np
from scipy.signal import find_peaks

def analyser_spectre(signal: np.ndarray, fs: int, n_pics_max: int = 100):
    N = len(signal)
    window = np.hanning(N)
    signal_windowed = signal * window
    
    fft_res = np.fft.rfft(signal_windowed)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    # Correction : facteur 2 pour l'amplitude crête d'un cosinus réel
    magnitude = 2.0 * np.abs(fft_res) / np.sum(window)
    phases = np.angle(fft_res)
    
    seuil = np.max(magnitude) * 0.005
    peaks, _ = find_peaks(magnitude, height=seuil)
    
    sorted_idx = peaks[np.argsort(magnitude[peaks])[::-1]]
    top_idx = sorted_idx[:n_pics_max]
    
    return freqs[top_idx], magnitude[top_idx], phases[top_idx]
