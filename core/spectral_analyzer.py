"""
Analyseur spectral par FFT.
Extrait les composantes dominantes (fréquences, amplitudes, phases) d'un signal.
"""

import numpy as np
from scipy.signal import find_peaks

def analyser_spectre(signal: np.ndarray, fs: int, n_pics_max: int = 100):
    """
    Extrait les N fréquences dominantes, leurs amplitudes et leurs phases.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal audio temporel 1D.
    fs : int
        Fréquence d'échantillonnage.
    n_pics_max : int
        Nombre maximum de pics à extraire.
        
    Returns
    -------
    freqs, amplitudes, phases : tuple de np.ndarray
    """
    N = len(signal)
    window = np.hanning(N)
    signal_windowed = signal * window
    
    # Transformée de Fourier (domaine fréquentiel complexe)
    fft_res = np.fft.rfft(signal_windowed)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    # Calcul de la magnitude et de la phase
    magnitude = np.abs(fft_res) / N
    phases = np.angle(fft_res)
    
    # Détection des pics (seuil à 0.5% de l'amplitude maximale)
    seuil = np.max(magnitude) * 0.005
    peaks, _ = find_peaks(magnitude, height=seuil)
    
    # Tri par amplitude décroissante
    sorted_idx = peaks[np.argsort(magnitude[peaks])[::-1]]
    top_idx = sorted_idx[:n_pics_max]
    
    return freqs[top_idx], magnitude[top_idx], phases[top_idx]
