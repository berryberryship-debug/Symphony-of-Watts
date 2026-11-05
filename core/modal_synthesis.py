import numpy as np

def synthetiser_spectre_complet(frequences, amplitudes, phases, duree, fs=44100):
    """
    Synthèse additive massive.
    Reconstruit un signal audio complexe à partir de N composantes spectrales.
    """
    n_echantillons = int(fs * duree)
    t = np.linspace(0, duree, n_echantillons, endpoint=False)
    signal = np.zeros(n_echantillons, dtype=np.float64)
    
    # On vectorise pour calculer toutes les ondes d'un coup
    for f, A, phi in zip(frequences, amplitudes, phases):
        signal += A * np.sin(2 * np.pi * f * t + phi)
        
    # Normalisation douce
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = (signal / max_val) * 0.95
        
    return signal
