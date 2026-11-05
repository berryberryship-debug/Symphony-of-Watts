import numpy as np

def synthetiser_modal(frequences_cibles, amplitudes, phases, duree, fs=44100):
    """
    Synthèse additive pure.
    Génère un signal audio à partir des fréquences, amplitudes et phases données.
    Erreur de transposition : 0.00 cents (précision float64).
    """
    t = np.linspace(0, duree, int(fs * duree), endpoint=False)
    signal = np.zeros_like(t)
    
    # On somme toutes les composantes spectrales (fondamentales + harmoniques)
    for f, A, phi in zip(frequences_cibles, amplitudes, phases):
        signal += A * np.sin(2 * np.pi * f * t + phi)
        
    # Normalisation pour éviter la saturation
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.9
        
    return signal
