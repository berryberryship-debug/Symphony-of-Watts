import numpy as np
from scipy.signal import stft, istft, resample

def transposer_audio_brut(signal: np.ndarray, ratio: float, fs: int = 44100) -> np.ndarray:
    """
    Transposition audio (Pitch Shifting) avec Vocodeur de Phase.
    Utilise un étirement temporel (time-stretch) suivi d'un rééchantillonnage.
    """
    if abs(ratio - 1.0) < 1e-4:
        return signal

    nperseg = 2048  
    noverlap = nperseg * 3 // 4
    
    # 1. Transformée de Fourier à court terme (STFT)
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 2. Modification du spectre (Time-stretching)
    # On inverse le ratio pour l'étirement temporel (si on monte le pitch, on doit étirer le temps)
    time_steps = np.arange(0, Zxx.shape[1], 1.0 / ratio)
    time_steps = time_steps[time_steps < Zxx.shape[1] - 1]
    
    # Interpolation des parties réelle et imaginaire
    Zxx_stretched = np.zeros((Zxx.shape[0], len(time_steps)), dtype=complex)
    for i in range(Zxx.shape[0]):
        real_interp = np.interp(time_steps, np.arange(Zxx.shape[1]), np.real(Zxx[i, :]))
        imag_interp = np.interp(time_steps, np.arange(Zxx.shape[1]), np.imag(Zxx[i, :]))
        Zxx_stretched[i, :] = real_interp + 1j * imag_interp
        
    # Reconstruit le signal étiré
    _, signal_stretched = istft(Zxx_stretched, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 3. Rééchantillonnage (Resampling) pour corriger la durée et obtenir le bon pitch
    n_samples_final = len(signal)
    
    # Si le signal étiré est plus court que l'original, on le pad
    if len(signal_stretched) < n_samples_final:
        signal_stretched = np.pad(signal_stretched, (0, n_samples_final - len(signal_stretched)))
    
    # On rééchantillonne pour revenir à la durée initiale
    signal_transpose = resample(signal_stretched, n_samples_final)
    
    # Normalisation douce optionnelle
    max_val = np.max(np.abs(signal_transpose))
    if max_val > 0:
        signal_transpose = signal_transpose / max_val * np.max(np.abs(signal))
        
    return signal_transpose[:len(signal)]
