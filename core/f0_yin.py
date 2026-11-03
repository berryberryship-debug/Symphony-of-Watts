"""
Extraction de la fréquence fondamentale (F0) par l'algorithme YIN
(de Cheveigné & Kawahara, 2002), avec score de confiance.

Conçu pour être utilisé dans un pipeline industriel :
- validations strictes des entrées
- plancher RMS configurable
- retour explicite (f0, confiance) ou None
- bornes de recherche respectant strictement Nyquist
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


def _interpole_parabole(array: np.ndarray, idx: int) -> float:
    """Interpolation parabolique sub-échantillon."""
    if idx <= 0 or idx >= len(array) - 1:
        return float(idx)
    alpha, beta, gamma = array[idx - 1], array[idx], array[idx + 1]
    denom = alpha - 2.0 * beta + gamma
    if denom == 0.0:
        return float(idx)
    return idx + 0.5 * (alpha - gamma) / denom


def extraire_f0_yin(
    sig: np.ndarray,
    fs: int,
    f_min: float = 50.0,
    f_max: float = 2000.0,
    seuil: float = 0.1,
    seuil_rms: float = 1e-4,
) -> Optional[Tuple[float, float]]:
    """
    Estime la fréquence fondamentale d'un signal mono.

    Parameters
    ----------
    sig : np.ndarray
        Signal 1D réel.
    fs : int
        Fréquence d'échantillonnage (Hz), entier strictement positif.
    f_min, f_max : float
        Bornes de recherche (doivent respecter 0 < f_min < f_max < fs/2).
    seuil : float
        Seuil absolu sur la CMNDF (plus bas = plus strict).
    seuil_rms : float
        Plancher d'énergie. En dessous, le signal est considéré comme silence.

    Returns
    -------
    Optional[Tuple[float, float]]
        (f0_hz, confiance) où confiance ∈ [0, 1],
        ou None si aucune fondamentale fiable n'a été trouvée.
    """
    # --- Validations strictes ---
    if not isinstance(sig, np.ndarray) or sig.ndim != 1:
        raise TypeError("Le signal doit être un np.ndarray 1D.")
    if not (np.issubdtype(sig.dtype, np.floating) or np.issubdtype(sig.dtype, np.integer)):
        raise TypeError("Le signal doit contenir des valeurs réelles.")
    if not np.all(np.isfinite(sig)):
        raise ValueError("Le signal contient des valeurs non finies.")
    if type(fs) is not int or fs <= 0:
        raise TypeError("fs doit être un entier strictement positif.")
    if not (np.isfinite(f_min) and np.isfinite(f_max) and 0.0 < f_min < f_max < fs / 2.0):
        raise ValueError("Bornes fréquentielles invalides (doivent respecter Nyquist).")
    if not (0.0 < seuil <= 1.0):
        raise ValueError("seuil doit être dans (0, 1].")
    if seuil_rms < 0.0:
        raise ValueError("seuil_rms doit être ≥ 0.")

    # --- Prétraitement ---
    sig = sig.astype(float) - np.mean(sig)
    rms = np.sqrt(np.mean(sig ** 2))
    if rms < seuil_rms:
        return None

    lag_min = max(1, int(np.ceil(fs / f_max)))
    lag_max = int(np.floor(fs / f_min))
    W = len(sig) // 2
    if lag_max >= W or lag_min >= lag_max:
        return None

    # --- Différence au carré ---
    diff = np.zeros(lag_max + 1, dtype=float)
    for tau in range(1,
