"""
Synthèse additive stéréo avec évolution de phase (Kuramoto) et fade.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from .coupling_matrix import evolution_phase_kuramoto
from .harmonic_resolution import generer_grille_pentatonique, projeter_sur_grille


def _appliquer_fade(signal: np.ndarray, fs: int, fade_sec: float = 0.05) -> np.ndarray:
    """Applique un fade-in et fade-out linéaire."""
    n_fade = int(fade_sec * fs)
    if n_fade <= 0 or n_fade * 2 >= len(signal):
        return signal
    fade_in = np.linspace(0.0, 1.0, n_fade)
    fade_out = np.linspace(1.0, 0.0, n_fade)
    signal = signal.copy()
    signal[:n_fade] *= fade_in
    signal[-n_fade:] *= fade_out
    return signal


def synthetiser(
    freqs: np.ndarray,
    amps: np.ndarray,
    phases: np.ndarray,
    kappa: np.ndarray,
    duree: float = 3.0,
    fs: int = 44100,
    dt_controle: float = 0.01,
    max_cents: float = 100.0,
    appliquer_resolution: bool = True,
    tonique: float = 440.0,
    fade_sec: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthèse additive stéréo avec évolution de phase via Kuramoto.

    Returns
    -------
    signal_stereo : np.ndarray de shape (n_samples, 2)
    phases_finales : phases à la fin de la simulation
    """
    N = len(freqs)
    if not (len(amps) == N and len(phases) == N):
        raise ValueError("Dimensions incohérentes.")
    if kappa.shape != (N, N):
        raise ValueError("kappa doit être N×N.")
    if duree <= 0 or fs <= 0 or dt_controle <= 0:
        raise ValueError("Paramètres temporels invalides.")

    # Résolution harmonique optionnelle
    if appliquer_resolution:
        grille = generer_grille_pentatonique(tonique=tonique, n_octaves=5)
        freqs, _, _ = projeter_sur_grille(freqs, amps, grille, max_cents=max_cents)

    n_echantillons = int(duree * fs)
    n_pas = max(1, int(duree / dt_controle))
    echantillons_par_pas = max(1, int(dt_controle * fs))

    signal_l = np.zeros(n_echantillons, dtype=float)
    signal_r = np.zeros(n_echantillons, dtype=float)
    phases_courantes = phases.copy().astype(float)
    omegas = 2.0 * np.pi * freqs

    # Pan stéréo simple selon l'index du mode
    pans = np.linspace(-0.8, 0.8, N)

    idx = 0
    for _ in range(n_pas):
        # Évolution des phases
        phases_courantes = evolution_phase_kuramoto(
            phases_courantes, omegas, kappa, dt_controle
        )

        t_local = np.arange(echantillons_par_pas) / fs
        seg_l = np.zeros(echantillons_par_pas, dtype=float)
        seg_r = np.zeros(echantillons_par_pas, dtype=float)

        for i in range(N):
            onde = amps[i] * np.sin(omegas[i] * t_local + phases_courantes[i])
            # Loi de pan constante de puissance
            g_l = np.cos((pans[i] + 1) * 0.25 * np.pi)
            g_r = np.sin((pans[i] + 1) * 0.25 * np.pi)
            seg_l += onde * g_l
            seg_r += onde * g_r

        fin = min(idx + echantillons_par_pas, n_echantillons)
        length = fin - idx
        signal_l[idx:fin] = seg_l[:length]
        signal_r[idx:fin] = seg_r[:length]
        idx = fin
        if idx >= n_echantillons:
            break

    # Normalisation + fade
    signal = np.stack([signal_l, signal_r], axis=1)
    max_abs = np.max(np.abs(signal))
    if max_abs > 0:
        signal /= max_abs * 1.02

    signal[:, 0] = _appliquer_fade(signal[:, 0], fs, fade_sec)
    signal[:, 1] = _appliquer_fade(signal[:, 1], fs, fade_sec)

    return signal, phases_courantes
