"""
Résolution harmonique : projection sur une grille musicale
avec allocation exclusive des slots et contrainte ±100 cents.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def generer_grille_pentatonique(
    tonique: float = 440.0,
    n_octaves: int = 4,
    octave_basse: int = -1,
) -> np.ndarray:
    """Génère une grille pentatonique majeure (tempérament égal)."""
    if not np.isfinite(tonique) or tonique <= 0:
        raise ValueError("Tonique invalide.")
    if not isinstance(n_octaves, int) or n_octaves <= 0:
        raise ValueError("n_octaves doit être un entier > 0.")

    degres = np.array([0, 2, 4, 7, 9])  # demi-tons
    freqs = []
    for oct in range(octave_basse, octave_basse + n_octaves):
        for d in degres:
            f = tonique * (2 ** ((oct * 12 + d) / 12))
            freqs.append(f)
    return np.array(sorted(freqs), dtype=float)


def distance_cents(f_source: float, f_cible: float) -> float:
    """Distance en cents entre deux fréquences."""
    if f_source <= 0 or f_cible <= 0:
        return np.inf
    return 1200.0 * np.log2(f_cible / f_source)


def calculer_cout_deplacement(
    freqs: np.ndarray,
    amps: np.ndarray,
    freqs_projetees: np.ndarray,
) -> float:
    """Coût = Σ Aᵢ × |ΔCᵢ|"""
    if len(freqs) != len(amps) or len(freqs) != len(freqs_projetees):
        raise ValueError("Dimensions incohérentes.")
    cout = 0.0
    for f_s, a, f_p in zip(freqs, amps, freqs_projetees):
        cout += a * abs(distance_cents(f_s, f_p))
    return float(cout)


def projeter_sur_grille(
    freqs: np.ndarray,
    amps: np.ndarray,
    grille: np.ndarray,
    max_cents: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Projection avec allocation exclusive triée par amplitude décroissante.
    Si aucun slot libre dans ±max_cents, le partiel conserve sa fréquence.
    """
    if len(freqs) != len(amps):
        raise ValueError("freqs et amps de tailles différentes.")
    if len(grille) == 0:
        raise ValueError("Grille vide.")

    freqs_proj = np.copy(freqs).astype(float)
    deltas = np.zeros(len(freqs), dtype=float)
    indices_tries = np.argsort(amps)[::-1]
    slots_occupes = set()

    for i in indices_tries:
        f = freqs[i]
        dists = np.array([distance_cents(f, g) for g in grille])
        indices_proches = np.argsort(np.abs(dists))

        projet_reussi = False
        for idx in indices_proches:
            delta = dists[idx]
            if abs(delta) > max_cents:
                break
            if idx not in slots_occupes:
                freqs_proj[i] = grille[idx]
                deltas[i] = delta
                slots_occupes.add(idx)
                projet_reussi = True
                break

        if not projet_reussi:
            freqs_proj[i] = f
            deltas[i] = 0.0

    cout = calculer_cout_deplacement(freqs, amps, freqs_proj)
    return freqs_proj, deltas, cout


def calculer_rugosite_simple(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Rugosité Plomp-Levelt (Sethares) simplifiée."""
    if len(freqs) < 2:
        return 0.0
    rugosite = 0.0
    N = len(freqs)
    for i in range(N):
        for j in range(i + 1, N):
            f_min = min(freqs[i], freqs[j])
            f_max = max(freqs[i], freqs[j])
            a_min = min(amps[i], amps[j])
            a_max = max(amps[i], amps[j])
            if f_min <= 0:
                continue
            s = 0.24 / (0.0207 * f_min + 18.96)
            df = f_max - f_min
            R = (a_min * a_max) * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))
            if R > 0:
                rugosite += R
    return float(rugosite)


def cout_total(
    freqs: np.ndarray,
    amps: np.ndarray,
    grille: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    max_cents: float = 100.0,
) -> Tuple[float, np.ndarray, float, float]:
    """J_total = α·J_déplacement + β·J_rugosité"""
    freqs_proj, _, j_dep = projeter_sur_grille(freqs, amps, grille, max_cents)
    j_rug = calculer_rugosite_simple(freqs_proj, amps)
    j_tot = alpha * j_dep + beta * j_rug
    return j_tot, freqs_proj, j_dep, j_rug
