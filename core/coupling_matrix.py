"""
Matrices de couplage dynamique K(t) pour Symphony of Watts.

Trois topologies de base :
- Nul          : modes indépendants
- Kuramoto     : synchronisation de phase
- Dissipatif   : transfert / amortissement d'amplitude
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def valider_matrice_kappa(kappa: np.ndarray, N: int, nom: str = "kappa") -> None:
    """Vérifie les invariants d'une matrice de couplage."""
    if not isinstance(kappa, np.ndarray):
        raise TypeError(f"{nom} doit être un np.ndarray.")
    if not (np.issubdtype(kappa.dtype, np.floating) or np.issubdtype(kappa.dtype, np.integer)):
        raise TypeError(f"{nom} doit contenir des valeurs réelles.")
    if kappa.ndim != 2 or kappa.shape != (N, N):
        raise ValueError(f"{nom} doit être une matrice carrée {N}x{N}.")
    if not np.all(np.isfinite(kappa)):
        raise ValueError(f"{nom} contient des valeurs non finies.")


# ------------------------------------------------------------------
# 1. COUPLAGE NUL
# ------------------------------------------------------------------
def creer_couplage_nul(N: int) -> np.ndarray:
    """Matrice de couplage nulle (modes totalement indépendants)."""
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N doit être un entier strictement positif.")
    return np.zeros((N, N), dtype=float)


# ------------------------------------------------------------------
# 2. COUPLAGE DE KURAMOTO
# ------------------------------------------------------------------
def creer_couplage_kuramoto(
    N: int,
    force: float = 0.1,
    topologie: str = "all_to_all",
    graine: Optional[int] = None,
) -> np.ndarray:
    """
    Matrice de couplage pour le modèle de Kuramoto.

    Équation :
        dφᵢ/dt = ωᵢ + Σⱼ κᵢⱼ sin(φⱼ - φᵢ)
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N doit être un entier strictement positif.")
    if not np.isfinite(force) or force < 0:
        raise ValueError("La force de couplage doit être ≥ 0.")

    kappa = np.zeros((N, N), dtype=float)

    if topologie == "all_to_all":
        kappa[:] = force / max(N - 1, 1)
        np.fill_diagonal(kappa, 0.0)

    elif topologie == "ring":
        for i in range(N):
            kappa[i, (i - 1) % N] = force
            kappa[i, (i + 1) % N] = force

    elif topologie == "random":
        rng = np.random.default_rng(graine)
        mask = rng.random((N, N)) < 0.3
        kappa[mask] = force
        np.fill_diagonal(kappa, 0.0)

    else:
        raise ValueError(f"Topologie inconnue : '{topologie}'. Attendu : all_to_all | ring | random")

    valider_matrice_kappa(kappa, N)
    return kappa


def evolution_phase_kuramoto(
    phases: np.ndarray,
    omegas: np.ndarray,
    kappa: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Une étape d'intégration d'Euler pour le modèle de Kuramoto.
    Retourne les nouvelles phases (ramenées dans [-π, π]).
    """
    N = len(phases)
    valider_matrice_kappa(kappa, N)
    if len(omegas) != N:
        raise ValueError("omegas et phases doivent avoir la même taille.")

    dphi = omegas.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                dphi[i] += kappa[i, j] * np.sin(phases[j] - phases[i])

    nouvelles = phases + dphi * dt
    return (nouvelles + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------
# 3. COUPLAGE DISSIPATIF
# ------------------------------------------------------------------
def creer_couplage_dissipatif(
    N: int,
    taux: float = 0.05,
    symetrique: bool = True,
) -> np.ndarray:
    """
    Matrice de couplage dissipatif (transfert d'amplitude).

    Équation typique :
        dAᵢ/dt ≈ -Σⱼ γᵢⱼ (Aᵢ - Aⱼ)
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N doit être un entier strictement positif.")
    if not np.isfinite(taux) or taux < 0:
        raise ValueError("Le taux de dissipation doit être ≥ 0.")

    kappa = np.full((N, N), taux, dtype=float)
    np.fill_diagonal(kappa, 0.0)

    if not symetrique:
        for i in range(N):
            for j in range(i + 1, N):
                kappa[j, i] *= 0.6  # asymétrie directionnelle simple

    valider_matrice_kappa(kappa, N)
    return kappa


def evolution_amplitude_dissipative(
    amplitudes: np.ndarray,
    kappa: np.ndarray,
    dt: float,
    amortissement_propre: float = 0.01,
) -> np.ndarray:
    """
    Une étape d'évolution des amplitudes sous couplage dissipatif.
    Garantit la non-négativité.
    """
    N = len(amplitudes)
    valider_matrice_kappa(kappa, N)
    if np.any(amplitudes < 0):
        raise ValueError("Les amplitudes doivent rester ≥ 0.")

    dA = -amortissement_propre * amplitudes
    for i in range(N):
        for j in range(N):
            if i != j:
                dA[i] -= kappa[i, j] * (amplitudes[i] - amplitudes[j])

    nouvelles = amplitudes + dA * dt
    return np.maximum(nouvelles, 0.0)
