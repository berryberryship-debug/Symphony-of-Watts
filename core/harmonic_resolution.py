"""
Résolution harmonique : génération d'une grille pentatonique
et projection de fréquences spectrales sur cette grille.

Convention :
    - La grille est construite à partir d'une tonique donnée.
    - Une fréquence est projetée uniquement si elle se trouve
      à moins de `max_cents` d'une note de la grille.
    - Sinon, elle est conservée telle quelle.
"""

from __future__ import annotations

import numpy as np


def _freq_vers_cents(f_ref: float, f: float) -> float:
    """Calcule l'intervalle en cents entre une fréquence de référence et une fréquence."""
    return 1200.0 * np.log2(f / f_ref)


def generer_grille_pentatonique(
    tonique: float = 440.0,
    n_octaves: int = 5,
) -> np.ndarray:
    """
    Génère une grille pentatonique majeure autour de la tonique.

    Paramètres
    ----------
    tonique : float
        Fréquence de référence en Hz.
    n_octaves : int
        Nombre d'octaves générées au-dessus de la tonique.

    Retour
    ------
    np.ndarray
        Fréquences de la grille, triées par ordre croissant.
    """
    if not np.isfinite(tonique) or tonique <= 0:
        raise ValueError("La tonique doit être un nombre positif et fini.")
    if not isinstance(n_octaves, (int, np.integer)) or isinstance(n_octaves, bool):
        raise TypeError("n_octaves doit être un entier.")
    if n_octaves < 1:
        raise ValueError("n_octaves doit être au moins égal à 1.")

    # Degrés de la gamme pentatonique majeure : do, ré, mi, sol, la
    degres_cents = np.array([0.0, 200.0, 400.0, 700.0, 900.0], dtype=float)

    frequences = []

    for octave in range(n_octaves):
        for degre in degres_cents:
            cents = degre + 1200.0 * octave
            frequences.append(tonique * (2.0 ** (cents / 1200.0)))

    return np.asarray(sorted(frequences), dtype=float)


def projeter_sur_grille(
    frequences: np.ndarray,
    amplitudes: np.ndarray,
    grille: np.ndarray,
    max_cents: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projette chaque fréquence sur la note la plus proche de la grille,
    si l'écart est inférieur ou égal à `max_cents`.

    Paramètres
    ----------
    frequences : np.ndarray
        Fréquences à projeter.
    amplitudes : np.ndarray
        Amplitudes associées ( conservées telles quelles ).
    grille : np.ndarray
        Fréquences de la grille pentatonique.
    max_cents : float
        Écart maximal autorisé, en cents.

    Retour
    ------
    (frequences_proj, indices_grille, ecarts_cents) : tuple
        frequences_proj : fréquences après projection
        indices_grille : indice de la note retenue dans la grille,
                         ou -1 si aucune projection n'a été effectuée
        ecarts_cents : écart final entre la fréquence d'origine
                       et la fréquence projetée
    """
    frequences = np.asarray(frequences, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    grille = np.asarray(grille, dtype=float)

    if frequences.ndim != 1 or amplitudes.ndim != 1 or grille.ndim != 1:
        raise ValueError("frequences, amplitudes et grille doivent être des vecteurs 1D.")

    if frequences.size != amplitudes.size:
        raise ValueError("frequences et amplitudes doivent avoir la même taille.")

    if grille.size == 0:
        raise ValueError("La grille harmonique ne peut pas être vide.")

    if not np.all(np.isfinite(frequences)):
        raise ValueError("Les fréquences doivent être finies.")

    if not np.all(np.isfinite(grille)):
        raise ValueError("La grille doit contenir des fréquences finies.")

    if np.any(frequences <= 0):
        raise ValueError("Les fréquences doivent être strictement positives.")

    if np.any(grille <= 0):
        raise ValueError("Les fréquences de la grille doivent être strictement positives.")

    if not np.isfinite(max_cents) or max_cents < 0:
        raise ValueError("max_cents doit être un nombre positif ou nul.")

    frequences_proj = frequences.copy()
    indices_grille = np.full(frequences.size, -1, dtype=int)
    ecarts_cents = np.zeros(frequences.size, dtype=float)

    for i, f in enumerate(frequences):
        # Écart en cents avec chaque note de la grille
        ecarts = np.abs(_freq_vers_cents(grille, f))

        k = int(np.argmin(ecarts))
        ecart_min = float(ecarts[k])

        if ecart_min <= max_cents:
            frequences_proj[i] = grille[k]
            indices_grille[i] = k
            ecarts_cents[i] = _freq_vers_cents(f, grille[k])

    return frequences_proj, indices_grille, ecarts_cents
