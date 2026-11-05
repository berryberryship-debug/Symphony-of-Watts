"""
Harmonisation par résolution d'intervalle en cents.

Convention :
    A reste strictement fixe.
    B est ajusté vers l'intervalle consonant le plus proche parmi :
        unisson (0), tierce mineure (300), tierce majeure (400),
        quinte (700), octave (1200).
        
Les renversements (sixte mineure 900, etc.) peuvent être ajoutés
via le paramètre intervalles_cibles_cents.
"""

import numpy as np


def freq_to_cents(f_ref: float, f: float) -> float:
    """Calcule l'intervalle en cents entre une fréquence de référence et une fréquence."""
    return 1200.0 * np.log2(f / f_ref)


def cents_to_freq(f_ref: float, cents: float) -> float:
    """Convertit un intervalle en cents en fréquence absolue."""
    return f_ref * (2.0 ** (cents / 1200.0))


def resoudre_intervalle_continu(
    f_A: float,
    f_B: float,
    intervalles_cibles_cents: list[float] | None = None,
    plage_octaves: int = 3,
):
    """
    A reste fixe. B est ajusté vers l'intervalle consonant le plus proche,
    en tenant compte des octaves.

    Paramètres
    ----------
    f_A : float
        Fréquence de la voiture A (référence, reste fixe).
    f_B : float
        Fréquence de la voiture B (à ajuster).
    intervalles_cibles_cents : list[float], optionnel
        Liste des intervalles cibles en cents. Par défaut :
        [0.0, 300.0, 400.0, 700.0, 1200.0]
    plage_octaves : int, optionnel
        Nombre d'octaves explorées de part et d'autre. Défaut : 3.

    Retour
    ------
    (f_A_cible, f_B_cible, cout_cents) : tuple
        f_A_cible = f_A (inchangé)
        f_B_cible = fréquence ajustée de B
        cout_cents = déplacement absolu de B en cents
    """
    if intervalles_cibles_cents is None:
        # Unisson (0), tierce mineure (300), tierce majeure (400),
        # quinte juste (700), octave (1200)
        intervalles_cibles_cents = [0.0, 300.0, 400.0, 700.0, 1200.0]

    intervalle_actuel = freq_to_cents(f_A, f_B)

    # Génération des candidats sur plusieurs octaves
    candidats = [
        cible + octave * 1200.0
        for octave in range(-plage_octaves, plage_octaves + 1)
        for cible in intervalles_cibles_cents
    ]

    # Sélection du candidat le plus proche (déplacement minimal de B)
    meilleure_cible = min(candidats, key=lambda c: abs(intervalle_actuel - c))
    cout_cents = abs(intervalle_actuel - meilleure_cible)
    f_B_cible = cents_to_freq(f_A, meilleure_cible)

    return f_A, f_B_cible, cout_cents
