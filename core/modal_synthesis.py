import warnings
import numpy as np

def synthetiser_spectre_complet(frequences, amplitudes, phases, duree, fs, taille_bloc=16384):
    frequences = np.asarray(frequences, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)

    if any(v.ndim != 1 for v in (frequences, amplitudes, phases)):
        raise ValueError("Les trois entrées doivent être des vecteurs 1D.")

    if not (frequences.size == amplitudes.size == phases.size):
        raise ValueError("Les trois vecteurs doivent avoir la même taille.")

    if not all(np.all(np.isfinite(v)) for v in (frequences, amplitudes, phases)):
        raise ValueError("Les composantes doivent être finies.")

    if np.any(amplitudes < 0):
        raise ValueError("Les amplitudes doivent être positives ou nulles.")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs doit être positif et fini.")

    if not np.isfinite(duree) or duree <= 0:
        raise ValueError("La durée doit être positive et finie.")

    # Validation stricte du type de taille_bloc (correction test 28)
    if (
        isinstance(taille_bloc, (bool, np.bool_))
        or not isinstance(taille_bloc, (int, np.integer))
        or taille_bloc < 1
    ):
        raise ValueError("La taille des blocs doit être un entier strictement positif.")
    taille_bloc = int(taille_bloc)

    n = int(round(fs * duree))
    if n < 1:
        raise ValueError("La durée produit un signal vide.")

    valides = (frequences > 0) & (frequences < fs / 2)
    nombre_rejete = int(np.count_nonzero(~valides))

    if nombre_rejete:
        warnings.warn(
            f"{nombre_rejete} composante(s) hors bande écartée(s).",
            RuntimeWarning,
            stacklevel=2,
        )

    frequences = frequences[valides]
    amplitudes = amplitudes[valides]
    phases = phases[valides]

    sortie = np.zeros(n, dtype=np.float64)
    
    for debut in range(0, n, taille_bloc):
        fin = min(debut + taille_bloc, n)
        temps = np.arange(debut, fin, dtype=np.float64) / fs
        bloc = np.zeros(fin - debut, dtype=np.float64)

        for f, A, phi in zip(frequences, amplitudes, phases):
            bloc += A * np.cos(2.0 * np.pi * f * temps + phi)

        sortie[debut:fin] = bloc

    return sortie
