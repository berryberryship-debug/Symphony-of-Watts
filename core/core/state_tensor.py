import numpy as np
from dataclasses import dataclass

@dataclass
class EtatVibroAcoustique:
    temps: float
    frequences: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    couplages_kappa: np.ndarray
    domaine_physique: str = "NON_SPECIFIE"

    def __post_init__(self):
        if type(self.temps) not in (int, float) or not np.isfinite(self.temps):
            raise ValueError("Le temps doit être un réel fini.")
        if not isinstance(self.domaine_physique, str):
            raise TypeError("Le domaine physique doit être une chaîne de caractères.")

        for arr, nom in [
            (self.frequences, "frequences"),
            (self.amplitudes, "amplitudes"),
            (self.phases, "phases"),
            (self.couplages_kappa, "couplages_kappa"),
        ]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"'{nom}' doit être un np.ndarray.")
            if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
                raise TypeError(f"'{nom}' doit contenir des réels (pas de complexes ou booléens).")

        if self.frequences.ndim != 1 or self.amplitudes.ndim != 1 or self.phases.ndim != 1:
            raise ValueError("f, A, phi doivent être 1D.")

        N = len(self.frequences)
        if len(self.amplitudes) != N or len(self.phases) != N:
            raise ValueError("Dimensions asymétriques.")
        if self.couplages_kappa.shape != (N, N):
            raise ValueError("La matrice kappa doit être carrée N x N.")

        if (
            not np.all(np.isfinite(self.frequences))
            or not np.all(np.isfinite(self.amplitudes))
            or not np.all(np.isfinite(self.phases))
            or not np.all(np.isfinite(self.couplages_kappa))
        ):
            raise ValueError("Présence de valeurs non finies.")

        if np.any(self.frequences < 0) or np.any(self.amplitudes < 0):
            raise ValueError("Fréquences et amplitudes doivent être >= 0.")
