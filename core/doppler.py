import numpy as np

def appliquer_doppler(f_emise: float, v_radiale: float, c: float = 343.0) -> float:
    """
    Applique le décalage Doppler.
    Convention : v_radiale > 0 si la source s'approche de l'observateur.
    """
    if type(f_emise) not in (int, float) or not np.isfinite(f_emise) or f_emise <= 0:
        raise ValueError("Fréquence émise invalide.")
    if type(c) not in (int, float) or not np.isfinite(c) or c <= 0:
        raise ValueError("Célérité invalide.")
    if type(v_radiale) not in (int, float) or not np.isfinite(v_radiale):
        raise ValueError("Vitesse radiale non finie.")
    if abs(v_radiale) >= c:
        raise ValueError("Vitesse radiale critique ou supersonique.")
    return f_emise * (c / (c - v_radiale))
