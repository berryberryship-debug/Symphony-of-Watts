import numpy as np

def freq_to_cents(f_ref: float, f: float) -> float:
    return 1200.0 * np.log2(f / f_ref)

def cents_to_freq(f_ref: float, cents: float) -> float:
    return f_ref * (2.0 ** (cents / 1200.0))

def resoudre_intervalle_continu(f_A: float, f_B: float, intervalles_cibles_cents: list = None):
    """
    A reste strictement fixe. B cherche la consonance la plus proche en cents.
    """
    if intervalles_cibles_cents is None:
        # Tierce mineure (300), Tierce Majeure (400), Quinte (700)
        intervalles_cibles_cents = [300.0, 400.0, 700.0]
        
    f_A_cible = f_A
    intervalle_actuel = freq_to_cents(f_A, f_B)
    
    meilleure_cible = None
    distance_min = float('inf')
    
    # On teste sur une large plage d'octaves (de -3 à +3)
    for octave in range(-3, 4):
        for cible in intervalles_cibles_cents:
            for signe in [1, -1]:
                intervalle_candidat = (cible * signe) + (octave * 1200)
                distance = abs(intervalle_actuel - intervalle_candidat)
                
                if distance < distance_min:
                    distance_min = distance
                    meilleure_cible = intervalle_candidat
                    
    f_B_cible = cents_to_freq(f_A_cible, meilleure_cible)
    cout_cents = abs(intervalle_actuel - meilleure_cible)
    
    return f_A_cible, f_B_cible, cout_cents
