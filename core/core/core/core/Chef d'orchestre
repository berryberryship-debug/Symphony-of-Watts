import numpy as np
from scipy.io import wavfile
import os

# Importation de ton architecture modulaire
from core.state_tensor import EtatVibroAcoustique
from core.doppler import appliquer_doppler
from core.f0_yin import extraire_f0_yin
from core.coupling_matrix import creer_couplage_kuramoto
from core.harmonic_resolution import generer_grille_pentatonique, projeter_sur_grille
from core.synthesis import synthetiser

def generer_signal_moteur_brut(f_whine, duree, fs):
    """Génère un faux signal de moteur pour simuler l'entrée YIN."""
    t = np.linspace(0, duree, int(fs * duree), endpoint=False)
    # Bruit + sifflement
    bruit = np.random.normal(0, 0.1, len(t))
    signal = 0.8 * np.sin(2 * np.pi * f_whine * t) + bruit
    return signal

def main():
    print("🚗 Initialisation de la scène Symphony of Watts...")
    fs = 44100
    duree = 3.0
    
    # 1. SIMULATION DES DEUX VOITURES (Signaux bruts)
    # Voiture A (ex: 3150 Hz) et Voiture B (ex: 3500 Hz)
    sig_A_brut = generer_signal_moteur_brut(3150.0, duree, fs)
    sig_B_brut = generer_signal_moteur_brut(3500.0, duree, fs)
    
    # 2. ANALYSE YIN (Extraction de la fréquence fondamentale)
    res_A = extraire_f0_yin(sig_A_brut, fs, f_min=1000, f_max=8000)
    res_B = extraire_f0_yin(sig_B_brut, fs, f_min=1000, f_max=8000)
    
    if not res_A or not res_B:
        print("❌ Impossible d'extraire la fréquence fondamentale.")
        return
        
    f_A = res_A[0]
    f_B = res_B[0]
    print(f"  -> Voiture A détectée à {f_A:.2f} Hz (Confiance : {res_A[1]:.2f})")
    print(f"  -> Voiture B détectée à {f_B:.2f} Hz (Confiance : {res_B[1]:.2f})")
    
    # 3. PROJECTION HARMONIQUE (Grille pentatonique basée sur A)
    grille = generer_grille_pentatonique(tonique=f_A, n_octaves=3)
    
    # On regroupe les deux fréquences pour les projeter ensemble
    freqs_initiales = np.array([f_A, f_B])
    amps_initiales = np.array([1.0, 0.8])
    
    freqs_harmoniques, deltas, cout = projeter_sur_grille(
        freqs_initiales, amps_initiales, grille, max_cents=100.0
    )
    print(f"  -> Fréquences projetées : {freqs_harmoniques[0]:.2f} Hz et {freqs_harmoniques[1]:.2f} Hz (Coût : {cout:.2f} cents)")
    
    # 4. COUPLAGE KURAMOTO (Les voitures interagissent)
    # On crée une matrice d'interaction (les deux oscillateurs se synchronisent)
    kappa = creer_couplage_kuramoto(N=2, force=5.0, topologie="all_to_all")
    
    # 5. CRÉATION DE L'ÉTAT VIBROACOUSTIQUE (Tenseur protégé)
    etat = EtatVibroAcoustique(
        temps=0.0,
        frequences=freqs_harmoniques,
        amplitudes=amps_initiales,
        phases=np.array([0.0, 0.0]),
        couplages_kappa=kappa,
        domaine_physique="ACOUSTIQUE"
    )
    
    # 6. SYNTHÈSE AUDIO FINALE
    print("\n🎵 Synthèse audio en cours (couplage Kuramoto)...")
    signal_stereo, phases_finales = synthetiser(
        freqs=etat.frequences,
        amps=etat.amplitudes,
        phases=etat.phases,
        kappa=etat.couplages_kappa,
        duree=duree,
        fs=fs,
        appliquer_resolution=False, # Déjà fait manuellement à l'étape 3
        tonique=f_A
    )
    
    # Sauvegarde
    nom_fichier = "symphony_of_watts_demo.wav"
    wavfile.write(nom_fichier, fs, np.int16(signal_stereo * 32767))
    print(f"✅ Fichier généré : {nom_fichier}")

if __name__ == "__main__":
    main()
