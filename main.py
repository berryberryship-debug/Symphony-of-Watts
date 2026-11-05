import numpy as np
from scipy.io import wavfile
from core.f0_yin import extraire_f0_yin
from core.harmonisation import resoudre_intervalle_continu
from core.modal_synthesis import synthetiser_modal

def main():
    print("🚗 Initialisation du Moteur Modal...")
    fs = 44100
    duree = 3.0
    f_A_brut = 445.0 
    f_B_brut = 460.0 
    
    # Simulation des signaux bruts (juste pour l'extraction YIN)
    t = np.linspace(0, duree, int(fs * duree), endpoint=False)
    sig_A = 0.6 * np.sin(2 * np.pi * f_A_brut * t) + 0.2 * np.random.normal(0, 1, len(t))
    sig_B = 0.6 * np.sin(2 * np.pi * f_B_brut * t) + 0.2 * np.random.normal(0, 1, len(t))
    
    # Extraction YIN
    res_A = extraire_f0_yin(sig_A, fs, f_min=100, f_max=1000)
    res_B = extraire_f0_yin(sig_B, fs, f_min=100, f_max=1000)
    
    f_A_detecte, f_B_detecte = res_A[0], res_B[0]
    
    # Résolution harmonique
    f_A_cible, f_B_cible, cout = resoudre_intervalle_continu(f_A_detecte, f_B_detecte)
    print(f"   -> A reste a {f_A_cible:.2f} Hz")
    print(f"   -> B transposee a {f_B_cible:.2f} Hz (Cout : {cout:.2f} cents)")
    
    # SYNTHÈSE MODALE (plus de STFT)
    # Voiture A (fondamentale + 2 harmoniques)
    sig_A_final = synthetiser_modal(
        np.array([f_A_cible, f_A_cible*2, f_A_cible*3]), 
        np.array([0.6, 0.3, 0.1]), 
        np.array([0.0, 0.0, 0.0]), 
        duree, fs
    )
    
    # Voiture B (fondamentale transposée + 2 harmoniques)
    sig_B_final = synthetiser_modal(
        np.array([f_B_cible, f_B_cible*2, f_B_cible*3]), 
        np.array([0.6, 0.3, 0.1]), 
        np.array([0.0, 0.0, 0.0]), 
        duree, fs
    )
    
    # Assemblage stéréo
    signal_stereo = np.stack([sig_A_final, sig_B_final], axis=1)
    
    nom_fichier = "symphony_modale.wav"
    wavfile.write(nom_fichier, fs, np.int16(signal_stereo * 32767))
    print(f"\n✅ Fichier généré : {nom_fichier}")

if __name__ == "__main__":
    main()
