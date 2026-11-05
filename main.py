import numpy as np
from scipy.io import wavfile
from core.f0_yin import extraire_f0_yin
from core.harmonisation import resoudre_intervalle_continu
from core.pitch_shifter import transposer_audio_brut

def generer_moteur_brut(f_base, duree, fs):
    t = np.linspace(0, duree, int(fs * duree), endpoint=False)
    signal = (0.6 * np.sin(2 * np.pi * f_base * t) 
              + 0.3 * np.sin(2 * np.pi * f_base * 2 * t)
              + 0.1 * np.sin(2 * np.pi * f_base * 3 * t)
              + 0.2 * np.random.normal(0, 1, len(t)))
    return signal / np.max(np.abs(signal))

def main():
    print("🚗 Initialisation de Symphony of Watts...")
    fs = 44100
    duree = 3.0
    f_A_brut = 445.0 
    f_B_brut = 460.0 
    
    sig_A = generer_moteur_brut(f_A_brut, duree, fs)
    sig_B = generer_moteur_brut(f_B_brut, duree, fs)
    
    # 1. Extraction
    res_A = extraire_f0_yin(sig_A, fs, f_min=100, f_max=1000)
    res_B = extraire_f0_yin(sig_B, fs, f_min=100, f_max=1000)
    
    if not res_A or not res_B:
        print("❌ Échec de l'extraction YIN.")
        return
        
    f_A_detecte = res_A[0]
    f_B_detecte = res_B[0]
    
    # 2. Harmonisation
    print("\n🧠 Résolution harmonique continue :")
    f_A_cible, f_B_cible, cout = resoudre_intervalle_continu(f_A_detecte, f_B_detecte)
    
    print(f"   -> A reste à {f_A_cible:.2f} Hz (Coût : 0 cents)")
    print(f"   -> B transposée à {f_B_cible:.2f} Hz (Coût : {cout:.2f} cents)")
    
    # 3. Transposition
    ratio_B = f_B_cible / f_B_detecte
    sig_B_accorde = transposer_audio_brut(sig_B, ratio_B, fs)
    
    # 4. Synthèse
    signal_stereo = np.stack([sig_A, sig_B_accorde], axis=1)
    
    max_val = np.max(np.abs(signal_stereo))
    if max_val > 0:
        signal_stereo = signal_stereo / max_val * 0.9
        
    nom_fichier = "symphony_continu.wav"
    wavfile.write(nom_fichier, fs, np.int16(signal_stereo * 32767))
    print(f"\n✅ Fichier généré : {nom_fichier}")

if __name__ == "__main__":
    main()
