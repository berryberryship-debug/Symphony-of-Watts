import numpy as np
from scipy.io import wavfile
from core.f0_yin import extraire_f0_yin
from core.harmonisation import resoudre_intervalle_continu
from core.spectral_analyzer import analyser_spectre
from core.modal_synthesis import synthetiser_spectre_complet

def generer_moteur_ultra_riche(f_base, duree, fs):
    t = np.linspace(0, duree, int(fs * duree), endpoint=False)
    signal = np.zeros_like(t)
    
    for i in range(1, 51):
        amp = 1.0 / (i**1.2)
        signal += amp * np.sin(2 * np.pi * f_base * i * t)
        
    bruit = np.cumsum(np.random.normal(0, 0.05, len(t)))
    bruit = bruit / np.max(np.abs(bruit)) * 0.2
    
    signal_final = signal + bruit
    return signal_final / np.max(np.abs(signal_final))

def main():
    print("🚗 Initialisation du Moteur Spectral MAXIMAL...")
    fs = 44100
    duree = 3.0
    
    f_A_brut = 445.0 
    f_B_brut = 460.0 
    
    sig_A_brut = generer_moteur_ultra_riche(f_A_brut, duree, fs)
    sig_B_brut = generer_moteur_ultra_riche(f_B_brut, duree, fs)
    
    f_A_detecte = extraire_f0_yin(sig_A_brut, fs, f_min=100, f_max=1000)[0]
    f_B_detecte = extraire_f0_yin(sig_B_brut, fs, f_min=100, f_max=1000)[0]
    
    f_A_cible, f_B_cible, cout = resoudre_intervalle_continu(f_A_detecte, f_B_detecte)
    ratio_B = f_B_cible / f_B_detecte
    
    print(f"   -> Voiture A reste à {f_A_cible:.2f} Hz")
    print(f"   -> Voiture B transposée à {f_B_cible:.2f} Hz (Coût : {cout:.2f} cents)")
    
    print("🔍 Analyse FFT des deux moteurs...")
    freqs_A, amps_A, phases_A = analyser_spectre(sig_A_brut, fs, n_pics_max=100)
    freqs_B, amps_B, phases_B = analyser_spectre(sig_B_brut, fs, n_pics_max=100)
    
    freqs_B_transposees = freqs_B * ratio_B
    
    print("🎛️  Synthèse additive de 100 composantes spectrales par canal...")
    sig_A_final = synthetiser_spectre_complet(freqs_A, amps_A, phases_A, duree, fs)
    sig_B_final = synthetiser_spectre_complet(freqs_B_transposees, amps_B, phases_B, duree, fs)
    
    # 7. ASSEMBLAGE STÉRÉO ET SAUVEGARDE
    signal_stereo = np.stack([sig_A_final, sig_B_final], axis=1)
    
    # PROTECTION ANTI-CLIPPING (Ce qui manquait sur ta capture)
    crete = np.max(np.abs(signal_stereo))
    if crete > 0.98:
        signal_stereo = signal_stereo * (0.98 / crete)
        
    # Vérification finale
    if not np.all(np.isfinite(signal_stereo)):
        raise RuntimeError("Le signal de sortie contient des valeurs invalides (NaN ou Inf).")
        
    pcm = np.rint(signal_stereo * 32767.0).astype(np.int16)
    
    nom_fichier = "symphony_maximal.wav"
    wavfile.write(nom_fichier, fs, pcm)
    print(f"\n✅ Fichier généré : {nom_fichier}")

if __name__ == "__main__":
    main()
