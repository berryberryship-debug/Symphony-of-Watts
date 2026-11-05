import argparse
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

from core.f0_yin import extraire_f0_yin
from core.harmonisation import resoudre_intervalle_continu
from core.modal_synthesis import synthetiser_spectre_complet
from core.spectral_analyzer import analyser_spectre

FS = 44100
DUREE = 1.0
N = int(FS * DUREE)
T = np.arange(N, dtype=np.float64) / FS


def cents(f_reference, f_mesuree):
    valeurs = np.asarray([f_reference, f_mesuree], dtype=float)
    if not np.all(np.isfinite(valeurs)) or np.any(valeurs <= 0):
        raise ValueError("Fréquences invalides pour le calcul en cents.")
    return float(1200.0 * np.log2(f_mesuree / f_reference))


def verifier_proche(obtenu, attendu, tolerance):
    np.testing.assert_allclose(obtenu, attendu, rtol=0.0, atol=tolerance, equal_nan=False)


def mesurer_pic(signal, fs, f_min, f_max):
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("Signal mono trop court.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal non fini.")
    if not 0 < f_min < f_max < fs / 2:
        raise ValueError("Bande de recherche invalide.")

    x = x - np.mean(x)
    fenetre = np.hanning(x.size)
    n_fft = 4 * x.size

    spectre = np.abs(np.fft.rfft(x * fenetre, n=n_fft))
    frequences = np.fft.rfftfreq(n_fft, 1.0 / fs)

    indices = np.flatnonzero((frequences >= f_min) & (frequences <= f_max))
    if indices.size < 3:
        raise ValueError("Bande de recherche trop étroite.")

    k = int(indices[np.argmax(spectre[indices])])
    if spectre[k] <= 0:
        raise ValueError("Aucun signal mesurable dans cette bande.")
    if k == indices[0] or k == indices[-1]:
        raise ValueError("Maximum sur une limite de recherche.")

    logs = np.log(np.maximum(spectre[k - 1:k + 2], np.finfo(float).tiny))
    gauche, centre, droite = logs
    denom = gauche - 2.0 * centre + droite

    delta = 0.0
    if abs(denom) > 1e-15:
        delta = 0.5 * (gauche - droite) / denom
        delta = float(np.clip(delta, -0.5, 0.5))

    return float((k + delta) * fs / n_fft)


def moteur_test(f0):
    signal = np.zeros(N, dtype=np.float64)
    for harmonique in range(1, 13):
        frequence = harmonique * f0
        if frequence >= FS / 2:
            break
        signal += (0.4 / harmonique**1.2 * np.cos(2.0 * np.pi * frequence * T))
    return signal


class ValidationMoteur(unittest.TestCase):
    def test_01_cosinus_amplitude_phase(self):
        attendu = 0.6 * np.cos(2.0 * np.pi * 445.25 * T + 0.73)
        obtenu = synthetiser_spectre_complet([445.25], [0.6], [0.73], DUREE, FS)
        self.assertEqual(obtenu.shape, (N,))
        self.assertTrue(np.all(np.isfinite(obtenu)))
        verifier_proche(obtenu, attendu, 1e-10)

    def test_02_independance_des_blocs(self):
        arguments = ([445.25, 890.5, 1335.75], [0.4, 0.2, 0.1], [0.2, -0.7, 1.1], DUREE, FS)
        petit_bloc = synthetiser_spectre_complet(*arguments, taille_bloc=257)
        grand_bloc = synthetiser_spectre_complet(*arguments, taille_bloc=N)
        verifier_proche(petit_bloc, grand_bloc, 1e-10)

    def test_03_rejet_composantes_hors_bande(self):
        with warnings.catch_warnings(record=True) as captures:
            warnings.simplefilter("always")
            obtenu = synthetiser_spectre_complet(
                [445.0, 0.0, FS / 2, 23000.0, -100.0],
                [0.3, 0.5, 0.5, 0.5, 0.5],
                [0.0] * 5, DUREE, FS,
            )
        attendu = 0.3 * np.cos(2.0 * np.pi * 445.0 * T)
        verifier_proche(obtenu, attendu, 1e-10)
        self.assertTrue(any(issubclass(m.category, RuntimeWarning) for m in captures))

    def test_04_entrees_invalides_synthese(self):
        cas = [([445.0], [0.3, 0.2], [0.0]), ([np.nan], [0.3], [0.0]), ([445.0], [-0.3], [0.0])]
        for f, a, p in cas:
            with self.subTest(f=f):
                with self.assertRaises(ValueError):
                    synthetiser_spectre_complet(f, a, p, DUREE, FS)

    def test_05_analyse_amplitude_et_phase(self):
        signal = 0.6 * np.cos(2.0 * np.pi * 445.0 * T + 0.73)
        freq, amp, ph = analyser_spectre(signal, FS, n_pics_max=10)
        self.assertGreater(len(freq), 0)
        k = int(np.argmin(np.abs(freq - 445.0)))
        verifier_proche(freq[k], 445.0, 0.05)
        verifier_proche(amp[k], 0.6, 0.002)
        erreur_phase = np.angle(np.exp(1j * (ph[k] - 0.73)))
        self.assertLess(abs(erreur_phase), 0.01)

    def test_06_analyse_puis_resynthese(self):
        attendu = (0.4 * np.cos(2.0 * np.pi * 445.0 * T + 0.2) + 0.2 * np.cos(2.0 * np.pi * 890.0 * T - 0.7))
        freq, amp, ph = analyser_spectre(attendu, FS, n_pics_max=10)
        obtenu = synthetiser_spectre_complet(freq, amp, ph, DUREE, FS)
        erreur = np.linalg.norm(obtenu - attendu) / np.linalg.norm(attendu)
        self.assertLess(erreur, 0.01, f"Erreur : {erreur:.6f}")

    def test_07_yin_signaux_harmoniques(self):
        for f0 in (445.0, 460.0):
            with self.subTest(f0=f0):
                mesure, confiance = extraire_f0_yin(moteur_test(f0), FS, f_min=100.0, f_max=1000.0)
                self.assertTrue(np.isfinite(mesure))
                self.assertLess(abs(cents(f0, mesure)), 1.0)
                self.assertGreater(confiance, 0.8)

    def test_08_yin_silence(self):
        mesure, confiance = extraire_f0_yin(np.zeros(N), FS, f_min=100, f_max=1000)
        self.assertTrue(np.isnan(mesure))
        verifier_proche(confiance, 0.0, 0.0)

    def test_09_solveur_tierce_mineure(self):
        f_a, f_b, cout = resoudre_intervalle_continu(445.0, 460.0)
        verifier_proche(f_a, 445.0, 1e-12)
        verifier_proche(cents(f_a, f_b), 300.0, 1e-9)
        verifier_proche(cout, abs(cents(460.0, f_b)), 1e-9)

    def test_10_transposition_mesuree(self):
        f_a_detecte, _ = extraire_f0_yin(moteur_test(445.0), FS)
        f_b_detecte, _ = extraire_f0_yin(moteur_test(460.0), FS)
        f_a_cible, f_b_cible, _ = resoudre_intervalle_continu(f_a_detecte, f_b_detecte)

        sorties = []
        for signal, ratio in ((moteur_test(445.0), f_a_cible / f_a_detecte), (moteur_test(460.0), f_b_cible / f_b_detecte)):
            freq, amp, ph = analyser_spectre(signal, FS, n_pics_max=30)
            sorties.append(synthetiser_spectre_complet(freq * ratio, amp, ph, DUREE, FS))

        mesure_a = mesurer_pic(sorties[0], FS, 430, 460)
        mesure_b = mesurer_pic(sorties[1], FS, 510, 550)
        intervalle = cents(mesure_a, mesure_b)
        self.assertLess(abs(intervalle - 300.0), 1.0, f"Intervalle : {intervalle:.4f}")


class ValidationModalSupplementaire(unittest.TestCase):
    def test_11_spectre_vide_produit_silence(self):
        obtenu = synthetiser_spectre_complet([], [], [], DUREE, FS)
        self.assertEqual(obtenu.shape, (N,))
        self.assertEqual(obtenu.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(obtenu, np.zeros(N))

    def test_12_amplitudes_nulles(self):
        obtenu = synthetiser_spectre_complet([445.0, 890.0], [0.0, 0.0], [0.3, -1.2], DUREE, FS)
        np.testing.assert_array_equal(obtenu, np.zeros(N))

    def test_13_somme_de_plusieurs_composantes(self):
        freq, amp, ph = [125.3, 445.25, 1100.7], [0.2, 0.4, 0.1], [0.1, -0.8, 2.3]
        attendu = np.zeros(N)
        for f, a, p in zip(freq, amp, ph):
            attendu += a * np.cos(2.0 * np.pi * f * T + p)
        obtenu = synthetiser_spectre_complet(freq, amp, ph, DUREE, FS, taille_bloc=317)
        verifier_proche(obtenu, attendu, 1e-10)

    def test_14_pas_de_normalisation_ni_ecretage(self):
        obtenu = synthetiser_spectre_complet([445.0, 890.0], [0.8, 0.8], [0.0, 0.0], DUREE, FS)
        verifier_proche(obtenu[0], 1.6, 1e-12)
        self.assertGreater(np.max(np.abs(obtenu)), 1.0)

    def test_15_annulation_en_opposition_de_phase(self):
        obtenu = synthetiser_spectre_complet([445.0, 445.0], [0.5, 0.5], [0.0, np.pi], DUREE, FS)
        verifier_proche(obtenu, np.zeros(N), 1e-10)

    def test_16_periodicite_des_phases(self):
        ref = synthetiser_spectre_complet([445.25], [0.6], [0.73], DUREE, FS)
        obt = synthetiser_spectre_complet([445.25], [0.6], [0.73 + 4.0 * np.pi], DUREE, FS)
        verifier_proche(obt, ref, 1e-10)

    def test_17_duree_non_entiere(self):
        duree = 0.12345
        nombre = int(round(FS * duree))
        temps = np.arange(nombre, dtype=np.float64) / FS
        obtenu = synthetiser_spectre_complet([445.0], [0.3], [0.2], duree, FS)
        attendu = 0.3 * np.cos(2.0 * np.pi * 445.0 * temps + 0.2)
        self.assertEqual(len(obtenu), nombre)
        verifier_proche(obtenu, attendu, 1e-10)

    def test_18_signal_un_seul_echantillon(self):
        obtenu = synthetiser_spectre_complet([445.0], [0.4], [0.7], 1.0 / FS, FS)
        self.assertEqual(obtenu.shape, (1,))
        verifier_proche(obtenu[0], 0.4 * np.cos(0.7), 1e-12)

    def test_19_toutes_composantes_hors_bande(self):
        with warnings.catch_warnings(record=True) as captures:
            warnings.simplefilter("always")
            obtenu = synthetiser_spectre_complet([-1.0, 0.0, FS / 2, FS], [0.1, 0.2, 0.3, 0.4], [0.0]*4, DUREE, FS)
        np.testing.assert_array_equal(obtenu, np.zeros(N))
        self.assertEqual(len([m for m in captures if issubclass(m.category, RuntimeWarning)]), 1)

    def test_20_frequence_juste_sous_nyquist(self):
        f = FS / 2 - 0.25
        with warnings.catch_warnings(record=True) as captures:
            warnings.simplefilter("always")
            obtenu = synthetiser_spectre_complet([f], [0.3], [0.4], DUREE, FS)
        attendu = 0.3 * np.cos(2.0 * np.pi * f * T + 0.4)
        verifier_proche(obtenu, attendu, 1e-10)
        self.assertEqual(len(captures), 0)

    def test_21_entrees_non_modifiees(self):
        f, a, p = np.array([445.0, 890.0]), np.array([0.4, 0.2]), np.array([0.3, -0.2])
        copies = [t.copy() for t in (f, a, p)]
        synthetiser_spectre_complet(f, a, p, DUREE, FS)
        for orig, copie in zip((f, a, p), copies):
            np.testing.assert_array_equal(orig, copie)

    def test_22_entrees_multidimensionnelles_refusees(self):
        cas = [([[445.0]], [0.3], [0.0]), ([445.0], [[0.3]], [0.0]), ([445.0], [0.3], [[0.0]])]
        for f, a, p in cas:
            with self.subTest(f=f):
                with self.assertRaises(ValueError):
                    synthetiser_spectre_complet(f, a, p, DUREE, FS)

    def test_23_valeurs_non_finies_refusees(self):
        for val in (np.nan, np.inf, -np.inf):
            for pos in range(3):
                args = [[445.0], [0.3], [0.0]]
                args[pos] = [val]
                with self.subTest(val=val, pos=pos):
                    with self.assertRaises(ValueError):
                        synthetiser_spectre_complet(*args, DUREE, FS)

    def test_24_durees_invalides_refusees(self):
        for duree in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(duree=duree):
                with self.assertRaises(ValueError):
                    synthetiser_spectre_complet([445.0], [0.3], [0.0], duree, FS)

    def test_25_echantillonnages_invalides_refuses(self):
        for fs in (0.0, -44100.0, np.nan, np.inf):
            with self.subTest(fs=fs):
                with self.assertRaises(ValueError):
                    synthetiser_spectre_complet([445.0], [0.3], [0.0], DUREE, fs)

    def test_26_duree_arrondie_a_zero_refusee(self):
        with self.assertRaises(ValueError):
            synthetiser_spectre_complet([445.0], [0.3], [0.0], 0.1 / FS, FS)

    def test_27_blocs_non_positifs_refuses(self):
        for taille in (0, -1):
            with self.subTest(taille=taille):
                with self.assertRaises(ValueError):
                    synthetiser_spectre_complet([445.0], [0.3], [0.0], DUREE, FS, taille_bloc=taille)

    def test_28_blocs_non_entiers_refuses(self):
        for taille in (1.5, np.nan, np.inf):
            with self.subTest(taille=taille):
                with self.assertRaises((ValueError, TypeError)):
                    synthetiser_spectre_complet([445.0], [0.3], [0.0], DUREE, FS, taille_bloc=taille)


def auditer_wav(chemin):
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"WAV introuvable : {chemin}")
    fs, donnees = wavfile.read(chemin)
    if donnees.dtype != np.int16:
        raise ValueError(f"PCM int16 attendu, reçu : {donnees.dtype}")
    if donnees.ndim != 2 or donnees.shape[1] != 2:
        raise ValueError("Un fichier stéréo à deux canaux est attendu.")
    if fs != FS:
        raise ValueError(f"44 100 Hz attendus, reçu : {fs}")
    if donnees.shape[0] != 3 * fs:
        raise ValueError("Une durée de 3 secondes est attendue.")

    audio = donnees.astype(np.float64) / 32767.0
    crete = float(np.max(np.abs(audio)))
    if crete > 0.9801:
        raise AssertionError(f"Crête {crete:.6f} supérieure à la marge d'export attendue.")

    mesures = [mesurer_pic(audio[:, 0], fs, 430.0, 460.0), mesurer_pic(audio[:, 1], fs, 510.0, 550.0)]
    intervalle = cents(mesures[0], mesures[1])
    erreur = intervalle - 300.0

    print("\n--- AUDIT DU WAV EXISTANT ---")
    print(f"Format : {fs} Hz, stéréo, int16")
    print(f"Durée : {len(audio) / fs:.3f} s")
    print(f"Crête normalisée : {crete:.6f}")
    print(f"Pic A dans la bande attendue : {mesures[0]:.6f} Hz")
    print(f"Pic B dans la bande attendue : {mesures[1]:.6f} Hz")
    print(f"Intervalle mesuré : {intervalle:.4f} cents")
    print(f"Écart à 300 cents : {erreur:+.4f} cents")

    for canal, fondamentale in enumerate(mesures):
        x = audio[:, canal]
        spectre = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        frequences = np.fft.rfftfreq(len(x), 1.0 / fs)
        pics, _ = find_peaks(spectre, prominence=float(np.max(spectre)) * 0.01)

        for multiple in (2, 3):
            attendu = multiple * fondamentale
            candidats = pics[np.abs(frequences[pics] - attendu) <= 3.0]
            if candidats.size == 0:
                raise AssertionError(f"Canal {canal + 1} : pas de pic significatif près de l'harmonique {multiple}.")
            k = candidats[np.argmax(spectre[candidats])]
            mesure = mesurer_pic(x, fs, frequences[k] - 0.6, frequences[k] + 0.6)
            ecart = cents(attendu, mesure)
            print(f"Canal {canal + 1}, harmonique {multiple} : {mesure:.4f} Hz, écart {ecart:+.3f} cents")
            if abs(ecart) > 3.0:
                raise AssertionError(f"Harmonique {multiple} hors tolérance.")

    if abs(erreur) > 1.0:
        raise AssertionError("Intervalle final hors tolérance de 1 cent.")
    print("Audit WAV : critères testés satisfaits.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", help="WAV stéréo à auditer après les tests des modules.")
    args = parser.parse_args()

    chargeur = unittest.defaultTestLoader
    suite = unittest.TestSuite([
        chargeur.loadTestsFromTestCase(ValidationMoteur),
        chargeur.loadTestsFromTestCase(ValidationModalSupplementaire),
    ])
    
    resultat = unittest.TextTestRunner(verbosity=2).run(suite)
    succes = resultat.wasSuccessful()

    if args.wav:
        try:
            auditer_wav(args.wav)
        except Exception as erreur:
            print(f"\nÉCHEC AUDIT WAV : {erreur}", file=sys.stderr)
            succes = False

    return 0 if succes else 1

if __name__ == "__main__":
    sys.exit(main())
