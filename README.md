# Symphony of Watts

Moteur généralisé de traduction vibro-acoustique → structures musicales.

Le projet transforme des phénomènes vibratoires et acoustiques (moteurs, structures mécaniques, phonons, etc.) en matière musicale, sans confondre les domaines physiques d’origine.

## Architecture

```text
Symphony-of-Watts/
│
├── core/
│   ├── state_tensor.py          # Tenseur d'état générique
│   ├── doppler.py
│   ├── f0_yin.py                # Extraction F0 (YIN + confiance)
│   ├── coupling_matrix.py       # Kuramoto / Dissipatif / Nul
│   ├── harmonic_resolution.py   # Grille pentatonique + allocation exclusive
│   └── synthesis.py             # Synthèse additive stéréo + évolution de phase
│
├── audits/
│   ├── audit_v1_4_2.py
│   ├── audit_coupling.py
│   └── audit_v1_6.py
│
├── examples/
└── README.md
