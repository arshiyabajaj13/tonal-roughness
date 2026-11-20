# Tonal roughness across six music genres

Code and data for the paper **"Quantifying Tonal Roughness Across Six Music Genres Using a Spectral Partial-Pair Roughness Model"** by Arshiya Bajaj.

The study computes a Vassilakis-based roughness score for 72 recordings (12 each of Classical, Jazz, Pop, Rap, Rock, and EDM), both from each recording's average spectrum and frame by frame across its Short-Time Fourier Transform, and compares average roughness and within-clip roughness fluctuation across genres. The analysis is written in Python (librosa, NumPy, SciPy, statsmodels, pandas, and matplotlib).

## Contents

| Path | What it is |
|---|---|
| `roughness_analysis.py` | The full analysis: roughness, statistics, and figures |
| `download_previews.py` | Downloads the 72 iTunes preview clips used in the study |
| `requirements.txt` | Package versions used for the paper |
| `data/song_matches.csv` | The 72 recordings: genre, title, artist, performer, album, release year, version, and iTunes track ID |
| `results/per_recording_results.csv` | One row per recording: roughness, log10 roughness, within-clip fluctuation, frame counts, and the candidate predictors, for the main setting and the four sensitivity settings |
| `results/stats.json` | Every statistic reported in the paper and its Supplementary Material |
| `results/frame_series.json` | Frame-by-frame log10(roughness) for each recording (main setting) |
| `results/figures/` | Figures 1–4 and Supplementary Figures S1–S2 |

## How to reproduce the results

Requires Python 3.12. Reading the `.m4a` preview clips works out of the box on macOS; on Windows and Linux, install [ffmpeg](https://ffmpeg.org/download.html) first, which librosa uses (through audioread) to decode them.

```bash
pip install -r requirements.txt
python download_previews.py
python roughness_analysis.py data results
```

`download_previews.py` looks up each iTunes track ID and saves the official 30-second preview to `data/previews/`. `roughness_analysis.py` then writes the files in `results/`. The analysis of all 72 recordings under five settings takes several minutes. To recompute only the statistics and figures from an existing `results/per_recording_results.csv`, run `python roughness_analysis.py data results --reuse`.

## Method in brief

For each recording:

1. Audio is loaded at its native sampling rate with `librosa.load(path, sr=None)`.
2. A Short-Time Fourier Transform is computed with `librosa.stft(y, hop_length=int(sr * 0.05))` (50-millisecond hop).
3. **Average roughness:** peaks in the time-averaged magnitude spectrum are found with `scipy.signal.find_peaks` at a minimum height of 5% of the maximum, and the ten highest are kept. Roughness is summed over the 45 pairs of these partials, with
   x = f_beat / (0.021 · f_avg + 19) and
   r = (a_min · a_max)^0.1 · (2 a_min / (a_min + a_max))^3.11 · (e^(−3.5x) − e^(−5.75x)).
4. **Within-clip fluctuation:** the same calculation is applied to every STFT frame that has at least two qualifying peaks; fluctuation is the standard deviation of log10(roughness) across those frames.

Genres are compared with one-way ANOVA, Kruskal-Wallis tests, and Tukey HSD comparisons (`scipy.stats`, `statsmodels`). The sensitivity analyses repeat everything with peak thresholds of 1% and 10% and with 5 and 20 partials. At the 10% threshold, one recording ("Bad Romance") has only one qualifying peak in its average spectrum, so its average roughness is undefined and it is left out of that setting's average-roughness tests (n = 71).

## About the audio

The preview clips are copyrighted and are **not** included in this repository. They are retrieved directly from Apple's iTunes Search API using the track IDs in `data/song_matches.csv`. Apple may change or withdraw a preview over time; `download_previews.py` reports any track it can no longer find.

## License

The code is released under the MIT License (see `LICENSE`). The audio clips belong to their respective rights holders.
