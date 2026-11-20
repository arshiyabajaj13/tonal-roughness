"""
Roughness analysis for the revised paper, in Python, following the Methods step by step.

    Quantifying Tonal Roughness Across Six Music Genres Using a Spectral Partial-Pair Roughness Model

Packages (the versions named in the Methods):
    numpy 2.2.6, scipy 1.16.1, pandas 3.0.3, statsmodels 0.14.6, matplotlib 3.11.0, librosa 0.11.0

Usage:
    python download_previews.py                    # first: fetch the 72 preview clips into data/previews/
    python roughness_analysis.py data results      # then: run the analysis
        data    = folder with song_matches.csv and previews/*.m4a
        results = where results and figures are written

Outputs (in OUT_DIR):
    per_recording_results.csv   one row per recording (main setting)
    frame_series.json           frame-by-frame log10(roughness) per recording (main setting)
    stats.json                  every statistic reported in the paper and supplement
    figures/*.png               Figures 1-4 and S1-S2
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import scipy
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import brentq
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # librosa's audioread deprecation notice

GENRES = ["Classical", "Jazz", "Pop", "Rap", "Rock", "EDM"]
COLORS = {"Classical": "#2a78d6", "Jazz": "#eb6834", "Pop": "#1baf7a", "Rap": "#eda100", "Rock": "#e87ba4", "EDM": "#008300"}

# main setting + the four sensitivity settings (peak threshold as a fraction of the maximum, number of partials)
SETTINGS = {"main": (0.05, 10), "thr01": (0.01, 10), "thr10": (0.10, 10), "part05": (0.05, 5), "part20": (0.05, 20)}


# --------------------------------------------------------------------------- roughness
def pair_roughness(f1, f2, a1, a2):
    """The paper's pairwise formula: x = f_beat / (0.021 f_avg + 19)."""
    f_avg = (f1 + f2) / 2.0
    f_beat = abs(f1 - f2)
    x = f_beat / (0.021 * f_avg + 19)
    a_min, a_max = min(a1, a2), max(a1, a2)
    return (a_min * a_max) ** 0.1 * (2 * a_min / (a_min + a_max)) ** 3.11 * (np.exp(-3.5 * x) - np.exp(-5.75 * x))


def top_partials(spectrum, freqs, frac, n):
    """scipy.signal.find_peaks with height = frac * max, keep the n highest."""
    mx = spectrum.max()
    if mx <= 0:
        return np.array([]), np.array([]), 0
    peaks, _ = find_peaks(spectrum, height=mx * frac)
    order = peaks[np.argsort(spectrum[peaks])[::-1][:n]]
    return freqs[order], spectrum[order], len(peaks)


def total_roughness(f, a):
    r = 0.0
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            r += pair_roughness(f[i], f[j], a[i], a[j])
    return r


def analyse_recording(path):
    y, sr = librosa.load(path, sr=None)                     # native sample rate, mono
    hop = int(sr * 0.05)
    S = np.abs(librosa.stft(y, hop_length=hop))             # n_fft 2048, Hann, centred
    freqs = librosa.fft_frequencies(sr=sr)
    avg = S.mean(axis=1)
    T = S.shape[1]

    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    den = S.sum(axis=0)
    cent = (freqs[:, None] * S).sum(axis=0)[den > 0] / den[den > 0]
    tempo = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])
    rec = dict(sr=int(sr), total_frames=int(T), rms=rms, peak=peak, crest=peak / rms,
               centroid=float(cent.mean()), tempo=tempo, settings={})

    for name, (frac, n) in SETTINGS.items():
        f, a, count = top_partials(avg, freqs, frac, n)
        R = total_roughness(f, a)
        beats = [abs(f[i] - f[j]) for i in range(len(f)) for j in range(i + 1, len(f))]
        series = []
        for t in range(T):
            ft, at, _ = top_partials(S[:, t], freqs, frac, n)
            if len(ft) < 2:
                series.append(None)
                continue
            r = total_roughness(ft, at)
            series.append(float(np.log10(r)) if r > 0 else None)
        valid = np.array([v for v in series if v is not None])
        tt = np.array([t / (T - 1) for t, v in enumerate(series) if v is not None])
        rec["settings"][name] = dict(
            roughness=float(R), log10_roughness=float(np.log10(R)), n_partials_detected=int(count),
            partial_spread=float(np.std(f)), mean_beat=float(np.mean(beats)) if beats else 0.0,
            fluct_sd=float(np.std(valid, ddof=1)), trend_slope=float(stats.linregress(tt, valid).slope),
            valid_frames=int(len(valid)), series=series if name == "main" else None)
    return rec


# --------------------------------------------------------------------------- statistics
def eta2_ci(F, d1, d2, N, level=0.95):
    """95% CI for eta-squared by inverting the noncentral F distribution."""
    def cdf(L):  # noncentral F cdf; the central F at L = 0 (scipy's ncf returns NaN there)
        return stats.f.cdf(F, d1, d2) if L <= 0 else stats.ncf.cdf(F, d1, d2, L)

    def bound(q):
        if cdf(0) < q:
            return 0.0
        lam = brentq(lambda L: cdf(L) - q, 0, 1000)
        return lam / (lam + N)
    a = (1 - level) / 2
    return [bound(1 - a), bound(a)]


def oneway(groups):
    allv = np.concatenate(groups)
    F, p = stats.f_oneway(*groups)
    ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    sst = ((allv - allv.mean()) ** 2).sum()
    d1, d2 = len(groups) - 1, len(allv) - len(groups)
    return dict(F=float(F), d1=d1, d2=d2, p=float(p), eta2=float(ssb / sst), eta2_ci=eta2_ci(F, d1, d2, len(allv)))


def kruskal(groups):
    H, p = stats.kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    return dict(H=float(H), df=k - 1, p=float(p), eps2=float((H - k + 1) / (n - k)))


def tukey(values, labels):
    res = pairwise_tukeyhsd(values, labels, alpha=0.05)
    tab = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return [dict(g1=r.group1, g2=r.group2, diff=float(r.meandiff), lo=float(r.lower), hi=float(r.upper), p_adj=float(r["p-adj"]))
            for _, r in tab.iterrows()]


def boot_geo(logv, R=10000, seed=42):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(logv, size=len(logv), replace=True).mean() for _ in range(R)])
    return [float(10 ** np.percentile(means, 2.5)), float(10 ** np.percentile(means, 97.5))]


def lin(x, y):
    r = stats.linregress(x, y)
    return dict(slope=float(r.slope), r=float(r.rvalue), r2=float(r.rvalue ** 2), p=float(r.pvalue))


def ancova(df, y, covar):
    """Genre effect controlling for a covariate (type II sums of squares)."""
    m = smf.ols(f"{y} ~ C(genre) + {covar}", data=df).fit()
    a = anova_lm(m, typ=2)
    return dict(F_genre=float(a.loc["C(genre)", "F"]), p_genre=float(a.loc["C(genre)", "PR(>F)"]),
                F_cov=float(a.loc[covar, "F"]), p_cov=float(a.loc[covar, "PR(>F)"]))


def analyse_setting(df, s):
    # A recording whose average spectrum has fewer than two qualifying peaks has no partial pairs, so its
    # average roughness is undefined (log10 of 0); it is left out of the average-roughness tests for that setting.
    ok = np.isfinite(df[f"{s}_log10_roughness"].replace(-np.inf, np.nan))
    dr = df[ok]
    lr = dr[f"{s}_log10_roughness"].to_numpy()
    fl = df[f"{s}_fluct_sd"].to_numpy()
    gl = [dr.loc[dr.genre == g, f"{s}_log10_roughness"].to_numpy() for g in GENRES]
    gf = [df.loc[df.genre == g, f"{s}_fluct_sd"].to_numpy() for g in GENRES]
    out = dict(
        genre=[dict(genre=g, geo_mean=float(10 ** gl[i].mean()), geo_ci=boot_geo(gl[i]), mean_log=float(gl[i].mean()),
                    fluct_mean=float(gf[i].mean()), fluct_median=float(np.median(gf[i]))) for i, g in enumerate(GENRES)],
        shapiro=dict(zip(["W", "p"], map(float, stats.shapiro(lr)))),
        levene=dict(zip(["W", "p"], map(float, stats.levene(*gl)))),
        anova_rough=oneway(gl), kruskal_rough=kruskal(gl), tukey_rough=tukey(lr, dr.genre.to_numpy()),
        n_rough=int(len(dr)), excluded_rough=dr.index.symmetric_difference(df.index).map(lambda i: df.loc[i, "title"]).tolist(),
        anova_fluct=oneway(gf), kruskal_fluct=kruskal(gf), tukey_fluct=tukey(fl, df.genre.to_numpy()))
    return out


def power_table(N=72, k=6):
    d1, d2 = k - 1, N - k
    def power(e, a):
        fc = stats.f.ppf(1 - a, d1, d2)
        return 1 - stats.ncf.cdf(fc, d1, d2, e / (1 - e) * N)
    mde = lambda a: brentq(lambda e: power(e, a) - 0.8, 1e-4, 0.9)
    return dict(mde_05_80=mde(0.05), mde_025_80=mde(0.025), power_medium_06=power(0.06, 0.05), power_large_14=power(0.14, 0.05))


# --------------------------------------------------------------------------- figures (matplotlib)
def jitter(n):
    return ((np.arange(n) * 7919 % 12) / 12 - 0.5) * 0.32


def strip(ax, df, col, order, center, log=False):
    for i, g in enumerate(order):
        v = df.loc[df.genre == g, col].to_numpy()
        ax.scatter(i + jitter(len(v)), v, s=55, color=COLORS[g], edgecolor="white", linewidth=1, zorder=3)
        c = center(v)
        ax.plot([i - 0.27, i + 0.27], [c, c], color="black", lw=2.5, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color="#e3e3e3", zorder=0)


def make_figures(df, st, frames, figdir):
    figdir.mkdir(parents=True, exist_ok=True)
    m = st["main"]
    # Figure 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 6.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    df["main_roughness"] = 10 ** df["main_log10_roughness"]
    strip(a1, df, "main_roughness", GENRES, lambda v: 10 ** np.log10(v).mean(), log=True)
    a1.set_title("A. Individual songs by genre (iTunes previews only)", fontweight="bold")
    a1.set_ylabel("Roughness (dimensionless, log scale)")
    ranked = sorted(m["genre"], key=lambda g: -g["geo_mean"])
    ys = np.arange(len(ranked))[::-1]
    for yy, g in zip(ys, ranked):
        a2.barh(yy, g["geo_mean"], color=COLORS[g["genre"]], height=0.8, zorder=2)
        a2.errorbar(g["geo_mean"], yy, xerr=[[g["geo_mean"] - g["geo_ci"][0]], [g["geo_ci"][1] - g["geo_mean"]]],
                    color="#555555", capsize=6, lw=1.5, zorder=3)
    a2.set_xscale("log")
    a2.set_yticks(ys)
    a2.set_yticklabels([g["genre"] for g in ranked])
    a2.set_xlabel("Geometric mean roughness (log scale)")
    a2.set_title("B. Genre means with 95% CI (n=12/genre)", fontweight="bold")
    a2.grid(axis="x", color="#e3e3e3", zorder=0)
    fig.suptitle("Spectral roughness of 72 songs across six genres (one consistent acquisition method)", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(figdir / "Figure1_average_roughness.png", dpi=200)
    plt.close(fig)

    # Figure 2: Tukey HSD (statsmodels labels "A - B", value = mean(B) - mean(A))
    rows = sorted(m["tukey_rough"], key=lambda r: r["diff"])
    fig, ax = plt.subplots(figsize=(8.64, 5.42))
    for i, r in enumerate(rows):
        c = "#d62728" if r["p_adj"] < 0.05 else "#8c8c8c"
        lc = "#d62728" if r["p_adj"] < 0.05 else "#c8c8c0"
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="o", color=c, ecolor=lc, capsize=4, ms=9)
    ax.axvline(0, color="#c8c8c0", lw=1.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['g1']} - {r['g2']}" for r in rows])
    ax.set_xlabel("Difference in mean log10(roughness), Tukey HSD 95% CI")
    ax.set_title("Pairwise genre comparisons, n=72, single method (red = significant at α=0.05)", fontweight="bold", fontsize=11)
    ax.grid(axis="x", color="#e3e3e3")
    fig.tight_layout()
    fig.savefig(figdir / "Figure2_tukey_average_roughness.png", dpi=200)
    plt.close(fig)

    # Figure 3
    order = sorted(GENRES, key=lambda g: -df.loc[df.genre == g, "main_fluct_sd"].mean())
    fig, ax = plt.subplots(figsize=(8.8, 6.42))
    strip(ax, df, "main_fluct_sd", order, np.mean)
    a = m["anova_fluct"]
    ax.set_title(f"Within-clip roughness fluctuation by genre (F(5,66)={a['F']:.2f}, p={a['p']:.4f})", fontweight="bold")
    ax.set_ylabel("Within-clip roughness variability\n(SD of log10(roughness) across STFT frames)")
    fig.tight_layout()
    fig.savefig(figdir / "Figure3_fluctuation_by_genre.png", dpi=200)
    plt.close(fig)

    # Figure 4: the same six example recordings as the original
    ex = [("Classical", "Spring Four Seasons", "Spring (Four Seasons)"), ("Jazz", "So What", "So What"),
          ("Pop", "Rolling in the Deep", "Rolling in the Deep"), ("Rap", "Juicy", "Juicy"),
          ("Rock", "Stairway to Heaven", "Stairway to Heaven"), ("EDM", "Scary Monsters and Nice Sprites", "Scary Monsters and Nice Sprites")]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.41), sharex=True)
    for ax, (g, title, lab) in zip(axes.flat, ex):
        no = df.loc[df.title == title, "no"].iloc[0]
        s = np.array([np.nan if v is None else v for v in frames[no]])
        ax.plot(np.linspace(0, 1, len(s)), s, color=COLORS[g], lw=1)
        ax.set_title(f"{g}: {lab}", loc="left", fontweight="bold", fontsize=10)
        ax.grid(color="#e3e3e3")
    for ax in axes[1]:
        ax.set_xlabel("Normalized clip time (0-1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("log10(roughness)")
    fig.suptitle("Example within-clip roughness curves, one recording per genre", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figdir / "Figure4_example_roughness_curves.png", dpi=200)
    plt.close(fig)

    # Figure S1: excluded frames
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    strip(ax, df, "excluded_pct", GENRES, np.mean)
    ax.set_ylabel("Frames excluded (%)")
    ax.set_title(f"Excluded frames by genre (Kruskal–Wallis p={st['exclusion']['kruskal']['p']:.3f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figdir / "FigureS1_excluded_frames.png", dpi=200)
    plt.close(fig)

    # Figure S2: release year
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    strip(ax, df, "release_year", GENRES, np.median)
    ax.set_ylabel("Release year of recording")
    ax.set_title("Release year by genre (bar = median; Kruskal–Wallis p<0.0001)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figdir / "FigureS2_release_year.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main(prev_dir, out_dir):
    prev_dir, out_dir = Path(prev_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(prev_dir / "song_matches.csv", dtype=str)
    rows, frames = [], {}
    cache = out_dir / "per_recording_results.csv"
    if cache.exists() and (out_dir / "frame_series.json").exists() and "--reuse" in sys.argv:
        print("reusing", cache)
        meta = meta.iloc[0:0]
        rows = pd.read_csv(cache, dtype={"no": str, "itunes_track_id": str}).to_dict("records")
        frames = json.loads((out_dir / "frame_series.json").read_text())
    for _, m in meta.iterrows():
        rec = analyse_recording(str(prev_dir / m["file"]))
        row = dict(no=m["no"], genre=m["genre"], title=m["title"], artist=m["artist"], itunes_track_id=m["trackId"],
                   release_year=int(m["release_year"]), track_duration_s=float(m["track_duration_s"]),
                   **{k: rec[k] for k in ["sr", "total_frames", "rms", "peak", "crest", "centroid", "tempo"]})
        for s, v in rec["settings"].items():
            for k, val in v.items():
                if k != "series":
                    row[f"{s}_{k}"] = val
        row["excluded_pct"] = 100 * (1 - row["main_valid_frames"] / row["total_frames"])
        rows.append(row)
        frames[m["no"]] = rec["settings"]["main"]["series"]
        print(f"{m['no']} {m['genre']:9s} {m['title'][:28]:28s} roughness={row['main_roughness']:.3e}  sd={row['main_fluct_sd']:.3f}  frames={row['main_valid_frames']}/{row['total_frames']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)                       # saved before the statistics
    (out_dir / "frame_series.json").write_text(json.dumps(frames))

    st = {s: analyse_setting(df, s) for s in SETTINGS}
    pred = ["rms", "peak", "crest", "centroid", "main_partial_spread", "main_mean_beat", "track_duration_s", "tempo", "main_n_partials_detected", "release_year"]
    st["predictors"] = {p: lin(df[p], df["main_log10_roughness"]) for p in pred}
    st["trend"] = dict(zip(["t", "p"], map(float, stats.ttest_1samp(df["main_trend_slope"], 0)))) | dict(df=len(df) - 1, mean_slope=float(df["main_trend_slope"].mean()))
    st["frames"] = dict(min=int(df.main_valid_frames.min()), max=int(df.main_valid_frames.max()), median=float(df.main_valid_frames.median()))
    df["rank_fluct"] = stats.rankdata(df["main_fluct_sd"])
    zero = df[df.excluded_pct == 0]
    st["exclusion"] = dict(
        overall_mean=float(df.excluded_pct.mean()),
        by_genre={g: dict(mean=float(d.excluded_pct.mean()), min=float(d.excluded_pct.min()), max=float(d.excluded_pct.max())) for g, d in df.groupby("genre")},
        kruskal=kruskal([df.loc[df.genre == g, "excluded_pct"].to_numpy() for g in GENRES]),
        corr_with_fluct=lin(df.excluded_pct, df.main_fluct_sd),
        spearman=dict(zip(["rho", "p"], map(float, stats.spearmanr(df.excluded_pct, df.main_fluct_sd)))),
        n_with_exclusions=int((df.excluded_pct > 0).sum()),
        ancova=ancova(df, "main_fluct_sd", "excluded_pct"), rank_ancova=ancova(df, "rank_fluct", "excluded_pct"),
        kruskal_zero_excluded=kruskal([zero.loc[zero.genre == g, "main_fluct_sd"].to_numpy() for g in GENRES]),
        n_zero_excluded=int(len(zero)))
    st["era"] = dict(
        by_genre={g: dict(min=int(d.release_year.min()), max=int(d.release_year.max()), median=float(d.release_year.median())) for g, d in df.groupby("genre")},
        kruskal_year_by_genre=kruskal([df.loc[df.genre == g, "release_year"].to_numpy() for g in GENRES]),
        year_vs_fluct=lin(df.release_year, df.main_fluct_sd), year_vs_rough=lin(df.release_year, df.main_log10_roughness),
        ancova_fluct=ancova(df, "main_fluct_sd", "release_year"), rank_ancova_fluct=ancova(df, "rank_fluct", "release_year"))
    st["power"] = power_table()
    st["versions"] = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                          statsmodels=statsmodels.__version__, matplotlib=matplotlib.__version__, librosa=librosa.__version__)

    (out_dir / "stats.json").write_text(json.dumps(st, indent=1))
    make_figures(df, st, frames, out_dir / "figures")
    print("done:", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
