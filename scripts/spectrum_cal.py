"""Usage: python spectrum_cal.py MANIFEST.tsv OUT_DIR

Radially averaged power spectra of ink maps by group, to find the spatial scales that separate
known text from maps judged to be noise. Numbers only."""
import csv, math, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from stroke_metrics import load_q, load_valid

man, outdir = sys.argv[1], sys.argv[2]          # manifest (see stroke_metrics.py) and output dir
bins = np.geomspace(8, 512, 13)          # wavelength edges in px
rows = []
for line in open(man):
    name, label, vol, ink, win = line.rstrip("\n").split("\t")
    q = load_q(ink); valid = load_valid(vol, q.shape); win = int(win) or q.shape[0]
    step = win // 2 if win < q.shape[0] else win
    for y in range(0, q.shape[0] - win + 1, step):
        for x in range(0, q.shape[1] - win + 1, step):
            a, m = q[y:y + win, x:x + win], valid[y:y + win, x:x + win]
            if m.mean() < 0.5:
                continue
            a = np.where(m, a - a[m].mean(), 0.0)
            h = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
            P = np.abs(np.fft.fft2(a * h)) ** 2
            fy = np.fft.fftfreq(a.shape[0])[:, None]; fx = np.fft.fftfreq(a.shape[1])[None, :]
            wl = 1 / np.maximum(np.sqrt(fy ** 2 + fx ** 2), 1e-9)
            prof = [P[(wl >= lo) & (wl < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
            prof = np.array(prof); prof /= prof.sum()
            rows.append((label, name, prof))
labels = ["known_text", "blank", "0846B_reviewed_noise", "0125", "0846B"]
print("wavelength_px  " + "  ".join(f"{l[:12]:>12s}" for l in labels))
for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
    cells = []
    for l in labels:
        v = [r[2][i] for r in rows if r[0] == l]
        cells.append(f"{np.median(v):12.4f}" if v else " " * 12)
    print(f"{lo:5.0f}-{hi:4.0f}     " + "  ".join(cells))
np.save(f"{outdir}/spectra.npy", np.array([r[2] for r in rows]))
with open(f"{outdir}/spectra_index.csv", "w") as fh:
    for r in rows:
        fh.write(f"{r[0]},{r[1]}\n")
