"""Image-free CT statistics per scroll volume, to rank scrolls by similarity to a reference
scroll on which an ink model is known to work (PHerc0211 for ink_9um).

For each volume (read straight from the public S3 bucket):
  1. Read the coarsest pyramid level and keep level-0 chunks that lie fully inside the masked
     scroll and in the middle 60% of its height.
  2. Sample N such 128^3 level-0 chunks (fixed RNG seed).
  3. Per chunk, on non-zero voxels: Otsu threshold into papyrus (fg) and gaps (bg); record
     overall mean and SD, fg mean, bg mean, fg fraction, and the dominant sheet period from the
     3D power spectrum (wavelength 5-60 voxels, in voxels and micrometres) with its peak strength.
  4. Report medians over chunks, plus acquisition facts from the catalog.
Numbers only; no image is written.

Usage: python ct_fit_stats.py CATALOG.json OUT.csv SAMPLE [SAMPLE ...] [--n 32]
"""
import argparse, csv, json, os, sys
import concurrent.futures as cf
import numpy as np
import zarr

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"


def volumes(cat, sample):
    for vid, v in cat["samples"][sample]["volumes"].items():
        p = v["properties"]
        path = v["data"][0]["origins"][0]["path"]
        meta = v["creation"].get("metadata") or {}
        yield dict(sample=sample, volume=vid, path=path, um=p.get("pixel_size_um"), kev=p.get("energy_keV"),
                   win_lo=meta.get("target_window_f32_min"), win_hi=meta.get("target_window_f32_max"))


def otsu(x):
    h = np.bincount(x, minlength=256).astype(np.float64); h[0] = 0
    p = h / h.sum(); w = np.cumsum(p); m = np.cumsum(p * np.arange(256)); mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mt * w - m) ** 2 / (w * (1 - w))
    return int(np.nanargmax(s))


def period(c):
    v = c.astype(np.float32); v -= v.mean()
    w = np.hanning(c.shape[0]).astype(np.float32)
    v *= w[:, None, None] * w[None, :, None] * w[None, None, :]
    P = np.abs(np.fft.fftn(v)) ** 2
    f = [np.fft.fftfreq(n) for n in c.shape]
    r = np.sqrt(f[0][:, None, None] ** 2 + f[1][None, :, None] ** 2 + f[2][None, None, :] ** 2)
    bins = np.linspace(1 / 60, 1 / 5, 56)
    idx = np.digitize(r.ravel(), bins)
    prof = np.bincount(idx, weights=P.ravel(), minlength=len(bins) + 1)[1:len(bins)]
    cnt = np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)]
    prof = prof / np.maximum(cnt, 1)
    # Remove the 1/f background with a power-law fit, then find the peak.
    fc = 0.5 * (bins[1:] + bins[:-1])
    coef = np.polyfit(np.log(fc), np.log(prof + 1e-9), 1)
    resid = np.log(prof + 1e-9) - np.polyval(coef, np.log(fc))
    k = int(np.argmax(resid))
    return 1 / fc[k], float(resid[k])


def chunk_stats(arr, z, y, x):
    c = np.asarray(arr[z:z + 128, y:y + 128, x:x + 128])
    nz = c[c > 0]
    if nz.size < 0.9 * c.size:
        return None
    t = otsu(nz)
    fg, bg = nz[nz > t], nz[nz <= t]
    per, strength = period(c)
    return dict(mean=float(nz.mean()), sd=float(nz.std()), fg=float(fg.mean()) if fg.size else np.nan,
                bg=float(bg.mean()) if bg.size else np.nan, fg_frac=fg.size / nz.size, period_vox=per,
                period_strength=strength)


def run(v, n, seed=0):
    g = zarr.open_group(S3 + v["path"], mode="r")
    # HTTP stores cannot list keys, so take the levels from the OME-Zarr multiscales metadata.
    levels = sorted((d["path"] for d in g.attrs["multiscales"][0]["datasets"]), key=int)
    a0, lc = g["0"], g[levels[-1]]
    f = 2 ** int(levels[-1])
    m = np.asarray(lc[:]) > 0
    zs = np.where(m.any((1, 2)))[0]
    z0, z1 = zs[0] + 0.2 * (zs[-1] - zs[0]), zs[0] + 0.8 * (zs[-1] - zs[0])
    step = max(128 // f, 1)
    cand = []
    for cz in range(0, a0.shape[0] // 128):
        if not (z0 <= cz * step <= z1):
            continue
        for cy in range(0, a0.shape[1] // 128):
            for cx in range(0, a0.shape[2] // 128):
                b = m[cz * step:(cz + 1) * step, cy * step:(cy + 1) * step, cx * step:(cx + 1) * step]
                if b.size and b.all():
                    cand.append((cz * 128, cy * 128, cx * 128))
    rng = np.random.default_rng(seed)
    pick = [cand[i] for i in rng.choice(len(cand), size=min(n * 2, len(cand)), replace=False)] if cand else []
    stats = []
    with cf.ThreadPoolExecutor(8) as ex:
        for s in ex.map(lambda p: chunk_stats(a0, *p), pick):
            if s:
                stats.append(s)
            if len(stats) >= n:
                break
    out = dict(v, levels=len(levels), candidates=len(cand), chunks=len(stats))
    for k in ("mean", "sd", "fg", "bg", "fg_frac", "period_vox", "period_strength"):
        out[k] = float(np.nanmedian([s[k] for s in stats])) if stats else np.nan
    out["contrast"] = (out["fg"] - out["bg"]) / out["sd"] if stats else np.nan
    out["period_um"] = out["period_vox"] * (v["um"] or np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog"); ap.add_argument("out"); ap.add_argument("samples", nargs="+")
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    cat = json.load(open(a.catalog))
    done = set()
    if os.path.exists(a.out):
        done = {(r["sample"], r["volume"]) for r in csv.DictReader(open(a.out))}
    cols = ["sample", "volume", "um", "kev", "win_lo", "win_hi", "levels", "candidates", "chunks", "mean", "sd",
            "fg", "bg", "fg_frac", "contrast", "period_vox", "period_um", "period_strength", "path"]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        for s in a.samples:
            for v in volumes(cat, s):
                if (s, v["volume"]) in done:
                    continue
                try:
                    r = run(v, a.n)
                except Exception as e:
                    print(f"{s} {v['volume']}: FAILED {e}", flush=True); continue
                w.writerow({k: (f"{x:.4g}" if isinstance(x, float) else x) for k, x in r.items()}); fh.flush()
                print(f"{s} {v['volume']} {v['um']}um {v['kev']}keV chunks={r['chunks']} sd={r['sd']:.1f} "
                      f"contrast={r['contrast']:.2f} period={r['period_um']:.0f}um", flush=True)


if __name__ == "__main__":
    sys.exit(main())
