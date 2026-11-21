"""Measure a scroll's spiral_outward_sense from its horizontal surface tracks.

Convention (villa spiral-fitting, surface_orientation.py): "CW" means the spiral's
radius grows with atan2(y, x) in the volume's own array coordinates; "ACW" is the mirror.

Method. Each horizontal ("h") track lies in one z slice and follows one papyrus sheet.
Around the umbilicus at that z, take polar angle t = atan2(y - uy, x - ux) and log
radius L = log(r). For every track step, record dL/dt at the step's angle. For any
closed curve (circle, flattened ellipse, any similar-shape family) the integral of
dL/dt over a full turn is 0, and using log r makes concentric similar shapes share
one dL/dt(t) profile regardless of size. A spiral adds a net drift
(sheet spacing / radius per turn) whose sign is the sense: positive -> CW, negative -> ACW.

We bin step slopes by angle (median per bin, robust to outliers) and integrate over
the bins. Consistency is checked across z sub-bands, radius bands and a bootstrap
over tracks.

Usage:
  python measure_spiral_sense.py PACKED_TRACKS_DIR UMBILICUS_JSON [--zmin Z --zmax Z]
  python measure_spiral_sense.py --selftest
"""
import argparse, json, os
import numpy as np

NBINS = 72


def umbilicus_interp(umb_json):
    """Return f(z) -> (x, y) by linear interpolation of the umbilicus control points."""
    pts = json.load(open(umb_json))["control_points"]
    zs = np.array([p["z"] for p in pts], float); order = np.argsort(zs)
    zs = zs[order]
    xs = np.array([pts[i]["x"] for i in order], float)
    ys = np.array([pts[i]["y"] for i in order], float)
    return lambda z: (np.interp(z, zs, xs), np.interp(z, zs, ys))


def step_slopes(tracks_yx, centers):
    """tracks_yx: list of (n,2) arrays (y,x); centers: list of (uy,ux).
    Returns arrays of step mid-angles, dL/dt, step weights (|dt|), track ids, mean radius."""
    ang, slope, w, tid, rad = [], [], [], [], []
    for k, (p, (uy, ux)) in enumerate(zip(tracks_yx, centers)):
        if len(p) < 3:
            continue
        dy, dx = p[:, 0] - uy, p[:, 1] - ux
        t = np.unwrap(np.arctan2(dy, dx))
        L = np.log(np.hypot(dy, dx) + 1e-9)
        span = t.max() - t.min()
        if span < 0.02 or span > 1.0:      # need a measurable arc, but short vs one turn
            continue
        # One least-squares slope dL/dt per track: averages pixel noise and quantization.
        tc = t - t.mean()
        s = float((tc * (L - L.mean())).sum() / (tc * tc).sum())
        ang.append(np.array([np.mod(t.mean(), 2 * np.pi)])); slope.append(np.array([s]))
        w.append(np.array([span])); tid.append(np.array([k])); rad.append(np.array([np.exp(L.mean())]))
    if not ang:
        return [np.empty(0)] * 5
    return [np.concatenate(a) for a in (ang, slope, w, tid, rad)]


def turn_integral(ang, slope, w):
    """Integral over one turn of the per-angle-bin weighted median of dL/dt."""
    b = np.minimum((ang / (2 * np.pi) * NBINS).astype(int), NBINS - 1)
    vals, filled = np.zeros(NBINS), 0
    for i in range(NBINS):
        m = b == i
        if m.sum() < 8:
            continue
        s, ww = slope[m], w[m]
        o = np.argsort(s); c = np.cumsum(ww[o])
        vals[i] = s[o][np.searchsorted(c, 0.5 * c[-1])]; filled += 1
    return vals.sum() * (2 * np.pi / NBINS), filled


def summarize(name, ang, slope, w, tid, rad, rng):
    I, filled = turn_integral(ang, slope, w)
    # bootstrap over tracks
    utid = np.unique(tid); boots = []
    for _ in range(100):
        pick = rng.choice(utid, size=len(utid), replace=True)
        cnt = np.bincount(pick, minlength=tid.max() + 1)
        rep = cnt[tid]
        m = rep > 0
        boots.append(turn_integral(ang[m], slope[m], w[m] * rep[m])[0])
    boots = np.array(boots)
    frac_pos = (boots > 0).mean()
    sense = "CW" if I > 0 else "ACW"
    print(f"{name:28s} integral={I:+.5f}  bins={filled}/{NBINS}  boot95=[{np.percentile(boots,2.5):+.5f},"
          f"{np.percentile(boots,97.5):+.5f}]  P(>0)={frac_pos:.2f}  -> {sense}  (tracks={len(utid)})")
    return I, frac_pos


def synthetic(sense, n_turns=40, spacing=20.0, squash=0.35, rot=0.4, noise=1.0, seed=0):
    """Flattened Archimedean spiral cut into short horizontal tracks (synthetic test data)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi * n_turns, 2000000)
    r = 200 + spacing * t / (2 * np.pi)
    s = 1 if sense == "CW" else -1
    x, y = r * np.cos(s * t), r * np.sin(s * t)          # radius grows with atan2(y,x) iff CW
    x, y = x, y * squash                                  # flatten
    x, y = x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)
    pts = np.stack([y, x], 1) + rng.normal(0, noise, (len(t), 2))
    tracks, i = [], 0
    while i < len(pts) - 50:
        n = int(rng.integers(30, 300)); tracks.append(pts[i:i + n][::3]); i += n + int(rng.integers(0, 200))
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="?"); ap.add_argument("umbilicus", nargs="?")
    ap.add_argument("--zmin", type=int, default=None); ap.add_argument("--zmax", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(); rng = np.random.default_rng(0)

    if a.selftest:
        print("SELF-TEST on synthetic flattened spirals (synthetic data, code check only)")
        for sense in ("CW", "ACW"):
            for squash in (1.0, 0.35, 0.15):
                tr = synthetic(sense, squash=squash)
                res = step_slopes(tr, [(0.0, 0.0)] * len(tr))
                I, _ = summarize(f"true {sense} squash={squash}", *res, rng)
                assert (I > 0) == (sense == "CW"), "self-test FAILED"
        print("self-test passed"); return

    d = a.tracks
    off = np.fromfile(os.path.join(d, "offsets.i64"), "<i8")
    fam = np.fromfile(os.path.join(d, "family_codes.i8"), "i1")
    zb = np.fromfile(os.path.join(d, "z_bounds.i32"), "<i4").reshape(-1, 2)
    coords = np.memmap(os.path.join(d, "coordinates.i32"), "<i4", mode="r").reshape(-1, 3)
    sel = np.where((fam == 0) & (zb[:, 0] == zb[:, 1]))[0]
    if a.zmin is not None: sel = sel[zb[sel, 0] >= a.zmin]
    if a.zmax is not None: sel = sel[zb[sel, 0] <= a.zmax]
    print(f"tracks total={len(fam)}  horizontal single-z used={len(sel)}  "
          f"(families: h={int((fam==0).sum())}, v={int((fam==1).sum())})")
    umb = umbilicus_interp(a.umbilicus)
    tracks, centers, zs = [], [], []
    for k in sel:
        p = np.asarray(coords[off[k]:off[k + 1]])
        if len(p) < 3: continue
        z = p[0, 0]; ux, uy = umb(z)
        tracks.append(p[:, 1:].astype(float)); centers.append((uy, ux)); zs.append(z)
    zs = np.array(zs)
    ang, slope, w, tid, rad = step_slopes(tracks, centers)
    print(f"steps={len(ang)}")
    summarize("ALL", ang, slope, w, tid, rad, rng)
    tz = zs[tid]
    for lo, hi in [(q[0], q[-1]) for q in np.array_split(np.unique(zs), 4)]:
        m = (tz >= lo) & (tz <= hi)
        summarize(f"z {lo}-{hi}", ang[m], slope[m], w[m], tid[m], rad[m], rng)
    qs = np.quantile(rad, [0, 1/3, 2/3, 1])
    for i in range(3):
        m = (rad >= qs[i]) & (rad <= qs[i + 1])
        summarize(f"radius {qs[i]:.0f}-{qs[i+1]:.0f}", ang[m], slope[m], w[m], tid[m], rad[m], rng)


if __name__ == "__main__":
    main()
