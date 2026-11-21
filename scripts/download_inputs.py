"""Download the public PHerc0125 inputs for a zero-shot First Letters test.

Source: Vesuvius Challenge open data (CC-BY-NC 4.0).
  S3 bucket  https://vesuvius-challenge-open-data.s3.amazonaws.com/
  Tracks     https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0125/20250821151825/tracks/
  Ink model  https://huggingface.co/scrollprize/ink_9um (MIT)

Mirrors the S3 key layout under OUT. Resumable: a file whose size already
matches is skipped. Usage:
  python download_inputs.py [--out D:/DBBun/data/vesuvius/PHerc0125] [--parts s3,tracks,model]
"""
import argparse, concurrent.futures as cf, os, re, sys, time, urllib.parse, urllib.request

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
VOL = "PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/"
LAS = "PHerc0125/representations/predictions/lasagna/20250821151825-lasagna-20260419180421/"
UMB = "PHerc0125/representations/umbilicus/20250821151825-umbilicus-20260808111524.json"
TRK = "https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0125/20250821151825/tracks/"
TRK_FILES = ["PHerc0125_20250821151825_surface_m7_L0_th0.2.dbm",
             "PHerc0125_20250821151825_surface_m7_L0_th0.2.dbm.crossings.npz",
             "PHerc0125_20250821151825_surface_m7_L0_th0.2.extract.json"]
HF = "https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth"

# Level-0 CT band: z chunk rows 80..87 = slices 10240..11263 (128-voxel chunks).
Z_ROWS = range(80, 88)


def s3_list(prefix):
    tok = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if tok:
            q["continuation-token"] = tok
        for a in range(5):
            try:
                r = urllib.request.urlopen(S3 + "?" + urllib.parse.urlencode(q), timeout=120).read().decode()
                break
            except Exception:
                time.sleep(2 ** a)
        else:
            raise RuntimeError(f"listing failed: {prefix}")
        for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", r, re.S):
            yield k, int(s)
        m = re.search(r"<NextContinuationToken>([^<]+)", r)
        if not m:
            return
        tok = m.group(1)


def fetch(url, dest, size=None):
    if size is not None and os.path.exists(dest) and os.path.getsize(dest) == size:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for a in range(6):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            if size is not None and os.path.getsize(tmp) != size:
                raise IOError("size mismatch")
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except Exception as e:
            err = e
            time.sleep(2 ** a)
    raise RuntimeError(f"{url}: {err}")


def s3_jobs():
    # Small metadata files.
    meta = [VOL + n for n in (".zattrs", ".zgroup", "metadata.json", "0/.zarray", "0/.zattrs")]
    meta += [UMB, LAS + "PHerc0125.lasagna.json"]
    for g in ("nx", "ny", "grad_mag"):
        z = f"{LAS}PHerc0125_{g}.ome.zarr/"
        meta += [z + ".zattrs", z + ".zgroup", z + "2/.zarray", z + "2/.zattrs"]
    for k in meta:
        yield k, None
    for row in Z_ROWS:
        yield from s3_list(f"{VOL}0/{row}/")
    for g in ("nx", "ny", "grad_mag"):
        yield from s3_list(f"{LAS}PHerc0125_{g}.ome.zarr/2/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/DBBun/data/vesuvius/PHerc0125")
    ap.add_argument("--parts", default="s3,tracks,model")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    root = os.path.dirname(a.out.rstrip("/\\"))  # S3 keys already start with PHerc0125/
    t0, done, nbytes, fails = time.time(), 0, 0, 0

    if "model" in parts:
        nbytes += fetch(HF, os.path.join(a.out, "models/ink_9um/hybrid_3d2d-seed42/step-075000.pth"), 138360039)
        print("model done", flush=True)
    if "tracks" in parts:
        for n in TRK_FILES:
            nbytes += fetch(TRK + n, os.path.join(a.out, "tracks", n))
            print("track file done:", n, flush=True)
    if "s3" in parts:
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {}
            for k, s in s3_jobs():
                futs[ex.submit(fetch, S3 + k, os.path.join(root, *k.split("/")), s)] = k
                if len(futs) >= 4000:
                    for f in cf.as_completed(list(futs)):
                        k2 = futs.pop(f)
                        try:
                            nbytes += f.result(); done += 1
                        except Exception as e:
                            fails += 1
                            if not k2.endswith(".zattrs"):
                                print("FAIL", k2, e, flush=True)
                    print(f"{done} files, {nbytes/2**30:.2f} GiB new, {fails} fails, "
                          f"{(time.time()-t0)/60:.1f} min", flush=True)
            for f in cf.as_completed(list(futs)):
                k2 = futs.pop(f)
                try:
                    nbytes += f.result(); done += 1
                except Exception as e:
                    fails += 1
                    print("FAIL", k2, e, flush=True)
    print(f"FINISHED: {done} S3 files, {nbytes/2**30:.2f} GiB new, {fails} fails, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
