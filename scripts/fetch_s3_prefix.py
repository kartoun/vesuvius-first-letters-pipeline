"""Mirror prefixes of the public Vesuvius Challenge S3 bucket to a local root, keeping the
key layout. Resumable: files whose size already matches are skipped.

Usage: python fetch_s3_prefix.py LOCAL_ROOT PREFIX [PREFIX ...]
"""
import os, re, sys, time, urllib.parse, urllib.request, concurrent.futures as cf

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"


def listing(prefix):
    tok = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if tok:
            q["continuation-token"] = tok
        r = urllib.request.urlopen(B + "?" + urllib.parse.urlencode(q), timeout=120).read().decode()
        for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", r, re.S):
            yield k, int(s)
        m = re.search(r"<NextContinuationToken>([^<]+)", r)
        if not m:
            return
        tok = m.group(1)


def get(root, k, size):
    d = os.path.join(root, *k.split("/"))
    if os.path.exists(d) and os.path.getsize(d) == size:
        return 0
    os.makedirs(os.path.dirname(d), exist_ok=True)
    for a in range(6):
        try:
            urllib.request.urlretrieve(B + urllib.parse.quote(k), d + ".part")
            os.replace(d + ".part", d)
            return size
        except Exception:
            time.sleep(2 ** a)
    raise RuntimeError(k)


root = sys.argv[1]
t0 = time.time()
for prefix in sys.argv[2:]:
    jobs = list(listing(prefix))
    with cf.ThreadPoolExecutor(32) as ex:
        got = sum(ex.map(lambda ks: get(root, *ks), jobs))
    print(f"{prefix}: {len(jobs)} files, {sum(s for _, s in jobs)/2**30:.2f} GiB listed, "
          f"{got/2**30:.2f} GiB new, {(time.time()-t0)/60:.1f} min", flush=True)
