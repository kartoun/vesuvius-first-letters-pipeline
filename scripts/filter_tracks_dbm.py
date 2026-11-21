"""Write a z-band subset of a dbm.sqlite3 surface-track database.

Every key is kept (so convert_track_store.py assigns the same key ordinals and
therefore the same source IDs as for the full database). Inside each key, tracks
whose z range does not intersect [z_lo, z_hi] are replaced by empty lists, which
the converter skips. Track coordinates are (z, y, x) integers.

Usage: python filter_tracks_dbm.py SRC.dbm DST.dbm Z_LO Z_HI
"""
import pickle, sqlite3, sys, time
import numpy as np

src, dst, z_lo, z_hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t0 = time.time()
cin = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
cout = sqlite3.connect(dst)
cout.execute("CREATE TABLE IF NOT EXISTS Dict (key BLOB UNIQUE NOT NULL, value BLOB NOT NULL)")
kept = total = keys = 0
rows = []
for key, value in cin.execute("SELECT key, value FROM Dict"):
    entries = pickle.loads(value)
    out = []
    for e in entries:
        if len(e):
            total += 1
            z = np.asarray(e)[:, 0]
            if z.max() >= z_lo and z.min() <= z_hi:
                out.append(e); kept += 1
                continue
        out.append([])
    rows.append((key, pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)))
    keys += 1
    if len(rows) >= 2000:
        cout.executemany("INSERT INTO Dict (key, value) VALUES (?, ?)", rows); rows.clear()
if rows:
    cout.executemany("INSERT INTO Dict (key, value) VALUES (?, ?)", rows)
cout.commit(); cout.close()
print(f"keys={keys} tracks kept {kept}/{total} ({100*kept/max(total,1):.1f}%) "
      f"for z in [{z_lo},{z_hi}] in {time.time()-t0:.0f}s")
