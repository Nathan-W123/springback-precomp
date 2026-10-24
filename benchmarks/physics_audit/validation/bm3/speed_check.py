"""Speed-halving check (docs/forming.md 7.2 at commit 7d571e6): profile differences between two explicit runs, and the springback
(release minus unload) of each. usage: python3 speed_check.py cases/a cases/b -> speed_check.json"""
import sys, os, json
import numpy as np
a, b = sys.argv[1], sys.argv[2]; out = {}
xs = np.linspace(-75, 75, 601)
def P(c, step, tag): q = np.loadtxt(os.path.join(c, f'profile_{step}_{tag}_top.csv'), delimiter=',', skiprows=1); return np.interp(xs, q[:, 0], q[:, 1])
for tag in ('RD', 'TD'):
    for step in ('unload', 'release'):
        d = P(a, step, tag) - P(b, step, tag)
        out[f'{step}_{tag}_diff_rms_mm'] = float(np.sqrt(np.mean(d ** 2))); out[f'{step}_{tag}_diff_max_mm'] = float(np.max(np.abs(d)))
    for c in (a, b):
        sb = P(c, 'release', tag) - P(c, 'unload', tag)
        out[f'springback_{os.path.basename(c)}_{tag}_rms_mm'] = float(np.sqrt(np.mean(sb ** 2))); out[f'springback_{os.path.basename(c)}_{tag}_max_mm'] = float(np.max(np.abs(sb)))
        out[f'springback_{os.path.basename(c)}_{tag}_centre_mm'] = float(sb[300])
json.dump(out, open(f'speed_check_{os.path.basename(a)}_vs_{os.path.basename(b)}.json', 'w'), indent=1)
for k, v in out.items(): print(k, round(v, 3))
