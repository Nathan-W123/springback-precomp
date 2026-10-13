"""Figure: measured vs published FE vs SparLab profiles (RD), thickness and force vs depth.
usage: python3 plot_bm3.py out.png cases/<a> [cases/<b> ...]"""
import sys, os, csv, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import analyze_bm3 as A
out = sys.argv[1]; cases = sys.argv[2:]
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
e = A.load_exp('RD')
ax[0].plot(e['exp_marker'][:, 0], e['exp_marker'][:, 1], 'k.', ms=3, label='measured (Neto et al. 2016, RD)')
ax[0].plot(e['sim'][:, 0], e['sim'][:, 1], '-', color='0.5', lw=1, label='published FE (DD3IMP)')
ax[0].plot(e['cad'][:, 0], e['cad'][:, 1], ':', color='0.6', lw=1, label='CAD')
for c in cases:
    for step, ls in (('release', '-'), ('unload', '--')):
        f = os.path.join(c, f'profile_{step}_RD_top.csv')
        if os.path.exists(f):
            p = np.loadtxt(f, delimiter=',', skiprows=1); ax[0].plot(p[:, 0], p[:, 1], ls, lw=1.2, label=f'{os.path.basename(c)} {step}')
ax[0].set_xlabel('distance from centreline [mm]'); ax[0].set_ylabel('z [mm]'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[0].set_xlim(-90, 90)
thk = list(csv.DictReader(open(os.path.join(A.DIG, 'bm3_fig15_thickness.csv'))))
for ser, st in (('exp_marker', 'k.'), ('sim', '-')):
    q = np.array(sorted((float(r['x_mm']), float(r['t_mm'])) for r in thk if r['series'] == ser))
    ax[1].plot(q[:, 0], q[:, 1], st, ms=3, color='k' if ser == 'exp_marker' else '0.5', label={'exp_marker': 'measured', 'sim': 'published FE'}[ser])
for c in cases:
    for tag, sgn in (('RD', -1), ('TD', 1)):
        f = os.path.join(c, f'thickness_release_{tag}.csv')
        if os.path.exists(f):
            q = np.loadtxt(f, delimiter=',', skiprows=1); q = q[q[:, 0] * sgn > 0]; ax[1].plot(q[:, 0], q[:, 1], lw=1.2, label=f'{os.path.basename(c)} {tag}')
ax[1].set_xlabel('distance from centreline [mm] (left RD, right TD)'); ax[1].set_ylabel('thickness [mm]'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
ef = json.load(open(os.path.join(A.HERE, 'exp_force_by_depth.json')))
for ser, col in (('exp_Fz_line', 'k'), ('sim_Fz', '0.5')):
    d = sorted(float(k) for k in ef[ser]); ax[2].plot(d, [ef[ser][str(k) if str(k) in ef[ser] else k] for k in d], '-', color=col,
        label={'exp_Fz_line': 'measured Fz, contour mean (dz 0.5)', 'sim_Fz': 'published FE Fz'}[ser])
ax[2].axhline(A.aerens(0.5), color='k', ls=':', lw=1, label='Aerens et al. (dz 0.5): %.0f N' % A.aerens(0.5))
for c in cases:
    f = os.path.join(c, 'force_by_depth.csv')
    if os.path.exists(f):
        a = np.genfromtxt(f, delimiter=',', names=True)
        a = np.atleast_1d(a)
        ax[2].plot(a['depth_mm'], a['fz_mean_N'], 'o-', label=f'{os.path.basename(c)} (its dz)')
        ax[2].plot(a['depth_mm'], a['fz_mean_N'] * a['aerens_factor_to_dz05'], 's--', mfc='none', label=f'{os.path.basename(c)} x Aerens scallop factor -> dz 0.5')
ax[2].set_xlabel('tool depth [mm]'); ax[2].set_ylabel('axial force Fz [N]'); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(out, dpi=110)
