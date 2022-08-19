# ======================================================
## PPMD実行コード
# ======================================================

# from two.ppmd import PPMD
from three.ppmd import PPMD

# ------------------------------------------------------
### STEPS, OB_INTERVAL, dt
ppmd = PPMD(10, 10, 0.0020)
### N, xl, yl, cutoff
ppmd.set_box(200, 10, 10, 10, 3.5)
ppmd.set_margin(0.5)
ppmd.set_envs(8)
ppmd.set_sdd(1)
ppmd.make_data('three/droplet.dump')
ppmd.set_mode()

ppmd.run()
