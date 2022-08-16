# ======================================================
## PPMD実行コード
# ======================================================

from two.ppmd import PPMD

# ------------------------------------------------------
### STEPS, OB_INTERVAL, dt
ppmd = PPMD(10, 10, 0.0020)
### N, xl, yl, cutoff
ppmd.set_box(100, 20, 20, 3.0)
ppmd.set_margin(0.5)
ppmd.set_envs(9)
ppmd.set_sdd(0)
ppmd.make_data('two/droplet.dump')
ppmd.set_mode()

ppmd.run()
