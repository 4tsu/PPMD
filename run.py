# ======================================================
## PPMD実行コード
# ======================================================

from two.ppmd import PPMD
# from three.ppmd import PPMD

# ------------------------------------------------------
### STEPS, OB_INTERVAL, dt
ppmd = PPMD(1000, 10, 0.0020)
### N, xl, yl(,zl), cutoff
ppmd.set_box(100, 10, 10, 3.5)
ppmd.set_margin(0.5)
ppmd.set_envs(9)
ppmd.set_sdd(2)
ppmd.make_data('two/droplet.dump')

ppmd.set_mode()

ppmd.run()
