# ======================================================
## PPMD実行コード
# ======================================================

import two.ppmd

# ------------------------------------------------------
### STEPS, OB_INTERVAL, dt
ppmd = ppmd.PPMD(10, 10, 0.0020)
### N, xl, yl, cutoff
ppmd.set_box(100, 20, 20, 3.0)
ppmd.set_margin(0.5)
ppmd.set_envs(9)
ppmd.set_sdd(0)
ppmd.make_data('droplet.dump')
ppmd.set_mode()

ppmd.run()
