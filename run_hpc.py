# ======================================================
## PPMD実行コード(HPCバッチジョブ用)
# ======================================================

# from two.ppmd import PPMD
from three.ppmd import PPMD

# ------------------------------------------------------
def run(filename):
    for sdd in range(3):
        ### STEPS, OB_INTERVAL, dt
        for i in range(5):
            ppmd = PPMD(1000, 10, 0.0020)
            ### N, xl, yl(,zl), cutoff
            ppmd.set_box(100, 10, 10, 10, 3.5)
            ppmd.set_margin(0.5)
            ppmd.set_envs(27)
            ppmd.set_sdd(sdd)
            ppmd.set_trial(i)
            ppmd.make_data('filename')

            ppmd.set_mode()

            ppmd.run()



run('three/droplet.dump')
