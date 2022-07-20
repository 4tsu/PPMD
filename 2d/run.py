# ======================================================
## Pythonで書いた粒子系シミュレーションコードのmain
### 仮想的な並列計算環境をシミュレートし、その上でMDを走らせる
### 他のモジュールで定義したメソッドを呼び出すだけで、本体は書かない
# ======================================================
import envs
import particle
import box
import sdd
import sim

import sys
import random

random.seed(1)

# ------------------------------------------------------

## シミュレーションパラメータの設定
STEPS = 1000
OB_INTERVAL = 50
dt = 0.0010
N = 100

## シミュレーション環境の準備
Machine = envs.Machine(1)   ### 並列プロセス数

## シミュレーションする系の準備
Box = box.SimulationBox([10, 10], 2.0, N)
Machine.set_boxes(Box)
Machine = sim.make_conf(Machine)
Machine = sim.set_initial_velocity(1.0, Machine)
print(len(Machine.procs[0].particles))
"""
## ループ
t = 0
for step in range(STEPS):
    for proc in Computer.procs:
        ### 計算本体(シンプレクティック積分)
        Box = sim.update_position(Box, dt)
        Box = sim.calculate_force(Box, dt)
        Box = sim.update_position(Box, dt)
        Box = box.periodic(Box)
        if step % OB_INTERVAL == 0:
            box.export_cdview(Box, step)   ### 情報の出力
    k = Box.kinetic_energy()
    v = Box.potential_energy()
    print('{:10.5f} {} {} {}'.format(t, k, v, k+v))
    t += dt
    # Computer.communicate()   ### 1stepの計算が全て終わったら、同期通信をする
print('*** Simulation Ended! ***', file=sys.stderr)
"""