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
STEPS = 100
OB_INTERVAL = 100
dt = 0.001
N = 100

## シミュレーション環境の準備
Computer = envs.Computer(1)   ### 並列プロセス数

## シミュレーションする系の準備
Box = box.SimulationBox([10, 10], 2.0)
Box = sim.make_conf(N, Box)
Box = sim.set_initial_velocity(1.0, Box)

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
