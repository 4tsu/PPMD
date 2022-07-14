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

import random

random.seed(1)

# ------------------------------------------------------

## シミュレーションパラメータの設定
STEPS = 1000
OB_INTERVAL = 100
dt = 0.001
N = 300

## シミュレーション環境の準備
Computer = envs.Computer(1)   ### 並列プロセス数

## シミュレーションする系の準備
Box = box.SimulationBox([10, 10], 2.0)
sim.make_conf(N, Box)

## ループ
t = 0
for step in range(STEPS):
    for proc in Computer.procs:
        ### 計算本体(シンプレクティック積分)
        sim.update_position(proc)
        sim.calculate_force(proc)
        sim.update_position(proc)
        box.periodic(proc)
        box.export(proc)   ### 情報の出力
        t += dt
    # Computer.communicate()   ### 1stepの計算が全て終わったら、同期通信をする
print('*** Simulation Ended! ***')
