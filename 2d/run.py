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
import os
import random

random.seed(1)

# ------------------------------------------------------

## シミュレーションパラメータの設定
STEPS = 1000
OB_INTERVAL = 50
dt = 0.0010
N = 121
np = 4

## シミュレーション実行環境の準備
Machine = envs.Machine(np)   ### 並列プロセス数

## シミュレーションする系の準備
Box = box.SimulationBox([10, 10], 2.0, N)
Box.set_margin(0.5)
Machine.set_boxes(Box)   ### シミュレーションボックスのグローバルな設定はグローバルに共有
Machine = sim.make_conf(Machine)   ### 初期配置
Machine = sim.set_initial_velocity(1.0, Machine)   ### 初速

### 最初のペアリスト作成
for proc in Machine.procs:
    proc.subregion.calc_center(proc.Box)
    proc.subregion.calc_radius(proc.Box)
Machine = sim.make_pair(Machine)

for i in range(len(Machine.procs)):
    print('len my pairlist', len(Machine.procs[i].subregion.pairlist))
    print('len neigh pairlist', len(Machine.procs[i].pairlist_between_neighbor))

## 粒子の軌跡とエネルギー出力準備
### export_cdviewが上書き方式なので、.cdvファイルを事前にクリアしておく
for filename in os.listdir("."):
    if '.cdv' in filename:
        os.remove(filename)
### step 0 情報
t = 0
k = 0
v = 0
for proc in Machine.procs:
    sim.export_cdview(proc, 0)
"""
    k += box.kinetic_energy(proc)
    v += box.potential_energy(proc)
print('{:10.5f} {} {} {}'.format(t, k, v, k+v))


## ループ
for step in range(STEPS):
    t += dt
    k = 0
    v = 0
    for i,proc in enumerate(Machine.procs):
        ### 計算本体(シンプレクティック積分)
        proc = sim.update_position(proc, dt/2)
        proc = sim.check_pairlist(proc, dt)
        proc = sim.calculate_force(proc, dt)
        proc = sim.update_position(proc, dt/2)
        proc = box.periodic(proc)
        Machine.procs[i] = proc
        if (step+1) % OB_INTERVAL == 0:
            sim.export_cdview(proc, step+1)   ### 情報の出力
        k += box.kinetic_energy(proc)
        v += box.potential_energy(proc)
    print('{:10.5f} {} {} {}'.format(t, k, v, k+v))
    # Machine.communicate()   ### 1stepの計算が全て終わったら、同期通信をする.Box情報の共有も含む
print('*** Simulation Ended! ***', file=sys.stderr)
"""