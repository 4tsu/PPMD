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
import time
from numpy.lib.function_base import average

random.seed(1)

# ------------------------------------------------------


## シミュレーションパラメータの設定
STEPS = 1000
OB_INTERVAL = 10
dt = 0.0020
N = 100
np = 9
sdd_num = 2   ### ロードバランサーの種類



## シミュレーション実行環境の準備
Machine = envs.Machine(np)   ### 並列プロセス数

## シミュレーションする系の準備
Box = box.SimulationBox([20, 20], 2.0, N)
Box.set_margin(0.5)
Machine.set_boxes(Box)   ### シミュレーションボックスのグローバルな設定はグローバルに共有
# Machine = sim.make_conf(Machine)   ### 初期配置
Machine = box.read_lammps(Machine, 'droplet.dump')   ### 液滴のデータを読み込み
Machine = sdd.sdd_init(Machine, sdd_num)   ### 選択した番号のロードバランサーを実行
Machine = sim.set_initial_velocity(1.0, Machine)   ### 初速
### 最初のペアリスト作成
for proc in Machine.procs:
    proc.subregion.calc_center(proc.Box)
    proc.subregion.calc_radius(proc.Box)
Machine = sim.make_pair(Machine)



## 粒子の軌跡とエネルギー出力準備
### export_cdviewが上書き方式なので、.cdvファイルを事前にクリアしておく
### 他の追記式ファイルも同様
for filename in os.listdir("."):
    if '.cdv' in filename:
        os.remove(filename)
    elif 'cost_{}.dat'.format(sdd_num)==filename:
        os.remove(filename)
    elif 'load_balance_{}.dat'.format(sdd_num)==filename:
        os.remove(filename)
### step 0 情報
t = 0
k = 0
v = 0
for proc in Machine.procs:
    sim.export_cdview(proc, 0)
    k += box.kinetic_energy(proc)
    v += box.potential_energy(proc)
k /= Machine.procs[0].Box.N
v /= Machine.procs[0].Box.N
print('{:10.5f} {:12.8f} {:12.8f} {:12.8f}'.format(t, k, v, k+v))



calc_time_ave = []
comm_cost_ave = []
## ループ
for step in range(STEPS):
    t += dt
    calc_time = 0
    comm_cost = 0
    
    ### 計算本体(シンプレクティック積分)
    ### 位置の更新t/2
    T = []
    for i,proc in enumerate(Machine.procs):
        Machine.procs[i] = sim.update_position(proc, dt/2)
        T.append(Machine.procs[i].time_result)
    calc_time += max(T)
    comm_cost += Machine.communicate_particles()   ### 位置が動いたので通信

    ### 速度の更新
    T = []
    for i,proc in enumerate(Machine.procs):
        """
        ### debug
        for pair in proc.subregion.pairlist:
            print(pair.idi, pair.idj)
        for pair in proc.pairlist_between_neighbor:
            print(pair.idi, pair.idj)
        """
        Machine.procs[i] = sim.calculate_force(proc, dt)
        T.append(Machine.procs[i].time_result)
    calc_time += max(T)
    comm_cost += Machine.communicate_velocity() ### 速度を更新したので通信
    
    ### 位置の更新t/2
    T = []
    for i,proc in enumerate(Machine.procs):
        Machine.procs[i] = sim.update_position(proc, dt/2)
        T.append(Machine.procs[i].time_result)
    calc_time += max(T)
    comm_cost += Machine.communicate_particles() ### 位置が動いたので通信
    
    """
    ### debug
    for i,proc in enumerate(Machine.procs):
        for p in Machine.procs[i].subregion.particles:
            # print('{:8.6f} {:8.6f}'.format(p.x, p.y))
            # print('{:8.6f} {:8.6f}'.format(p.vx, p.vy))
            pass
    """

    ### 粒子座標のスナップショットとエネルギーの出力
    v_maxs = []   ### ペアリストチェック用
    k = 0
    v = 0
    for i,proc in enumerate(Machine.procs):
        if (step+1) % OB_INTERVAL == 0:
            sim.export_cdview(proc, step+1)   ### 情報の出力
        k += box.kinetic_energy(proc)
        v += box.potential_energy(proc)
        v_maxs.append(sim.check_vmax(proc))
    k /= Machine.procs[0].Box.N
    v /= Machine.procs[0].Box.N
    print('{:10.5f} {:12.8f} {:12.8f} {:12.8f}'.format(t, k, v, k+v))
    
    ### ペアリストの有効性チェック、必要があれば更新
    ### 空間分割は、ペアリストの更新と同じタイミングで行っている
    Machine, update = sim.check_pairlist(Machine, max(v_maxs), dt)
    if update:
        # print('Pairlist Update/Running Load Balancer')
        Machine = sdd.sdd(Machine, sdd_num)   ### 選択した番号のロードバランサーを実行
        Machine = sim.make_pair(Machine)   ### ペアリスト更新
        comm_cost += Machine.communicate_particles()
    
    ### このステップでのロードバランスや計算/通信コストを出力
    envs.export_cost(min(Machine.count()), max(Machine.count()), step+1, 'load_balance_{}.dat'.format(sdd_num))
    calc_time_ave.append(calc_time)
    comm_cost_ave.append(comm_cost)
    if (step+1) % OB_INTERVAL == 0:
        envs.export_cost(average(calc_time_ave), average(comm_cost_ave), step+1, 'cost_{}.dat'.format(sdd_num))
        calc_time_ave = []
        comm_cost_ave = []



print('*** Simulation has Ended! ***', file=sys.stderr)
