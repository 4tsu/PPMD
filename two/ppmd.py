# ======================================================
## Pythonで書いた粒子系シミュレーションコード
### 仮想的な並列計算環境をシミュレートし、その上でMDを走らせる
### 他のモジュールで定義したメソッドを呼び出すだけで、本体は書かない
# ======================================================
from two import envs
from two import particle
from two import box
from two import sdd
from two import sim

import sys
import os
import random
import time
from numpy.lib.function_base import average

random.seed(1)

# ------------------------------------------------------

class PPMD():
    ## シミュレーションパラメータの設定
    def __init__(self, STEPS=1000, OB_INTERVAL=10, dt=0.0020):
        self.STEPS = STEPS
        self.OB_INTERVAL = OB_INTERVAL
        self.dt = dt


    ## シミュレーションボックスの設定
    def set_box(self, N, xl, yl, cutoff):
        self.N = N
        self.xl = xl
        self.yl = yl
        self.cutoff = cutoff


    ## bookkeeping法の設定
    def set_margin(self, margin):
        self.margin = margin


    ## 並列プロセス数の設定
    def set_envs(self, np):
        self.np = np


    ## ロードバランサーのタイプを設定
    def set_sdd(self, sdd_type=0):
        self.sdd_type = sdd_type


    ## 系の粒子配置
    def make_data(self, config_data=None):
        self.config_data = config_data


    ## デバッグモード
    def set_mode(self, debug_flag=False):
        self.debug_flag = debug_flag


    ## 実行
    def run(self):
        start = time.time()

        ## シミュレーション実行環境の準備
        Machine = envs.Machine(self.np)   ### 並列プロセス数

        ## シミュレーションする系の準備
        Box = box.SimulationBox([self.xl, self.yl], self.cutoff, self.N)
        Box.set_margin(self.margin)
        Machine.set_boxes(Box)   ### シミュレーションボックスのグローバルな設定はグローバルに共有
        if self.config_data == None:
            Machine = sim.make_conf(Machine)   ### 初期配置
        else:
            Machine = box.read_lammps(Machine, self.config_data)   ### 液滴のデータを読み込み
        Machine = sdd.sdd_init(Machine, self.sdd_type)   ### 選択した番号のロードバランサーを実行
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
            elif 'cost_{}.dat'.format(self.sdd_type)==filename:
                os.remove(filename)
            elif 'load_balance_{}.dat'.format(self.sdd_type)==filename:
                os.remove(filename)
        ### step 0 情報
        t = 0
        k = 0
        v = 0
        sim.export_cdview(Machine.procs[0], 0, head=True)
        for proc in Machine.procs:
            sim.export_cdview(proc, 0)
            k += box.kinetic_energy(proc)
            v += box.potential_energy(proc)
        k /= Machine.procs[0].Box.N
        v /= Machine.procs[0].Box.N
        if not self.debug_flag:
            print('{:10.5f} {:12.8f} {:12.8f} {:12.8f}'.format(t, k, v, k+v))



        calc_time_ave = []
        comm_cost_ave = []
        ## ループ
        for step in range(self.STEPS):
            t += self.dt
            calc_time = 0
            comm_cost = 0
            
            ### 計算本体(シンプレクティック積分)
            ### 位置の更新t/2
            T = []
            for i,proc in enumerate(Machine.procs):
                Machine.procs[i] = sim.update_position(proc, self.dt/2)
                T.append(Machine.procs[i].time_result)
            calc_time += max(T)
            comm_cost += Machine.communicate_particles()   ### 位置が動いたので通信

            ### 速度の更新
            T = []
            for i,proc in enumerate(Machine.procs):
                if self.debug_flag:
                    debug_pairlist(proc)
                Machine.procs[i] = sim.calculate_force(proc, self.dt)
                T.append(Machine.procs[i].time_result)
            calc_time += max(T)
            comm_cost += Machine.communicate_velocity() ### 速度を更新したので通信
            
            ### 位置の更新t/2
            T = []
            for i,proc in enumerate(Machine.procs):
                Machine.procs[i] = sim.update_position(proc, self.dt/2)
                T.append(Machine.procs[i].time_result)
            calc_time += max(T)
            comm_cost += Machine.communicate_particles() ### 位置が動いたので通信
            
            if self.debug_flag:
                debug_particles()

            ### 粒子座標のスナップショットとエネルギーの出力
            v_maxs = []   ### ペアリストチェック用
            k = 0
            v = 0
            if (step+1) % self.OB_INTERVAL == 0:
                sim.export_cdview(Machine.procs[0], step+1, head=True)
                for i,proc in enumerate(Machine.procs):
                    sim.export_cdview(proc, step+1)   ### 情報の出力
            for i,proc in enumerate(Machine.procs):
                k += box.kinetic_energy(proc)
                v += box.potential_energy(proc)
                v_maxs.append(sim.check_vmax(proc))
            k /= Machine.procs[0].Box.N
            v /= Machine.procs[0].Box.N
            if not self.debug_flag:
                print('{:10.5f} {:12.8f} {:12.8f} {:12.8f}'.format(t, k, v, k+v))
            
            ### ペアリストの有効性チェック、必要があれば更新
            ### 空間分割は、ペアリストの更新と同じタイミングで行っている
            Machine, update = sim.check_pairlist(Machine, max(v_maxs), self.dt)
            if update:
                # print('Pairlist Update/Running Load Balancer')
                Machine = sdd.sdd(Machine, self.sdd_type)   ### 選択した番号のロードバランサーを実行
                Machine = sim.make_pair(Machine)   ### ペアリスト更新
                comm_cost += Machine.communicate_particles()
            
            ### このステップでのロードバランスや計算/通信コストを出力
            envs.export_cost(min(Machine.count()), max(Machine.count()), step+1, 'load_balance_{}.dat'.format(self.sdd_type))
            calc_time_ave.append(calc_time)
            comm_cost_ave.append(comm_cost)
            if (step+1) % self.OB_INTERVAL == 0:
                envs.export_cost(average(calc_time_ave), average(comm_cost_ave), step+1, 'cost_{}.dat'.format(self.sdd_type))
                calc_time_ave = []
                comm_cost_ave = []



        print('Walltime {}'.format(time.time()-start), file=sys.stderr)
        print('*** Simulation has Ended! ***', file=sys.stderr)



def debug_pairlist(proc):
    for pair in proc.subregion.pairlist:
        print('i,j=[{},{}]'.format(pair.idi, pair.idj))
    for pair in proc.pairlist_between_neighbor:
        print('i,j=[{},{}]'.format(pair.idi, pair.idj))

def debug_particles(Machine):
    for i,proc in enumerate(Machine.procs):
        for p in Machine.procs[i].subregion.particles:
            print('x,y=[{:8.6f} {:8.6f}]'.format(p.x, p.y))
            print('vx,vy=[{:8.6f} {:8.6f}]'.format(p.vx, p.vy))
