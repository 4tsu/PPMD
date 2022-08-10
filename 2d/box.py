# ====================================================
## 対象とする系、シミュレーションボックスについてのコード
## 系のエネルギーなどの観測コード
# ====================================================
import sim

import random

random.seed(1)
# ----------------------------------------------------
## シミュレーションボックスを定義するクラス
### すべてのプロセスで共有される最低限の情報を持つ
### lengthsは要素数2の配列[lx,ly]であり、ボックスの寸法を表す
class SimulationBox:
    def __init__(self, lengths, cutoff, N):
        self.xl = lengths[0]
        self.yl = lengths[1]
        self.x_max =  self.xl/2
        self.x_min = -self.xl/2
        self.y_max =  self.yl/2
        self.y_min = -self.yl/2
        
        self.cutoff = cutoff
        self.N = N

        self.Potential = Potential()

    ## 下の関数用
    def modify(self, x, x_max, x_min, L):
        if x > x_max:
            x -= L
        elif x < x_min:
            x += L
        return x

    ## 周期境界条件の適用
    def periodic_coordinate(self, x, y):
        if not (self.x_min <= x <= self.x_max):
            while not (self.x_min <= x <= self.x_max):
                x = self.modify(x, self.x_max, self.x_min, self.xl)
                # print(x)
        if not (self.y_min <= y <= self.y_max):
            while not (self.y_min <= y <= self.y_max):
                y = self.modify(y, self.y_max, self.y_min, self.yl)
        assert self.x_min<=x<=self.x_max and self.y_min<=y<=self.y_max, '周期境界補正に失敗しています'
        return x, y
    
    ## 周期境界条件を考慮した距離を返す
    def periodic_distance(self, x1, y1, x2, y2):
        rx = min((x1-x2)**2, (self.xl-abs(x1-x2))**2)
        ry = min((y1-y2)**2, (self.yl-abs(y1-y2))**2)
        assert 0<=rx<=(self.xl/2)**2 and 0<=ry<=(self.yl/2)**2, '周期境界条件の補正に失敗しています[rx={},ry={}]'.format(rx, ry)
        return (rx + ry)**0.5
    
    def add_particle(self, particle):
        self.particles.append(particle)

    ### bookkeeping法のマージン
    def set_margin(self, margin):
        self.margin = margin
        self.co_p_margin = self.cutoff + self.margin   ### マージン＋カットオフ距離
        self.margin_life = margin
    
    def subtract_margin(self, arg):
        self.margin_life -= arg

    ### 空間分割の仕方を保存
    ### この関数ここにいるべきなのか、要検討
    def set_subregion(self, xn, yn, sd_xl, sd_yl):
        self.xn = xn
        self.yn = yn
        self.sd_xl = sd_xl
        self.sd_yl = sd_yl


## ポテンシャル記述クラス
### 将来的にはこれを継承したユーザー定義クラスの使用を想定
### もしくはよく使うポテンシャルクラスを一通り実装しておく
class Potential:
    epsilon = 4.0
    rho     = 1.0

    def potential(self, r):
        v = self.epsilon * (self.rho/r**12 - self.rho/r**6)
        return v

# -----------------------------------------------------------------------
def periodic(proc):
    for i in range(len(proc.particles)):
        x, y = proc.Box.periodic_coordinate(proc.particles[i].x, proc.particles[i].y)
        proc.particles[i].x = x
        proc.particles[i].y = y
    return proc



def kinetic_energy(proc):
    k = 0
    for p in proc.subregion.particles:
        k += p.vx ** 2
        k += p.vy ** 2
    k /= 2
    return k



## ポテンシャルエネルギーの算出
### ペアリストを使用するので、事前に要構築
def potential_energy(proc):
    v = 0
    
    ### 自領域内粒子間ポテンシャル
    for pl in proc.subregion.pairlist:
        ip = proc.subregion.particles[pl.i]
        jp = proc.subregion.particles[pl.j]
        assert ip.id==pl.idi and jp.id==pl.idj, '参照している粒子IDが一致しません'
        assert ip.id != jp.id, '同一粒子ペアがリストに含まれています'
        r = proc.Box.periodic_distance(ip.x, ip.y, jp.x, jp.y)
        if r > proc.Box.cutoff:
            continue
        v += proc.Box.Potential.potential(r) - 4.0*(1/proc.Box.cutoff**12 - 1/proc.Box.cutoff**6) 
    
    ### 領域をまたいだポテンシャル
    for pl in proc.pairlist_between_neighbor:
        ip = proc.subregion.particles[pl.i]
        jp = proc.particles_in_neighbor[pl.j]
        assert ip.id==pl.idi and jp.id==pl.idj, '粒子IDが一致しません'
        assert ip.id != jp.id, '同一粒子ペアがリストに含まれています'
        r = proc.Box.periodic_distance(ip.x, ip.y, jp.x, jp.y)
        if r > proc.Box.cutoff:
            continue
        v += proc.Box.Potential.potential(r) - 4.0*(1/proc.Box.cutoff**12 - 1/proc.Box.cutoff**6) 
    
    return v

# ==============================================================