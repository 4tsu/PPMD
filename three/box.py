# ====================================================
## 対象とする系、シミュレーションボックスについてのコード
## 系のエネルギーなどの観測コード
# ====================================================
from three import sim
from three import particle
from three import sdd

from numba import jit, njit
import random

random.seed(1)
# ----------------------------------------------------
## シミュレーションボックスを定義するクラス
### すべてのプロセスで共有される最低限の情報を持つ
### lengthsは要素数3の配列[lx,ly,lz]であり、ボックスの寸法を表す
class SimulationBox:
    def __init__(self, lengths, cutoff, N):
        self.xl = lengths[0]
        self.yl = lengths[1]
        self.zl = lengths[2]
        self.x_max =  self.xl/2
        self.x_min = -self.xl/2
        self.y_max =  self.yl/2
        self.y_min = -self.yl/2
        self.z_max =  self.zl/2
        self.z_min = -self.zl/2
        
        self.cutoff = cutoff
        self.N = N

        self.Potential = Potential()



    ## 周期境界条件の適用
    def periodic_coordinate(self, x, y, z):
        if not (self.x_min <= x <= self.x_max):
            while not (self.x_min <= x <= self.x_max):
                x = modify(x, self.x_max, self.x_min, self.xl)
                # print(x)
        
        if not (self.y_min <= y <= self.y_max):
            while not (self.y_min <= y <= self.y_max):
                y = modify(y, self.y_max, self.y_min, self.yl)
        
        if not (self.z_min <= z <= self.z_max):
            while not (self.z_min <= z <= self.z_max):
                z = modify(z, self.z_max, self.z_min, self.zl)
        
        assert self.x_min<=x<=self.x_max and self.y_min<=y<=self.y_max and self.z_min<=z<=self.z_max, '周期境界補正に失敗しています'
        return x, y, z
    
    ## 周期境界条件を考慮した距離を返す
    def periodic_distance(self, x1, y1, z1, x2, y2, z2):
        rx,ry,rz = periodic_d(x1,x2,y1,y2,z1,z2,self.xl,self.yl,self.zl)
        return (rx + ry + rz)**0.5
    
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
    def set_subregion(self, xn, yn, zn, sd_xl, sd_yl, sd_zl):
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.sd_xl = sd_xl
        self.sd_yl = sd_yl
        self.sd_zl = sd_zl



## ポテンシャル記述クラス
class Potential:
    epsilon = 4.0
    rho     = 1.0

    def potential(self, r):
        v = self.epsilon * (self.rho/r**12 - self.rho/r**6)
        return v

# -----------------------------------------------------------------------

@njit("Tuple((f8,f8,f8))(f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def periodic_d(x1,x2,y1,y2,z1,z2,xl,yl,zl):
    rx = min((x1-x2)**2, (xl-abs(x1-x2))**2)
    ry = min((y1-y2)**2, (yl-abs(y1-y2))**2)
    rz = min((z1-z2)**2, (zl-abs(z1-z2))**2)
    return rx, ry, rz



@njit("f8(f8,f8,f8,f8)")
def periodic(x, xl, x_min, x_max):
    if not (x_min <= x <= x_max):
        if x > x_max:
            x -= xl
        elif x < x_min:
            x += xl
    assert x_min<=x<=x_max, '周期境界補正に失敗しています'
    return x


## periodic_coordinate用
@njit("f8(f8,f8,f8,f8)")
def modify(x, x_max, x_min, L):
    if x > x_max:
        x -= L
    elif x < x_min:
        x += L
    return x



def kinetic_energy(proc):
    k = 0
    for p in proc.subregion.particles:
        k += p.vx ** 2
        k += p.vy ** 2
        k += p.vz ** 2
    k /= 2
    return k


## 下で使う用
@njit("f8(f8,f8,f8,f8)")
def potential_one_pair(epsilon, rho, cutoff, r):
    v = epsilon * (rho/r**12 - rho/r**6) - 4.0*(1/cutoff**12 - 1/cutoff**6) 
    return v

## ポテンシャルエネルギーの算出
### ペアリストを使用するので、事前に要構築
def potential_energy(proc):
    epsilon = proc.Box.Potential.epsilon
    rho     = proc.Box.Potential.rho
    cutoff  = proc.Box.cutoff
    v = 0
    
    ### 自領域内粒子間ポテンシャル
    for pl in proc.subregion.pairlist:
        ip = proc.subregion.particles[pl.i]
        jp = proc.subregion.particles[pl.j]
        assert ip.id==pl.idi and jp.id==pl.idj, '参照している粒子IDが一致しません'
        r = proc.Box.periodic_distance(ip.x, ip.y, ip.z, jp.x, jp.y, jp.z)
        if r > proc.Box.cutoff:
            continue
        v += potential_one_pair(epsilon, rho, cutoff, r)

    ### 領域をまたいだポテンシャル
    for pl in proc.pairlist_between_neighbor:
        ip = proc.subregion.particles[pl.i]
        jp = proc.particles_in_neighbor[pl.j]
        assert ip.id==pl.idi and jp.id==pl.idj, '粒子IDが一致しません'
        r = proc.Box.periodic_distance(ip.x, ip.y, ip.z, jp.x, jp.y, jp.z)
        if r > proc.Box.cutoff:
            continue
        v += potential_one_pair(epsilon, rho, cutoff, r)

    return v



### LAMMPSの.dump形式のファイルを読み込み
def read_lammps(Machine, filename):
    ### box情報の読み込み
    x_min = ''
    x_max = ''
    y_min = ''
    y_max = ''
    z_min = ''
    z_max = ''
    N = ''
    with open(filename) as f:
        Line = [s.strip() for s in f.readlines()]
        for i,l in enumerate(Line):
            if i == 3:
                index = 0
                for s in l:
                    if s == ' ':
                        index += 1
                        continue
                    if index == 0:
                        N += s
                    else:
                        break
                continue

            if i == 5:
                index = 0
                for s in l:
                    if s == ' ':
                        index += 1
                        continue
                    if index == 0:
                        x_min += s
                    elif index == 1:
                        x_max += s
                    else:
                        break
                continue

            if i == 6:
                index = 0
                for s in l:
                    if s == ' ':
                        index += 1
                        continue
                    if index == 0:
                        y_min += s
                    elif index == 1:
                        y_max += s
                    else:
                        break
            
            if i == 7:
                index = 0
                for s in l:
                    if s == ' ':
                        index += 1
                        continue
                    if index == 0:
                        z_min += s
                    elif index == 1:
                        z_max += s
                    else:
                        break
                break

    ### 系の情報を入力に合わせて変更
    box = Machine.procs[0].Box
    x_min = float(x_min)
    y_min = float(y_min)
    z_min = float(z_min)
    x_max = float(x_max)
    y_max = float(y_max)
    z_max = float(z_max)
    xl = x_max - x_min
    yl = y_max - y_min
    zl = z_max - z_min

    box.xl = xl
    box.yl = yl
    box.zl = zl
    box.x_min = x_min
    box.y_min = y_min
    box.z_min = z_min
    box.x_max = x_max
    box.y_max = y_max
    box.z_max = z_max
    box.N = float(N)
    
    ### 等間隔分割の仕方を取得
    box = sdd.get_simple_array(box, len(Machine.procs))
    xn = box.xn
    yn = box.yn
    zn = box.zn
    sd_xl = box.sd_xl
    sd_yl = box.sd_yl
    sd_zl = box.sd_zl
    Machine.set_boxes(box)

    ### 粒子情報の読み込み
    with open(filename) as f:
        Line = [s.strip() for s in f.readlines()]
        is_last = 0
        for l in range(0,len(Line)):
            if Line[l] == 'ITEM: ATOMS id x y z vx vy vz':
                is_last += 1
                continue
            if is_last != 2:
                continue
            ids = ''
            xs  = ''
            ys  = ''
            zs  = ''
            vxs = ''
            vys = ''
            vzs = ''
            index = 0
            for s in Line[l]:
                if s == ' ':
                    index += 1
                    continue
                if index == 0:
                    ids += s
                elif index == 1:
                    xs  += s
                elif index == 2:
                    ys  += s
                elif index == 3:
                    zs  += s
                elif index == 4:
                    vxs += s
                elif index == 5:
                    vys += s
                elif index == 6:
                    vzs += s
            
            ### 原点が端っこになる座標系
            x = float(xs) - x_min
            y = float(ys) - y_min
            z = float(zs) - z_min
            
            ### 粒子をどの領域=プロセスに分配するか
            ip = int((z//sd_zl)*(xn*yn) + (y//sd_yl)*xn + x//sd_xl)

            ### 原点を中心にした座標系
            x += x_min
            y += y_min
            z += z_min

            x = modify(x, x_max, x_min, xl)
            y = modify(y, y_max, y_min, yl)
            z = modify(z, z_max, z_min, zl)

            p = particle.Particle(int(ids), x, y, z)
            p.set_velocity(float(vxs), float(vys), float(vzs))

            Machine.procs[ip].subregion.particles.append(p)
            assert x_min <= x <= x_max, '初期配置が適切ではありません x={}'.format(x)
            assert y_min <= y <= y_max, '初期配置が適切ではありません y={}'.format(y)
            assert z_min <= z <= z_max, '初期配置が適切ではありません z={}'.format(z)
    
    assert max(Machine.count())!=0, '粒子情報が正しく読み込まれていません'
    
    for i,proc in enumerate(Machine.procs):
        left   = (i%yn) * sd_xl + x_min
        right  = (i%yn + 1) * sd_xl + x_min
        front  = ((i%(xn*yn))//yn) * sd_yl + y_min
        back   = ((i%(xn*yn))//yn + 1) * sd_yl + y_min
        top    = (i//(xn*yn) + 1) * sd_zl + z_min
        bottom = i//(xn*yn) * sd_zl + z_min
        Machine.procs[i].subregion.set_limit(top, bottom, right, left, front, back)
   
    return Machine

# ==============================================================