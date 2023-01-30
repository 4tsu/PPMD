# ===================================================
## シミュレーション本体の各過程をメソッドにまとめておく
# ===================================================
from three import particle
from three import box
from three import sdd

from numba import jit, njit
from math import pi, sin, cos, ceil, sqrt
import random

random.seed(1)

# ---------------------------------------------------

class Pair:
    def __init__(self, i, j, idi, idj):
        self.i = i
        self.j = j
        self.idi = idi
        self.idj = idj



## ペアリストのメッシュ探索
### プロセス並列化しているので、空間分割領域を利用した探索
### プロセス数npに関してだけO(np^2)
### 各SubRegionのcenterとradiusが必要
class DomainPairList:
    def __init__(self, Machine):
        Box = Machine.procs[0].Box
        np = Machine.np
        self.search_length = Box.cutoff + Box.margin
        self.num_mesh_total = int(np)
        self.counts = [0 for _ in range(self.num_mesh_total)]
        self.head_i = [0 for _ in range(self.num_mesh_total)]
        self.radii = []
        self.centers = []
        self.ranks = []
        for i, proc in enumerate(Machine.procs):
            self.radii.append(proc.subregion.radius)
            self.centers.append(proc.subregion.center)
            self.ranks.append(proc.rank)
            assert i == proc.rank, "プロセスのインデックスとrankが一致しません"
        self.make_domian_pair_list(Box)

    ### 一個下の関数用
    def judge(self, i, j, box):
        if i == j:
            return True
        if not (self.centers[i] and self.centers[j]):
            return True
        diff = box.periodic_distance(self.centers[i][0], self.centers[i][1], self.centers[i][2], self.centers[j][0], self.centers[j][1], self.centers[j][2])
        if (diff - self.radii[i] - self.radii[j]) > box.co_p_margin:
            return True

    ## 相互作用する可能性のある領域ペア検出
    def make_domian_pair_list(self, box):
        ### O(np^2)で全探索
        ### 作用反作用を考慮して半分に落とすのを、計算が効率よくなるようにやる
        ### なるべくすべての領域で同じくらいの長さの領域ペアリストを持ちたい
        dpl = []

        for i in range(int(len(self.ranks)/2)):
            i_dpl = []
            for j in range(i+1, int(len(self.ranks)/2)+i+1):
                if self.judge(i,j,box):
                    continue
                i_dpl.append([i,j])
            dpl.append(i_dpl)

        for i in range(int(len(self.ranks)/2), len(self.ranks)-1):
            i_dpl = []
            for j in range(0, i-int(len(self.ranks)/2)):
                if self.judge(i,j,box):
                    continue
                i_dpl.append([i,j])
            for j in range(i+1, len(self.ranks)):
                if self.judge(i,j,box):
                    continue
                i_dpl.append([i,j])
            dpl.append(i_dpl)

        for i in range(len(self.ranks)-1, len(self.ranks)):
            i_dpl = []
            for j in range(0, i-int(len(self.ranks)/2)):
                if self.judge(i,j,box):
                    continue
                i_dpl.append([i,j])
            dpl.append(i_dpl)

        self.list = dpl
        # print(dpl)



    ### リストを使用したペア探索
    ### 実行前にdomain_pair_listに基づいた通信をし、自プロセスに周辺粒子の情報を持ってくること。
    def search_pair(self, proc):
        rank = proc.rank
        ### 他の領域と相互作用する可能性があるようだったら…
        if len(self.list[rank]) != 0:
            ### 隣接する領域との間で粒子ペア探索
            ### 密度分布差が大きいときは領域粒子がゼロでも領域ペアリストにあってよい
            # assert len(proc.particles_in_neighbor) != 0, "周辺粒子情報がプロセスに登録されていません"
            proc = self.search_other_region(proc)
        ### なければクリア
        else:
            proc.pairlist_between_neighbor.clear()
        
        ### 自分の領域内の粒子ペアリスト作成
        pairlist = []
        box = proc.Box
        particles = proc.subregion.particles
        for i in range(len(particles)-1):
            for j in range(i+1, len(particles)):
                ip = particles[i]
                jp = particles[j]
                r = box.periodic_distance(ip.x, ip.y, ip.z, jp.x, jp.y, jp.z)
                if r > box.co_p_margin:
                    continue
                P = Pair(i, j, ip.id, jp.id)
                pairlist.append(P)
        ### ペアリストを追加して返す
        proc.subregion.pairlist.clear()
        proc.subregion.pairlist = pairlist
        return proc

    def search_other_region(self, proc):
        pairlist = []
        box = proc.Box
        my_particles = proc.subregion.particles
        other_particles = proc.particles_in_neighbor
        for i in range(len(my_particles)):
            for j in range(len(other_particles)):
                ip = my_particles[i]
                jp = other_particles[j]   ### jが他領域の粒子
                r = box.periodic_distance(ip.x, ip.y, ip.z, jp.x, jp.y, jp.z)
                if r > box.co_p_margin:
                    continue
                P = Pair(i, j, ip.id, jp.id)
                pairlist.append(P)
        proc.pairlist_between_neighbor.clear()
        proc.pairlist_between_neighbor = pairlist
        return proc



# ---------------------------------------------------
## 初期配置を生成する
### 一様分布なので、最初から空間分割して配置したい
def make_conf(Machine):
    Box = Machine.procs[0].Box
    N = Box.N
    xl = Box.xl   ### 液滴を作りたいときは、ここの値を減らす
    yl = Box.yl
    zl = Box.zl
    x_min = Box.x_min
    y_min = Box.y_min
    z_min = Box.z_min
    x_max = Box.x_max
    y_max = Box.y_max
    z_max = Box.z_max
    xppl = ceil((xl**2*N/(yl*zl))**(1/3))
    yppl = ceil((yl**2*N/(xl*zl))**(1/3))
    zppl = ceil((zl**2*N/(xl*yl))**(1/3))
    pitch = min(xl/xppl, yl/yppl, zl/zppl)

    ### 等間隔分割の仕方を取得
    Box = sdd.get_simple_array(Box, len(Machine.procs))
    xn = Box.xn
    yn = Box.yn
    zn = Box.zn
    sd_xl = Box.sd_xl
    sd_yl = Box.sd_yl
    sd_zl = Box.sd_zl
    Machine.set_boxes(Box)

    for j in range(N):
        ### 粒子位置決める
        jz = j//(xppl*yppl)
        jy = (j%(xppl*yppl))//xppl
        jx = (j%(xppl*yppl))%xppl
        x = jx*pitch
        y = jy*pitch
        z = jz*pitch
        ### 粒子をどの領域=プロセスに分配する
        ip = int((z//sd_zl)*(xn*yn) + (y//sd_yl)*xn + x//sd_xl)
        ### 原点を中心にした座標系
        x += x_min
        y += y_min
        z += z_min
        p = particle.Particle(j,x,y,z)
        Machine.procs[ip].subregion.particles.append(p)
        assert x_min <= x < x_max, '初期配置が適切ではありません'
        assert y_min <= y < y_max, '初期配置が適切ではありません'
        assert z_min <= z < z_max, '初期配置が適切ではありません'
    for i,proc in enumerate(Machine.procs):
        left   = (i%yn) * sd_xl + x_min
        right  = (i%yn + 1) * sd_xl + x_min
        front  = ((i%(xn*yn))//yn) * sd_yl + y_min
        back   = ((i%(xn*yn))//yn + 1) * sd_yl + y_min
        top    = (i//(xn*yn) + 1) * sd_zl + z_min
        bottom = i//(xn*yn) * sd_zl + z_min
        Machine.procs[i].subregion.set_limit(top, bottom, right, left, front, back)
    return Machine



## 初速の大きさだけ受け取って、ランダムな方向に向ける
def set_initial_velocity(v0, Machine, xmove=0):
    avx = 0.0
    avy = 0.0
    avz = 0.0
    for i, proc in enumerate(Machine.procs):
        for j, p in enumerate(proc.subregion.particles):
            xi    = random.random() * 1.0 * pi
            theta = random.random() * 2.0 * pi
            vx = v0 * cos(theta) * sin(xi)
            vy = v0 * sin(theta) * sin(xi)
            vz = v0 * cos(xi)
            p.vx = vx
            p.vy = vy
            p.vz = vz
            proc.subregion.particles[j] = p
            avx += vx
            avy += vy
            avz += vz
        Machine.procs[i] = proc
    avx /= Machine.procs[0].Box.N
    avy /= Machine.procs[0].Box.N
    avz /= Machine.procs[0].Box.N
    for i, proc in enumerate(Machine.procs):
        for j in range(len(proc.subregion.particles)):
            proc.subregion.particles[j].vx -= avx
            proc.subregion.particles[j].vy -= avy
            proc.subregion.particles[j].vz -= avz
            proc.subregion.particles[j].vx += xmove
    return Machine



## 一方向についての、周期境界を考慮した距離
@njit
def periodic_od(L,dx):
    LH = L/2
    if dx < -LH:
        dx += L
    elif dx > LH:
        dx -= L
    return dx



## 力の計算
def calculate_force(proc, dt):
    proc.start_watch()
    Box = proc.Box

    ### 自領域内での力積計算
    my_particles = proc.subregion.particles
    for pair in proc.subregion.pairlist:
        ip = my_particles[pair.i]
        jp = my_particles[pair.j]
        assert ip.id==pair.idi and jp.id==pair.idj, 'ペアリストIDと選択された粒子IDが一致しません'
        r = Box.periodic_distance(ip.x, ip.y, ip.z, jp.x, jp.y, jp.z)
        if r > Box.cutoff:
            continue
        df = (24.0 * r**6 - 48.0) / r**14 * dt

        dfdx = periodic_od(Box.xl,jp.x-ip.x) * df
        dfdy = periodic_od(Box.yl,jp.y-ip.y) * df
        dfdz = periodic_od(Box.zl,jp.z-ip.z) * df
        ip.vx += dfdx
        ip.vy += dfdy
        ip.vz += dfdz
        jp.vx -= dfdx
        jp.vy -= dfdy
        jp.vz -= dfdz
        my_particles[pair.i] = ip
        my_particles[pair.j] = jp
    
    ### 領域をまたいだ力積計算
    other_particles = proc.particles_in_neighbor
    sending_velocities = []
    for pair in proc.pairlist_between_neighbor:
        ip = my_particles[pair.i]
        jp = other_particles[pair.j]
        assert ip.id==pair.idi and jp.id==pair.idj, 'ペアリストIDと選択された粒子IDが一致しません'
        r = Box.periodic_distance(ip.x, ip.y, ip.z, jp.x, jp.y, jp.z)
        if r > Box.cutoff:
            continue
        df = (24.0 * r**6 - 48.0) / r**14 * dt

        dfdx = periodic_od(Box.xl,jp.x-ip.x) * df
        dfdy = periodic_od(Box.yl,jp.y-ip.y) * df
        dfdz = periodic_od(Box.zl,jp.z-ip.z) * df
        ip.vx += dfdx
        ip.vy += dfdy
        ip.vz += dfdz
        ### 他領域の粒子の力積は追加分しか記録しない
        kp = particle.Particle(jp.id, jp.x, jp.y, jp.z)
        kp.vx = -dfdx
        kp.vy = -dfdy
        kp.vz = -dfdz
        my_particles[pair.i] = ip
        kp.set_process(jp.proc)
        sending_velocities.append(kp)

    proc.subregion.particles = my_particles
    proc.set_sending_velocity(sending_velocities)

    proc.stop_watch()
    return proc



def update_position(proc, dt):
    proc.start_watch()
    Box = proc.Box
    for i,p in enumerate(proc.subregion.particles):
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.z += p.vz * dt
        p.x = box.periodic(p.x, Box.xl, Box.x_min, Box.x_max)
        p.y = box.periodic(p.y, Box.yl, Box.y_min, Box.y_max)
        p.z = box.periodic(p.z, Box.zl, Box.z_min, Box.z_max)
        proc.subregion.particles[i] = p
    
    proc.stop_watch()
    return proc



## 可視化ソフトcdview用のダンプ出力
### 並列化のため上書き形式
def export_cdview(proc, step, head=False):
    filename = 'conf{:0=6}.cdv'.format(step)
    if head:
        with open(filename, 'a') as f:
            f.write('#box_sx={}\n'.format(proc.Box.x_min))
            f.write('#box_ex={}\n'.format(proc.Box.x_max))
            f.write('#box_sy={}\n'.format(proc.Box.y_min))
            f.write('#box_ey={}\n'.format(proc.Box.y_max))
            f.write('#box_sz={}\n'.format(proc.Box.z_min))
            f.write('#box_ez={}\n'.format(proc.Box.z_max))
    else:
        rank = proc.rank
        with open(filename, 'a') as f:
            for i,p in enumerate(proc.subregion.particles):
                f.write('{} {} {} {} {}\n'.format(p.id, rank, p.x, p.y, p.z))



## ペアリスト作成
def make_pair(Machine):
    for i,proc in enumerate(Machine.procs):
        Machine.procs[i].subregion.calc_center(proc.Box)
        Machine.procs[i].subregion.calc_radius(proc.Box)
    dpl = DomainPairList(Machine)
    for i,proc in enumerate(Machine.procs):
        proc.set_domain_pair_list(dpl)
    Machine.communicate_particles()
    for i,proc in enumerate(Machine.procs):
        Machine.procs[i] = proc.domain_pair_list.search_pair(proc)
    return Machine



def check_vmax(proc):
    vmax2 = 0.0
    for p in proc.subregion.particles:
        v2 = p.vx**2 + p.vy**2
        if v2 > vmax2:
            vmax2 = v2
    vmax = sqrt(vmax2)
    return vmax



def check_pairlist(Machine, vmax, dt):
    box = Machine.procs[0].Box
    box.subtract_margin(vmax*2.0*dt)
    if box.margin_life < 0.0:
        box.set_margin(box.margin)
        Machine.set_boxes(box)
        return Machine, True
    else:
        Machine.set_boxes(box)
        return Machine, False

# ---------------------------------------------------