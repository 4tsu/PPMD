# ===================================================
## シミュレーション本体の各過程をメソッドにまとめておく
# ===================================================
import particle
import box
import sdd

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
class Meshlist:
    def __init__(self, Box):
        self.search_length = Box.cutoff + Box.margin
        self.num_mesh_x = Box.xl//self.search_length - 1
        self.mesh_size_x = Box.xl / self.num_mesh_x
        self.num_mesh_y = Box.yl//self.search_length - 1
        self.mesh_size_y = Box.yl / self.num_mesh_y
        assert self.num_mesh_x > 2, 'ペアリスト探索メッシュの数が適切ではありません'
        assert self.mesh_size_x > self.search_length, 'ペアリスト探索メッシュの大きさが適切ではありません'
        assert self.num_mesh_y > 2, 'ペアリスト探索メッシュの数が適切ではありません'
        assert self.mesh_size_y > self.search_length, 'ペアリスト探索メッシュの大きさが適切ではありません'
        self.num_mesh_total = int(self.num_mesh_x * self.num_mesh_y)
        self.counts = [0 for _ in range(self.num_mesh_total)]
        self.head_i = [0 for _ in range(self.num_mesh_total)]

    ### メッシュの周期境界補正
    def periodic_mesh(self, ix, iy):
        if ix < 0:
            ix += self.num_mesh_x
        if ix >= self.num_mesh_x:
            ix -= self.num_mesh_x
        if iy < 0:
            iy += self.num_mesh_y
        if iy >= self.num_mesh_y:
            iy -= self.num_mesh_y
        return ix, iy

    ### メッシュリストを使った粒子の住所録を作成
    def make_mesh(self, particles):
        particle_position = [0 for _ in range(len(particles))]
        for i,p in enumerate(particles):
            ix = p.x // self.mesh_size_x
            iy = p.y // self.mesh_size_y
            ix, iy = self.periodic_mesh(ix, iy)
            i_mesh = int(ix + iy * self.num_mesh_x)
            assert i_mesh >= 0, '割り当てられたメッシュ領域が不正です'
            assert i_mesh < self.num_mesh_total, '割り当てられたメッシュ領域が不正です'
            self.counts[i_mesh] += 1
            particle_position[i] = i_mesh

        ### 各メッシュ領域の先頭粒子インデックスの登録
        s = 0
        for i in range(self.num_mesh_total-1):
            s += self.counts[i]
            self.head_i[i+1] = s
            
        ### メッシュ順の粒子インデックスの作成
        self.sorted_particle_i = [0 for _ in range(len(particles))]
        pointer = [0 for _ in range(self.num_mesh_total)]
        for i in range(len(particles)):
            i_mesh = particle_position[i]
            j = self.head_i[i_mesh] + pointer[i_mesh]
            self.sorted_particle_i[j] = i
            pointer[i_mesh] += 1

    ### メッシュリストを使用したペア探索
    def search_pair(self, i_mesh, proc):
        ix = i_mesh%self.num_mesh_x
        iy = i_mesh//self.num_mesh_x

        ### 隣接するメッシュ領域との間で粒子ペア探索
        ### 作用反作用を考えて半分
        proc = self.search_other_region(i_mesh, ix+1, iy-1, proc)
        proc = self.search_other_region(i_mesh, ix+1, iy,   proc)
        proc = self.search_other_region(i_mesh, ix+1, iy+1, proc)
        proc = self.search_other_region(i_mesh, ix,   iy+1, proc)
        
        ### 自分の領域内の粒子ペアリスト作成
        for k in range(self.head_i[i_mesh], self.head_i[i_mesh]+self.counts[i_mesh]-1):
            for l in range(k+1, self.head_i[i_mesh]+self.counts[i_mesh]):
                i = self.sorted_particle_i[k]
                j = self.sorted_particle_i[l]
                ip = proc.particles[i]
                jp = proc.particles[j]
                r = proc.Box.periodic_distance(ip.x, ip.y, jp.x, jp.y)
                if r > proc.Box.co_p_margin:
                    continue
                P = Pair(i, j, ip.id, jp.id)
                proc.pairlist.append(P)
        ### ペアリストを追加して返す
        return proc

    def search_other_region(self, i_mesh, ix, iy, proc):
        ix, iy = self.periodic_mesh(ix, iy)
        j_mesh = int(ix + iy * self.num_mesh_x)
        for k in range(self.head_i[i_mesh], self.head_i[i_mesh]+self.counts[i_mesh]):
            for l in range(self.head_i[j_mesh], self.head_i[j_mesh]+self.counts[j_mesh]):
                i = self.sorted_particle_i[k]
                j = self.sorted_particle_i[l]
                ip = proc.particles[i]
                jp = proc.particles[j]
                r = proc.Box.periodic_distance(ip.x, ip.y, jp.x, jp.y)
                if r > proc.Box.co_p_margin:
                    continue
                P = Pair(i, j, ip.id, jp.id)
                proc.pairlist.append(P)
        return proc

    def make_pair(self, proc):
        self.make_mesh(proc.particles)
        for i in range(self.num_mesh_total):
            proc = self.search_pair(i, proc)
        return proc



# ---------------------------------------------------
## 初期配置を生成する
### 一様分布なので、最初から空間分割して配置したい
def make_conf(Machine):
    Box = Machine.procs[0].Box
    N = Box.N
    xl = Box.xl
    yl = Box.yl
    x_min = Box.x_min
    y_min = Box.y_min
    x_max = Box.x_max
    y_max = Box.y_max
    xppl = ceil(N/yl)
    yppl = ceil(N/xl)
    pitch = xl/xppl

    ### 等間隔分割の仕方を取得
    Box = sdd.get_simple_array(Box, len(Machine.procs))
    xn = Box.xn
    yn = Box.yn
    sd_xl = Box.sd_xl
    sd_yl = Box.sd_yl
    Machine.set_boxes(Box)

    for j in range(N):
        ### 粒子位置決める
        jy = j//xppl
        jx = j%xppl
        x = jx*pitch
        y = jy*pitch
        ### 粒子をどの領域=プロセスに分配する
        ip = int((y//sd_yl)*xn + x//sd_xl)
        ### 原点を中心にした座標系
        x += x_min
        y += y_min
        Particle = particle.Particle(j,x,y)
        Machine.procs[ip].subregion.particles.append(Particle)
        assert x_min <= x < x_max, '初期配置が適切ではありません'
        assert y_min <= y < y_max, '初期配置が適切ではありません'
    return Machine



## 初速の大きさだけ受け取って、ランダムな方向に向ける
def set_initial_velocity(v0, Machine):
    avx = 0.0
    avy = 0.0
    for i, proc in enumerate(Machine.procs):
        for j, p in enumerate(proc.subregion.particles):
            theta = random.random() * 2.0 * pi
            vx = v0 * cos(theta)
            vy = v0 * sin(theta)
            p.vx = vx
            p.vy = vy
            proc.subregion.particles[j] = p
            avx += vx
            avy += vy
        Machine.procs[i] = proc
    avx /= Machine.procs[0].Box.N
    avy /= Machine.procs[0].Box.N
    for i, proc in enumerate(Machine.procs):
        for j in range(len(proc.subregion.particles)):
            proc.subregion.particles[j].vx -= avx
            proc.subregion.particles[j].vy -= avy
    return Machine



## 一方向についての、周期境界を考慮した距離
def periodic_od(L,dx):
    LH = L/2
    if dx < -LH:
        dx += L
    elif dx > LH:
        dx -= L
    return dx



## 力の計算
def calculate_force(proc, dt):
    Box = proc.Box
    pl = proc.pairlist
    for k in range(len(pl)):
        ip = proc.particles[pl[k].i]
        jp = proc.particles[pl[k].j]
        assert ip.id == pl[k].idi, 'ペアリストIDと選択された粒子IDが一致しません'
        assert jp.id == pl[k].idj, 'ペアリストIDと選択された粒子IDが一致しません'
        r = Box.periodic_distance(ip.x, ip.y, jp.x, jp.y)
        if r > Box.cutoff:
            continue
        df = (24.0 * r**6 - 48.0) / r**14 * dt

        dx = periodic_od(Box.xl,jp.x-ip.x)
        dy = periodic_od(Box.yl,jp.y-ip.y)
        ip.vx += df * dx
        ip.vy += df * dy
        jp.vx -= df * dx
        jp.vy -= df * dy
        proc.particles[pl[k].i] = ip
        proc.particles[pl[k].j] = jp
    return proc



def update_position(proc, dt):
    for i,p in enumerate(proc.particles):
        p.x += p.vx * dt
        p.y += p.vy * dt
        proc.particles[i] = p
    return proc



## 可視化ソフトcdview用のダンプ出力
### 並列化のため上書き形式
def export_cdview(proc, step):
    filename = 'conf{:0=4}.cdv'.format(step)
    with open(filename, 'a') as f:
        for i,p in enumerate(proc.particles):
            f.write('{} 0 {} {} 0\n'.format(p.id, p.x, p.y))



## ペアリスト作成
def make_pair(proc):
    proc.pairlist.clear()
    proc.Box.set_mesh()
    proc = proc.Box.Meshlist.make_pair(proc)
    return proc



def check_pairlist(proc, dt):
    vmax2 = 0.0
    for p in proc.particles:
        v2 = p.vx**2 + p.vy**2
        if v2 > vmax2:
            vmax2 = v2
    vmax = sqrt(vmax2)
    proc.Box.subtract_margin(vmax*2.0*dt)
    if proc.Box.margin_life < 0.0:
        proc.Box.set_margin(proc.Box.margin)
        proc = make_pair(proc)
    return proc

# ---------------------------------------------------