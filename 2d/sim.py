# ===================================================
## シミュレーション本体の各過程をメソッドにまとめておく
# ===================================================
import particle
import box

from math import pi, sin, cos, ceil, sqrt
import random

random.seed(1)

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
    xppl = ceil(N/yl)
    yppl = ceil(N/xl)
    pitch = xl/xppl

    for i,p in enumerate(Machine.procs):
        for j in range(N):
            jy = j//xppl
            jx = j%xppl
            x = jx*pitch
            y = jy*pitch
            Particle = particle.Particle(j,x,y)
            Machine.procs[i].particles.append(Particle)
        
    return Machine


## 初速の大きさだけ受け取って、ランダムな方向に向ける
def set_initial_velocity(v0, Machine):
    avx = 0.0
    avy = 0.0
    for i, proc in enumerate(Machine.procs):
        for j, p in enumerate(proc.particles):
            theta = random.random() * 2.0 * pi
            vx = v0 * cos(theta)
            vy = v0 * sin(theta)
            p.vx = vx
            p.vy = vy
            proc.particles[j] = p
            avx += vx
            avy += vy
        Machine.procs[i] = proc
    avx /= Machine.procs[0].Box.N
    avy /= Machine.procs[0].Box.N
    for i, proc in enumerate(Machine.procs):
        for j in range(len(proc.particles)):
            proc.particles[j].vx -= avx
            proc.particles[j].vy -= avy
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
    for i in range(len(proc.particles)-1):
        for j in range(i+1, len(proc.particles)):
            ip = proc.particles[i]
            jp = proc.particles[j]
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
            proc.particles[i] = ip
            proc.particles[j] = jp
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

# ---------------------------------------------------