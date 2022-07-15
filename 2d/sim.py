# ===================================================
## シミュレーション本体の各過程をメソッドにまとめておく
# ===================================================
import particle
import box

from math import pi, sin, cos, ceil, sqrt
import random

random.seed(1)

# ---------------------------------------------------

def make_conf(N, Box):
    xl = Box.xl
    yl = Box.yl
    x_min = Box.x_min
    y_min = Box.y_min
    xppl = ceil(N/yl)
    yppl = ceil(N/xl)
    pitch = xl/xppl
    for i in range(N):
        iy = i//xppl
        ix = i%xppl
        x = ix*pitch
        y = iy*pitch
        Particle = particle.Particle(i,x,y)
        Box.particles.append(Particle)
    return Box


## 初速の大きさだけ受け取って、ランダムな方向に向ける
def set_initial_velocity(v0, Box):
    avx = 0.0
    avy = 0.0
    for i, p in enumerate(Box.particles):
        theta = random.random() * 2.0 * pi
        vx = v0 * cos(theta)
        vy = v0 * sin(theta)
        p.vx = vx
        p.vy = vy
        Box.particles[i] = p
        avx += vx
        avy += vy
    avx /= len(Box.particles)
    avy /= len(Box.particles)
    for i in range(len(Box.particles)):
        Box.particles[i].vx -= avx
        Box.particles[i].vy -= avy
    return Box



## 一方向についての、周期境界を考慮した距離
def periodic_od(L,dx):
    LH = L/2
    if dx < -LH:
        dx += L
    elif dx > LH:
        dx -= L
    return dx



## 力の計算
def calculate_force(Box, dt):
    for i in range(len(Box.particles)-1):
        for j in range(i+1, len(Box.particles)):
            ip = Box.particles[i]
            jp = Box.particles[j]
            r = Box.periodic_distance(ip.x, ip.y, jp.x, jp.y)
            if r > Box.cutoff:
                continue
            df = (24.0 * r**6 - 48.0) / r**14 * dt

            dx = periodic_od(Box.xl,ip.x-jp.x)
            dy = periodic_od(Box.yl,ip.y-jp.y)
            ip.vx += df * dx
            ip.vy += df * dy
            jp.vx -= df * dx
            jp.vy -= df * dy
            Box.particles[i] = ip
            Box.particles[j] = jp
    return Box





def update_position(Box):
    return Box
# ---------------------------------------------------