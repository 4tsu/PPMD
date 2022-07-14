# ===================================================
## シミュレーション本体の各過程をメソッドにまとめておく
# ===================================================
import particle
import box

from math import pi, sin, cos
import random

random.seed(1)

# ---------------------------------------------------

def make_conf(N, Box):
    xl = Box.xl
    yl = Box.yl
    x_min = Box.x_min
    y_min = Box.y_min
    for i in range(N):
        x = random.random()*xl - x_min
        y = random.random()*yl - y_min
        Particle = particle.Particle(i,x,y)
        Box.particles.append(Particle)
    return Box



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



def calculate_force(proc):
    pass



def update_position(proc):
    pass
# ---------------------------------------------------