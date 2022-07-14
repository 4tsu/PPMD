# ===================================================
## シミュレーション本体の各過程をメソッドにまとめておく
# ===================================================
import particle
import box

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



def set_initial_velocity(proc):
    pass



def calculate_force(proc):
    pass



def update_position(proc):
    pass
# ---------------------------------------------------