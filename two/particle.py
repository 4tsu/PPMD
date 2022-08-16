# ===================================================
## シミュレーションを行う粒子・エージェントについてのコード
# ===================================================

# ---------------------------------------------------

class Particle:
    vx = 0
    vy = 0

    def __init__(self, ID, x_init, y_init):
        self.id = ID
        self.x  = x_init
        self.y  = y_init
    
    def set_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy
