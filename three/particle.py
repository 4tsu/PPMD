# ===================================================
## シミュレーションを行う粒子・エージェントについてのコード
# ===================================================

# ---------------------------------------------------

class Particle:
    vx = 0
    vy = 0
    vz = 0

    def __init__(self, ID, x_init, y_init, z_init):
        self.id = ID
        self.x  = x_init
        self.y  = y_init
        self.z  = z_init
    
    def set_velocity(self, vx, vy, vz):
        self.vx = vx
        self.vy = vy
        self.vz = vz
