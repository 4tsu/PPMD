# ====================================================
## 対象とする系、シミュレーションボックスについてのコード
## 系のエネルギーなどの観測コード
# ====================================================
import random

random.seed(1)
# ----------------------------------------------------
## シミュレーションボックスを定義するクラス
### すべてのプロセスで共有される最低限の情報を持つ
### lengthsは要素数2の配列[lx,ly]であり、ボックスの寸法を表す
class SimulationBox:
    def __init__(self, lengths, cutoff):
        self.xl = lengths[0]
        self.yl = lengths[1]
        self.x_max =  self.xl/2
        self.x_min = -self.xl/2
        self.y_max =  self.yl/2
        self.y_min = -self.yl/2
        
        self.cutoff = cutoff
        self.particles = []

        self.Potential = Potential()

    ## 周期境界条件の適用
    def periodic_coordinate(self, x, y):
        if x > self.x_max:
            x -= self.xl
        elif x < self.x_min:
            x += self.xl
        if y > self.y_max:
            y -= self.yl
        elif y < self.y_min:
            y += self.yl
        return x, y
    
    ## 周期境界条件を考慮した距離を返す
    def periodic_distance(self, x1, y1, x2, y2):
        rx = min((x1-x2)**2, (self.xl-abs(x1-x2))**2)
        ry = min((y1-y2)**2, (self.yl-abs(y1-y2))**2)
        return (rx + ry)**0.5
    
    def add_particle(self, particle):
        self.particles.append(particle)

    def kinetic_energy(self):
        k = 0
        for p in self.particles:
            k += p.vx ** 2
            k += p.vy ** 2
        k /= len(self.particles)
        k /= 2
        return k
    
    def potential_energy(self):
        v = 0
        for i in range(len(self.particles)-1):
            for j in range(i+1, len(self.particles)):
                ip = self.particles[i]
                jp = self.particles[j]
                r = self.periodic_distance(ip.x, ip.y, jp.x, jp.y)
                if r > self.cutoff:
                    continue
                v += self.Potential.potential(r) - 4.0*(1/self.cutoff**12 - 1/self.cutoff**6) 
        v /= len(self.particles)
        return v

## ポテンシャル記述クラス
### 将来的にはこれを継承したユーザー定義クラスの使用を想定
### もしくはよく使うポテンシャルクラスを一通り実装しておく
class Potential:
    epsilon = 4.0
    rho     = 1.0

    def potential(self, r):
        v = self.epsilon * (self.rho/r**12 - self.rho/r**6)
        return v

# ----------------------------------------------------
def periodic(Box):
    for i in range(len(Box.particles)):
        x, y = Box.periodic_coordinate(Box.particles[i].x, Box.particles[i].y)
        Box.particles[i].x = x
        Box.particles[i].y = y
    return Box

def export_cdview(Box, step):
    filename = 'conf{:0=4}.cdv'.format(step)
    with open(filename, 'w') as f:
        for i,p in enumerate(Box.particles):
            f.write('{} 0 {} {} 0\n'.format(p.id, p.x, p.y))

# ==============================================================