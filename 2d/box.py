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
        xl = lengths[0]
        yl = lengths[1]
        x_max =  xl/2
        x_min = -xl/2
        y_max =  yl/2
        y_min = -yl/2
        self.cutoff = cutoff

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
        rx = min((x1-x2)**2, (xl-abs(x1-x2))**2)
        ry = min((y1-y2)**2, (yl-abs(y1-y2))**2)
        return (rx + ry)**0.5
    
# ----------------------------------------------------
def periodic(proc):
    pass

def export(proc):
    pass

def kinetic_energy():
    pass

def potential_energy():
    pass
