import numpy as np

class Atom:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.type = 1
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

# 液滴の配置を作る
def add_droplet(atoms, offset=0):
    # r = 20
    r = 16
    s = 3.0   # 原子間距離
    h = 0.5 * s
    for ix in range(-r, r+1):
        for iy in range(-r, r+1):
            x = ix * s
            y = iy * s
            z = 0
            if (x**2 + y**2 + z**2 > r**2):     # 原子を球状に配置するため
                continue
            atoms.append(Atom(x+offset, y+offset, z))   # 原子を配置



# 泡の形を作る
def add_bubble(atoms, rho):
    r =  0.18   # シミュレーションボックスの一辺長を0.9としたときの球の半径
    M = 16             # 格子の数を最初に指定
    s = (0.9/M)**1.0   # 格子間距離のルート2倍
    h = 0.5 * s
    # 周期境界条件により，ぎっちり詰めると端っこ同士がくっついて吹っ飛ぶ
    ix = -0.45
    while(ix < 0.45):   
        iy = -0.45
        while(iy < 0.45):
            x = ix
            y = iy
            z = 0
            if (x**2 + y**2 + z**2 < r**2):   # 原子を球の外側のみに配置するため
                iy += 0.9/M
                continue
            atoms.append(Atom(x, y, z))   # まっすぐに原子を配置
            iy += 0.9/M
        ix += 0.9/M

    N = len(atoms)
    L = (N/rho)**(1/2)
    for i, a in enumerate(atoms):
        a.x *= L
        a.y *= L
        a.z *= L



# 一様な液体の原子配置を読み込んで，急減圧して返す
def rapid_decomp(atoms, rate, path, output, orgL):
    with open(path) as f:
        Line = [s.strip() for s in f.readlines()]
        X = []
        Y = []
        Z = []
        VX = []
        VY = []
        VZ = []
        for l in range(len(Line)):
            id = ''
            x = ''
            y = ''
            z = ''
            vx = ''
            vy = ''
            vz = ''
            index = 0
            for s in Line[l]:
                if s == ' ':
                    index += 1
                    continue
                if index == 0:
                    x += s
                elif index == 1:
                    y += s
                elif index == 2:
                    z += s
                elif index == 3:
                    vx += s
                elif index == 4:
                    vy += s
                elif index == 5:
                    vz += s
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
            VX.append(float(vx))
            VY.append(float(vy))
            VZ.append(float(vz))

    for i in range(len(X)):
        a = Atom(X[i]*rate, Y[i]*rate, Z[i]*rate)
        a.vx = VX[i]
        a.vy = VY[i]
        a.vz = VZ[i]
        atoms.append(a)

    print("# of atoms",len(atoms))
    print("rho = ", rho)
    with open(output, "w") as f:
        f.write("Position Data\n\n")
        f.write("{} atoms\n".format(len(atoms)))
        f.write("1 atom types\n\n")
        f.write("0 {} xlo xhi\n".format(orgL*rate))
        f.write("0 {} ylo yhi\n".format(orgL*rate))
        f.write("0 {} zlo zhi\n".format(orgL*rate))
        f.write("\n")
        f.write("Atoms\n\n")
        for i, a in enumerate(atoms):
            f.write("{} {} {} {} {}\n".format(i+1, a.type, a.x, a.y, a.z))
        f.write("\n")
        f.write("Velocities\n\n")
        for i, a in enumerate(atoms):
            f.write("{} {} {} {}\n".format(i+1, a.vx, a.vy, a.vz))
    print("Generated {}".format(output))



def save_file(filename, atoms, rho):
    # 密度と原子数から系の大きさを決定
    N = len(atoms)
    L = (N/rho)**(1/2)
    print("# of atoms",len(atoms))
    print("rho = ", rho)
    with open(filename, "w") as f:
        f.write("Position Data\n\n")
        f.write("{} atoms\n".format(len(atoms)))
        f.write("1 atom types\n\n")
        f.write("-{0} {0} xlo xhi\n".format(L/2))
        f.write("-{0} {0} ylo yhi\n".format(L/2))
        f.write("-{0} {0} zlo zhi\n".format(L/2))
        f.write("\n")
        f.write("Atoms\n\n")
        for i, a in enumerate(atoms):
            f.write("{} {} {} {} {}\n".format(i+1, a.type, a.x, a.y, a.z))
        f.write("\n")
        f.write("Velocities\n\n")
        for i, a in enumerate(atoms):
            f.write("{} {} {} {}\n".format(i+1, a.vx, a.vy, a.vz))
    print("Generated {}".format(filename))

atoms = []
# rho = 0.05
# add_droplet(atoms)
# save_file("droplet.atoms", atoms, rho)

# rho = 0.7
# add_bubble(atoms, rho)
# save_file("bubble.atoms", atoms, rho)

rho = 0.05
add_droplet(atoms, offset=-16)
add_droplet(atoms, offset=16)
save_file("droplet.atoms", atoms, rho)

# rate = (0.65/0.575)**(1/3)   # 密度の変化を長さの次元に．
# orgL = 29.320330466373726
# rapid_decomp(atoms, rate, 'liquid.dump', "decompression.atoms", orgL)
