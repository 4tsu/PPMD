# =================================================
## シミュレーション実行環境のシミュレート用コード
# =================================================
from os import stat_result
import random
from re import X
from matplotlib.pyplot import phase_spectrum
import numpy as np

# --------------------------------------------------------------------

class Process:
    def __init__(self, rank):
        self.rank = rank
        self.packets = []   ### 通信時に送信する情報を詰めておく
        self.receivebox = []   ### 通信時に受信するポスト
        self.communicatable = []   ### ハードウェア上で隣接するコア
        self.subdomain = SubDomain()

    def set_box(self, Box):
        self.Box = Box

    def receive(self):
        received_list = []
        for i,pckt in enumerate(self.receivebox):
            if pckt.direction[0] == 0 and pckt.direction[1] == 0:
                # print('data #', i, 'received')
                received_list.append(i)
        self.packets = [self.receivebox[i] for i in range(len(self.receivebox)) if i not in received_list]
        self.receivebox.clear()



class SubDomain:
    def __init__(self):
        self.particles = []   ### 所属粒子
        self.pairlist = []   ### 粒子ペアリスト
        self.neighbors = []   ### シミュレーション上で隣接するプロセッサ
        self.center = []   ### 領域中心
        self.boundaries = []   ### 領域境界

    def calc_center(self, xi, yi):
        Q = self.particles
        if not len(Q) == 0:
            sx = 0
            sy = 0
            for j in range(len(Q)):
                sx += Q[j][xi]
                sy += Q[j][yi]
            sx /= len(Q)
            sy /= len(Q)
            sx, sy = periodic_coordinate(sx, sy)
            assert 1>sx>0 and 1>sy>0, 'center value is out of range!'
            self.center.clear()
            self.center.append(sx)
            self.center.append(sy)

    def calc_perimeter(self):
        prmtr = 0
        C = []
        B = self.boundaries
        for j in range(len(B)):
            a1 = B[j-1][0]
            b1 = B[j-1][1]
            c1 = B[j-1][2]
            a2 = B[j][0]
            b2 = B[j][1]
            c2 = B[j][2]
            x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
            y = (c1*a2 - c2*a1) / (a1*b2 - a2*b1)
            C.append([x, y])
        for j in range(len(C)):
            prmtr += ((C[j][0]-C[j-1][0])**2 + (C[j][1]-C[j-1][1])**2)**0.5
        self.perimeter = prmtr



class Packet:
    def __init__(self):
        self.passengers = []
        self.origin = -1
        self.destination = -1
        self.current = -1
        self.direction = []

    def get_num_passengers(self):
        return len(self.passengers)

    def send(self):
        h = self.direction[0]
        v = self.direction[1]
        if h >= 0 and v > 0:
            step = [0,1]
        elif h > 0 and v <= 0:
            step = [1,0]
        elif h <= 0 and v < 0:
            step = [0,-1]
        elif h < 0 and v >= 0:
            step = [-1,0]
        return step

    def move(self, cell, step):
        crnt = self.current
        if crnt%cell[0]==0 and step[0]==-1:
            next = crnt + cell[0] - 1
        elif crnt%cell[0]==cell[0]-1 and step[0]==1:
            next = crnt - cell[0] + 1
        elif crnt//cell[0]==cell[1]-1 and step[1]==1:
            next = crnt - cell[0]*(cell[1]-1)
        elif crnt//cell[0]==0 and step[1]==-1:
            next = crnt + cell[0]*(cell[1]-1)
        else:
            next = crnt + step[0] + step[1]*cell[0]
        self.current = next
        self.direction[0] -= step[0]
        self.direction[1] -= step[1]



class Machine:
    def __init__(self, num_of_procs):
        self.procs = []
        for i in range(num_of_procs):
            P = Process(i)
            self.procs.append(P)

    ### すべてのプロセッサでシミュレーションボックスのグローバルな設定を共有する
    def set_boxes(self, Box):
        for i in range(len(self.procs)):
            self.procs[i].set_box(Box)
    


    ## 周期境界を考えた隣接セル検出
    def detect_neighbors(self, cutoff):
        for i,p in enumerate(self.procs):
            self.procs[i].SubDomain.neighbors.clear()
            pr = random.sample(p.particles, len(p.particles))
            for j,q in enumerate(self.procs):
                if i == j:
                    continue
                flag = False
                qr = random.sample(q.particles, len(q.particles))
                for m in pr:
                    for n in qr:
                        r = periodic_distance(m[0], m[1], n[0], n[1])
                        if r < cutoff:
                            self.procs[i].SubDomain.neighbors.append(j)
                            flag = True
                            break
                    if flag == True:
                        break
    
    ## 周期境界を考えない隣接セル検出
    def detect_adjacent(self, cutoff):
        for i,p in enumerate(self.procs):
            self.procs[i].SubDomain.neighbors.clear()
            pr = random.sample(p.particles, len(p.particles))
            for j,q in enumerate(self.procs):
                if i == j:
                    continue
                flag = False
                qr = random.sample(q.particles, len(q.particles))
                for m in pr:
                    for n in qr:
                        r = ((m[0]-n[0])**2 + (m[1]-n[1])**2)**0.5
                        if r < cutoff:
                            self.procs[i].SubDomain.neighbors.append(j)
                            flag = True
                            break
                    if flag == True:
                        break

    def calc_direction(self, origin, destination):
        cell = self.cell
        xo = origin%cell[0]
        yo = origin//cell[0]
        xd = destination%cell[0]
        yd = destination//cell[0]
        rx = xd-xo
        if abs(rx + cell[0]) < abs(rx):
            rx = rx + cell[0]
        elif abs(rx - cell[0]) < abs(rx):
            rx = rx - cell[0]
        else:
            pass
        ry = yd-yo
        if abs(ry + cell[1]) < abs(ry):
            ry = ry + cell[1]
        elif abs(ry - cell[1]) < abs(ry):
            ry = ry - cell[1]
        else:
            pass
        return [rx, ry]

    # not for particles move across processors
    # for transmit particles' position to other processors
    def packing(self, cutoff):
        for i,p in enumerate(self.procs):
            for j in p.neighbors:
                q = self.procs[j]
                packlist = []
                for k,m in enumerate(p.particles):
                    for n in q.particles:
                        r = periodic_distance(m[0], m[1], n[0], n[1])
                        if r < cutoff:
                            packlist.append(k)
                            break
                pckt = Packet()
                pckt.passengers = packlist
                pckt.origin = i
                pckt.destination = j
                pckt.current = i
                pckt.direction = self.calc_direction(i,j)
                self.procs[i].SubDomain.packets.append(pckt)

    def calc_all_center(self, across_border=False):
        if across_border:
            xi = 3
            yi = 4
        else:
            xi = 0
            yi = 1
        for p in self.procs:
            p.calc_center(xi, yi)

    # this process is not faithful to actual operation
    def reallocate(self):
        D = data
        for i,p in enumerate(self.procs):
            self.procs[i].SubDomain.particles.clear()
            B = p.boundaries
            M = []
            ctr = p.center
            for j in range(len(D[0])):
                flag = True
                x = D[0][j]
                y = D[1][j]
                for k in B:
                    if (k[0]*x+k[1]*y+k[2])*(k[0]*ctr[0]+k[1]*ctr[1]+k[2]) < 0:
                        flag = False
                        break
                if flag == True:
                    M.append(j)
                    self.procs[i].SubDomain.particles.append([x, y, D[2][j]])
            D = np.delete(D, M, 1)
            
    def count(self):
        counts = []
        for p in self.procs:
            counts.append(len(p.particles))
        return counts

    def calc_all_perimeter(self):
        for p in self.procs:
            p.calc_perimeter()

    def communicate(self):
        self.commtable = np.zeros((len(self.procs), len(self.procs)))
        cell = self.cell
        for i,p in enumerate(self.procs):
            for pckt in p.packets:
                step = pckt.send()
                pckt.move(cell, step)
                self.procs[pckt.current].receivebox.append(pckt)
                print('processor #', i, '-> #', pckt.current, 'with', pckt.get_num_passengers(), 'particles')
                self.commtable[i,pckt.current] += pckt.get_num_passengers()
            self.procs[i].SubDomain.packets.clear()
        comm_flag = False
        for i in range(len(self.procs)):
            # print('processor # ', i, "'s receiving")
            self.procs[i].SubDomain.receive()
            if self.procs[i].SubDomain.packets:
                comm_flag = True
        return comm_flag

    def get_comm_max(self):
        return np.max(self.commtable)

