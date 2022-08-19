# =================================================
## シミュレーション実行環境のシミュレート用コード
# =================================================
from three import sdd
from three import sim

from os import stat_result
import random
from re import X
from matplotlib.pyplot import phase_spectrum
import numpy as np
import time

# --------------------------------------------------------------------

class Process:
    def __init__(self, rank):
        self.rank = rank
        self.particles_in_neighbor = []   ### 相互作用する可能性のある周辺領域の粒子
        self.pairlist_between_neighbor = []   ### 領域をまたいだペアリスト
        self.packets = []   ### 通信時に送信する情報を詰めておく
        self.receivebox = []   ### 通信時に受信するポスト
        self.communicatable = []   ### ハードウェア上で隣接するコア
        self.subregion = sdd.subregion()

    def set_box(self, Box):
        self.Box = Box
    
    def set_domain_pair_list(self, dpl):
        self.domain_pair_list = dpl

    ### calc_forceによって計算した力積を、他領域に通信するようのメモリに退避させる
    def set_sending_velocity(self, sending_velocities):
        self.sending_velocities = sending_velocities

    def receive(self):
        received_list = []
        for i,pckt in enumerate(self.receivebox):
            if pckt.direction[0] == 0 and pckt.direction[1] == 0:
                # print('data #', i, 'received')
                received_list.append(i)
        self.packets = [self.receivebox[i] for i in range(len(self.receivebox)) if i not in received_list]
        self.receivebox.clear()

    def start_watch(self):
        self.time_result = 0
        self.start = time.time()

    def stop_watch(self):
        assert self.start!=0, '計測が始まっていません'
        self.time_result = time.time() - self.start
        self.start = 0



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
        self.np = num_of_procs
        self.procs = []
        for i in range(num_of_procs):
            P = Process(i)
            self.procs.append(P)

    ### すべてのプロセッサでシミュレーションボックスのグローバルな設定を共有する
    def set_boxes(self, Box):
        for i in range(len(self.procs)):
            self.procs[i].set_box(Box)
    
    def set_domain_pair_lists(self):
        for i in range(len(self.procs)):
            self.procs[i].set_domain_pair_list(self.proc[i].Box, self.np)

    ### 相互作用する可能性のある領域の粒子を全部持ってくる
    def communicate_particles(self):
        comms = np.zeros((self.np, self.np))   ### どの領域からどこへが最も通信量が多かったか。
        for i,proc in enumerate(self.procs):
            assert proc.rank == i, 'ランクとプロセスリストのインデックスが一致しません'
            self.procs[i].particles_in_neighbor.clear()
            for j_pair in proc.domain_pair_list.list[i]:
                assert j_pair[0] == i, '領域ペアリストが不適切です'
                for p in self.procs[j_pair[1]].subregion.particles:
                    p.set_process(j_pair[1])
                    self.procs[i].particles_in_neighbor.append(p)
        
        ### 最大通信量だけ返す
        ### 双方向同時通信が可能と仮定
                comms[i,j_pair[1]] = self.procs[j_pair[1]].subregion.particles.__sizeof__()
        return np.max(comms)
    
    ### 他領域の粒子との間の力積計算結果を他領域に書き戻す
    def communicate_velocity(self):
        comms = np.zeros((self.np, self.np))   ### どの領域からどこへが最も通信量が多かったか。
        ### 各領域に格納してある、
        for i, proci in enumerate(self.procs):
            ### 相互作用した粒子について、
            # print(len(proci.sending_velocities))
            for j, pj in enumerate(proci.sending_velocities):
                for k, pk in enumerate(self.procs[pj.proc].subregion.particles):
                    if pj.id != pk.id:
                        continue
                    # print('b {:8.6f} {:8.6f}'.format(pl.vx, pl.vy))
                    # print('j {:8.6f} {:8.6f}'.format(pj.vx, pj.vy))
                    pk.vx += pj.vx
                    pk.vy += pj.vy
                    pk.vz += pj.vz
                    # print('a {:8.6f} {:8.6f}'.format(pl.vx, pl.vy))
                    self.procs[pj.proc].subregion.particles[k] = pk
                    comms[i,pj.proc] = pk.__sizeof__()
                    break

        comm2 = self.communicate_particles()
        return np.max(comms)+comm2



    ## 粒子が行方不明になってないかチェックする
    def check_particles(self):
        particle_count = 0
        for i,p in enumerate(self.procs):
            particle_count += len(p.subregion.particles)
        assert particle_count == self.procs[0].Box.N, '粒子が失われています'



    ## 周期境界を考えた隣接セル検出
    def detect_neighbors(self, cutoff):
        for i,p in enumerate(self.procs):
            self.procs[i].subregion.neighbors.clear()
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
                            self.procs[i].subregion.neighbors.append(j)
                            flag = True
                            break
                    if flag == True:
                        break
    
    ## 周期境界を考えない隣接セル検出
    def detect_adjacent(self, cutoff):
        for i,p in enumerate(self.procs):
            self.procs[i].subregion.neighbors.clear()
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
                            self.procs[i].subregion.neighbors.append(j)
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
                self.procs[i].subregion.packets.append(pckt)

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
            self.procs[i].subregion.particles.clear()
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
                    self.procs[i].subregion.particles.append([x, y, D[2][j]])
            D = np.delete(D, M, 1)
            
    def count(self):
        counts = []
        for proc in self.procs:
            counts.append(len(proc.subregion.particles))
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
            self.procs[i].subregion.packets.clear()
        comm_flag = False
        for i in range(len(self.procs)):
            # print('processor # ', i, "'s receiving")
            self.procs[i].subregion.receive()
            if self.procs[i].subregion.packets:
                comm_flag = True
        return comm_flag

    def get_comm_max(self):
        return np.max(self.commtable)

# --------------------------------------------------------------------------------------------------------------------------

### 計算/通信の時間/量を記録する
def export_cost(t, c, step, filename):
    with open(filename, 'a') as f:
        f.write('{:0=7} {:12.10f} {}\n'.format(step, t, c))

# ==========================================================================================================================