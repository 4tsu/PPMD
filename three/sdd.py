# ================================================================================================================================
## 並列シミュレーション時の空間分割手法コード
## ロードバランサー
# ===============================================================================================================================
from three import particle
from three import sim

from itertools import count
from locale import ABDAY_1
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
import numpy as np
from numpy.lib.function_base import average
from numba import jit
from math import ceil, floor

# --------------------------------------------------------------------------------------------------------------------------------

class subregion:
    def __init__(self):
        self.particles = []   ### 所属粒子
        self.pairlist = []   ### 粒子ペアリスト
        self.neighbors = []   ### シミュレーション上で隣接するプロセッサ
        self.center = []   ### 領域中心
        self.boundaries = []   ### 領域境界

    ### 領域中心を計算する
    def calc_center(self, box):
        xl = box.xl
        yl = box.yl
        zl = box.zl
        P = self.particles
        
        if not len(P) == 0:
            ### 一点てきとうに決めて、そこからの相対位置ベクトルで中心を計算
            ### シミュレーションボックスの境界をまたいでも正しく計算できるよう
            origin_px = P[0].x
            origin_py = P[0].y
            origin_pz = P[0].z
            sx = 0
            sy = 0
            sz = 0
            for i,p in enumerate(P):
                rx_list = np.array([abs(p.x-origin_px), abs(p.x-origin_px-xl), abs(p.x-origin_px+xl)])
                if np.argmin(rx_list)==0:
                    rx = p.x - origin_px
                elif np.argmin(rx_list)==1:
                    rx = p.x - origin_px - xl
                else:
                    rx = p.x - origin_px + xl
                    
                ry_list = np.array([abs(p.y-origin_py), abs(p.y-origin_py-yl), abs(p.y-origin_py+yl)])
                if np.argmin(ry_list)==0:
                    ry = p.y - origin_py
                elif np.argmin(ry_list)==1:
                    ry = p.y - origin_py - yl
                else:
                    ry = p.y - origin_py + yl

                rz_list = np.array([abs(p.z-origin_pz), abs(p.z-origin_pz-zl), abs(p.z-origin_pz+zl)])
                if np.argmin(rz_list)==0:
                    rz = p.z - origin_pz
                elif np.argmin(rz_list)==1:
                    rz = p.z - origin_pz - zl
                else:
                    rz = p.z - origin_pz + zl

                sx += rx
                sy += ry
                sz += rz

            sx /= len(P)
            sy /= len(P)
            sz /= len(P)
            sx += origin_px
            sy += origin_py
            sz += origin_py
            sx, sy, sz = box.periodic_coordinate(sx, sy, sz)
            assert box.x_max>=sx>=box.x_min and box.y_max>=sy>=box.y_min and box.z_max>=sz>=box.z_min, 'center value is out of range!'
            self.center.clear()
            self.center.append(sx)
            self.center.append(sy)
            self.center.append(sz)
    
    def calc_radius(self, box):
        r_max = 0
        for p in self.particles:
            r = box.periodic_distance(p.x, p.y, p.z, self.center[0], self.center[1], self.center[2])
            if r > r_max:
                r_max = r
        self.radius = r_max

    def calc_perimeter(self):
        perimeter = 0
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
            perimeter += ((C[j][0]-C[j-1][0])**2 + (C[j][1]-C[j-1][1])**2)**0.5
        self.perimeter = perimeter

    def set_limit(self, top, bottom, right, left, front, back):
        self.top    = top
        self.bottom = bottom
        self.right  = right
        self.left   = left
        self.back   = back
        self.front  = front

    def set_bias(self, bias):
        self.bias = bias

# ------------------------------------------------------------------------------------------

def sdd_init(Machine, sdd_num):
    if sdd_num==0:
        return Machine
    elif sdd_num==1:
        return global_sort(Machine)
    elif sdd_num==2:
        Machine = simple(Machine)   ### 最初は等間隔分割
        Machine = voronoi_init(Machine)   ### 等間隔分割で不具合が出たらカバー
        return voronoimc(Machine)
    elif sdd_num==3:
        return rcb(Machine)
    elif sdd_num==4:
        Machine = odp_init(Machine)
        return one_d_parallel(Machine)
    elif sdd_num==5:
        Machine = sb_init(Machine)
        return skew_boundary(Machine)



def sdd(Machine, sdd_num):
    if sdd_num==0:
        return simple(Machine)
    elif sdd_num==1:
        return global_sort(Machine)
    elif sdd_num==2:
        return voronoimc(Machine)
    elif sdd_num==3:
        return rcb(Machine)
    elif sdd_num==4:
        return one_d_parallel(Machine)
    elif sdd_num==5:
        return skew_boundary(Machine)



def get_simple_array(Box, np):
    ## 空間分割の形状=領域の並び方を決める
    xl = Box.xl
    yl = Box.yl
    zl = Box.zl
    ### 何x何x何の配列が良いか決める。領域は立方体に近い方が良い
    xyzn_list = []
    for xn in range(1, np+1):
        if np%xn == 0:
            yzn = np/xn
            for yn in range(1, int(yzn)+1):
                if yzn%yn == 0:
                    zn = yzn/yn
                    xyzn_list.append([xn, yn, zn])
    n_best = np
    i_best = -1
    for i,xyzn in enumerate(xyzn_list):
        if abs(max(xyzn)-min(xyzn)) < n_best:
            n_best = abs(max(xyzn)-min(xyzn))
            i_best = i
    xn = int(xyzn_list[i_best][0])
    yn = int(xyzn_list[i_best][1])
    zn = int(xyzn_list[i_best][2])
    Box.set_subregion(xn, yn, zn, xl/xn, yl/yn, zl/zn)
    return Box



def simple(Machine):
    for i,proc in enumerate(Machine.procs):
        top    = proc.subregion.top
        bottom = proc.subregion.bottom
        right  = proc.subregion.right
        left   = proc.subregion.left
        back   = proc.subregion.back
        front  = proc.subregion.front
        box = proc.Box
        x_min = box.x_min
        y_min = box.y_min
        z_min = box.z_min
        sd_xl = box.sd_xl
        sd_yl = box.sd_yl
        sd_zl = box.sd_zl
        xn = box.xn
        yn = box.yn
        zn = box.zn

        particles = proc.subregion.particles
        new_particles = []
        for j in range(len(particles)):
            jp = particles.pop()
            if jp.x>right or jp.x<left or jp.y>back or jp.y<front or jp.z>top or jp.z<bottom:
                iproc = int(((jp.z-z_min)//sd_zl)*(xn*yn) + ((jp.y-y_min)//sd_yl)*xn + (jp.x-x_min)//sd_xl)
                Machine.procs[iproc].subregion.particles.append(jp)
            else:
                new_particles.append(jp)
        Machine.procs[i].subregion.particles = new_particles
    return Machine



def global_sort(Machine):
    ### 準備
    datalist = []
    for proc in Machine.procs:
        for p in proc.subregion.particles:
            datalist.append([p.id, p.x, p.y, p.z, p.vx, p.vy, p.vz])
    data = np.array(datalist)
    ### まずはZ軸方向についてソート
    sortedIndexZ = np.argsort(data, axis=0)

    box = Machine.procs[0].Box
    xn = box.xn
    yn = box.yn
    zn = box.zn

    all_particles = []
    boundaries_z = []
    numz = int(box.N//zn)
    for i in range(zn):
        z_group = []

        for j in range(numz+1):
            k = i*(numz+1) + j
            if k >= len(data):
                break
            z_group.append(datalist[sortedIndexZ[k,3]])

        all_particles.append(z_group)

    ## 次にYX軸方向についてのソート
    for h,z_group in enumerate(all_particles):
        ### Y軸方向
        data = np.array(z_group)
        sortedIndexY = np.argsort(data, axis=0)
        numy = int(len(z_group)//yn)

        z_particles = []
        for i in range(yn):
            y_group = []
            for j in range(numy+1):
                k = i*(numy+1) + j
                if k >= len(data):
                    break
                y_group.append(z_group[sortedIndexY[k,2]])
            z_particles.append(y_group)
        
        for i,y_group in enumerate(z_particles):
            data = np.array(y_group)
            sortedIndexX = np.argsort(data, axis=0)
            numx = int(len(y_group)//xn)

            for j in range(xn):
                proc_index = h*(yn*xn) + i*yn + j
                Machine.procs[proc_index].subregion.particles.clear()
                new_particles = []

                ### ぴったりより一個余分に詰めていき、最後は少し少なくなるのでbreakで脱出
                for k in range(numx+1):
                    l = j*(numx+1) + k
                    if len(data) <= l:
                        break
                    pdata = y_group[sortedIndexX[l,1]]
                    p = particle.Particle(pdata[0], pdata[1], pdata[2], pdata[3])
                    p.set_velocity(pdata[4], pdata[5], pdata[6])
                    new_particles.append(p)
                
                Machine.procs[proc_index].subregion.particles = new_particles
   
    return Machine



## Voronoi分割の不具合を避けるための、初期等間隔分割やりなおし
## 空になってしまった領域を、必要に応じて他に割り当てるので、分散した計算
## 空の領域を、最も粒子数の多い領域(target)に突っ込んで、半分ずつ分ける
def voronoi_init(Machine):
    while min(Machine.count()) == 0:
        for i in range(len(Machine.procs)):
            counts_np = np.array(Machine.count())
            
            if counts_np[i] == 0:
                target_j = np.argmax(counts_np)
                target_right  = Machine.procs[target_j].subregion.right
                target_left   = Machine.procs[target_j].subregion.left
                target_top    = Machine.procs[target_j].subregion.top
                target_bottom = Machine.procs[target_j].subregion.bottom
                target_front  = Machine.procs[target_j].subregion.front
                target_back   = Machine.procs[target_j].subregion.back
                target_width = target_right - target_left
            
                center_line = target_right - target_width*0.5
                ### 領域の左側は、空だったプロセスに 
                Machine.procs[i].subregion.set_limit(target_top, target_bottom, center_line, target_left, target_front, target_back)
                ### 領域の右側は、もともといたプロセスに
                Machine.procs[target_j].subregion.left = center_line

                ### 実際の粒子割当
                target_j_particles = []
                i_particles = []
                for p in Machine.procs[target_j].subregion.particles:
                    if p.x > center_line:
                        target_j_particles.append(p)
                    elif p.x <= center_line:
                        i_particles.append(p)
                
                Machine.procs[i].subregion.particles        = i_particles
                Machine.procs[target_j].subregion.particles = target_j_particles

            assert sum(Machine.count()) == Machine.procs[0].Box.N, '初期空間分割に失敗しています'
    
    ## バイアスをゼロにしておく
    for i in range(Machine.np):
        Machine.procs[i].subregion.set_bias(0.0)

    return Machine



## ボロノイ分割の、各粒子を最も近いボロノイ中心点の領域に所属させるメソッド
def voronoi_allocate(Machine, bias):
    xl = Machine.procs[0].Box.xl
    yl = Machine.procs[0].Box.yl
    zl = Machine.procs[0].Box.zl
    ### 後の計算用に、中心点のデータをnumpyにしておく
    center_np = np.zeros((Machine.np,3))
    for i,proc in enumerate(Machine.procs):
        center_np[i] = np.array([proc.subregion.center])
    
    ### 各粒子を、最もボロノイ中心点が近い領域に所属させる
    for i,proc in enumerate(Machine.procs):
        ### 周期境界を考えた、最も近いボロノイ中心点算出
        for j in range(len(proc.subregion.particles)):
            p = Machine.procs[i].subregion.particles.pop(0)

            r2_np = (p.x - center_np[:,0])**2 + (p.y - center_np[:,1])**2 + (p.z - center_np[:,2])**2 - bias

            r2_np_x_reverse = (xl - abs(p.x - center_np[:,0]))**2 + (p.y - center_np[:,1])**2 + (p.z - center_np[:,2])**2 - bias
            r2_np_y_reverse = (p.x - center_np[:,0])**2 + (yl - abs(p.y - center_np[:,1]))**2 + (p.z - center_np[:,2])**2 - bias
            r2_np_z_reverse = (p.x - center_np[:,0])**2 + (p.y - center_np[:,1])**2 + (zl - abs(p.z - center_np[:,2]))**2 - bias

            r2_np_xy_reverse = (xl - abs(p.x - center_np[:,0]))**2 + (yl - abs(p.y - center_np[:,0]))**2 + (p.z - center_np[:,2])**2 - bias
            r2_np_yz_reverse = (p.x - center_np[:,0])**2 + (yl - abs(p.y - center_np[:,0]))**2 + (zl - abs(p.z - center_np[:,2]))**2 - bias
            r2_np_zx_reverse = (xl - abs(p.x - center_np[:,0]))**2 + (p.y - center_np[:,0])**2 + (zl - abs(p.z - center_np[:,2]))**2 - bias
            
            r2_np_xyz_reverse = (xl - abs(p.x - center_np[:,0]))**2 + (yl - abs(p.y - center_np[:,0]))**2 + (zl - abs(p.z - center_np[:,2]))**2 - bias
            minimums = np.array([r2_np.min(), r2_np_x_reverse.min(), r2_np_y_reverse.min(), r2_np_z_reverse.min(), r2_np_xy_reverse.min(), r2_np_yz_reverse.min(), r2_np_zx_reverse.min(), r2_np_xyz_reverse.min()])

            ### シミュレーションボックスの境界をまたがない
            if np.argmin(minimums) == 0:
                Machine.procs[np.argmin(r2_np)].subregion.particles.append(p)
            ### x方向はまたぐ
            elif np.argmin(minimums) == 1:
                Machine.procs[np.argmin(r2_np_x_reverse)].subregion.particles.append(p)
            ### y方向はまたぐ
            elif np.argmin(minimums) == 2:
                Machine.procs[np.argmin(r2_np_y_reverse)].subregion.particles.append(p)
            ### z方向はまたぐ
            elif np.argmin(minimums) == 3:
                Machine.procs[np.argmin(r2_np_z_reverse)].subregion.particles.append(p)
            ### xyともにまたぐ
            elif np.argmin(minimums) == 4:
                Machine.procs[np.argmin(r2_np_xy_reverse)].subregion.particles.append(p)
            ### yzともにまたぐ
            elif np.argmin(minimums) == 5:
                Machine.procs[np.argmin(r2_np_yz_reverse)].subregion.particles.append(p)
            ### xzともにまたぐ
            elif np.argmin(minimums) == 6:
                Machine.procs[np.argmin(r2_np_zx_reverse)].subregion.particles.append(p)
            ### xyzずべてまたぐ
            else:
                Machine.procs[np.argmin(r2_np_xyz_reverse)].subregion.particles.append(p)

    return Machine



## モンテカルロ式に計算負荷分配を調節する
### iteration：アルゴリズムの最大繰り返し回数、alpha：biasの変化係数
### early_stop_range：繰り返し時のearly stopを、理想値のどれくらいで発動させるか
def voronoimc(Machine,
              iteration=500, alpha=0.008, early_stop_range=0.10):
    ## 各粒子は、最も近い中心点の領域の所属となる。これをそのまま実装している。
    ## preparation
    method_type_name = 'voronoimc'

    ## 1.assigned to the cluster with the closest center
    ## 2.bias is initially set to zero 
    ### ↑シミュレーション途中のときはバイアスは前回の値を引き継ぐ
    bias = np.zeros(Machine.np)
    for i,proc in enumerate(Machine.procs):
        Machine.procs[i].subregion.calc_center(proc.Box)
        Machine.procs[i].subregion.calc_radius(proc.Box)
        bias[i] = proc.subregion.bias
    Machine = voronoi_allocate(Machine, bias)
    # plot_fig(Machine, 0, method_type_name)  ### 分割初期状態の図
    
    for proc in Machine.procs:
        Machine.procs[i].subregion.calc_center(proc.Box)
    counts = Machine.count()   ### estimates work-load by using # of particles
    
    ## iteration
    ### early stopのために、理想的な計算負荷を見積もっておく
    ideal_count_max = ceil(average(counts)*(1+early_stop_range))
    for s in range(iteration):
        dpl = sim.DomainPairList(Machine)
        counts = Machine.count()
        n = np.array(counts)
        for proc_list in dpl.list:
            for domain_pair in proc_list:
                i = domain_pair[0]
                j = domain_pair[1]
                ## 3.modifying "bi"
                bias[i] -= (alpha*(n[i] - n[j]))**3
                bias[j] += (alpha*(n[i] - n[j]))**3
        ### バイアスが極端な値にならないように。
        for i,proc in enumerate(Machine.procs):
            if proc.subregion.bias < -proc.subregion.radius:
                Machine.procs[i].subregion.bias = -proc.subregion.radius
            elif proc.subregion.bias > min(proc.Box.xl, proc.Box.yl, proc.Box.zl):
                Machine.procs[i].subregion.bias = min(proc.Box.xl, proc.Box.yl, proc.Box.zl)

        ## 4.Atoms move to another cluster or stay
        Machine = voronoi_allocate(Machine, bias)
        for proc in Machine.procs:
            Machine.procs[i].subregion.calc_center(proc.Box)
            Machine.procs[i].subregion.calc_radius(proc.Box)
       
        # print('step', s+1, 'bias', bias)
        # with open('bias.dat', 'a') as f:
        #     f.write('{}'.format(s+1))
        #     for b in bias:
        #         f.write(' {}'.format(b))
        #     f.write('\n')
        
        # if (s+1)%5 == 0:
            # print('step', s+1, 'count', Machine.count())
            # plot_fig(Machine, s+1, method_type_name)
        
        if max(Machine.count()) <= ideal_count_max:
            # print('***Early Stop***')
            break

    ### 次回の空間分割のために、バイアスは保存しておく
    for i in range(Machine.np):
        Machine.procs[i].subregion.set_bias(bias[i])

    return Machine



def plot_fig(Machine, s, method_type_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorlist =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for c,proc in enumerate(Machine.procs):
        clr = colorlist[c%10]
        P = []
        for p in proc.subregion.particles:
            P.append([p.x, p.y])
        D = np.array(P)
        if not len(D) == 0:
            plt.scatter(D[:,0], D[:,1], c=clr)

    box = Machine.procs[0].Box
    plt.xlim(box.x_min, box.x_max)
    plt.ylim(box.y_min, box.y_max)
    plt.xticks([])
    plt.yticks([])
    ax.set_aspect('equal')
    figname = "{}_iteration#{:0=3}.png".format(method_type_name, s+1)
    plt.title(figname)
    plt.savefig(figname)
    # plt.show()
    plt.close()



# Recursive Coordinate Bisectioning
def rcb(Machine):
    method_type_name = "rcb"

    Machine.procs[0].subregion.right  = Machine.procs[0].Box.x_max
    Machine.procs[0].subregion.left   = Machine.procs[0].Box.x_min
    Machine.procs[0].subregion.back   = Machine.procs[0].Box.y_max
    Machine.procs[0].subregion.front  = Machine.procs[0].Box.y_min
    Machine.procs[0].subregion.top    = Machine.procs[0].Box.z_max
    Machine.procs[0].subregion.bottom = Machine.procs[0].Box.z_min
    for i in range(1,Machine.np):
        Machine.procs[0].subregion.particles += Machine.procs[i].subregion.particles
        Machine.procs[i].subregion.particles.clear()
    # plot_fig(Machine, -1, method_type_name)

    directions = ["x" for _ in range(Machine.np)]
    number_of_particles = Machine.procs[0].Box.N
    for i in range(1,Machine.np):
        counts_np = np.array(Machine.count())
        target_j = np.argmax(counts_np)
        datalist = []
        for p in Machine.procs[target_j].subregion.particles:
            datalist.append([p.id, p.x, p.y, p.z, p.vx, p.vy, p.vz])
        target_data = np.array(datalist)

        # 分割の方向はxyz順番に
        sorted_index = np.argsort(target_data, axis=0)
        hi = counts_np[target_j]//2
        ## x方向のとき
        if directions[target_j] == "x":
            border = (target_data[sorted_index[hi,1],1] + target_data[sorted_index[hi+1,1],1])/2

            ### 領域の左側は、空だったプロセスに
            Machine.procs[i].subregion.right  = border
            Machine.procs[i].subregion.left   = Machine.procs[target_j].subregion.left
            Machine.procs[i].subregion.back   = Machine.procs[target_j].subregion.back
            Machine.procs[i].subregion.front  = Machine.procs[target_j].subregion.front
            Machine.procs[i].subregion.top    = Machine.procs[target_j].subregion.top
            Machine.procs[i].subregion.bottom = Machine.procs[target_j].subregion.bottom
            for jp in range(hi):
                Machine.procs[i].subregion.particles.append(Machine.procs[target_j].subregion.particles[sorted_index[jp,1]])
            ### 領域の右側は、もともとのプロセスに
            Machine.procs[target_j].subregion.left = border
            new_target_particles = []
            for jp in range(hi, len(target_data)):
                new_target_particles.append(Machine.procs[target_j].subregion.particles[sorted_index[jp,1]])
            Machine.procs[target_j].subregion.particles.clear()
            Machine.procs[target_j].subregion.particles = new_target_particles

            directions[target_j] = "y"
            directions[i] = "y"
        
        ## y方向のとき
        elif directions[target_j] == "y":
            border = (target_data[sorted_index[hi,2],2] + target_data[sorted_index[hi+1,2],2])

            ### 領域の奥側は、空だったプロセスに
            Machine.procs[i].subregion.back   = Machine.procs[target_j].subregion.back
            Machine.procs[i].subregion.front  = border
            Machine.procs[i].subregion.right  = Machine.procs[target_j].subregion.right
            Machine.procs[i].subregion.left   = Machine.procs[target_j].subregion.left
            Machine.procs[i].subregion.top    = Machine.procs[target_j].subregion.top
            Machine.procs[i].subregion.bottom = Machine.procs[target_j].subregion.bottom
            for jp in range(hi):
                Machine.procs[i].subregion.particles.append(Machine.procs[target_j].subregion.particles[sorted_index[jp,2]])
            ### 領域の手前側は、もともとのプロセスに
            Machine.procs[target_j].subregion.back = border
            new_target_particles = []
            for jp in range(hi, len(target_data)):
                new_target_particles.append(Machine.procs[target_j].subregion.particles[sorted_index[jp,2]])
            Machine.procs[target_j].subregion.particles.clear()
            Machine.procs[target_j].subregion.particles = new_target_particles

            directions[target_j] = "z"
            directions[i] = "z"
        
        ## z方向のとき
        else:
            border = (target_data[sorted_index[hi,3],3] + target_data[sorted_index[hi+1,3],3])

            ### 領域の上側は、空だったプロセスに
            Machine.procs[i].subregion.top    = Machine.procs[target_j].subregion.top
            Machine.procs[i].subregion.bottom = border
            Machine.procs[i].subregion.right  = Machine.procs[target_j].subregion.right
            Machine.procs[i].subregion.left   = Machine.procs[target_j].subregion.left
            Machine.procs[i].subregion.back   = Machine.procs[target_j].subregion.back
            Machine.procs[i].subregion.front  = Machine.procs[target_j].subregion.front
            for jp in range(hi):
                Machine.procs[i].subregion.particles.append(Machine.procs[target_j].subregion.particles[sorted_index[jp,3]])
            ### 領域の下側は、もともとのプロセスに
            Machine.procs[target_j].subregion.top = border
            new_target_particles = []
            for jp in range(hi, len(target_data)):
                new_target_particles.append(Machine.procs[target_j].subregion.particles[sorted_index[jp,3]])
            Machine.procs[target_j].subregion.particles.clear()
            Machine.procs[target_j].subregion.particles = new_target_particles

            directions[target_j] = "x"
            directions[i] = "x"

        # print('proc', i, 'count', Machine.count())
        assert(number_of_particles == sum(Machine.count()))
        # plot_fig(Machine, i-1, method_type_name)

    return Machine



def odp_init(Machine):
    num_cell = Machine.np
    box = Machine.procs[0].Box
    # 初期分割
    lxp = box.xl/num_cell
    back   = box.y_max
    front  = box.y_min
    top    = box.z_max
    bottom = box.z_min
    migration_particles = [[] for _ in range(Machine.np)]
    for i in range(Machine.np):
        left = i*lxp + box.x_min
        right = (i+1)*lxp + box.x_min
        Machine.procs[i].subregion.left   = left
        Machine.procs[i].subregion.right  = right
        Machine.procs[i].subregion.back   = back
        Machine.procs[i].subregion.front  = front
        Machine.procs[i].subregion.top    = top
        Machine.procs[i].subregion.bottom = bottom
        for p in Machine.procs[i].subregion.particles:
            migration_particles[int((p.x-box.x_min)/lxp)].append(p)
    for i in range(Machine.np):
        Machine.procs[i].subregion.particles.clear()
        Machine.procs[i].subregion.particles = migration_particles[i]
    assert(sum(Machine.count()) == box.N)
    return Machine



def one_d_parallel(Machine, iteration=500, alpha=0.005, early_stop_range=0.02):
    method_type_name = "one_d_parallel"
    print("Iteration =", iteration)
    box = Machine.procs[0].Box

    # plot_fig(Machine, -1, method_type_name)
    # print("step", 0, "count", Machine.count())

    ideal_count_max = ceil(average(Machine.count())*(1+early_stop_range))
    for s in range(iteration):
        counts = Machine.count()
        for i in range(Machine.np-1):
            dx = alpha*(counts[i] - counts[i+1])
            Machine.procs[i].subregion.right  -= dx
            Machine.procs[i+1].subregion.left -= dx

            left_limit  = (Machine.procs[i].subregion.left + Machine.procs[i].subregion.right)/2
            right_limit = (Machine.procs[i+1].subregion.left + Machine.procs[i+1].subregion.right)/2

            if Machine.procs[i].subregion.right < left_limit:
                Machine.procs[i].subregion.right = left_limit 
                Machine.procs[i+1].subregion.left = left_limit
            elif Machine.procs[i].subregion.right > right_limit:
                Machine.procs[i].subregion.right = right_limit
                Machine.procs[i+1].subregion.left = right_limit

            i_particles = []
            for p in Machine.procs[i].subregion.particles:
                if p.x < Machine.procs[i].subregion.left:
                    Machine.procs[i-1].subregion.particles.append(p)
                elif p.x > Machine.procs[i].subregion.right:
                    Machine.procs[i+1].subregion.particles.append(p)
                else:
                    i_particles.append(p)
            Machine.procs[i].subregion.particles.clear()
            Machine.procs[i].subregion.particles = i_particles
 
        last_particles = []
        for p in Machine.procs[Machine.np-1].subregion.particles:
            if p.x < Machine.procs[Machine.np-1].subregion.left:
                Machine.procs[Machine.np-2].subregion.particles.append(p)
            else:
                last_particles.append(p)
        Machine.procs[Machine.np-1].subregion.particles.clear()
        Machine.procs[Machine.np-1].subregion.particles = last_particles

        """
        if (s+1)%5 == 0:
            print('step', s+1, 'count', Machine.count())
            plot_fig(Machine, s, method_type_name)
        """
    
        assert(box.N == sum(Machine.count()))
        if max(Machine.count()) <= ideal_count_max:
            break

    return Machine



def sb_init(Machine):
    box = Machine.procs[0].Box
    # 初期分割は等間隔
    Machine = simple(Machine)
    # ただし、境界の定義だけ変更する
    for i in range(Machine.np):
        ip_yz = (i//box.xn)
        Machine.procs[i].subregion.left  += ip_yz*box.xl - box.x_min
        Machine.procs[i].subregion.right += ip_yz*box.xl - box.x_min
        Machine.procs[i].subregion.back   = box.y_max
        Machine.procs[i].subregion.front  = box.y_min
        Machine.procs[i].subregion.top    = box.z_max
        Machine.procs[i].subregion.bottom = box.z_min
    assert(sum(Machine.count()) == box.N)

    return Machine



def skew_boundary(Machine, iteration=600, alpha=0.018, early_stop_range=0.02):
    method_type_name = "skew_boundary"
    print("Iteration =", iteration)
    box = Machine.procs[0].Box
    
    assert(sum(Machine.count()) == box.N)
    # plot_fig(Machine, -1, method_type_name)
    # print("step", 0, "count", Machine.count())

    ideal_count_max = ceil(average(Machine.count())*(1+early_stop_range))
    for s in range(iteration):
        counts = Machine.count()
        for i in range(Machine.np-1):
            dx = (alpha*(counts[i] - counts[i+1]))**3
            Machine.procs[i].subregion.right  -= dx
            Machine.procs[i+1].subregion.left -= dx

            left_limit  = (Machine.procs[i].subregion.left + Machine.procs[i].subregion.right)/2
            right_limit = (Machine.procs[i+1].subregion.left + Machine.procs[i+1].subregion.right)/2

            if Machine.procs[i].subregion.right < left_limit:
                Machine.procs[i].subregion.right = left_limit 
                Machine.procs[i+1].subregion.left = left_limit
            elif Machine.procs[i].subregion.right > right_limit:
                Machine.procs[i].subregion.right = right_limit
                Machine.procs[i+1].subregion.left = right_limit

            i_particles = []
            for p in Machine.procs[i].subregion.particles:
                sbx = floor((p.z-box.z_min)/box.sd_zl)*box.yn*box.xl + floor((p.y-box.y_min)/box.sd_yl)*box.xl + p.x-box.x_min
                if sbx < Machine.procs[i].subregion.left:
                    Machine.procs[i-1].subregion.particles.append(p)
                elif sbx > Machine.procs[i].subregion.right:
                    Machine.procs[i+1].subregion.particles.append(p)
                else:
                    i_particles.append(p)
            Machine.procs[i].subregion.particles.clear()
            Machine.procs[i].subregion.particles = i_particles
 
        last_particles = []
        for p in Machine.procs[Machine.np-1].subregion.particles:
            sbx = floor((p.z-box.z_min)/box.sd_zl)*box.yn*box.xl + floor((p.y-box.y_min)/box.sd_yl)*box.xl + p.x-box.x_min
            if sbx < Machine.procs[Machine.np-1].subregion.left:
                Machine.procs[Machine.np-2].subregion.particles.append(p)
            else:
                last_particles.append(p)
        Machine.procs[Machine.np-1].subregion.particles.clear()
        Machine.procs[Machine.np-1].subregion.particles = last_particles

        """
        if (s+1)%5 == 0:
            print('step', s+1, 'count', Machine.count())
            plot_fig(Machine, s, method_type_name)
        """
    
        assert(box.N == sum(Machine.count()))
        if max(Machine.count()) <= ideal_count_max:
            break

    return Machine


# ========================================================================================
