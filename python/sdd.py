# ================================================================================================================================
## 並列シミュレーション時の空間分割手法コード
## ロードバランサー
# ===============================================================================================================================
from itertools import count
from locale import ABDAY_1
from matplotlib.pyplot import axis
import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
from numba import jit
from systm import System, Processor
from math import ceil

# --------------------------------------------------------------------------------------------------------------------------------

def plot_fig(S, s, method_type_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorlist =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for c,p in enumerate(S.Processors):
        clr = colorlist[c%10]
        D = np.array(p.members)
        if not len(D) == 0:
            plt.scatter(D[:,0], D[:,1], c=clr)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks([])
    plt.yticks([])
    ax.set_aspect('equal')
    fgnm = "{}_iteration#{:0=3}.png".format(method_type_name, s+1)
    plt.title(fgnm)
    plt.savefig(fgnm)
    # plt.show()
    plt.close()
def ideal(data, cell):
    N = cell[0]*cell[1]
    icount = len(data[0])/N
    return icount

def simple(data, S):
    cell = S.cell
    dx = 1.0/cell[0]
    dy = 1.0/cell[1]
    for i in range(cell[1]):
        bn = dy*(i+1)
        bs = dy*i
        for j in range(cell[0]):
            be = dx*(j+1)
            bw = dx*j
            p = i*cell[0]+j
            S.Processors[p].boundaries.append([0.0, 1.0, -1*bn])
            S.Processors[p].boundaries.append([1.0, 0.0, -1*be])
            S.Processors[p].boundaries.append([0.0, 1.0, -1*bs])
            S.Processors[p].boundaries.append([1.0, 0.0, -1*bw])
            S.Processors[p].center.append((bw+be)/2)
            S.Processors[p].center.append((bn+bs)/2)
    S.reallocate(data)
    return S

def xybin(data, S):
    cell = S.cell
    Py = []
    By = []
    sortedIndexY = np.argsort(data, axis=1)
    dny = len(data[0])//cell[1]
    for i in range(cell[1]):
        Oy = []
        for j in range(dny+1):
            k = i*(dny+1) + j
            if k >= len(data[0]):
                break
            Oy.append([data[0,sortedIndexY[1,k]], data[1,sortedIndexY[1,k]]])
        Py.append(Oy)
    for i in range(len(Py)):
        if i == 0:
            b = 0.0
        else:
            b = (Py[i-1][-1][1] + Py[i][0][1])/2
        if i == len(Py)-1:
            t = 1.0
        else:
            t = (Py[i][-1][1] + Py[i+1][0][1])/2
        By.append([t, b])
    
    P = []
    B = np.zeros((len(S.Processors), 4))
    count = np.zeros(cell[0]*cell[1])
    for h,Q in enumerate(Py):
        idy = np.array(Q)
        sortedIndexX = np.argsort(idy, axis=0)
        dnx = len(Q)//cell[0]
        for i in range(cell[0]):
            j = (dnx+1)*i
            k = (dnx+1)*(i+1) - 1
            if i == 0:
                l = 0.0
            else:
                l = (Q[sortedIndexX[j-1,0]][0] + Q[sortedIndexX[j,0]][0])/2
            if i == cell[0]-1:
                r = 1.0
            else:
                r = (Q[sortedIndexX[k,0]][0] + Q[sortedIndexX[k+1,0]][0])/2
            p = h*cell[0] + i
            S.Processors[p].boundaries.append([0.0, 1.0, -1*By[h][0]])
            S.Processors[p].boundaries.append([1.0, 0.0, -1*r])
            S.Processors[p].boundaries.append([0.0, 1.0, -1*By[h][1]])
            S.Processors[p].boundaries.append([1.0, 0.0, -1*l])
            S.Processors[p].center.append((r+l)/2)
            S.Processors[p].center.append((By[h][0]+By[h][1])/2)
    S.reallocate(data)
    return S

##  Voronoi分割の不具合を避けるための、初期分割やりなおし
## 空になってしまった領域を、必要に応じて他に割り当てるので、分散した計算
## 空の領域を、最も粒子数の多い領域(target)に突っ込んで、半分ずつ分ける
def voronoi_init(data, S):
    while min(S.count()) == 0:
        for i in range(len(S.Processors)):
            counts_np = np.array(S.count())
            if counts_np[i] == 0:
                target_j = np.argmax(counts_np)
                target_right  = S.Processors[target_j].boundaries[1][2]*-1
                target_left   = S.Processors[target_j].boundaries[3][2]*-1
                target_top    = S.Processors[target_j].boundaries[0][2]*-1
                target_bottom = S.Processors[target_j].boundaries[2][2]*-1
                target_width = target_right - target_left
                ### 領域の左側は、空だったプロセスに 
                S.Processors[i].boundaries[3][2] = target_left*-1
                S.Processors[i].boundaries[1][2] = (target_right - target_width*0.5)*-1
                S.Processors[i].boundaries[0][2] = target_top*-1
                S.Processors[i].boundaries[2][2] = target_bottom*-1
                S.Processors[i].center[0] = target_left + target_width*0.25
                S.Processors[i].center[1] = (target_bottom + target_top)*0.5
                ### 領域の右側は、もともといたプロセスに
                S.Processors[target_j].boundaries[3][2] = (target_right - target_width*0.5)*-1
                S.Processors[target_j].center[0] = target_right - target_width*0.25
                S.reallocate(data)
    return S

## ボロノイ分割の、各粒子を最も近いボロノイ中心点の領域に所属させるメソッド
def voronoi_allocate(data, S, bias):
    ### 後の計算用に、中心点のデータをnumpyにしておく
    center_np = np.zeros((len(S.Processors),2))
    for i in range(len(S.Processors)):
        S.Processors[i].members.clear()
        center_np[i] = np.array([S.Processors[i].center])
    ### 各粒子を、最もボロノイ中心点が近い領域に所属させる
    for d in range(len(data[0])):
        ### 周期境界を考えた、最も近いボロノイ中心点算出
        r2_np = (data[0][d] - center_np[:,0])**2 + (data[1][d] - center_np[:,1])**2 - bias
        r2_np_xreverse = (1 - abs(data[0][d] - center_np[:,0]))**2 + (data[1][d] - center_np[:,0])**2 - bias
        r2_np_yreverse = abs(data[0][d] - center_np[:,0])**2 + (1 - abs(data[1][d] - center_np[:,0]))**2 - bias
        r2_np_xyreverse = (1 - abs(data[0][d] - center_np[:,0]))**2 + (1 - abs(data[1][d] - center_np[:,0]))**2 - bias
        minimums = np.array([r2_np.min(), r2_np_xreverse.min(), r2_np_yreverse.min(), r2_np_xyreverse.min()])
        ### calc_center()用に、境界をまたいで中心点にアクセスする粒子は、周期境界を作用させた座標を入れておく
        if np.argmin(minimums) == 0:
            S.Processors[np.argmin(r2_np)].members.append([data[0][d], data[1][d], data[2][d], data[0][d], data[1][d]])
        elif np.argmin(minimums) == 1:
            if data[0][d] < 0.5:
                x_cross = data[0][d] + 1.0
            else:
                x_cross = data[0][d] - 1.0
            S.Processors[np.argmin(r2_np_xreverse)].members.append([data[0][d], data[1][d], data[2][d], x_cross, data[1][d]])
        elif np.argmin(minimums) == 2:
            if data[1][d] < 0.5:
                y_cross = data[0][d] + 1.0
            else:
                y_cross = data[1][d] - 1.0
            S.Processors[np.argmin(r2_np_yreverse)].members.append([data[0][d], data[1][d], data[2][d], data[0][d], y_cross])
        else:
            if data[0][d] < 0.5:
                x_cross = data[0][d] + 1.0
            else:
                x_cross = data[0][d] - 1.0
            if data[1][d] < 0.5:
                y_cross = data[0][d] + 1.0
            else:
                y_cross = data[1][d] - 1.0
            S.Processors[np.argmin(r2_np_xyreverse)].members.append([data[0][d], data[1][d], data[2][d], x_cross, y_cross])
    return S

## モンテカルロ式に計算負荷分配を調節する
### iteration：アルゴリズムの最大繰り返し回数、alpha：biasの変化係数
### early_stop_range：繰り返し時のearly stopを、理想値のどれくらいで発動させるか
def voronoimc(data, S, cutoff, 
              iteration=50, alpha=0.0005, early_stop_range=0.01):
    ## 各粒子は、最も近い中心点の領域の所属となる。これをそのまま実装している。
    ## preparation
    method_type_name = 'voronoimc'
    print('Iteration =', iteration)
    S = simple(data, S)   ### 最初は等間隔分割
    S = voronoi_init(data, S)
    S.calc_all_center()   ### ボロノイ中心点を計算
    bias = np.zeros(len(S.Processors))

    ## 1.assigned to the cluster with the closest center
    S = voronoi_allocate(data, S, bias)
    plot_fig(S, -1, method_type_name)   ### 分割初期状態の図
    ## 2.bias is initially set to zero

    S.calc_all_center(across_border=True)
    counts = S.count()   ### estimates work-load by using # of particles
    print('step', 0, 'count', S.count())
    
    ## iteration
    ### early stopのために、理想的な計算負荷を見積もっておく
    ideal_count_max = ceil(average(counts)*(1+early_stop_range))
    for s in range(iteration):
        # S.detect_adjacent(cutoff)
        S.detect_neighbors(cutoff)
        counts = S.count()
        n = np.array(counts)
        for i,p in enumerate(S.Processors):
            for j in p.neighbors:
                ## 3.modifying "bi"
                bias[j] += alpha*0.01*(n[i] - n[j])

        ## 4.Atoms move to another cluster or stay
        S = voronoi_allocate(data, S, bias)
        S.calc_all_center(across_border=True)
       
        #  print('step', s+1, 'bias', bias, 'count', S.count())
        print('step', s+1, 'count', S.count())
        if (s+1)%1 == 0:
            plot_fig(S, s, method_type_name)
        if max(S.count()) <= ideal_count_max:
            print('***Early Stop***')
            break
    return S

def voronoi2(data, S, cutoff, iteration=50, alpha=0.05):
    ## method from "R. Koradi et al. Comput. Phys. Commun. 124(2000) 139"
    ## 各粒子は、最も近い中心点の領域の所属となる。これをそのまま実装している。
    ## preparation
    method_type_name = 'voronoi2'
    print('Iteration =', iteration)
    S = simple(data, S)   ### 最初は等間隔分割
    S = voronoi_init(data, S)
    S.calc_all_center()   ### ボロノイ中心点を計算
    bias = np.zeros(len(S.Processors))

    ## 1.assigned to the cluster with the closest center
    ### 後の計算用に、中心点のデータをnumpyにしておく
    voronoi_allocate(data, S, bias)
    plot_fig(S, -1, method_type_name)   ### 分割初期状態の図
    ## 2.bias is initially set to zero

    a = np.zeros(len(S.Processors))   ### 周りの領域の計算負荷（粒子数）保管用配列
    r = np.zeros(len(S.Processors))   ### 領域半径保管用配列
    S.calc_all_center()
    counts = S.count()   ### estimates work-load by using # of particles
    ## iteration
    print('step', 0, 'count', S.count())

    for s in range(iteration):
        S.detect_adjacent(cutoff)
        for i,p in enumerate(S.Processors):
            counts = S.count()
            n = np.array(counts)
            pmem_np = np.array(p.members)   ### 計算用の所属粒子配列のnumpy
            ### 中心から最も離れた粒子までの距離が、その領域の半径
            if not len(pmem_np) == 0:
                radii2 = (pmem_np[:,0] - p.center[0])**2 + (pmem_np[:,1] - p.center[1])**2
                r[i] = (radii2[np.argmax(radii2)])**0.5
            else:
                r[i] = 0
            a[i] = average([n[j] for j in range(len(n)) if j in p.neighbors])   ### 周囲の計算負荷の平均
            ## 3.modifying "bi"
            bias[i] += alpha*(r[i]**2)*((a[i]/n[i])**(2/3) - 1)

        ## 4.Atoms move to another cluster or stay
        voronoi_allocate(data, S, bias)
        ## 5.calculates the new center and the new radius
        S.calc_all_center()
        r_new = np.zeros(len(S.Processors))
        for i,p in enumerate(S.Processors):
            pmem_np = np.array(p.members)
            if not len(pmem_np) == 0:
                radii2 = (pmem_np[:,0] - p.center[0])**2 + (pmem_np[:,1] - p.center[1])**2
                r[i] = (radii2[np.argmax(radii2)])**0.5
            else:
                r[i] = 0
        ## 6.the modification of the radius requires another adaption of the bias
        # bias -= (r_new - r)/2
        
        #  print('step', s+1, 'bias', bias, 'count', S.count())
        print('step', s+1, 'count', S.count())
        if (s+1)%2 == 0:
            plot_fig(S, s, method_type_name)
    return S
# ========================================================================================
