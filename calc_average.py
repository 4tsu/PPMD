# =======================================================================
## .datファイルの値の平均を取る
# =======================================================================
import os
import numpy as np

# -----------------------------------------------------------------------

def lb_average(number):
    datalist = []
    count = 0
    for filename in os.listdir('.'):
        if 'load_balance_{}_'.format(number) in filename:
            data = []
            with open(filename, 'r') as f:
                Line = [s.strip() for s in f.readlines()]
                for l in Line:
                    steps = ''
                    mins  = ''
                    maxs  = ''
                    index = 0
                    for s in l:
                        if s == ' ':
                            index += 1
                            continue
                        if   index == 0:
                            steps += s
                        elif index == 1:
                            mins  += s
                        else:
                            maxs  += s
                    data.append([int(steps), float(mins), float(maxs), 0, 0])
            datalist.append(data)
            count += 1
    if count == 0:
        return 0

    ### 平均
    result = np.zeros_like(datalist[0])
    for data in datalist:
        datanp = np.array(data)
        result += datanp
    result /= len(datalist)
    ### 標準偏差
    for data in datalist:
        datanp = np.array(data)
        result[:,3] += (datanp[:,1]-result[:,1])**2
        result[:,4] += (datanp[:,2]-result[:,2])**2
    result[:,3] /= len(datalist)
    result[:,4] /= len(datalist)
    result[:,3] = result[:,3]**0.5
    result[:,4] = result[:,4]**0.5

    with open('load_balance_{}.dat'.format(number), 'w') as f:
        for i in range(len(result)):
            f.write('{:.0f} {:.0f} {:.0f} {:.5f} {:.5f}\n'.format(result[i,0], result[i,1], result[i,2], result[i,3], result[i,4]))



def cost_average(number):
    datalist = []
    count = 0
    for filename in os.listdir('.'):
        if 'cost_{}_'.format(number) in filename:
            data = []
            with open(filename, 'r') as f:
                Line = [s.strip() for s in f.readlines()]
                for l in Line:
                    steps = ''
                    times = ''
                    costs = ''
                    index = 0
                    for s in l:
                        if s == ' ':
                            index += 1
                            continue
                        if   index == 0:
                            steps += s
                        elif index == 1:
                            times += s
                        else:
                            costs += s
                    data.append([int(steps), float(times), float(costs), 0, 0])
            datalist.append(data)
            count += 1
    if count == 0:
        return 0

    ### 平均
    result = np.zeros_like(datalist[0])
    for data in datalist:
        datanp = np.array(data)
        result += datanp
    result /= len(datalist)
    ### 標準偏差
    for data in datalist:
        datanp = np.array(data)
        result[:,3] += (datanp[:,1]-result[:,1])**2
        result[:,4] += (datanp[:,2]-result[:,2])**2
    result[:,3] /= len(datalist)
    result[:,4] /= len(datalist)
    result[:,3] = result[:,3]**0.5
    result[:,4] = result[:,4]**0.5

    with open('cost_{}.dat'.format(number), 'w') as f:
        for i in range(len(result)):
            f.write('{:.0f} {:.8f} {:.0f} {:.8f} {:.8f}\n'.format(result[i,0], result[i,1], result[i,2], result[i,3], result[i,4]))

# ------------------------------------------------------------------------------------------

lb_average(0)
lb_average(1)
lb_average(2)
cost_average(0)
cost_average(1)
cost_average(2)
