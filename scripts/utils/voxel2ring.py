import copy
import random

import numpy as np
import model.config as config
import math

'''
return:  the coordinate data of the rings in each not-null level
         the number of not-null levels
size  :  (levels(not-null), RING_SIZE, 3 (dimensions) )

attention : the valid voxel's value is set to -1
'''

'''
get the z-th plane
input: a- 3D-array
       z- index of z
output: 2D plane of array

'''
V_VALID = -1
LEVEL_SIZE =  config.cfg['n_levels']
dim = 32
RING_SIZE_FIX = config.cfg['n_rings']

RING_SIZE=RING_SIZE_FIX
# edge_rate=config.cfg['edge_rate']
# ring_rate=config.cfg['ring_rate']
# af1=config.cfg['af1']
# af2=config.cfg['af2']
# edge_size=int(math.floor((dim*4-4)*edge_rate)//(af1*af2) *(af1*af2))
# ring_size = int((dim*4-4)*ring_rate//(af1*af2) *(af1*af2))
def get_plane(a, z):
    for i in range(len(a)):
        for j in range(len(a)):
            if a[i][j][z] == 1:
                array_zero = a[:, :, z]
                return array_zero
    return []


'''
get inside and outside edge
input: array - 2D plane array
       z - index of z axis

output: a - the edge voxel plane (2D) 
        edge_list - the list of edge voxel coordinate (in tuple)

'''
def get_edge(array):

    flag = 0  # the first not empty voxel?
    edge_list = []
    edge_out_list = []
    for i in range(dim):
        start = 1
        for j in range(dim - 1):
            if array[i][j] == 1:
                if start == 1:
                    edge_out_list.append((i, j))
                    start = 0
                if flag == 0:
                    edge_list.append((i, j))
                if array[i][j + 1] == 1:
                    flag = 1
                else:
                    flag = 0

    flag = 0
    for j in range(dim):
        start = 1
        for i in range(dim - 1, 0, -1):
            if array[i][j] == 1:
                if start == 1:
                    edge_out_list.append((i, j))
                    start = 0
                if flag == 0:
                    edge_list.append((i, j))
                if array[i - 1][j] == 1:
                    flag = 1
                else:
                    flag = 0

    flag = 0
    for i in range(dim):
        start = 1
        for j in range(dim - 1, 0, -1):

            if array[i][j] == 1:
                if start == 1:
                    edge_out_list.append((i, j))
                    start = 0
                if flag == 0:
                    edge_list.append((i, j))
                if array[i][j - 1] == 1:
                    flag = 1
                else:
                    flag = 0

    flag = 0
    for j in range(dim):
        start = 1
        for i in range(dim - 1):

            if array[i][j] == 1:
                if start == 1:
                    edge_out_list.append((i, j))
                    start = 0
                if flag == 0:
                    edge_list.append((i, j))
                if array[i + 1][j] == 1:
                    flag = 1
                else:
                    flag = 0
    edge_out_list = list(set(edge_out_list))
    edge_list = list(set(edge_list))

    return  edge_list, edge_out_list


'''
get 2 vectors' distance
output: distance float
'''


def calcu_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


'''
calculate the cos value of 2 vectors
'''
def calcu_cos(x, y):
    if (len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1 = 0.0;
    result2 = 0.0;
    result3 = 0.0;
    for i in range(len(x)):
        result1 += x[i] * y[i]  # sum(X*Y)
        result2 += x[i] ** 2  # sum(X*X)
        result3 += y[i] ** 2  # sum(Y*Y)

    return result1 / ((result2 * result3) ** 0.5)


'''
get center point
input : l - the coordinate list of plane array (2D)
        z - z axis of model
output : p - the coordinate of the center point
'''
def get_centerpoint_axis(l):
    lx = []
    ly = []
    for cor in l:
        lx.append(cor[0])
        ly.append(cor[1])
    return ((max(lx) + min(lx)) // 2, (max(ly) + min(ly)) // 2)


def get_centerpoint_elem(l):
    l_dis1 = []
    l_dis2 = []
    cor1 = np.array(l[0])
    for cor in l:
        l_dis1.append(calcu_distance(cor, cor1))
    l_dis1 = np.array(l_dis1)
    cor2 = np.array(l[l_dis1.argmax()])
    cor3 = np.array(l[l_dis1.argmax() // 2])

    for cor in l:
        l_dis2.append(calcu_distance(cor, cor3))
    l_dis2 = np.array(l_dis2)
    cor4 = np.array(l[l_dis2.argmax()])
    p = (cor1 + cor2 + cor3 + cor4) / 4
    return p


'''
get the vector made up with 2 points
input : the coordinates of the 2 points 
        b - the start point
output : the result vector
'''


def get_vector(a, b):
    v = ((a[0] - b[0]), (a[1] - b[1]))
    return v


'''
main function
input : the chunk voxel data arrays ( for example 2048 * 32*32*32 chunk_size , dimension) 
output : arr_cors - the ring of each level(not null)  in coordinate
                    format : (levels, coordinates)
         arr_lsize - the level length of each model in the chunk
'''


def load_data(array):
    ls_cors = []
    chunk_size = array.shape[0]
    for index in range(chunk_size):  # for each model
        if index%128==0:
            print 'the',index,'model is preparing'
        lt = []  # l - levels in every model
        x = array[index, 0, :]
        for z in range(dim):  # for each level

            ar_p = get_plane(x, z)
            if ar_p == []:
                continue

            ls_e, ls_eo = get_edge(ar_p, z)
            # get center point
            cor_center = get_centerpoint_axis(ls_eo)
            # get 2 parts of the ouside edge list (1:up 2:down)
            ls_eo_p1 = []
            ls_eo_p2 = []
            for cor in ls_eo:
                # cor= list(cor)   # change coordinate tuple into list
                if cor[0] <= cor_center[0]:
                    ls_eo_p1.append(cor)
                else:
                    ls_eo_p2.append(cor)
            # sorted the coordinate sequence
            ls_cos = []
            ls_cor = []
            v1 = (0, -1)  # v1 : the start vector

            for cor in ls_eo_p1:
                if cor == cor_center:  # when the center point and the point  are coincide , set the cos value 1
                    ls_cos.append((cor, 1))
                    continue
                cos = calcu_cos(get_vector(cor, cor_center), v1)
                ls_cos.append((cor, cos))

            ls_sort = sorted(ls_cos, key=lambda x: x[1], reverse=True)
            for i in ls_sort:
                ls_cor.append(i[0])
            ls_cos = []
            for cor in ls_eo_p2:
                if cor == cor_center:  # when the center point and the point  are coincide , set the cos value 1
                    ls_cos.append((cor, 1))
                    continue
                cos = calcu_cos(get_vector(cor, cor_center), v1)
                ls_cos.append((cor, cos))
            ls_sort = sorted(ls_cos, key=lambda x: x[1])
            for i in ls_sort:
                ls_cor.append(i[0])
            # set looping
            l = len(ls_cor)
            if l < RING_SIZE:
                for i in range(RING_SIZE - l):
                    ls_cor.append(ls_cor[i])
            elif l > RING_SIZE:
                raise Exception("RING_SIZE is not big enough .The ring size here :", l, " index:", index, " z : ", z)
            if len(ls_cor) != RING_SIZE:
                raise Exception("Size erro here : length", len(ls_cor), " index:", index, " z : ", z)
            lt.append(ls_cor)

        # extend the levels which is not enough
        flag = 0
        while len(lt) < LEVEL_SIZE:
            lt.insert(-flag, lt[-flag])
            flag = (flag + 1) % 2

        # delete the levels which is oversize
        flag = -1
        while len(lt) > LEVEL_SIZE:
            # step1 : delete the coinsided elements
            if flag == -1:
                flag = 0
                for ix, x in enumerate(lt):

                    if ix == len(lt) - 2:
                        break
                    if lt[ix] == lt[ix + 1]:
                        lt.pop(ix + 1)
                    if len(lt) == LEVEL_SIZE:
                        break
            # if step 1 is not enough ,delete the elements from start and end
            else:
                lt.pop(-flag)
                flag = (flag + 1) % 2

        if len(lt) != LEVEL_SIZE:
            raise Exception("the length of level is wrong:" + len(lt))

        ls_cors.append(lt)
    arr_cors = np.array(ls_cors, dtype=np.float32)
    arr_cors = arr_cors.reshape((-1, LEVEL_SIZE, RING_SIZE, 2))
    return arr_cors

'''
input  : 32 * 32 * 32 model array
output : levesize * ringsize * 3
'''
def load_data_permodel(array):
    lt = []  # l - levels in every model
    for z in range(dim):  # for each level

        ar_p = get_plane(array, z)
        if ar_p == []:
            continue

        ls_e, ls_eo = get_edge(ar_p)
        # get center point
        cor_center = get_centerpoint_axis(ls_eo)
        # get 2 parts of the ouside edge list (1:up 2:down)
        ls_eo_p1 = []
        ls_eo_p2 = []
        for cor in ls_eo:
            # cor= list(cor)   # change coordinate tuple into list
            if cor[0] <= cor_center[0]:
                ls_eo_p1.append(cor)
            else:
                ls_eo_p2.append(cor)
        # sorted the coordinate sequence
        ls_cos = []
        ls_cor = []
        v1 = (0, -1)  # v1 : the start vector

        for cor in ls_eo_p1:
            if cor == cor_center:  # when the center point and the point  are coincide , set the cos value 1
                ls_cos.append((cor, 1))
                continue
            cos = calcu_cos(get_vector(cor, cor_center), v1)
            ls_cos.append((cor, cos))

        ls_sort = sorted(ls_cos, key=lambda x: x[1], reverse=True)
        for i in ls_sort:
            ls_cor.append(i[0])
        ls_cos = []
        for cor in ls_eo_p2:
            if cor == cor_center:  # when the center point and the point  are coincide , set the cos value 1
                ls_cos.append((cor, 1))
                continue
            cos = calcu_cos(get_vector(cor, cor_center), v1)
            ls_cos.append((cor, cos))
        ls_sort = sorted(ls_cos, key=lambda x: x[1])
        for i in ls_sort:
            ls_cor.append(i[0])
        # set looping
        l = len(ls_cor)
        if l < RING_SIZE:
            for i in range(RING_SIZE - l):
                ls_cor.append(ls_cor[i])
        elif l > RING_SIZE:
            raise Exception("RING_SIZE is not big enough .The ring size here :", l,  " z : ", z)
        if len(ls_cor) != RING_SIZE:
            raise Exception("Size erro here : length", len(ls_cor),  " z : ", z)
        lt.append(ls_cor)

    # extend the levels which is not enough
    flag = 0
    while len(lt) < LEVEL_SIZE:
        lt.insert(-flag, lt[-flag])
        flag = (flag + 1) % 2

    # delete the levels which is too enough
    flag = -1
    while len(lt) > LEVEL_SIZE:
        # step1 : delete the coinsided elements
        if flag == -1:
            flag = 0
            for ix, x in enumerate(lt):

                if ix >= len(lt) - 2:
                    break
                if lt[ix] == lt[ix + 1]:
                    lt.pop(ix + 1)
                if len(lt) == LEVEL_SIZE:
                    break
        # if step 1 is not enough ,delete the elements from start and end
        else:
            lt.pop(-flag)
            flag = (flag + 1) % 2

    if len(lt) != LEVEL_SIZE:
        raise Exception("the length of level is wrong:" + len(lt))


    arr_cors = np.array(lt, dtype=np.float32)
    arr_cors = arr_cors.reshape((LEVEL_SIZE, RING_SIZE, 2))
    return arr_cors

def load_data_permodel_new(array):
    ls_cors = []
    lt = []  # l - levels in every model
    for z in range(dim):  # for each level

        ar_p = get_plane(array, z)
        if ar_p == []:
            continue

        ls_e, ls_eo = get_edge(ar_p)
        # get center point
        cor_center = get_centerpoint_axis(ls_eo)
        # get 2 parts of the ouside edge list (1:up 2:down)
        ls_eo_p1 = []
        ls_eo_p2 = []
        for cor in ls_eo:
            # cor= list(cor)   # change coordinate tuple into list
            if cor[0] <= cor_center[0]:
                ls_eo_p1.append(cor)
            else:
                ls_eo_p2.append(cor)
        # sorted the coordinate sequence
        ls_cos = []
        ls_cor = []
        v1 = (0, -1)  # v1 : the start vector

        for cor in ls_eo_p1:
            if cor == cor_center:  # when the center point and the point  are coincide , set the cos value 1
                ls_cos.append((cor, 1))
                continue
            cos = calcu_cos(get_vector(cor, cor_center), v1)
            ls_cos.append((cor, cos))

        ls_sort = sorted(ls_cos, key=lambda x: x[1], reverse=True)
        for i in ls_sort:
            ls_cor.append(i[0])
        ls_cos = []
        for cor in ls_eo_p2:
            if cor == cor_center:  # when the center point and the point  are coincide , set the cos value 1
                ls_cos.append((cor, 1))
                continue
            cos = calcu_cos(get_vector(cor, cor_center), v1)
            ls_cos.append((cor, cos))
        ls_sort = sorted(ls_cos, key=lambda x: x[1])
        for i in ls_sort:
            ls_cor.append(i[0])
        # set looping
        l = len(ls_cor)
        if l < RING_SIZE:
            for i in range(RING_SIZE - l):
                ls_cor.append(ls_cor[i])
        elif l > RING_SIZE:
            raise Exception("RING_SIZE is not big enough .The ring size here :", l,  " z : ", z)
        if len(ls_cor) != RING_SIZE:
            raise Exception("Size erro here : length", len(ls_cor),  " z : ", z)
        lt.append(ls_cor)

    # extend the levels which is not enough
    flag = 0
    while len(lt) < LEVEL_SIZE:
        lt.insert(-flag, lt[-flag])
        flag = (flag + 1) % 2

    # delete the levels which is too enough
    flag = -1
    while len(lt) > LEVEL_SIZE:
        # step1 : delete the coinsided elements
        if flag == -1:
            flag = 0
            for ix, x in enumerate(lt):

                if ix >= len(lt) - 2:
                    break
                if lt[ix] == lt[ix + 1]:
                    lt.pop(ix + 1)
                if len(lt) == LEVEL_SIZE:
                    break
        # if step 1 is not enough ,delete the elements from start and end
        else:
            lt.pop(-flag)
            flag = (flag + 1) % 2

    if len(lt) != LEVEL_SIZE:
        raise Exception("the length of level is wrong:" + len(lt))


    arr_cors = np.array(lt, dtype=np.float32)
    arr_cors = arr_cors.reshape((LEVEL_SIZE, RING_SIZE, 2))
    return arr_cors

