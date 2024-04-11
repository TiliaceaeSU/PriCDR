
"""
This is a python file to implement dp publishing of graph with Johnson-Lindenstrauss Transform 
and Fast Jonhson-Lindenstrauss Tranform. 
Visualize Laplacian matrix at the end. 
The visualization shows our method gets a good approximation of original one. 

Author: Huiwen Wu
Date: 08/26/2021
"""



import numpy as np 
from sklearn import random_projection as rp 
from sklearn.metrics.pairwise import euclidean_distances
import torch 
from matplotlib import pyplot as plt
import math
np.random.seed(234)
from scipy import linalg 
import scipy 
import pandas as pd

def get_parameters(eps, delta, eta, nu):
    """
    get parameters from privacy parameters 
    Input: --- eps: privacy parameter, ranging in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
           --- delta: privacy parameter, fix = 1/(number_of_data)
           --- eta: approximation parameter, determinng subspace dimesion in (0,0.5)
           --- nu: approximation parameter, determing the concentration probability 
    Output: --- r: dimension of subspace 
            --- w: weights to generate new graph 
    """

    r = 8* math.log(2/nu)/eta/eta
    # w = 16 * math.sqrt(r*math.log(2/delta))*math.log(16*r/delta)/eps
    # print('(eps, delta)-DP: (%2.2f, %2.5f)' %(eps, delta))
    # print('subspace dim is: %3d:' %r)
    # print('weight is %2.2f:' %w)
    eps0 = eps
    epss = eps0 * 2 * math.sqrt(r * math.log(2/delta))
    delta0 = delta/2/r 
    w = 1./(math.sqrt(eps0/ 2 / math.log(4 /delta0) + 0.25) - 0.5)

    print("eps is :"+str(epss))
    print("eps0 is :"+str(eps0))
    print("w is :"+str(w))
    

    return int(r), w


def random_projection(mat, dim=3):
    """
    Projects a matrix of dimensions m x oldim to m x dim using a random Projection
    """

    # project 
    m, oldim = mat.shape 
    # t = rp.GaussianRandomProjection()
    t = np.random.randn(dim , oldim)
    # proj_mat = torch.tensor(t._make_random_matrix(dim, oldim))
    proj_mat = torch.tensor(t)
    # proj_mat = proj_mat.to(device)
    output = torch.matmul(mat.double(), proj_mat.t())

    # check new dim 
    assert(output.shape[0] == m)
    assert(output.shape[1] == dim)

    return output 

def sparse_projection(mat, dim=3, p=1/3.0):
    """
    Projects a matrix of dimensions m x oldim to m x dim using a sparse random Projection
    """

    # project
    m, oldim = mat.shape
    t = rp.SparseRandomProjection(n_components = dim, density = p)
    output1 = t.fit_transform(mat)
    output = torch.tensor(output1)
    # proj_mat = torch.tensor(t._make_random_matrix(dim, oldim))
    # output = torch.matmul(mat.double(), proj_mat.t())

    # check new dim 
    assert(output.shape[0] == m)
    assert(output.shape[1] == dim)

    return output 



def graph_publish(edge, w, p, k, transform):
    """
    python file to output pertured edge matrix/ laplacian matrix 
    input: --- edge: edge/incidence matrix with 
                --- shape[0]: number of edges 
                --- shape[1]: number of nodes 
           --- w : privacy parameters
           --- p : sparse parameters 
           --- k : reduced dims 
           --- transform: JLT/FJLT 
    output: --- new_edge: perturbed edge/incidence matrix 
            --- new_lap: perturbed laplacian matrix 
    """
    print('transform is '+transform)
    print('subspace dim is: %3d' %k)
    print('weight is: %2.2f' %w)

    m, n = edge.shape 
    I = torch.tensor(np.ones(shape = (m,1))).long()
    edge = edge.long()

    # subtract mean from edge 
    # temp_edge = edge 
    temp_edge = torch.subtract(edge ,torch.mul(1/n, torch.matmul(I, torch.matmul(I.t(), edge))))

    # svd of temp_edge 
    U, s, Vh = linalg.svd(temp_edge, full_matrices=False, compute_uv=True)
    U = torch.tensor(U)
    s = torch.tensor(s)

    # s = torch.diag(torch.tensor(s))
    Vh = torch.tensor(Vh)

    # modify s 
    s2 = torch.sqrt(torch.add(torch.square(s), torch.square(torch.tensor(w))))
    s2 = torch.diag(s2)

    # ensemble to new edge 
    temp_edge2 = torch.matmul(U, torch.matmul(s2, Vh))

    # random transform 

    if transform == 'JLT':
        new_edge = random_projection(temp_edge2, dim=k)
    elif transform == 'FJLT':
        new_edge = sparse_projection(temp_edge2, dim=k, p = p)


    new_edge = torch.mul(1/math.sqrt(k), new_edge)

    new_lap = torch.mul(1/k, torch.matmul(new_edge, new_edge.t()))
    

    return new_edge, new_lap 

def get_dp_matrix_and_plot(w, train_mat, epss, r):
    r = int(r)
    edge_w, edge_w1 = graph_publish(torch.tensor(train_mat).t(), w, 0.9, r, 'JLT' )
    # random_w = np.random.randn(edge_w.shape[0], edge_w.shape[1])
    svd_original = linalg.svd(edge_w, compute_uv=False)
    svd_w = linalg.svd(edge_w1, compute_uv=False)
    # svd_random = linalg.svd(random_w, compute_uv=False)
    plt.plot(svd_original[:r], label = 'original')
    plt.plot(svd_w[:r], label = 'perturbed')
    # plt.plot(svd_random[:r] ,label = 'random' )
    plt.legend()
    plt.title('spectrum with epss = {}'.format(epss))
    plt.savefig('spectrum_{}.png'.format(epss))
    plt.show()


def prosvd(edge_0,num,pp,mode,et):
    ## step1 -- get parameters 
    # epss = [ 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    epss = [0.25, 0.5, 1, 2, 4, 8, 16,32,64]
    eta = et
    nu = 0.2
    r_list = []
    w_list = []
    ww_list = []
    p = pp # sparsity parameter 


    # COL_USER = "UserId"
    # COL_ITEM = "MovieId"
    # COL_RATING = "Rating"
    # COL_PREDICTION = "Rating"
    # COL_TIMESTAMP = "Timestamp"
    # data = pd.read_csv('data.txt', sep="\t", names=[COL_USER, COL_ITEM, COL_RATING, COL_TIMESTAMP])
    # col_user = data[COL_USER].to_numpy()
    # col_item = data[COL_ITEM].to_numpy()
    # col_rating = data[COL_RATING].to_numpy()
    # edge_0 = scipy.sparse.coo_matrix((col_rating,(col_user,col_item))).todense()

    n,m = edge_0.shape 
    print('number of users :', n)
    print('number of items :', m)

    # edge_0 = torch.tensor(edge_0)
    # edge_0 = edge_0.t()

    deltas = [1/n]

    for eps in epss:
        for delta in deltas:
            r, w = get_parameters(eps, delta, eta, nu)
            r_list.append(r)
            w_list.append(w)
    

    edge_1, lap_1 = graph_publish(edge_0, torch.tensor(w_list[num]), p, r_list[num], mode)
    return edge_1



## step2 -- generate test matrix and perturb it 

## step3 -- visualize Laplacian matrix 

# for r, w, eps in zip(r_list, w_list, epss):
#     lap_0 = torch.mul(math.sqrt(1/r),torch.matmul(edge_0, edge_0.t()))
#     edge_1, lap_1 = graph_publish(edge_0, torch.tensor(w), p, r, 'JLT' )
#     edge_2, lap_2 = graph_publish(edge_0, torch.tensor(w), p, r, 'FJLT' )
#     fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
#     for ax, status, mat in zip((ax1, ax2, ax3), ('Original', 'Random', 'Sparse'),(lap_0, lap_1, lap_2)):
#         cells = ax.pcolor(mat, cmap=plt.cm.PuBu)
#         ax.set_title('{}_eps_{}_w_{}'.format(status, eps, w))
#     plt.savefig('laplacian_matrix_{}_{}.png'.format(status,eps))
#     plt.show()

    



