# cython: language_level=3

import cython
from cython.parallel cimport prange, parallel
cimport numpy as cnp
import numpy as np

# Cython typedefs
ctypedef cnp.int64_t DTYPE_t

def floyd_warshall(cnp.ndarray[DTYPE_t, ndim=2] adjacency_matrix):

    cdef int nrows = adjacency_matrix.shape[0]
    cdef int ncols = adjacency_matrix.shape[1]
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(np.int64, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']

    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] M = adj_mat_copy
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] path = np.zeros([n, n], dtype=np.int64)

    cdef unsigned int i, j, k
    cdef DTYPE_t M_ij, M_ik, cost_ikkj
    cdef DTYPE_t* M_ptr = &M[0,0]
    cdef DTYPE_t* M_i_ptr
    cdef DTYPE_t* M_k_ptr

    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path


def get_all_edges(cnp.ndarray[DTYPE_t, ndim=2] path, int i, int j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(int max_dist, cnp.ndarray[DTYPE_t, ndim=2] path, cnp.ndarray[DTYPE_t, ndim=3] edge_feat):

    cdef int nrows = path.shape[0]
    cdef int ncols = path.shape[1]
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(np.int64, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(np.int64, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef cnp.ndarray[DTYPE_t, ndim=4, mode='c'] edge_fea_all = -1 * np.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=np.int64)

    cdef unsigned int i, j, k, num_path
    cdef list path_list

    for i in range(n):
        for j in range(n):
            if i == j or path_copy[i][j] == 510:
                continue

            path_list = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path_list) - 1

            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path_list[k], path_list[k+1], :]

    return edge_fea_all
