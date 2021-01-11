# distutils: language=c++
cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport calloc, free

# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef target_mean_v3(cnp.ndarray[long, ndim=2] data):
    cdef cnp.ndarray[dtype=long, ndim=2, mode='fortran'] matrix = np.asfortranarray(data, dtype=np.int)
    cdef:
        int i = 0
        dict value_dict = {}
        dict count_dict = {}
        int n = matrix.shape[0]

    for i in range(n):
        key = matrix[i][1]
        value = matrix[i][0]
        if key not in value_dict.keys():
            value_dict[key] = value
            count_dict[key] = 1
        else:
            value_dict[key] += value
            count_dict[key] += 1

    cdef list ret = [i for i in range(n)]
    for i in range(n):
        key = matrix[i][1]
        value = matrix[i][0]
        ret[i] = (value_dict[key] - value) / (count_dict[key] - 1)

    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef target_mean_v4(cnp.ndarray data, int n, int m):
    cdef cnp.ndarray[dtype=long, ndim=2, mode='fortran'] matrix = np.asfortranarray(data, dtype=np.int)
    cdef:
        int i = 0
        long *value_array = <long *> calloc(m * sizeof(long), sizeof(long))
        long *count_array = <long *> calloc(m * sizeof(long), sizeof(long))

    for i in range(n):
        index = matrix[i][1]
        value = matrix[i][0]
        value_array[index] += value
        count_array[index] += 1

    cdef list ret = [i for i in range(n)]
    for i in range(n):
        index = matrix[i][1]
        value = matrix[i][0]
        ret[i] = (value_array[index] - value) / (count_array[index] - 1)
    free(value_array)
    free(count_array)
    return ret