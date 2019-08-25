import os
import numpy as np
from numba import njit, cuda
import timeit


def matmul_trivial(X, Y):
    res = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                res[i][j] += X[i][k] * Y[k][j]
    return res


@njit
def matmul_numba(X, Y):
    res = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                res[i][j] += X[i][k] * Y[k][j]
    return res


def matmul_gpu(X, Y):
    # Allocate the output matrix in GPU memory using cuda.to_device
    ary = np.zeros((X.shape[0], Y.shape[1]))
    C = cuda.to_device(ary)

    # invoke the dot kernel with 1 threadBlock with 1024 threads
    matmul_kernel[1, 1024](X, Y, C)

    # copy the output matrix from GPU to cpu using copy_to_host()
    return C.copy_to_host()


@cuda.jit
def matmul_kernel(A, B, C):
    # Thread id in a 1D block:
    tx = cuda.threadIdx.x
    # Block width, i.e. number of threads per block:
    bw = cuda.blockDim.x

    # Size of C (equal to A.shape[0] x B.shape[1]):
    C_size = C.shape[0] * C.shape[1]

    # The number of cells in C that under the responsibility of the current thread:
    range_size = C_size // bw
    # The start-index and end-index of all the cells in C that under the responsibility of the current thread:
    start = tx * range_size
    end = start + range_size - 1

    # In case C size is smaller than the number of threads: Each thread responsible for only one cell in C
    if range_size == 0:
        start = end = tx

    # In case C size is not divisible by the number of threads: The last thread is responsible for the remainder
    if (tx == bw - 1) & (C_size % bw > 0):
        end = C_size - 1

    # Perform the matrices multiplication:
    for Ci in range(start, end + 1):
        row = Ci // C.shape[1]
        col = Ci % C.shape[1]
        if row < C.shape[0] and col < C.shape[1]:
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]
            C[row, col] = tmp
        else:
            break


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Y = np.random.randn(128, 64)

    def timer(f):
        return min(timeit.Timer(lambda: f(X, Y)).repeat(3, 100))

    # print('Python:', timer(matmul_trivial)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul))
    print('Numba:', timer(matmul_numba))
    print('CUDA:', timer(matmul_gpu))


if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
    matmul_comparison()
