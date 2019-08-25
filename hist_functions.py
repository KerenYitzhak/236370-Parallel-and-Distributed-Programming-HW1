import os
import numpy as np
from numba import int32, cuda, njit
import timeit


def hist_cpu(A):
    """
     Returns
     -------
     np.array
         histogram of A of size 256
     """
    res = np.zeros(256)
    for i in range(len(A)):
        res[A[i]] += 1
    return res


@njit
def hist_numba(A):
    """
     Returns
     -------
     np.array
         histogram of A of size 256
     """
    res = np.zeros(256)
    for i in range(len(A)):
        res[A[i]] += 1
    return res


def hist_gpu(A):
    # Allocate the output np.array histogram C in GPU memory using cuda.to_device
    ary = np.zeros(256, dtype=int)

    C = cuda.to_device(ary)

    # invoke the hist kernel with 1000 threadBlocks with 1024 threads each
    hist_kernel[1000, 1024](A, C)

    # copy the output histogram C from GPU to cpu using copy_to_host()
    return C.copy_to_host()


@cuda.jit
def hist_kernel(A, C):
    # Thread id in a 1D block:
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    bx = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x

    # Compute flattened index inside the array
    pos = tx + bx * bw
    # Check array boundaries
    if pos < A.size:
        # Update the histogram: C[A[pos]]++
        cuda.atomic.add(C, A[pos], 1)
        # Wait until all threads finish
        cuda.syncthreads()


# this is the comparison function - keep it as it is, don't change A.
def hist_comparison():
    A = np.random.randint(0, 256, 1000 * 1024)

    def timer(f):
        return min(timeit.Timer(lambda: f(A)).repeat(3, 20))

    print('CPU:', timer(hist_cpu))
    print('Numba:', timer(hist_numba))
    print('CUDA:', timer(hist_gpu))


if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
    hist_comparison()
