#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <cassert>

/* Helper function to round up to a power of 2. */
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

#define BLOCK_SIZE 128
#define UPSWEEP_STEP 8
#define DOWNSWEEP_STEP 32
// 128 Additions per block, 4 warps
// Each block takes cares of 256 interested values

__device__ void upsweepNaive(int* data, int N, int width) {
    const int halfWidth = width >> 1;
    const int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int dataIdx = threadId * width;
    int srcIdx = (halfWidth - 1) + dataIdx;
    int destIdx = srcIdx + halfWidth;
    if (destIdx >= N) return;
    data[destIdx] += data[srcIdx];
}

// Performs up sweep by summing adjacent [beginStep, endStep) elements
// Arguments: data - Data in device memory 
//            N    - Total size of the array (rounded up)
//            beginStep - distance between elements in first iteratrion
//            endStep - distance between elements in last iteration
// Requires:  beginStep <= endStep, both are powers of 2
__global__ void upsweep(int* data, int N, int startWidth, int endWidth) {
    __shared__ int buf[BLOCK_SIZE];

    const int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    // Each thread takes care of adjacent 2 values
    int dataIdx = threadId * startWidth;

    // Check for array out of bound
    if (dataIdx >= N) return;

    if (2 * startWidth == endWidth) {
        upsweepNaive(data, N, startWidth);
        return;
    }

    int bufWidth = 1;
    int dataWidth = startWidth;
    while (dataWidth < endWidth) {
        if (dataWidth == startWidth) {
            // Take data into shared memory while compacting them.
            // The "first" iteration is also performed here
            int halfWidth = dataWidth >> 1;
            buf[threadIdx.x] = data[dataIdx + halfWidth - 1] + data[dataIdx + dataWidth - 1];
        } else {
            int halfWidth = bufWidth >> 1;
            int srcIdx = (halfWidth - 1) + threadIdx.x * bufWidth;
            int destIdx = srcIdx + halfWidth;
            if (destIdx < BLOCK_SIZE) buf[destIdx] += buf[srcIdx];
        }
        dataWidth <<= 1;
        bufWidth <<= 1;
        __syncthreads();
    }

    // Return the data to global memory
    data[dataIdx + startWidth - 1] = buf[threadIdx.x];
}

__global__ void cudaSet(int* device_data, int val) {
    *device_data = val;
}

__device__ void downsweepNaive(int* data, int N, int width) {
    const int halfWidth = width >> 1;
    const int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int dataIdx = threadId * width;
    int midIdx = (halfWidth - 1) + dataIdx;
    int lastIdx = midIdx + halfWidth;
    if (lastIdx >= N) return;
    int t = data[lastIdx];
    data[lastIdx] += data[midIdx];
    data[midIdx] = t;
}

__global__ void downsweep(int* data, int N, int startWidth, int endWidth) {
    assert(blockDim.x == BLOCK_SIZE);

    __shared__ int buf[2 * BLOCK_SIZE];

    const int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    // Each thread takes care of adjacent 2 values
    int dataIdx = 2 * threadId * endWidth;

    // Check for array out of bound
    if (dataIdx >= N) return;

    // Load data to share memory
    int mid = dataIdx + endWidth - 1;
    buf[2 * threadIdx.x] = data[mid];
    buf[2 * threadIdx.x + 1] = data[mid + endWidth];
    __syncthreads();

    int width = startWidth;
    int bufWidth = startWidth / endWidth; // There are two elements per endWidth
    while (width > endWidth) {
        int halfWidth = bufWidth >> 1;
        int accIdx = bufWidth * threadIdx.x + bufWidth - 1;
        int midIdx = accIdx - halfWidth;
        if (accIdx < 2 * BLOCK_SIZE) {
            int t = buf[accIdx];
            buf[accIdx] += buf[midIdx];
            buf[midIdx] = t;
        }
        bufWidth >>= 1;
        width >>= 1;
        __syncthreads();
    }

    // Return data to global memory
    data[mid] = buf[2 * threadIdx.x];
    data[mid + endWidth] = buf[2 * threadIdx.x + 1];

    return;
}

__global__ void cudaMoveIndexByMask(int* mask, int* indicies, int* target, int length, int* count) {
    const int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId >= length) return;
    if (mask[threadId]) target[indicies[threadId]] = threadId;
    // The last thread will be in charge of put the count into 
    if (threadId == length - 1 && count != nullptr) {
        if (mask[threadId]) {
          *count = indicies[threadId] + 1;
        } else {
          *count = indicies[threadId];
        }
    }
}

void exclusive_scan(int* device_data, int length) {
    int pow2 = nextPow2(length);

    int width = 2;
    while (width < pow2) {
        int blockCount = pow2 / (BLOCK_SIZE * width) + 1;
        int endWidth = width * UPSWEEP_STEP >= pow2 ? pow2 : width * UPSWEEP_STEP;
        upsweep<<<blockCount, BLOCK_SIZE>>>(device_data, pow2, width, endWidth);
        width = endWidth;
    }

    cudaSet<<<1, 1>>>(device_data + pow2 - 1, 0);

    width = pow2;
    while (width > 1) {
        int t = width / DOWNSWEEP_STEP;
        int nextWidth = (t > 0) ? t : 1;
        int blockCount = pow2 / (BLOCK_SIZE * nextWidth * 2) + 1;
        downsweep<<<blockCount, BLOCK_SIZE>>>(device_data, pow2, width, nextWidth);
        width = nextWidth;
    }
}

// Requires device_list to be at least as long as length
// Scratch must be as large as next power-of-two of lenght
void compact_mask_to_list(int* deviceMask, int* deviceList, int* scratch, int length, int* count) {
    // Copy center information to a temp array
    int* indicies = scratch;
    cudaMemcpy(indicies, deviceMask, sizeof(int) * length, cudaMemcpyDeviceToDevice);
    // Perform exclusive scan
    exclusive_scan(indicies, length);
    int blockCount = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Copy items into their indicies
    cudaMoveIndexByMask<<<blockCount, BLOCK_SIZE>>>(deviceMask, indicies, deviceList, length, count);
}