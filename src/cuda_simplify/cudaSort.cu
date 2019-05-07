#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cstdio>

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

#define TH_PER_BLOCK 64

// Sets keys with idx [length, nextPow2(length) ) as -1
// Sets keys with idx [0, length - 1] with its idx
// -1 signifies maximum possible weight
__global__ void cudaSortInit(int* keys, int length, int maxSize) {
    if (maxSize & (maxSize - 1)) {
        printf("WARINING: Length is not a power of two.");
        return;
    }
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Only going to modify ones past max length
    if (idx >= maxSize) {
        return;
    } else if (idx < length) {
        keys[idx] = idx;
    } else {
        keys[idx] = -1;
    }
}

// Requires length to be a power of 2
__global__ void cudaBitonicKernel(
    int* keys, float* value, int length, int step, int maxStep
) {
    if (length & (length - 1)) {
        printf("WARINING: Length is not a power of two.");
        return;
    }
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId >= length / 2) return;
    // The whole range is subdivided into regions of 2 * $step size.
    // Each region will use $step number of threads
    int regionSize = 2 * step;
    // Each region contains exactly $step threads
    int regionId = threadId / step;
    int idxInRegion = threadId % step;
    int thisIdx = regionId * regionSize + idxInRegion;
    int otherIdx = regionId * regionSize + step + idxInRegion; 
    int thisKey = keys[thisIdx], otherKey = keys[otherIdx];
    // Each maxStep region contains exactly maxStep active threads
    int maxStepRegionId = threadId / maxStep;
    bool needSwap = false;
    if (maxStepRegionId % 2 == 1) {
        // If we are in the upsweep phase, and region is odd, 
        // We need to ensure larger one is in lower index
        needSwap = (otherKey == -1 || 
            (thisKey != -1 && value[thisKey] < value[otherKey]));
    } else {
        // We need to ensure larger one is in higher index
        needSwap = (thisKey == -1 || 
            (otherKey != -1 && value[thisKey] > value[otherKey]));
    }
    if (needSwap) {
        keys[thisIdx] = otherKey;
        keys[otherIdx] = thisKey;
    }
}

__global__ void cudaRandomData(float* values, int length) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId >= length) return;
    values[threadId] = (threadId * 17167 + (17183 - threadIdx.x) * 15271) % 32768;
}

void generateRandomData(float* values, int length) {
    int nBlocks = (length + TH_PER_BLOCK - 1) / TH_PER_BLOCK;
    cudaRandomData<<<nBlocks, TH_PER_BLOCK>>>(values, length);
}

void prepareIndiciesForSort(int* keys, int length) {
    int p2len = nextPow2(length); // Power-of-2 length
    if (p2len > length) {
        // Needs to pad the keys
        int nBlocks = (p2len + TH_PER_BLOCK - 1) / TH_PER_BLOCK;
        cudaSortInit<<<nBlocks, TH_PER_BLOCK>>>(keys, length, p2len);
    }
}

void cudaSortKV(int* keys, float* values, int length) {
    int p2len = nextPow2(length); // Power-of-2 length
    // Only p2len / 2 of swaps may be required
    int nBlocks = (p2len / 2 + TH_PER_BLOCK - 1) / TH_PER_BLOCK;

    // Up sweep pahse
    for (int maxStep = 1; maxStep < p2len; maxStep *= 2) {
        for (int step = maxStep; step >= 1; step /= 2) {
            // printf("nblocks = %d, TH_PER_BLOCK = %d.\n", nBlocks, TH_PER_BLOCK);
            cudaBitonicKernel<<<nBlocks, TH_PER_BLOCK>>>(
                keys, values, p2len, step, maxStep
            );
        }
    }
}
