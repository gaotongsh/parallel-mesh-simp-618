#include <cassert>
#include <cstdio>
#include <cfloat>

#include <utility>
#include <string>
#include <unordered_map>
#include <map>
#include <unordered_set>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "gpuSimplify.h"
#define TIME 1
#include "cycleTimer.h"

#define IDX2C(i,j) ((3*(j))+(i))

#define INCIDENCE_DATA_MAGIC 99912345

#define NDEBUG

struct IncidenceHeader {
  int size, cont;
};

struct CudaMesh {
  int nVertices;
  int nVerticesRemaining;
  float3* vertices;

  int nTriangles;
  int nTrianglesRemaining;
  int3* triangles;

  float* trianglesQuadric;
  float* verticesQuadricBuffer;
  float* verticesQuadric;

  int* verticesIndex;
  int* verticesEdgeToDelete;
  float* verticesError;
  float3* verticesNewPos;

  int* trianglesMask; // Is the triangle still in mesh?
  int* verticesMask; // Is the current vertex still in mesh?

  int              incidenceCount;
  IncidenceHeader* incidenceHeader;
  int*             incidenceData;
  // When we compact, we copy and switch buffer
  IncidenceHeader* incidenceHeaderCompact; 
  int*             incidenceDataCompact;

  // Size of nVertices. Keeps track of independence info
  int* independenceUsed; 
  int* independenceCenterMask;
  int* independenceCenterList;
  int  nIndependenceCenters;

  // Error of independence center information 
  // Use float* verticesError

  GpuMeshSimplifyParams params;
};

// Detects CUDA device status.
void cudaDetect();

// Verifies data structures are initialized properly
void verifyInitialization(GpuMesh* meshRef, CudaMesh* deviceData);

// Verifies incidence datat generated is consistent with original data
void verifyIncidenceData(GpuMesh* meshRef, CudaMesh* deviceData);

// This only checks if incidence data structure is locally consistent
void verifyIncidenceDataInProgress(GpuMesh* meshRef, CudaMesh* deviceData);

// Verifies super-independence property holds for current device data
void verifySuperIndependence(GpuMesh* meshRef, CudaMesh* deviceData);

// Verifies bitonic sort generates correct data
void verifySorting(GpuMesh* meshRef, CudaMesh* deviceData);

// In place exclusive scan, implemented in cudaScan.cu
// Input array size must be at least as large as next power-of-2 of length
void exclusive_scan(int* device_data, int length);

// Requires device_list to be at least as long as length
// Scratch must be as large as next power-of-two of lenght
void compact_mask_to_list(int* deviceMask, int* deviceList, int* scratch, int length, int* deviceCount);

// Keys are associate with a weight of value[key]
// In place sorts the key so that their values are in ascending order
// Requires keys list to be as large as next power-of-two of length
void cudaSortKV(int* keys, float* value, int length);
// Populates values with fixed set of random data
void generateRandomData(float* values, int length);
// Populates keys array with [0, ... , length - 1, -1, -1, ... ]
void prepareIndiciesForSort(int* keys, int length);

// Main function of mesh simplification
void cudaMeshSimplify(GpuMesh *from, GpuMesh *to, GpuMeshSimplifyParams params);

static __device__ __inline__ uint64_t primePermute(uint64_t prime, int gdim, int bdim, int bid, int tid) {
  uint64_t d = bdim, b = bid, t = tid, g = gdim;
  uint64_t threadId = d * b + t;
  return (threadId * prime) % (g * d);
}

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

template <typename S, typename T>
static T copyFieldFromGPU(S* sLocal, T* fieldPtr, S* sDevice) {
  int ptrDiff = (uint8_t*)(fieldPtr) - (uint8_t*)sLocal;
  T* devicePtr = (T*)((uint8_t*)sDevice + ptrDiff);
  T temp;
  cudaMemcpy(&temp, devicePtr, sizeof(T), cudaMemcpyDeviceToHost);
  return temp;
}

template <typename S, typename T>
static void copyFieldToGpu(S* sLocal, T* fieldPtr, S* sDevice, T value) {
  int ptrDiff = (uint8_t*)(fieldPtr) - (uint8_t*)sLocal;
  T* devicePtr = (T*)((uint8_t*)sDevice + ptrDiff);
  cudaMemcpy(devicePtr, &value, sizeof(T), cudaMemcpyHostToDevice);
}

double timeDiff(double current, double& last, double scale = 1000000.0) {
  auto t = scale * (current - last);
  last = current;
  return t;
}


// Device code
__device__  int atomicLoad(const int *addr) {
  const volatile int *vaddr = addr; // volatile to bypass cache
  __threadfence(); // for seq_cst loads. Remove for acquire semantics.
  const int value = *vaddr;
  // fence to ensure that dependent reads are correctly ordered
  __threadfence(); 
  return value; 
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator/(const float3 &a, const float &b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}


__device__ float3 cudaMatMul3(const float* A, const float3 &v) {
  return make_float3( A[IDX2C(0,0)]*v.x + A[IDX2C(0,1)]*v.y + A[IDX2C(0,2)]*v.z,
                      A[IDX2C(1,0)]*v.x + A[IDX2C(1,1)]*v.y + A[IDX2C(1,2)]*v.z,
                      A[IDX2C(2,0)]*v.x + A[IDX2C(2,1)]*v.y + A[IDX2C(2,2)]*v.z );
}

__device__ float3 cudaCross3(const float3 &u, const float3 &v) {
  return make_float3( u.y*v.z - u.z*v.y,
                      u.z*v.x - u.x*v.z,
                      u.x*v.y - u.y*v.x );
}

__device__ float cudaDot3(const float3 &u, const float3 &v) {
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

__device__ float cudaDet3x3(const float* A) {
  return -A[IDX2C(0,2)]*A[IDX2C(1,1)]*A[IDX2C(2,0)]
         +A[IDX2C(0,1)]*A[IDX2C(1,2)]*A[IDX2C(2,0)]
         +A[IDX2C(0,2)]*A[IDX2C(1,0)]*A[IDX2C(2,1)]
         -A[IDX2C(0,0)]*A[IDX2C(1,2)]*A[IDX2C(2,1)]
         -A[IDX2C(0,1)]*A[IDX2C(1,0)]*A[IDX2C(2,2)]
         +A[IDX2C(0,0)]*A[IDX2C(1,1)]*A[IDX2C(2,2)];
}

__device__ void cudaInv3x3(const float* A, float* B) {
  B[IDX2C(0,0)] = -A[IDX2C(1,2)]*A[IDX2C(2,1)] + A[IDX2C(1,1)]*A[IDX2C(2,2)];
  B[IDX2C(0,1)] =  A[IDX2C(0,2)]*A[IDX2C(2,1)] - A[IDX2C(0,1)]*A[IDX2C(2,2)];
  B[IDX2C(0,2)] = -A[IDX2C(0,2)]*A[IDX2C(1,1)] + A[IDX2C(0,1)]*A[IDX2C(1,2)];
  B[IDX2C(1,0)] =  A[IDX2C(1,2)]*A[IDX2C(2,0)] - A[IDX2C(1,0)]*A[IDX2C(2,2)];
  B[IDX2C(1,1)] = -A[IDX2C(0,2)]*A[IDX2C(2,0)] + A[IDX2C(0,0)]*A[IDX2C(2,2)];
  B[IDX2C(1,2)] =  A[IDX2C(0,2)]*A[IDX2C(1,0)] - A[IDX2C(0,0)]*A[IDX2C(1,2)];
  B[IDX2C(2,0)] = -A[IDX2C(1,1)]*A[IDX2C(2,0)] + A[IDX2C(1,0)]*A[IDX2C(2,1)];
  B[IDX2C(2,1)] =  A[IDX2C(0,1)]*A[IDX2C(2,0)] - A[IDX2C(0,0)]*A[IDX2C(2,1)];
  B[IDX2C(2,2)] = -A[IDX2C(0,1)]*A[IDX2C(1,0)] + A[IDX2C(0,0)]*A[IDX2C(1,1)];

  float det = cudaDet3x3(A);

  for (int i = 0; i < 9; ++i) {
    B[i] /= det;
  }
}

__device__ void cudaComputeQuadric(const float* n, const float d,  float* q) {
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      q[IDX2C(i,j)] = n[j] * n[i];
    }
    q[9 + j] = d * n[j];
  }
  q[12] = d * d;
}

__device__ void cudaAddQuadric(float* q, const float* q2) {
  for (int i = 0; i < 13; ++i) {
    q[i] += q2[i];
  }
}

__device__ void cudaAddQuadric(float* q, const float* q2, const float* q3) {
  for (int i = 0; i < 13; ++i) {
    q[i] = q2[i] + q3[i];
  }
}

__device__ float3 cudaNormal(const float3 &v1, const float3 &v2, const float3 &v3) {
  float3 nv = cudaCross3(v2 - v1, v3 - v2);
  float norm = normf(3, (float*)(&nv));
  return make_float3(nv.x / norm, nv.y / norm, nv.z / norm);
}

__device__ __inline__ int cudaFindEdgeOfTriangle(const int vertex, const int3& triangle) {
  if (vertex == triangle.x) return triangle.y;
  if (vertex == triangle.y) return triangle.z;
  if (vertex == triangle.z) return triangle.x;
  // printf("Error. Edge not in triangle.\n");
  return -1;
}

__device__ float cudaEvaluateQuadric(const float* quadric, const float3 v) {
  float3 b = make_float3(quadric[9], quadric[10], quadric[11]);
  float error = cudaDot3(v, cudaMatMul3(quadric, v)) + 2 * cudaDot3(b, v) + quadric[12];
  return error;
}

__device__ void cudaPrintQuadric(const float* q) {
  printf("Q: [");
  for (int i = 0; i < 3; ++i) {
    printf("[");
    for (int j = 0; j < 3; ++j) {
      printf("%f", q[IDX2C(i,j)]);
      if (j != 2) {
        printf(", ");
      }
    }
    printf("]");
    if (i != 2) {
      printf(", ");
    }
  }
  printf("], b: [%f, %f, %f], d: %f\n", q[9], q[10], q[11], q[12]);
}

// This function computes the quadric of a triangle.
// It has been verified to be correct.
__global__ void cudaComputeTriangleQuadric(CudaMesh* mesh) {
  // Each cuda thread will in charge of processing one triangle
  int triangleId = blockDim.x * blockIdx.x + threadIdx.x;
  if (triangleId >= mesh->nTriangles) return;

  int3 triangle = mesh->triangles[triangleId];
  float3 v1 = mesh->vertices[triangle.x],
         v2 = mesh->vertices[triangle.y],
         v3 = mesh->vertices[triangle.z];
  float3 norm = cudaNormal(v1, v2, v3);
  float d = -cudaDot3(norm, v1);
  float* quadric = &mesh->trianglesQuadric[13 * triangleId];
  cudaComputeQuadric((float*)(&norm), d, quadric);
}

// This function computes the quadric of a vertex.
// It has been verified to be correct.
__global__ void cudaComputeVertexQuadric(CudaMesh* mesh) {
  int vertexId = blockDim.x * blockIdx.x + threadIdx.x;
  if (vertexId >= mesh->nVertices) return;

  int* incData = mesh->incidenceData;
  float* quadric = &mesh->verticesQuadric[13 * vertexId];

  int headerIdx = vertexId;
  while (headerIdx >= 0) {
    IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
    int base = headerIdx * mesh->params.incidenceDataPerBlock;
    for (int i = 0; i < header->size; ++i) {
      int otherTriangleIdx = incData[base + i];
      float* otherTriangleQ = &mesh->trianglesQuadric[13 * otherTriangleIdx];
      cudaAddQuadric(quadric, otherTriangleQ);
    }
    // Go on and check next header block
    headerIdx = header->cont;
  };
}

__global__ void cudaFindBestEdgeForZone(CudaMesh* mesh) {
  int centerId = blockDim.x * blockIdx.x + threadIdx.x;
  if (centerId >= mesh->nIndependenceCenters) return;

  int vertexId = mesh->independenceCenterList[centerId];
  int* incData = mesh->incidenceData;
  float* quadric = &mesh->verticesQuadric[13 * vertexId];

  // Delete degenerated vertex here
  int size = (mesh->incidenceHeader + vertexId)->size;
  if (size == 0) {
    mesh->verticesMask[vertexId] = 0;
    atomicSub(&mesh->nVerticesRemaining, 1);
    mesh->verticesError[centerId] = FLT_MAX;
    return;
  }

  int* trianglesMask = mesh->trianglesMask;

#ifndef NDEBUG
  float* best_q_buffer = new float[13];
  float* best_q_inv_buffer = new float[9];
#endif

  float min_error = FLT_MAX;
  float3 best_vertex_pos;
  int best_edge;
  int headerIdx = vertexId;
  float* quadricBuffer = &mesh->verticesQuadricBuffer[26 * vertexId];
  float* quadricInvBuffer = quadricBuffer + 13;
  int global_edge_count = 0;
  while (headerIdx >= 0) {
    IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
    int base = headerIdx * mesh->params.incidenceDataPerBlock;
    for (int i = 0; i < header->size; ++i) {
      // Calculate the edge vertex
      int otherTriangleIdx = incData[base + i];
      if (trianglesMask[otherTriangleIdx] == 0) continue;

      // Delete degenerated triangles here
      int3 tri = mesh->triangles[otherTriangleIdx];
      if (tri.x == tri.y || tri.y == tri.z || tri.z == tri.x) {
        trianglesMask[otherTriangleIdx] = 0;
        atomicSub(&mesh->nTrianglesRemaining, 1);
        continue;
      }

      int otherVertex = cudaFindEdgeOfTriangle(vertexId, mesh->triangles[otherTriangleIdx]);
      float* otherEdgeQ = &mesh->verticesQuadric[13 * otherVertex];
      cudaAddQuadric(quadricBuffer, quadric, otherEdgeQ);
      float3 best_vertex;

      // Determine if to use special algorithm
      float det = cudaDet3x3(quadricBuffer);
      if (det < 1e-3) {
        float3 p0 = mesh->vertices[vertexId];
        float3 p1 = mesh->vertices[otherVertex];
        float3 delta = p0 - p1;
        while (normf(3, (float*)(&delta)) > 1e-4) {
          float cost0 = cudaEvaluateQuadric(quadricBuffer, p0);
          float cost1 = cudaEvaluateQuadric(quadricBuffer, p1);
          if (cost0 > cost1) {
            p0 = (p0 + p1) / 2;
          } else {
            p1 = (p0 + p1) / 2;
          }
          delta = p0 - p1;
        }
        best_vertex = p0;
      } else {
        // Solve for the best position and error
        cudaInv3x3(quadricBuffer, quadricInvBuffer);
//      if (centerId == -1) {
//        printf("NewQ:\n");
//        cudaPrintQuadric(quadricBuffer);
//        printf("ReversedQ:\n");
//        cudaPrintQuadric(quadricInvBuffer);
//        printf("Error on this side: %e, Error on the other side: %e\n",
//                cudaEvaluateQuadric(quadric, mesh->vertices[vertexId]),
//                cudaEvaluateQuadric(otherEdgeQ, mesh->vertices[vertexId])
//                );
//      }
        best_vertex = cudaMatMul3(quadricInvBuffer, *(float3*)(quadricBuffer + 9));
        best_vertex.x = -best_vertex.x;
        best_vertex.y = -best_vertex.y;
        best_vertex.z = -best_vertex.z;
      }

      //float error = cudaDot3(best_vertex, *(float3*)(quadricBuffer + 9)) + *(quadricBuffer + 12);
      float error = cudaEvaluateQuadric(quadricBuffer, best_vertex);

      if (error < min_error) {
        min_error = error;
        best_edge = otherTriangleIdx;
        best_vertex_pos = best_vertex;
#ifndef NDEBUG
        memcpy(best_q_buffer, quadricBuffer, 13);
        memcpy(best_q_inv_buffer, quadricInvBuffer, 9);
#endif
      }
      global_edge_count++;
    }
    // Go on and check next header block
    headerIdx = header->cont;
  };

  //if (global_edge_count == 0) {
  //  mesh->verticesMask[vertexId] = 0;
  //  atomicSub(&mesh->nVerticesRemaining, 1);
  //  mesh->verticesError[centerId] = FLT_MAX;
  //  return;
  //}

#ifndef NDEBUG
  float3 orig = mesh->vertices[vertexId];
  //float delta[] = {best_vertex_pos.x, best_vertex_pos.y, best_vertex_pos.z};
  float delta[] = {orig.x - best_vertex_pos.x, orig.y - best_vertex_pos.y, orig.z - best_vertex_pos.z};
  if (centerId < 100 && normf(3, delta) > 1) {
    printf("Center Id: %d\n", centerId);
    printf("Vertex Id: %d\n", vertexId);
    printf("Degree is at least: %d\n", (mesh->incidenceHeader + vertexId)->size);
    printf("Degree is actually: %d\n", global_edge_count);
    cudaPrintQuadric(best_q_buffer);
    cudaPrintQuadric(best_q_inv_buffer);
    printf("Det:  %f\n", cudaDet3x3(best_q_buffer));
    printf("Orig: %f %f %f\n", mesh->vertices[vertexId].x, mesh->vertices[vertexId].y, mesh->vertices[vertexId].z);
    printf("New:  %f %f %f\n", best_vertex_pos.x, best_vertex_pos.y, best_vertex_pos.z);
    printf("Err:  %e\n", min_error);
    printf("============\n");
  }
#endif

  // Store the calculated data
  mesh->verticesEdgeToDelete[centerId] = best_edge;
  mesh->verticesError[centerId] = min_error;
  mesh->verticesNewPos[centerId] = best_vertex_pos;

#ifndef NDEBUG
  delete[] best_q_buffer;
  delete[] best_q_inv_buffer;
#endif
}

// Is not affected by masks
__global__ void cudaComputeTriangleIncidence(CudaMesh* mesh) {
  // Each cuda thread will in charge of processing one triangle
  // int triangleId = blockDim.x * blockIdx.x + threadIdx.x;
  int triangleId = primePermute(10111u, gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
  // triangleId = (triangleId * 10111) % (gridDim.x * blockDim.x);
  if (triangleId >= mesh->nTriangles) return;
  // printf("%d, %d.\n", blockDim.x, gridDim.x);
  if (triangleId == 17) {
      printf("Added triangle %d to its vertex's list.\n", triangleId);
  }

  int3 triangle = mesh->triangles[triangleId];
  int bsize = mesh->params.incidenceDataPerBlock;

  int vertices[] = {triangle.x, triangle.y, triangle.z};

  for (int i = 0; i < 3; ++i) {
    int headerIdx = vertices[i];
    while(1) {
      // Follow through the linked list to find latestNode
      IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
      int nextHeader = atomicCAS(&header->cont, INT_MAX, 0);
      if (nextHeader >= 0) {
        headerIdx = nextHeader;
        continue;
      }

      do {
        // May go over max size. We need to fix it later on
        int place = atomicAdd(&header->size, 1);
        if (place < bsize) {
          size_t baseIdx = headerIdx * bsize;
          mesh->incidenceData[baseIdx + place] = triangleId;
          goto success;
        } else {
          // Restore the header->size
          atomicMin(&header->size, bsize);
          int nextBlock = atomicAdd(&mesh->incidenceCount, 1);
          // printf("Incidence count becomes %d due to %d.\n", nextBlock, headerIdx);
          if (nextBlock >= mesh->params.incidenceHeaderSize) {
            printf("WARNING: Incidence header size (%d) exceeded! Cont = %d\n", 
                nextBlock, header->cont);
          }
          // Updade header->cont iff no one updated it before me
          atomicCAS(&header->cont, -1, nextBlock);
          goto failed;
        }
      } while(1);

    failed:
      continue;
    success:
      break;
    }
  }
}

// Masked vertex does not need to set its independence
__global__ void cudaComputeIndependence(CudaMesh* mesh) {
  int vertexId = primePermute(4637u, gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
  // int vertexId = blockDim.x * blockIdx.x + threadIdx.x;
  // // Use a prime to scatter access to avoid huge collisions
  // int prime = 4637;
  // vertexId = (vertexId * prime) % (blockDim.x * gridDim.x);
  if (vertexId >= mesh->nVertices) return;
  if (!mesh->verticesMask[vertexId]) return;

  int* center = mesh->independenceCenterMask;
  int* indep = mesh->independenceUsed;
  int* incData = mesh->incidenceData;

  // If a vertex is already used, we have nothing to do
  if (indep[vertexId] == 1) return;

  int headerIdx = vertexId;
  while (headerIdx >= 0) {
    IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
    int base = headerIdx * mesh->params.incidenceDataPerBlock;
    for (int i = 0; i < header->size; ++i) {
      int otherTriangleIdx = incData[base + i];
      // Skips triangles that are no longer valid
      if (!mesh->trianglesMask[otherTriangleIdx]) continue;
      int3 triangle = mesh->triangles[otherTriangleIdx];
      // If any of incident triangle includes a vertex used, abort
      if (atomicCAS(&indep[triangle.x], 1, 1)) return;
      if (atomicCAS(&indep[triangle.y], 1, 1)) return;
      if (atomicCAS(&indep[triangle.z], 1, 1)) return;
    }
    // Go on and check next header block
    headerIdx = header->cont;
  };

  // At this point we have checked all adjacent nodes.
  // It looks like nobody is using them. Let's take them.
  center[vertexId] = 1; // Current vertex is independence center
  headerIdx = vertexId;
  while (headerIdx >= 0) {
    IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
    int base = headerIdx * mesh->params.incidenceDataPerBlock;
    for (int i = 0; i < header->size; ++i) {
      int otherTriangleIdx = incData[base + i];
      if (!mesh->trianglesMask[otherTriangleIdx]) continue;
      int3 triangle = mesh->triangles[otherTriangleIdx];
      // If any of incident triangle includes a vertex used, abort
      atomicCAS(&indep[triangle.x], 0, 1);
      atomicCAS(&indep[triangle.y], 0, 1);
      atomicCAS(&indep[triangle.z], 0, 1);
    }
    // Go on and check next header block
    headerIdx = header->cont;
  };
}

// Device function that scans independence information
// and fixup any false independence region
// Assuming we have scanned and compacted all centers into mesh->intTemp;
// Note: This algorithm itself, unfortunately is problematic, because if
//       center A and center B reaches each other, we have 2 choices:
//       1) We remove one of them 2) We remove both of them.
//       If both of them are removed, we might end-up with too few nodes.
__global__ void cudaFixupSuperIndependence(CudaMesh* mesh) {
  // int centerIdx = blockDim.x * blockIdx.x + threadIdx.x;
  int centerIdx = primePermute(5737u, gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
  // centerIdx = (centerIdx * 5737) % (gridDim.x * blockDim.x);
  if (centerIdx >= mesh->nIndependenceCenters) return;
  // Centers must be valid!

  int* center = mesh->independenceCenterMask;
  int* data = mesh->incidenceData;
  int* centerList = mesh->independenceCenterList;

  // For each candidate center do
  int currentCenter = centerList[centerIdx];
  int headerIdx = currentCenter;
  do {
    IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
    int base = headerIdx * mesh->params.incidenceDataPerBlock;
    for (int i = 0; i < header->size; ++i) {
      // For every other triangle incident to center
      int otherTriangleIdx = data[base + i];
      if (!mesh->trianglesMask[otherTriangleIdx]) continue;
      int3 triangle = mesh->triangles[otherTriangleIdx];
      
      int vertices[] = {triangle.x, triangle.y, triangle.z};
      for (int j = 0; j < 3; ++j) {
        // If the vertex is the center, we do nothing
        int vertex = vertices[j];
        if (vertex == currentCenter) continue;
        // Traverse adjacent triangles of other nodes
        // through = vertex;
        int hIdx = vertex;
        do {
          IncidenceHeader* h = mesh->incidenceHeader + hIdx;
          int base = hIdx * mesh->params.incidenceDataPerBlock;
          for (int k = 0; k < h->size; ++k) {
            int triangleIdx = data[base + k];
            if (!mesh->trianglesMask[triangleIdx]) continue;
            int3 triangle = mesh->triangles[triangleIdx];
            // If any of the vertices are center, and that node is not current center
            if (triangle.x != currentCenter && center[triangle.x]) goto not_a_center;
            if (triangle.y != currentCenter && center[triangle.y]) goto not_a_center;
            if (triangle.z != currentCenter && center[triangle.z]) goto not_a_center;
          }
          hIdx = h->cont;
        } while(hIdx >= 0);
      }
    } 
    // Go on and check next header block */
    headerIdx = header->cont;
  } while(headerIdx >= 0);
  // If all other nodes checks out, we 
  return;
  // Current node is no a center
not_a_center:
  center[centerList[centerIdx]] = 0;
}

// Initialize the data structure
__global__ void cudaSimplifyInit(CudaMesh* mesh) {
  int nBlocks = gridDim.x;
  int nWorkers = blockDim.x * nBlocks;
  int workerId = blockDim.x * blockIdx.x + threadIdx.x;

#ifndef NDEBUG
  if (workerId == 0) {
    printf("Simplification initialization (%d, %d) (n = %d).\n", 
        nBlocks, blockDim.x, nWorkers);
  }
#endif

  int vPerWorker = (mesh->nVertices + nWorkers - 1) / nWorkers;
  for (int i = 0; i < vPerWorker; ++i) {
    int idx = vPerWorker * workerId + i;
    if (idx >= mesh->nVertices) break;
    mesh->verticesMask[idx] = 1;
  }

  int tPerWorker = (mesh->nTriangles + nWorkers - 1) / nWorkers;
  for (int i = 0; i < tPerWorker; ++i) {
    int idx = tPerWorker * workerId + i;
    if (idx >= mesh->nTriangles) break;
    mesh->trianglesMask[idx] = 1;
  }

  // Sets size of incidence header to 0, cont to -1
  int nHeaders = mesh->params.incidenceHeaderSize;
  int headerPerWorker = (nHeaders + nWorkers - 1) / nWorkers;
  for (int i = 0; i < headerPerWorker; ++i) {
    int idx = headerPerWorker * workerId + i;
    if (idx >= nHeaders) break;
    mesh->incidenceHeader[idx].size = 0;
    mesh->incidenceHeader[idx].cont = -1;
  }

  // Initializes incidence data to magic number to faciliate debugging
  int bsize = mesh->params.incidenceDataPerBlock;
  int nData = mesh->params.incidenceHeaderSize * bsize;
  int incDataPerWorkder = (nHeaders + nWorkers - 1) / nWorkers;
  for (int i = 0; i < incDataPerWorkder; ++i) {
    int idx = incDataPerWorkder * nWorkers + i;
    if (idx >= nData) break;
    mesh->incidenceData[idx] = INCIDENCE_DATA_MAGIC;
    mesh->incidenceDataCompact[idx] = INCIDENCE_DATA_MAGIC;
  }
}


__global__ void cudaSimplify(CudaMesh* mesh) {
  int listId = blockDim.x * blockIdx.x + threadIdx.x;
  float carefulness = mesh->params.simplificationCarefulness;
  if (listId >= mesh->nIndependenceCenters * carefulness) return;
  // if (centerId > 10) return;

  int centerId = mesh->verticesIndex[listId];
  int vertexId = mesh->independenceCenterList[centerId];
  int vertexEdgeIdx = mesh->verticesEdgeToDelete[centerId];
  int* incData = mesh->incidenceData;
  int bsize = mesh->params.incidenceDataPerBlock;
  // printf("%d %d\n", vertexId, vertexEdgeIdx);

  int* trianglesMask = mesh->trianglesMask;
  int* verticesMask = mesh->verticesMask;

  int headerIdx = vertexId;
  int triangleIdx1 = vertexEdgeIdx;

  // Find the index of the other vertex
  int3 triangle1 = mesh->triangles[triangleIdx1];
  int otherVertexId = cudaFindEdgeOfTriangle(vertexId, triangle1);

  // Find the second triangle
  headerIdx = otherVertexId;
  int triangleIdx2 = -1;
  while (headerIdx >= 0) {
    IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
    int base = headerIdx * bsize;
    for (int i = 0; i < header->size; ++i) {
      int triIdx = incData[base + i];
      if (mesh->trianglesMask[triIdx] == 0) continue;
      int3& tri = mesh->triangles[triIdx];
      if (cudaFindEdgeOfTriangle(otherVertexId, tri) == vertexId) {
        triangleIdx2 = triIdx;
        goto found_triangle_2;
      }
    }
    // Go on and check next header block
    headerIdx = header->cont;
  }

found_triangle_2:
  // Temporary fix. What exactly happened?
  // if (vertexId == otherVertexId) return;
  // Mask deleted triangles and vertices
  trianglesMask[triangleIdx1] = 0;
  atomicSub(&mesh->nTrianglesRemaining, 1);
  if (triangleIdx2 != -1) {
    trianglesMask[triangleIdx2] = 0;
    atomicSub(&mesh->nTrianglesRemaining, 1);
  } 
// #ifndef NDEBUG
//   // else {
//   //   printf("Unable to find the other triangle. Not a surface mesh?\n");
//   // }
// #endif
  verticesMask[vertexId] = 0;
  // printf("Deleted triangles: (%d, %d)\nMerge vertexId %d into vertexId %d\n", 
  //         triangleIdx1, triangleIdx2, vertexId, otherVertexId);
  atomicSub(&mesh->nVerticesRemaining, 1);

  if (vertexId == otherVertexId) {
    printf("ERROR: TRYTING to delete an edge containing two identical vertices.\n");
  }

  // Move connectivity from vertex to otherVertex
  headerIdx = vertexId;
  while (headerIdx >= 0) {
    IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
    int base = headerIdx * bsize;
    for (int i = 0; i < header->size; ++i) {
      int triIdx = incData[base + i];
      if (trianglesMask[triIdx] == 0) continue;
      int3& tri = mesh->triangles[triIdx];
      // if (tri.x == vertexId) { 
      //   printf("Changed a triangle: %d (%d -> %d)\n", triIdx, tri.x, otherVertexId);
      //   tri.x = otherVertexId; 
      // }
      // if (tri.y == vertexId) {
      //   printf("Changed a triangle: %d (%d -> %d)\n", triIdx, tri.y, otherVertexId);
      //   tri.y = otherVertexId;
      // }
      // if (tri.z == vertexId) {
      //   printf("Changed a triangle: %d (%d -> %d)\n", triIdx, tri.z, otherVertexId);
      //   tri.z = otherVertexId;
      // }
      if (tri.x == vertexId) {tri.x = otherVertexId;}
      if (tri.y == vertexId) {tri.y = otherVertexId;}
      if (tri.z == vertexId) {tri.z = otherVertexId;}
      // mesh->triangles[triIdx] = tri;
    }
    // Go on and check next header block
    headerIdx = header->cont;
  }

  // Update incidence list
  headerIdx = vertexId;
  int headerIdx2 = otherVertexId;
  
  IncidenceHeader* header = nullptr;
  int idx = headerIdx2;
  do {
    header = mesh->incidenceHeader + idx;
    idx = header->cont;
  } while(idx >= 0);
  // This should work. No one says the list has to be fully populated
  header->cont = headerIdx;

  // Reposition the vertex
  float3 vertexNewPos = mesh->verticesNewPos[centerId];
  // if (centerId == 0) {
  //   printf("Old position: %f %f %f\n", mesh->vertices[otherVertexId].x, mesh->vertices[otherVertexId].y, mesh->vertices[otherVertexId].z);
  //   printf("New position: %f %f %f\n", vertexNewPos.x, vertexNewPos.y, vertexNewPos.z);
  // }
  mesh->vertices[otherVertexId] = vertexNewPos;

  // Update errors (quadrics)
  cudaAddQuadric(mesh->verticesQuadric + 13 * otherVertexId, mesh->verticesQuadric + 13 * vertexId);
}

// Compacts Incidence table. Removed deleted triangles
__global__ void cudaCompactIncidenceTable(CudaMesh* mesh) {
  int vertexId = blockDim.x * blockIdx.x + threadIdx.x; 
  // If a vertex is no longer alive, we may never compact it.
  if (!mesh->verticesMask[vertexId]) return;

  int bsize = mesh->params.incidenceDataPerBlock;

  // Where should we copy to?
  int toHeaderIdx = vertexId;
  auto* toHeader = mesh->incidenceHeaderCompact + toHeaderIdx;
  auto* toData   = mesh->incidenceDataCompact;
  int toBase = toHeaderIdx * bsize;
  int toHeaderDataIdx = 0;

  IncidenceHeader* fromHeader = nullptr;
  int* fromData = mesh->incidenceData;
  int fromHeaderIdx = vertexId;
  do {
    fromHeader = mesh->incidenceHeader + fromHeaderIdx;
    int base = fromHeaderIdx * bsize;
    for (int i = 0; i < fromHeader->size; ++i) {
      int triIdx = fromData[base + i];
      if (!mesh->trianglesMask[triIdx]) continue;
      // Triangle tirIdx needs to be copied
      if (toHeaderDataIdx == bsize) {
        // This target block is full
        toHeader->size = bsize;
        // Use original cont, no need to allocate
        int originalCont = mesh->incidenceHeader[toHeaderIdx].cont;
        toHeader->cont = originalCont;
        // Move on to next block
        toHeaderIdx = originalCont;
        toHeader = mesh->incidenceHeaderCompact + toHeaderIdx;
        toBase = toHeaderIdx * bsize;
        toHeaderDataIdx = 0;
      }
      toData[toBase + toHeaderDataIdx] = triIdx;
      toHeaderDataIdx++;
    }
    fromHeaderIdx = fromHeader->cont;
  } while(fromHeaderIdx >= 0);

  // Finalize by setting the one last block
  toHeader->cont = -1;
  toHeader->size = toHeaderDataIdx;
}

// Exchanges buffer. Must be run with one thread and one block
__global__ void cudaExchangeIncidenceBuffer(CudaMesh* mesh) {
  auto tHeader = mesh->incidenceHeader;
  mesh->incidenceHeader = mesh->incidenceHeaderCompact;
  mesh->incidenceHeaderCompact = tHeader;

  auto tData = mesh->incidenceData;
  mesh->incidenceData = mesh->incidenceDataCompact;
  mesh->incidenceDataCompact = tData;
}

void cudaMeshSimplify(GpuMesh *from, GpuMesh *to, GpuMeshSimplifyParams params) {
    assert(from != nullptr && to != nullptr);
    assert(to->nVertices >= from->nVertices);
    assert(to->nTriangles >= from->nTriangles);

    cudaDetect();

#if TIME
    FILE* fp = fopen("time.log", "w+");
    fprintf(fp, "Simplification Rate = %f.\n", params.simplificationRate);
    double last = currentSeconds();
    watchStart((char*)"GpuSimpify");
#endif


    cudaError_t errCode;

    CudaMesh mesh;
    mesh.params = std::move(params);
    mesh.nVertices = from->nVertices;
    mesh.nTriangles = to->nTriangles;
    mesh.nVerticesRemaining = mesh.nVertices;
    mesh.nTrianglesRemaining = mesh.nTriangles;

    /* Allocate device storage for vertices */
    cudaMalloc((void**)(&mesh.vertices), sizeof(float3) * mesh.nVertices);
    cudaMemcpy(mesh.vertices, from->vertices, sizeof(float3) * mesh.nVertices, cudaMemcpyHostToDevice);

    /* Allocatte device storage for triangles */
    cudaMalloc((void**)(&mesh.triangles), sizeof(int3) * mesh.nTriangles);
    cudaMemcpy(mesh.triangles, from->triangles, sizeof(float3) * mesh.nTriangles, cudaMemcpyHostToDevice);

    /* Allocatte device storage for quadrics and errors*/
    cudaMalloc((void**)(&mesh.trianglesQuadric), 13 * sizeof(float) * mesh.nTriangles);
    cudaMalloc((void**)(&mesh.verticesQuadric), 13 * sizeof(float) * mesh.nVertices);
    cudaMemset((void*)(mesh.verticesQuadric), 0, 13 * sizeof(float) * mesh.nVertices);
    cudaMalloc((void**)(&mesh.verticesQuadricBuffer), 26 * sizeof(float) * mesh.nVertices);

    /* Allocate device storage for two masks */
    cudaMalloc((void**)(&mesh.verticesMask), sizeof(int) * mesh.nTriangles);
    cudaMalloc((void**)(&mesh.trianglesMask), sizeof(int) * mesh.nTriangles);

    /* Allocate device storage for incidence information */
    int hSize = mesh.params.incidenceHeaderSize;
    int bSize = mesh.params.incidenceDataPerBlock;
    cudaMalloc((void**)(&mesh.incidenceHeader), sizeof(IncidenceHeader) * hSize);
    cudaMalloc((void**)(&mesh.incidenceData), sizeof(int) * hSize * bSize);
    cudaMalloc((void**)(&mesh.incidenceHeaderCompact), sizeof(IncidenceHeader) * hSize);
    cudaMalloc((void**)(&mesh.incidenceDataCompact), sizeof(int) * hSize * bSize);
    mesh.incidenceCount = mesh.nVertices; // These are list "headers"

    /* Allocate device storage for super-indepence regeion info */
    int po2nVertices = nextPow2(mesh.nVertices);
    cudaMalloc((void**)(&mesh.independenceUsed), sizeof(int) * mesh.nVertices);
    cudaMalloc((void**)(&mesh.independenceCenterMask), sizeof(int) * mesh.nVertices);
    // Needs to be a next power of two because it will be used in sorting
    cudaMalloc((void**)(&mesh.independenceCenterList), sizeof(int) * po2nVertices);
    int* indepScratch = nullptr;
    cudaMalloc((void**)&indepScratch, sizeof(int) * po2nVertices);

    /* Allocate device storage for per-independence space */
    cudaMalloc((void**)(&mesh.verticesIndex), sizeof(int) * mesh.nVertices);
    cudaMalloc((void**)(&mesh.verticesEdgeToDelete), sizeof(int) * mesh.nVertices);
    cudaMalloc((void**)(&mesh.verticesError), sizeof(float) * mesh.nVertices);
    cudaMalloc((void**)(&mesh.verticesNewPos), sizeof(float3) * mesh.nVertices);

    /* Allocate and copy arguments to device */
    CudaMesh* deviceArgs = nullptr;
    cudaMalloc((void**)(&deviceArgs), sizeof(CudaMesh));
    cudaMemcpy(deviceArgs, &mesh, sizeof(CudaMesh), cudaMemcpyHostToDevice);

#if TIME
    fprintf(fp, "Allocation completes in %.2lf.\n", timeDiff(currentSeconds(), last));
    watchTake((char*)"Allocation");
#endif

    int thPerBlock = params.threadsPerBlock;

    /* Initialization code */
    cudaSimplifyInit<<<20, thPerBlock>>>(deviceArgs);
#if TIME
    cudaDeviceSynchronize();
    fprintf(fp, "Initialization completes in %.2lf.\n", timeDiff(currentSeconds(), last));
    watchTake((char*)"Initialization");
#endif
#ifndef NDEBUG
    cudaDeviceSynchronize();
    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
      fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
    }
     
    // Verify initialization is successful
    verifyInitialization(from, deviceArgs);
#endif

    /* Compute incidence information */
    int nBlocks = (mesh.nTriangles + thPerBlock - 1) / thPerBlock;
    cudaComputeTriangleIncidence<<<nBlocks, thPerBlock>>>(deviceArgs);
#if TIME
    cudaDeviceSynchronize();
    fprintf(fp, "Incidence calculated in %.2lf.\n", timeDiff(currentSeconds(), last));
    watchTake((char*)"Build incidence");
#endif
#ifndef NDEBUG
    cudaDeviceSynchronize();
    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
      fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      abort();
    }

    /* Check the generated incidence data is coherent */
    verifyIncidenceData(from, deviceArgs);
    printf("Incidence Computed.\n");
#endif

    /* Compute triangle quadrics */ 
    nBlocks = (mesh.nTriangles + thPerBlock - 1) / thPerBlock;
    cudaComputeTriangleQuadric<<<nBlocks, thPerBlock>>>(deviceArgs);
    /* Compute Vertex Quadric */
    cudaComputeVertexQuadric<<<nBlocks, thPerBlock>>>(deviceArgs);

#if TIME
    cudaDeviceSynchronize();
    fprintf(fp, "Quadric calculated in %.2lf.\n", timeDiff(currentSeconds(), last));
#endif
#ifndef NDEBUG
    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
      fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
    }
#endif

    int round = 1;
    int nVerticesRemaining = mesh.nVertices;
#if TIME
    fputs("Round\tIndep\tError\tSort\tSimp\tRemains\n", fp);
    watchTake((char*)"Initial quadric");
#endif
    while(nVerticesRemaining > params.simplificationRate * mesh.nVertices) {
      last = currentSeconds();
#ifndef NDEBUG
      printf("====  Executing Round %d. =====\n", round);
      /* Check consistency of incidence table */
      verifyIncidenceDataInProgress(from, deviceArgs);
#endif
#if TIME
      fprintf(fp, "%d\t", round);
#endif

      /* Compute Vertex Quadric */
      //if (round > 1) cudaComputeVertexQuadric<<<nBlocks, thPerBlock>>>(deviceArgs);
#if TIME
      //cudaDeviceSynchronize();
      //fprintf(fp, "%.2lf\t", timeDiff(currentSeconds(), last));
#endif
#ifndef NDEBUG
      errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      }
#endif

      // Compute independence regeion
      int indepThPerBlock = params.threadsPerBlockIndep;
      nBlocks = (mesh.nVertices + indepThPerBlock - 1) / indepThPerBlock;
      int* deviceCountAddr = (int*)((uint8_t*)deviceArgs + ((uint8_t*)(&mesh.nIndependenceCenters) - (uint8_t*)(&mesh)));
      cudaMemset(mesh.independenceUsed, 0, sizeof(int) * mesh.nVertices);
      cudaMemset(mesh.independenceCenterMask, 0, sizeof(int) * mesh.nVertices);
      cudaComputeIndependence<<<nBlocks, indepThPerBlock>>>(deviceArgs);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      }
#endif
      compact_mask_to_list(mesh.independenceCenterMask, mesh.independenceCenterList, 
        indepScratch, mesh.nVertices, deviceCountAddr);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      }
#endif
      // May use one or two extra blocks. May be smaller block size would be better?
      cudaFixupSuperIndependence<<<nBlocks, indepThPerBlock>>>(deviceArgs);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      }
#endif
      compact_mask_to_list(mesh.independenceCenterMask, mesh.independenceCenterList, 
        indepScratch, mesh.nVertices, deviceCountAddr);
#if TIME
      cudaDeviceSynchronize();
      fprintf(fp, "%.2lf\t", timeDiff(currentSeconds(), last));
#endif
#ifndef NDEBUG
      cudaDeviceSynchronize();
      errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      }
      // Check if the super-independence information is correct
      verifySuperIndependence(from, deviceArgs);
#endif
    
      /* Compute Best Edge for each Region */
      int nIndepCenters = copyFieldFromGPU(&mesh, &mesh.nIndependenceCenters, deviceArgs);
      if (nIndepCenters * params.simplificationCarefulness <= 1) {
        printf("Error, found %d independence centers. Not enough for simplification!.\n", nIndepCenters);
        abort();
      }
      nBlocks = (nIndepCenters + thPerBlock - 1) / thPerBlock;
      cudaFindBestEdgeForZone<<<nBlocks, thPerBlock>>>(deviceArgs);
#if TIME
      cudaDeviceSynchronize();
      fprintf(fp, "%.2lf\t", timeDiff(currentSeconds(), last));
#endif
#ifndef NDEBUG
      cudaDeviceSynchronize();
      errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      }
#endif
    
      /* Sort the independence center according to their error */
      prepareIndiciesForSort(mesh.verticesIndex, nIndepCenters);
      cudaSortKV(mesh.verticesIndex, mesh.verticesError, nIndepCenters);
#if TIME
      cudaDeviceSynchronize();
      fprintf(fp, "%.2lf\t", timeDiff(currentSeconds(), last));
#endif
#ifndef NDEBUG
      cudaDeviceSynchronize();
      errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      }

      // Check if the sorting succeeds.
      // Interestingly, if the errors array is unitialized, then it 
      // contains (random) garbage data, perfect for testing.
      verifySorting(from, deviceArgs);
#endif

      /* Execute the simplification */
      // TODO: Insert code to determine number of block required
      cudaSimplify<<<nBlocks, thPerBlock>>>(deviceArgs);
#if TIME
      cudaDeviceSynchronize();
      fprintf(fp, "%.2lf\t", timeDiff(currentSeconds(), last));
#endif
#ifndef NDEBUG
      cudaDeviceSynchronize();
      errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
      }
#endif

      nVerticesRemaining = copyFieldFromGPU(&mesh, &mesh.nVerticesRemaining, deviceArgs);
#if TIME
      cudaDeviceSynchronize();
      fprintf(fp, "%d", nVerticesRemaining);
#endif
#ifndef NDEBUG
      printf("Current Vertices Remaining: %d\n", nVerticesRemaining);
#endif

      if (round % params.compactIncidenceInterval == 0 && round != 0) {
#ifndef NDEBUG
        printf("Compact at round: %d.\n", round);
        verifyIncidenceDataInProgress(from, deviceArgs);
#endif
        int nBlocks = (mesh.nVertices + thPerBlock - 1) / thPerBlock;
        cudaCompactIncidenceTable<<<nBlocks, thPerBlock>>>(deviceArgs);
        cudaExchangeIncidenceBuffer<<<1, 1>>>(deviceArgs);
#if TIME
        cudaDeviceSynchronize();
        fprintf(fp, "\t%.2lf", timeDiff(currentSeconds(), last));
#endif
#ifndef NDEBUG
        verifyIncidenceDataInProgress(from, deviceArgs);
#endif
      }

#if TIME
      fprintf(fp, "\n"); // Just a newline.
#endif

      round++;
    } 
#if TIME
    watchTake((char*)"Simplification");
    last = currentSeconds();
#endif

    /* Copy results metadata back from device */
    cudaMemcpy((void*)&mesh, deviceArgs, sizeof(CudaMesh), cudaMemcpyDeviceToHost);
    /* Copy results arrays from device */
    to->nVertices = mesh.nVertices;
    from->nTriangles = mesh.nTriangles;
    int* verticesMask = new int[mesh.nVertices];
    int* trianglesMask = new int[mesh.nTriangles];
    cudaMemcpy((void*)(to->triangles), mesh.triangles, sizeof(int3) * mesh.nTriangles, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)(to->vertices), mesh.vertices, sizeof(int3) * mesh.nVertices, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)(verticesMask), mesh.verticesMask, sizeof(int) * mesh.nVertices, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)(trianglesMask), mesh.trianglesMask, sizeof(int) * mesh.nTriangles, cudaMemcpyDeviceToHost);
    compactGpuMeshWithMask(to, verticesMask, trianglesMask);
#if TIME
    fprintf(fp, "Fetch and compact: %.2lf\n", timeDiff(currentSeconds(), last));
    watchTake((char*)"Fetch and compact.");
    watchReport();
#endif
#ifndef NDEBUG
    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
      fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
    }
#endif
    delete[] verticesMask;
    delete[] trianglesMask;


    cudaFree(deviceArgs);
    cudaFree(mesh.vertices);
    cudaFree(mesh.triangles);
    cudaFree(mesh.trianglesQuadric);
    cudaFree(mesh.verticesQuadricBuffer);
    cudaFree(mesh.verticesQuadric);
    cudaFree(mesh.verticesEdgeToDelete);
    cudaFree(mesh.verticesError);
    cudaFree(mesh.verticesNewPos);
    cudaFree(mesh.trianglesMask);
    cudaFree(mesh.verticesMask);
    cudaFree(mesh.incidenceHeader);
    cudaFree(mesh.incidenceData);
    cudaFree(mesh.independenceUsed);
    cudaFree(mesh.independenceCenterMask);
    cudaFree(mesh.independenceCenterList);
#if TIME
    fclose(fp);
#endif
}

void cudaDetect() {
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce GTX 1080") == 0) {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU) {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA GTX 1080.\n");
        printf("---------------------------------------------------------\n");
    }
}

void verifySuperIndependence(GpuMesh* meshRef, CudaMesh* deviceData) {
  CudaMesh localDeviceData;
  /* Copy results metadata back from device */
  cudaMemcpy((void*)&localDeviceData, deviceData, sizeof(CudaMesh), cudaMemcpyDeviceToHost);

  int nVertices = meshRef->nVertices;
  int nTriangles = meshRef->nTriangles;
  // int nTriangles = meshRef->nTriangles;
  int headerSize = localDeviceData.params.incidenceHeaderSize; 
  int bSize = localDeviceData.params.incidenceDataPerBlock;

  int* centerMask = new int[nVertices]; // Centers, read from device
  int* centerList = new int[nVertices]; // Centers, read from device
  int* verticesMask = new int[nVertices];  // read from device
  int* trianglesMask = new int[nTriangles];  // read from device
  i3Vec* triangles = meshRef->triangles; // list of triangles, use meshRef
  IncidenceHeader* incidenceHeader = new IncidenceHeader[headerSize]; // Incidence header, read from device
  int* incidenceData = new int[bSize * headerSize]; // Incidence data, read from device
  
  cudaMemcpy((void*)incidenceHeader, localDeviceData.incidenceHeader, sizeof(IncidenceHeader) * headerSize, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)incidenceData, localDeviceData.incidenceData, sizeof(int) * bSize * headerSize, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)centerMask, localDeviceData.independenceCenterMask, sizeof(int) * nVertices, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)centerList, localDeviceData.independenceCenterList, sizeof(int) * nVertices, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)verticesMask, localDeviceData.verticesMask, sizeof(int) * nVertices, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)trianglesMask, localDeviceData.trianglesMask, sizeof(int) * nTriangles, cudaMemcpyDeviceToHost);

  auto errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
    abort();
  }

  int nCentersDevice = localDeviceData.nIndependenceCenters;

  std::unordered_set<int> centerSet;
  for (int i = 0; i < nVertices; ++i) {
    if (centerMask[i]) centerSet.insert(i);
  }
  if (centerSet.size() != nCentersDevice) {
    printf("Error. Center mask count (%d) and nIndepCenters (%d) mismatch!\n", centerSet.size(), nCentersDevice);
  }

  auto removeIfContains = [&](std::unordered_set<int>& set, int val) -> bool {
    auto find = set.find(val);
    if (find != set.end()) {
      set.erase(find);
      return true;
    } else return false;
  };

  for (int i = 0; i < nCentersDevice; ++i) {
    int centerIdx = centerList[i];
    if (centerMask[centerIdx]) {
      int contains = removeIfContains(centerSet, centerList[i]);
      if (!contains) 
        printf("Error. %d (at %d) appeared more than once in the center list.\n", centerIdx, i);
    } else {
      printf("Error. %d is in center list but not marked as center. Duplicates?\n", centerIdx);
    }
  }

  assert(centerSet.empty());

  printf("Program found %d super-independent centers.\n", nCentersDevice);
  
  // Reconstruct adjacency information from incidence information
  std::unordered_map<int, std::unordered_set<int>> adjacency;
  for (int i = 0; i < nVertices; ++i) {
    if (!verticesMask[i]) continue;
    int headerIdx = i;
    do {
      IncidenceHeader* header = incidenceHeader + headerIdx;
      int base = headerIdx * bSize;
      for (int k = 0; k < header->size; ++k) {
        int triangleIdx = incidenceData[base + k];
        if (!trianglesMask[triangleIdx]) continue;
        auto triangle = triangles[triangleIdx];
        adjacency[triangle.x].insert(triangle.y);
        adjacency[triangle.x].insert(triangle.z);
        adjacency[triangle.y].insert(triangle.x);
        adjacency[triangle.y].insert(triangle.z);
        adjacency[triangle.z].insert(triangle.x);
        adjacency[triangle.z].insert(triangle.y);
      }
      headerIdx = header->cont;
    } while(headerIdx >= 0);
  }

  for (int i = 0; i < nVertices; ++i) {
    if (centerMask[i]) {
      auto& adjs = adjacency[i];
      if (adjs.size() == 0) {
        // printf("WARNING: Vertex %d seems to be dangling.\n", i);
      }
      for (auto adj : adjs) {
        if (centerMask[adj]) {
          printf("Error. Vertex %d adjacent to %d. Both are center.\n", i, adj);
          abort();
        }
        auto& secondAdjs = adjacency.at(adj);
        for (auto adj2 : secondAdjs) {
          if (adj2 != i && centerMask[adj2]) {
            printf("Error. Vertex %d collides with %d through %d.\n", i, adj2, adj);
            abort();
          }
        }
      }
    }
  }

  delete[] centerMask;
  delete[] centerList;
  delete[] verticesMask;
  delete[] trianglesMask;
  delete[] incidenceHeader;
  delete[] incidenceData;
}

// This function verifies that computed incidence data is consistent
void verifyIncidenceDataInProgress(GpuMesh* meshRef, CudaMesh* deviceData) {
  CudaMesh localDeviceData;
  /* Copy results metadata back from device */
  cudaMemcpy((void*)&localDeviceData, deviceData, sizeof(CudaMesh), cudaMemcpyDeviceToHost);

  int headerSize = localDeviceData.params.incidenceHeaderSize;
  int bSize = localDeviceData.params.incidenceDataPerBlock;
  IncidenceHeader* incidenceHeader = new IncidenceHeader[headerSize];
  int* incidenceData = new int[bSize * headerSize];
  int* verticesMask = new int[localDeviceData.nVertices];
  int* trianglesMask = new int[localDeviceData.nTriangles];
  int3* triangles = new int3[localDeviceData.nTriangles];

  cudaMemcpy((void*)incidenceHeader, localDeviceData.incidenceHeader, sizeof(IncidenceHeader) * headerSize, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)incidenceData, localDeviceData.incidenceData, sizeof(int) * bSize * headerSize, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)verticesMask, localDeviceData.verticesMask, sizeof(int) * localDeviceData.nVertices, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)trianglesMask, localDeviceData.trianglesMask, sizeof(int) * localDeviceData.nTriangles, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)triangles, localDeviceData.triangles, sizeof(int3) * localDeviceData.nTriangles, cudaMemcpyDeviceToHost);

  std::unordered_map<int, std::unordered_set<int>> incidence;
  std::map<std::pair<int, int>, std::unordered_set<int>> edgeCount;

  auto mpair = [](int x, int y) -> std::pair<int, int> {
    if (x < y) return std::make_pair(x, y);
    else return std::make_pair(y, x);
  };

  auto inRangeVert = [&](int vert) -> void {
    if (vert < 0 || vert >= localDeviceData.nVertices) {
      printf("Error. Edge %d out of range [0, %d].\n", vert, localDeviceData.nVertices);
    }
  };

  auto inRangeTri = [&](int tri) -> void {
    if (tri < 0 || tri >= localDeviceData.nTriangles) {
      printf("Error. Triangle %d out of range.\n", tri);
    }
  };

  // Check every triangle resides in the incidence of its nodes
  for (int i = 0; i < meshRef->nTriangles; ++i) {
    // If triangle has already been removed, skip it
    if (!trianglesMask[i]) continue;
    auto triangle = triangles[i];
    incidence[triangle.x].insert(i);
    incidence[triangle.y].insert(i);
    incidence[triangle.z].insert(i);
    
    inRangeVert(triangle.x);
    inRangeVert(triangle.y);
    inRangeVert(triangle.z);

    edgeCount[mpair(triangle.x, triangle.y)].insert(i);
    edgeCount[mpair(triangle.x, triangle.z)].insert(i);
    edgeCount[mpair(triangle.y, triangle.z)].insert(i);
  }

  for (auto& p : edgeCount) {
    if (p.second.size() != 2 && p.second.size() != 4) {
      auto idx1 = p.first.first;
      auto idx2 = p.first.second;
      // printf("Edge (%d, %d) shared by %d triangles.\n", idx1, idx2, p.second.size());
      // Well, got to admit it is indeed possible...
      // if (p.second.size() > 2) {
      //   // Should never be in a situation where an edge is shared by more than 2 triangles!
      //   printf("Triangles: ");
      //   for (auto tri : p.second) printf("%d ", tri);
      //   puts("");
      //   abort();
      // }
    }
  }

  for (int i = 0; i < meshRef->nVertices; ++i) {
    if (!verticesMask[i]) {
        if (incidence[i].size() != 0) {
          printf("Error. Deleted verteces in surviving triangle: %d\n", i);
          for (auto triangle : incidence[i]) {
            printf("Surviving triangle: %d.\n", triangle);
          }
          abort();
        }
        continue;
    }
    if (incidence[i].size() == 0) {
      // printf("WARNING. Dangling vertex %d does not adjacent to anyone.\n", i);
    }
    std::unordered_set<int> headerIncidence;
    int headerIdx = i;
    do {
      IncidenceHeader* header = incidenceHeader + headerIdx;
      int base = headerIdx * bSize;
      for (int k = 0; k < header->size; ++k) {
        int triangleIdx = incidenceData[base + k];
        if (!trianglesMask[triangleIdx]) continue;
        headerIncidence.insert(triangleIdx);
      }
      headerIdx = header->cont;
    } while(headerIdx >= 0);
    if (headerIncidence != incidence[i]) {
      printf("Error! Vertex %d's incidence triangle set mismatch.\n", i);
      for (auto &j : headerIncidence) {
        printf("Incidence:%d\n", j);
      }
      for (auto &j : incidence[i]) {
        printf("Triangle:%d\n", j);
      }
      // abort();
    }
  }

  delete[] verticesMask;
  delete[] trianglesMask;
  delete[] triangles;
  delete[] incidenceHeader;
  delete[] incidenceData;
}

// This function verifies that computed incidence data corresponds back to original data
void verifyIncidenceData(GpuMesh* meshRef, CudaMesh* deviceData) {
  CudaMesh localDeviceData;
  /* Copy results metadata back from device */
  cudaMemcpy((void*)&localDeviceData, deviceData, sizeof(CudaMesh), cudaMemcpyDeviceToHost);

  int headerSize = localDeviceData.params.incidenceHeaderSize;
  int bSize = localDeviceData.params.incidenceDataPerBlock;
  IncidenceHeader* incidenceHeader = new IncidenceHeader[headerSize];
  int* incidenceData = new int[bSize * headerSize];
  cudaMemcpy((void*)incidenceHeader, localDeviceData.incidenceHeader, sizeof(IncidenceHeader) * headerSize, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)incidenceData, localDeviceData.incidenceData, sizeof(int) * bSize * headerSize, cudaMemcpyDeviceToHost);

  printf("localDeviceData.incidenceCount: %d\n", localDeviceData.incidenceCount);

  // Check every triangle resides in the incidence of its nodes
  for (int i = 0; i < meshRef->nTriangles; ++i) {
    auto triangle = meshRef->triangles[i];
    int vertices[] = {triangle.x, triangle.y, triangle.z};
    for (int j = 0; j < 3; ++j) {
      int vertex = vertices[j];
      int headerIdx = vertex;
      do {
        IncidenceHeader* header = incidenceHeader + headerIdx;
        int base = headerIdx * bSize;
        if (header->size > bSize) {
          printf("Error. Size (%d) of header (vertex %d) is greater than bSize.\n", header->size, headerIdx);
        }
        for (int k = 0; k < header->size; ++k) {
          if (incidenceData[base + k] == i) goto nextVertex;
        }
        headerIdx = header->cont;
      } while (headerIdx >= 0);
      // Not found in the entire list
      printf("Error. Triangle %d is not found in list of its vertex %d.\n", i, vertex);
      abort();
    nextVertex:
      // Nothings needs to be done
      continue;
    }
  }

  // Check every vertex is in the triangle it listed as incidence
  for (int i = 0; i < meshRef->nVertices; ++i) {
    int headerIdx = i;
    do {
      IncidenceHeader* header = incidenceHeader + headerIdx;
      int base = headerIdx * bSize;
      for (int k = 0; k < header->size; ++k) {
        int triangleIdx = incidenceData[base + k];
        auto triangle = meshRef->triangles[triangleIdx];
        if (triangle.x == i || triangle.y == i || triangle.z == i) continue;
        printf("Error. Triangle %d in list of vertex %d, yet does not have such vertex.\n", triangleIdx, i);
      }
      headerIdx = header->cont;
    } while(headerIdx >= 0);
  }

  delete[] incidenceHeader;
  delete[] incidenceData;
}

// Verifies centers are properly sorted according to their respective size
void verifySorting(GpuMesh* meshRef, CudaMesh* deviceData) {
  CudaMesh localDeviceData;
  /* Copy results metadata back from device */
  cudaMemcpy((void*)&localDeviceData, deviceData, sizeof(CudaMesh), cudaMemcpyDeviceToHost);

  int nCenters = localDeviceData.nIndependenceCenters;
  int nVertices = localDeviceData.nVertices;
  int* centerList = new int[nCenters];
  float* centerErros = new float[nVertices];

  cudaMemcpy((void*)centerList, localDeviceData.verticesIndex, sizeof(int) * nCenters, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)centerErros, localDeviceData.verticesError, sizeof(float) * nVertices, cudaMemcpyDeviceToHost);
  auto errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
  }

  std::unordered_set<int> centers;
  for (int i = 0; i < nCenters; ++i) {
    if (centerList[i] < 0) {
      printf("WARNING: Sort mixed pseudo values.\n");
    } else {
      centers.insert(centerList[i]);
    }
  }
  
  if (centers.size() < nCenters) {
    printf("Collected center size = %d, nCenters = %d.\n", centers.size(), nCenters);
    printf("WARNING: There may be duplicated center item.\n");
  }

  // Verifies proper ordering of nodes
  for (int i = 0; i < nCenters - 1; ++i) {
    int idx = centerList[i];
    int idx2 = centerList[i + 1];
    if (centerErros[idx] > centerErros[idx2]) {
      printf("Error. Error of center %d (%f, idx = %d) is greater than %d (%f, idx = %d).\n", 
          i, idx, centerErros[idx], i + 1, idx2, centerErros[idx2]);
    }
  }

  printf("Verifying %d sorted centers complete!.\n", nCenters);

  delete[] centerList;
  delete[] centerErros;
}

void verifyInitialization(GpuMesh* meshRef, CudaMesh* deviceData) {
  CudaMesh localDeviceData;
  /* Copy results metadata back from device */
  cudaMemcpy((void*)&localDeviceData, deviceData, sizeof(CudaMesh), cudaMemcpyDeviceToHost);

  int nVertices = localDeviceData.nVertices;
  int nTriangles = localDeviceData.nTriangles;

  printf("Total vertices = %d, triangles = %d.\n", nVertices, nTriangles);

  int* trianglesMask = new int[localDeviceData.nTriangles];
  int* verticesMask = new int[localDeviceData.nVertices];
  cudaMemcpy((void*)verticesMask, localDeviceData.verticesMask, sizeof(int) * localDeviceData.nVertices , cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)trianglesMask, localDeviceData.trianglesMask, sizeof(int) * localDeviceData.nTriangles , cudaMemcpyDeviceToHost);

  auto errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "LINE %d: WARNING: A CUDA error occured: code=%d, %s\n", __LINE__, errCode, cudaGetErrorString(errCode));
  }

  for (int i = 0; i < meshRef->nTriangles; ++i) {
    if (!trianglesMask[i]) {
      printf("WARNING. Triangles %d not marked as enabled at beginning.\n", i);
      abort();
    }
  }

  for (int i = 0; i < meshRef->nVertices; ++i) {
    if (!verticesMask[i]) {
      printf("WARNING. Vertex %d not marked as enabled at beginning.\n", i);
      abort();
    }
  }

  delete[] trianglesMask;
  delete[] verticesMask;
}

// void buildIncidenceSequential(GpuMesh* meshRef, CudaMesh* mesh) {
//   int bsize = mesh->params.incidenceDataPerBlock;
//   for (int i = 0; i < mesh->nTriangles; ++i) {
//     i3Vec triangle = meshRef.triangles[i];
//     int vertices[] = {triangle.x, triangle.y, triangle.z};
//     for (int i = 0; i < 3; ++i) {
//       int headerIdx = vertices[i];
//       while(1) {
//         // Follow through the linked list to find latestNode
//         IncidenceHeader* header = mesh->incidenceHeader + headerIdx;
//         int nextHeader = header;
//         if (nextHeader >= 0) {
//           headerIdx = nextHeader;
//           continue;
//         }

//       do {
//         // May go over max size. We need to fix it later on
//         int place = atomicAdd(&header->size, 1);
//         if (place < bsize) {
//           size_t baseIdx = headerIdx * bsize;
//           mesh->incidenceData[baseIdx + place] = triangleId;
//           goto success;
//         } else {
//           // Restore the header->size
//           atomicMin(&header->size, bsize);
//           int nextBlock = atomicAdd(&mesh->incidenceCount, 1);
//           // printf("Incidence count becomes %d due to %d.\n", nextBlock, headerIdx);
//           if (nextBlock >= mesh->params.incidenceHeaderSize) {
//             printf("WARNING: Incidence header size (%d) exceeded!\n", nextBlock);
//           }
//           // Updade header->cont iff no one updated it before me
//           atomicCAS(&header->cont, 0, nextBlock);
//           goto failed;
//         }
//       } while(1);

//     failed:
//       continue;
//     success:
//       break;
//     }
//   }
// }
// }

// void cpuIndependenceSolver(CudaMesh* deviceArgs) {
//   CudaMesh localDeviceData;
//   /* Copy results metadata back from device */
//   cudaMemcpy((void*)&localDeviceData, deviceData, sizeof(CudaMesh), cudaMemcpyDeviceToHost);

//   int headerSize = localDeviceData.params.incidenceHeaderSize;
//   int bSize = localDeviceData.params.incidenceDataPerBlock;
//   int* verticesMask = new int[localDeviceData.nVertices];
//   int* trianglesMask = new int[localDeviceData.nTriangles];
//   int3* triangles = new int3[localDeviceData.nTriangles];

//   cudaMemcpy((void*)verticesMask, localDeviceData.verticesMask, sizeof(int) * localDeviceData.nVertices, cudaMemcpyDeviceToHost);
//   cudaMemcpy((void*)trianglesMask, localDeviceData.trianglesMask, sizeof(int) * localDeviceData.nTriangles, cudaMemcpyDeviceToHost);
//   cudaMemcpy((void*)triangles, localDeviceData.triangles, sizeof(int3) * localDeviceData.nTriangles, cudaMemcpyDeviceToHost);

//   std::unordered_map<int, std::unordered_set<int>> adjacency;

//   // auto mpair = [](int x, int y) -> std::pair<int, int> {
//   //   if (x < y) return std::make_pair(x, y);
//   //   else return std::make_pair(y, x);
//   // };

//   // Check every triangle resides in the incidence of its nodes
//   for (int i = 0; i < meshRef->nTriangles; ++i) {
//     // If triangle has already been removed, skip it
//     if (!trianglesMask[i]) continue;
//     auto triangle = triangles[i];
//     adjacency[triangle.x].insert(triangle.y);
//     adjacency[triangle.x].insert(triangle.z);
//     adjacency[triangle.y].insert(triangle.x);
//     adjacency[triangle.y].insert(triangle.z);
//     adjacency[triangle.z].insert(triangle.x);
//     adjacency[triangle.z].insert(triangle.y);
//   }
  
//   std::unordered_set<int> usedVerticies;
//   std::unordered_set<int> candidateCenters;
//   std::vector<int> centerList; 

//   double last = currentSeconds();

//   // Initialize candidate center with arbitrary vertice
//   candidateCenters.insert(adjacency.begin()->first);
//   while (candidateCenters.begin() != candidateCenters.end()) {
//     // While we have candidates, pop and remove
//     auto cand = *candidateCenters.begin();
//     candidateCenters.erase(candidateCenters.begin());
//     auto candAdjacent = adjacency[cand];
//     for (auto adjV : candAdjacent) {
//       if (usedVerticies.count(adjV)) {
//         goto not_a_candidate;
//       }
//     }
//     // We have a valid candidate
//     for (auto adjV : candAdjacent) {
//       usedVerticies.insert(adjV);
//       usedVerticies.insert(cand);
//     }
//     centerList.push_back(cand);
//   not a candidate:
//     continue;
//   }

//   printf("Found %d centers in %.2lf us.\n", 
//       centerList.size(), timeDiff(currentSeconds(), last));
// }

