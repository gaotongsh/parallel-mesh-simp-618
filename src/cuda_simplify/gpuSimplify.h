#ifndef SCOTTY3D_GPUSIMPLIFY_H
#define SCOTTY3D_GPUSIMPLIFY_H

#include <vector>
#include "../vector3D.h"
using std::vector;
using CMU462::Vector3D;

struct f3Vec {
  float x, y, z;
};

struct i3Vec {
  int x, y, z;
};

struct GpuMesh {
  int nVertices;
  f3Vec* vertices;

  int nTriangles;
  i3Vec* triangles;
};

struct GpuMeshSimplifyParams {
  float simplificationRate;
  int threadsPerBlock;
  int threadsPerBlockIndep;

  size_t incidenceHeaderSize;
  size_t incidenceDataPerBlock;
  
  float simplificationCarefulness;

  int compactIncidenceInterval;

public:
  GpuMeshSimplifyParams() :
    simplificationRate(0.5),
    simplificationCarefulness(0.1),
    threadsPerBlock(128),
    threadsPerBlockIndep(4),
    incidenceHeaderSize(30000),
    incidenceDataPerBlock(10),
    compactIncidenceInterval(10)
  { }
};

// This function builds a mesh from the given data structures.
GpuMesh buildCudaMesh(vector<vector<size_t>> &triangles, vector<Vector3D> &vertexPositions);

// This function reads mesh from the first operand, peform simplification,
// then exports results into the second argument. The second argument is required
// to be at least as large as the first argument.
void cudaMeshSimplify(GpuMesh* from, GpuMesh* to, GpuMeshSimplifyParams params);

// Generate some fake data to test the functionality
void testCudaMeshSimplify(int size);

// Use real data to test functionality
void testCudaMeshSimplify(vector<vector<size_t>> &triangles, vector<Vector3D> &vertexPositions);

// Actually do the simplify
void doCudaMeshSimplify(vector<vector<size_t>> &triangles, vector<Vector3D> &vertexPositions,
        vector<vector<size_t>> &triangles_out, vector<Vector3D> &vertexPositions_out, float simpRate);

// Compact the GpuMesh data structure by removing unused 
void compactGpuMeshWithMask(GpuMesh* dest, int* verticesMask, int* triangleMask);

#endif //SCOTTY3D_GPUSIMPLIFY_H
