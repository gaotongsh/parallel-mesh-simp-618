#include <cassert>
#include <cstdio>
#include <unordered_map>

#include "gpuSimplify.h"

static void assertFloatNear(float x, float y) {
  assert(x - y <= 0.001f);
}

GpuMesh buildCudaMesh(vector<vector<size_t>> &triangles, vector<Vector3D> &vertexPositions) {
  GpuMesh mesh;
  mesh.nVertices = vertexPositions.size();
  mesh.nTriangles = triangles.size();

  mesh.vertices = new f3Vec[mesh.nVertices];
  mesh.triangles = new i3Vec[mesh.nTriangles];

  for (int i = 0; i < mesh.nVertices; ++i) {
    mesh.vertices[i].x = vertexPositions[i].x;
    mesh.vertices[i].y = vertexPositions[i].y;
    mesh.vertices[i].z = vertexPositions[i].z;
  }

  for (int i = 0; i < mesh.nTriangles; ++i) {
    mesh.triangles[i].x = triangles[i][0];
    mesh.triangles[i].y = triangles[i][1];
    mesh.triangles[i].z = triangles[i][2];
  }

  return mesh;
}

// Generates a group of random data to test the function
// Also serves as an example of how to use the function
void testCudaMeshSimplify(int size) {
  GpuMesh meshIn;
  meshIn.nVertices = size;
  meshIn.nTriangles = size / 3;
  meshIn.vertices = new f3Vec[meshIn.nVertices];
  meshIn.triangles = new i3Vec[meshIn.nTriangles];

  for (int i = 0; i < meshIn.nVertices; ++i) {
    meshIn.vertices[i].x = i;
    meshIn.vertices[i].y = i + 0.33f;
    meshIn.vertices[i].z = i + 0.66f;
  }

  for (int i = 0; i < meshIn.nTriangles; ++i) {
    meshIn.triangles[i].x = 3 * i;
    meshIn.triangles[i].y = 3 * i + 1;
    meshIn.triangles[i].z = 3 * i + 2;
  }

  GpuMesh meshOut;
  meshOut.nVertices = meshIn.nVertices;
  meshOut.nTriangles = meshIn.nTriangles;
  meshOut.vertices = new f3Vec[meshOut.nVertices];
  meshOut.triangles = new i3Vec[meshOut.nTriangles];

  cudaMeshSimplify(&meshIn, &meshOut, GpuMeshSimplifyParams());

  for (int i = 0; i < meshIn.nVertices; ++i) {assertFloatNear(meshOut.vertices[i].x, i);
    assertFloatNear(meshOut.vertices[i].y, i + 0.33f);
    assertFloatNear(meshOut.vertices[i].z, i + 0.66f);
  }

  for (int i = 0; i < meshIn.nTriangles; ++i) {
    assert(meshOut.triangles[i].x == 3 * i);
    assert(meshOut.triangles[i].y == 3 * i + 1);
    assert(meshOut.triangles[i].z == 3 * i + 2);
  }

  delete[] meshIn.vertices;
  delete[] meshIn.triangles;
  delete[] meshOut.vertices;
  delete[] meshOut.triangles;

  printf("Test success.\n");
}

void testCudaMeshSimplify(vector<vector<size_t>> &triangles, vector<Vector3D> &vertexPositions) {
  GpuMesh meshIn = buildCudaMesh(triangles, vertexPositions);

  GpuMesh meshOut;
  meshOut.nVertices = meshIn.nVertices;
  meshOut.nTriangles = meshIn.nTriangles;
  meshOut.vertices = new f3Vec[meshOut.nVertices];
  meshOut.triangles = new i3Vec[meshOut.nTriangles];

  GpuMeshSimplifyParams params;
  params.incidenceHeaderSize = meshIn.nVertices * 50;
  cudaMeshSimplify(&meshIn, &meshOut, params);

  for (int i = 0; i < meshIn.nVertices; ++i) {
    assertFloatNear(meshOut.vertices[i].x, vertexPositions[i].x);
    assertFloatNear(meshOut.vertices[i].y, vertexPositions[i].y);
    assertFloatNear(meshOut.vertices[i].z, vertexPositions[i].z);
  }

  for (int i = 0; i < meshIn.nTriangles; ++i) {
    assert(meshOut.triangles[i].x == triangles[i][0]);
    assert(meshOut.triangles[i].y == triangles[i][1]);
    assert(meshOut.triangles[i].z == triangles[i][2]);
  }

  delete[] meshIn.vertices;
  delete[] meshIn.triangles;
  delete[] meshOut.vertices;
  delete[] meshOut.triangles;

  printf("Test success.\n");
}

void doCudaMeshSimplify(
  vector<vector<size_t>> &triangles, vector<Vector3D> &vertexPositions,
  vector<vector<size_t>> &triangles_out, vector<Vector3D> &vertexPositions_out, float simpRate) {
  GpuMesh meshIn = buildCudaMesh(triangles, vertexPositions);

  GpuMesh meshOut;
  meshOut.nVertices = meshIn.nVertices;
  meshOut.nTriangles = meshIn.nTriangles;
  meshOut.vertices = new f3Vec[meshOut.nVertices];
  meshOut.triangles = new i3Vec[meshOut.nTriangles];

  GpuMeshSimplifyParams params;
  params.incidenceHeaderSize = meshIn.nVertices * 50;
  params.simplificationRate = simpRate;
  cudaMeshSimplify(&meshIn, &meshOut, params);

  triangles_out.clear();
  vertexPositions_out.clear();

  for (int i = 0; i < meshOut.nVertices; ++i) {
    auto pos = meshOut.vertices[i];
    vertexPositions_out.emplace_back(pos.x, pos.y, pos.z);
  }

  for (int i = 0; i < meshOut.nTriangles; ++i) {
    auto tri = meshOut.triangles[i];
    vector<size_t> v = {tri.x, tri.y, tri.z};
    triangles_out.push_back(v);
  }

  delete[] meshIn.vertices;
  delete[] meshIn.triangles;
  delete[] meshOut.vertices;
  delete[] meshOut.triangles;
}

template<typename T> 
std::pair<int, std::unordered_map<int, int>>
compactArrayWithMask(T* data, int* mask, int length) {
  std::unordered_map<int, int> movementMap; 
  int numValid = 0;
  int currLoc = 0, currFrom = length - 1;
  while (currLoc <= currFrom) {
    // printf("currLoc, currFrom = %d, %d\n", currLoc, currFrom);
    if (!mask[currFrom]) {
      currFrom -= 1;
      continue;
    }
    // Now currFrom is at pointing at 
    if (mask[currLoc]) {
      numValid += 1;
      currLoc += 1;
      continue;
    }
    if (mask[currLoc] || !mask[currFrom]) {
      printf("Error. Moving from %d (%d) to %d (%d).\n", 
          currFrom, currLoc, mask[currFrom], mask[currLoc]);
      abort();
    }
    data[currLoc] = data[currFrom];
    movementMap[currFrom] = currLoc;
    numValid += 1;
    currLoc += 1;
    currFrom -= 1;
  }

  printf("Total valid %d valid elements, %d elements moved.\n", numValid, movementMap.size());
  for (auto p : movementMap) {
    assert(mask[p.first] && !mask[p.second]);
  }

  return std::make_pair(numValid, movementMap);
}

void compactGpuMeshWithMask(GpuMesh* dest, int* verticesMask, int* trianglesMask) {
  auto vertCompact = 
      compactArrayWithMask(dest->vertices, verticesMask, dest->nVertices);
  auto triCompact =
      compactArrayWithMask(dest->triangles, trianglesMask, dest->nTriangles);

  int nVert = vertCompact.first, nTri = triCompact.first;
  auto& verticesMoved = vertCompact.second;
  auto& trianglesMoved = triCompact.second;

  printf("Remaining %d vertices, %d triangles.\n", nVert, nTri);

  dest->nVertices = nVert;
  dest->nTriangles = nTri;

  auto update = [&](int& vertex) -> void {
    if (vertex < nVert) return;
    auto it = verticesMoved.find(vertex);
    if (it == verticesMoved.end()) {
      printf("Error. Vertex not moved, yet larger than nVerticies (%d > %d).\n"
          ,vertex ,nVert);
      abort();
    }
    // Replace vertex with the move target
    vertex = it->second;
  };

  for (int i = 0; i < nTri; ++i) {
    auto& triangle = dest->triangles[i];
    update(triangle.x);
    update(triangle.y);
    update(triangle.z);
  }
}
