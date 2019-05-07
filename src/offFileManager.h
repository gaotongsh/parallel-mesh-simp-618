//
// Created by Tong Gao on 2019-04-27.
//

#ifndef SCOTTY3D_OFFFILEMANAGER_H
#define SCOTTY3D_OFFFILEMANAGER_H

#include <string>
#include <vector>
#include "vector3D.h"

using namespace std;
using namespace CMU462;

int read_off(string &filename, vector<vector<size_t>> &polygons, vector<Vector3D> &vertexPositions);

int write_off(const string &filename, const vector<vector<size_t>> &polygons, const vector<Vector3D> &vertexPositions);

#endif //SCOTTY3D_OFFFILEMANAGER_H
